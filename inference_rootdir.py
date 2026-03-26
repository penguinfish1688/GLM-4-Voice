#!/usr/bin/env python3
"""
Batch inference for GLM-4-Voice on dataset layout: root_dir/*/input.wav.

This script reads each input wav and writes output wav in the same folder.
It reuses GLM-4-Voice's speech tokenizer + decoder and calls model_server.py
for streamed token generation.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
from glob import glob
from pathlib import Path
from typing import Any, Iterable, List
from urllib.parse import urlparse
import uuid

import requests
import torch
import torchaudio
from transformers import AutoTokenizer, WhisperFeatureExtractor

from flow_inference import AudioDecoder
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("glm4voice_rootdir_inference")
    parser.add_argument("--root-dir", required=True, help="Dataset root containing */input.wav")
    parser.add_argument("--prefix", default="", help="Input/output filename prefix. Example: clean_")
    parser.add_argument("--input-name", default="input.wav", help="Input wav filename")
    parser.add_argument("--output-name", default="output.wav", help="Output wav filename")
    parser.add_argument("--hidden-name", default="output_hidden.pt", help="Hidden payload filename")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output wav")
    parser.add_argument("--save-hidden", action="store_true", help="Save hidden payload next to output wav")
    parser.add_argument(
        "--inference-with-steering",
        action="store_true",
        help="Read root_dir/*/steering_vector.json and run steering inference",
    )
    parser.add_argument(
        "--inject-layer",
        type=int,
        default=None,
        help="Layer index for steering injection (required with --inference-with-steering)",
    )

    parser.add_argument("--model-path", default="THUDM/glm-4-voice-9b", help="GLM model path")
    parser.add_argument("--tokenizer-path", default="THUDM/glm-4-voice-tokenizer", help="Speech tokenizer path")
    parser.add_argument("--flow-path", default="./glm-4-voice-decoder", help="Decoder checkpoint directory")
    parser.add_argument("--server-url", default="http://localhost:10000/generate_stream", help="Model server endpoint")
    parser.add_argument(
        "--server-hidden-url",
        default="http://localhost:10000/generate_with_hidden",
        help="Model server endpoint returning token ids and hidden states",
    )
    parser.add_argument(
        "--server-steering-url",
        default="http://localhost:10000/generate_with_steering",
        help="Model server endpoint for steering inference",
    )

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument(
        "--system-prompt",
        default=(
            "User will provide you with a speech instruction. Do it step by step. "
            "First, think about the instruction and respond in a interleaved manner, "
            "with 13 text token followed by 26 audio tokens."
        ),
    )
    parser.add_argument("--device", default="cuda", help="Inference device for tokenizer/decoder")
    return parser.parse_args()


def list_input_files(root_dir: str, prefix: str, input_name: str) -> List[Path]:
    root = Path(root_dir).expanduser()
    pattern = str(root / f"*/{prefix}{input_name}")
    return [Path(p) for p in sorted(glob(pattern))]


def assert_server_reachable(endpoint_url: str, timeout_sec: float = 3.0) -> None:
    parsed = urlparse(endpoint_url)
    host = parsed.hostname
    port = parsed.port

    if host is None:
        raise RuntimeError(f"Invalid endpoint URL: {endpoint_url}")

    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec):
            return
    except OSError as exc:
        raise RuntimeError(
            "Cannot connect to GLM model server at "
            f"{host}:{port} (from {endpoint_url}). "
            "Start model_server.py first, for example: "
            "python model_server.py --host localhost --port 10000 --model-path THUDM/glm-4-voice-9b --dtype bfloat16 --device cuda:0"
        ) from exc


def iter_token_ids(response: requests.Response) -> Iterable[int]:
    for raw in response.iter_lines():
        if not raw:
            continue
        payload = json.loads(raw)
        token_id = payload.get("token_id")
        if token_id is None:
            continue
        yield int(token_id)


def build_prompt(audio_path: Path, whisper_model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor) -> str:
    audio_tokens = extract_speech_token(whisper_model, feature_extractor, [str(audio_path)])[0]
    if len(audio_tokens) == 0:
        raise RuntimeError(f"No audio tokens extracted from: {audio_path}")

    token_str = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    return "<|begin_of_audio|>" + token_str + "<|end_of_audio|>"


def generate_audio(
    prompt: str,
    args: argparse.Namespace,
    glm_tokenizer: Any,
    audio_decoder: AudioDecoder,
) -> torch.Tensor:
    inputs = f"<|system|>\n{args.system_prompt}<|user|>\n{prompt}<|assistant|>streaming_transcription\n"

    with torch.no_grad():
        response = requests.post(
            args.server_url,
            data=json.dumps(
                {
                    "prompt": inputs,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                }
            ),
            stream=True,
            timeout=600,
        )
        response.raise_for_status()

        audio_offset = glm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
        end_token_id = glm_tokenizer.convert_tokens_to_ids("<|user|>")

        audio_tokens: List[int] = []
        tts_speechs: List[torch.Tensor] = []
        tts_mels: List[torch.Tensor] = []
        prompt_speech_feat = torch.zeros(1, 0, 80, device=args.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64, device=args.device)

        stream_uuid = str(uuid.uuid4())
        is_finalize = False
        block_size_list = [25, 50, 100, 150, 200]
        block_size_idx = 0
        block_size = block_size_list[block_size_idx]

        for token_id in iter_token_ids(response):
            if token_id == end_token_id:
                is_finalize = True

            if token_id >= audio_offset and not is_finalize:
                audio_tokens.append(token_id - audio_offset)

            if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                if block_size_idx < len(block_size_list) - 1:
                    block_size_idx += 1
                    block_size = block_size_list[block_size_idx]

                tts_token = torch.tensor(audio_tokens, device=args.device).unsqueeze(0)

                if tts_mels:
                    prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                tts_speech, tts_mel = audio_decoder.token2wav(
                    tts_token,
                    uuid=stream_uuid,
                    prompt_token=flow_prompt_speech_token,
                    prompt_feat=prompt_speech_feat,
                    finalize=is_finalize,
                )
                tts_speechs.append(tts_speech.squeeze())
                tts_mels.append(tts_mel)
                flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                audio_tokens = []

        if not tts_speechs:
            raise RuntimeError("Model returned no audio tokens for this sample")

        return torch.cat(tts_speechs, dim=-1).cpu()


def generate_tokens_and_hidden(prompt: str, args: argparse.Namespace) -> tuple[List[int], torch.Tensor]:
    inputs = f"<|system|>\n{args.system_prompt}<|user|>\n{prompt}<|assistant|>streaming_transcription\n"
    response = requests.post(
        args.server_hidden_url,
        json={
            "prompt": inputs,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()

    if int(payload.get("error_code", 1)) != 0:
        raise RuntimeError(f"generate_with_hidden failed: {payload}")

    token_ids = payload.get("token_ids", [])
    if not isinstance(token_ids, list):
        raise RuntimeError("Invalid response: token_ids is not a list")

    hidden_layers_shape = payload.get("hidden_layers_shape", None)
    hidden_layers_b64 = payload.get("hidden_layers_b64", "")

    if isinstance(hidden_layers_shape, list) and len(hidden_layers_shape) == 3:
        raw = base64.b64decode(hidden_layers_b64) if hidden_layers_b64 else b""
        if hidden_layers_shape[0] == 0:
            hidden_states = torch.empty((0, 0, 0), dtype=torch.float16)
        else:
            hidden_states = (
                torch.frombuffer(raw, dtype=torch.float16)
                .clone()
                .reshape(hidden_layers_shape[0], hidden_layers_shape[1], hidden_layers_shape[2])
            )
    else:
        # Backward compatibility for older server payloads that only return last-layer hidden.
        hidden_shape = payload.get("hidden_shape", [0, 0])
        if not isinstance(hidden_shape, list) or len(hidden_shape) != 2:
            raise RuntimeError("Invalid response: hidden_shape/hidden_layers_shape")

        hidden_b64 = payload.get("hidden_b64", "")
        raw = base64.b64decode(hidden_b64) if hidden_b64 else b""
        if hidden_shape[0] == 0:
            hidden_states = torch.empty((0, 0, 0), dtype=torch.float16)
        else:
            hidden_2d = torch.frombuffer(raw, dtype=torch.float16).clone().reshape(hidden_shape[0], hidden_shape[1])
            hidden_states = hidden_2d.unsqueeze(1)

    return [int(x) for x in token_ids], hidden_states


def decode_audio_from_token_ids(
    token_ids: List[int],
    glm_tokenizer: Any,
    audio_decoder: AudioDecoder,
    device: str,
) -> torch.Tensor:
    audio_offset = int(glm_tokenizer.convert_tokens_to_ids("<|audio_0|>"))
    end_token_id = int(glm_tokenizer.convert_tokens_to_ids("<|user|>"))

    audio_tokens: List[int] = []
    for token_id in token_ids:
        if token_id == end_token_id:
            break
        if token_id >= audio_offset:
            audio_tokens.append(int(token_id - audio_offset))

    if not audio_tokens:
        raise RuntimeError("No audio tokens returned from model")

    token_tensor = torch.tensor(audio_tokens, dtype=torch.int64, device=device).unsqueeze(0)
    return audio_decoder.offline_inference(token_tensor).squeeze(0).cpu()


def save_hidden_payload(
    hidden_path: Path,
    input_path: Path,
    output_path: Path,
    token_ids: List[int],
    hidden_states: torch.Tensor,
    glm_tokenizer: Any,
) -> None:
    out_token_ids = torch.tensor(token_ids, dtype=torch.long).view(-1, 1)
    in_token_ids = torch.full_like(out_token_ids, fill_value=-1)
    if out_token_ids.shape[0] > 1:
        in_token_ids[1:, 0] = out_token_ids[:-1, 0]

    if hidden_states.dim() == 2:
        text_hidden_layers = hidden_states.unsqueeze(1)
    elif hidden_states.dim() == 3:
        text_hidden_layers = hidden_states
    else:
        raise RuntimeError(f"Expected hidden_states with dim 2 or 3, got {hidden_states.dim()}")

    t = int(text_hidden_layers.shape[0])
    frame_rate_hz = 12.5
    times = torch.arange(t, dtype=torch.float32) / frame_rate_hz
    token_time_ranges = torch.stack([times, times + (1.0 / frame_rate_hz)], dim=1) if t > 0 else torch.empty((0, 2))

    if text_hidden_layers.shape[1] > 0:
        last_hidden = text_hidden_layers[:, -1, :]
    else:
        last_hidden = torch.empty((t, 0), dtype=text_hidden_layers.dtype)

    payload = {
        "schema_version": 1,
        "input_wav": str(input_path),
        "output_wav": str(output_path),
        "frame_rate": float(frame_rate_hz),
        "token_ids": out_token_ids.view(-1),
        "token_names": glm_tokenizer.convert_ids_to_tokens(token_ids),
        "input_token_ids": in_token_ids,
        "input_token_width": 1,
        "output_token_ids": out_token_ids,
        "output_token_width": 1,
        "times": times,
        "token_time_ranges_sec": token_time_ranges,
        # Keep legacy key as last-layer hidden for downstream compatibility.
        "hidden_states": last_hidden.float().cpu(),
        "text_hidden_layers": text_hidden_layers.float().cpu(),
    }
    torch.save(payload, hidden_path)


def load_steering_map(sample_dir: Path, inject_layer: int) -> dict[str, list[float] | None]:
    steering_path = sample_dir / "steering_vector.json"
    if not steering_path.exists():
        raise FileNotFoundError(f"Missing steering file: {steering_path}")

    payload = json.loads(steering_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid steering payload in {steering_path}: expected dict")

    layer_key = f"layer_{inject_layer}"
    if layer_key not in payload:
        keys = ", ".join(sorted(payload.keys()))
        raise KeyError(f"{steering_path} has no '{layer_key}'. Available keys: [{keys}]")

    layer_payload = payload[layer_key]
    if not isinstance(layer_payload, dict):
        raise RuntimeError(f"Invalid {layer_key} payload in {steering_path}: expected dict")
    return layer_payload


def generate_tokens_with_steering(
    prompt: str,
    args: argparse.Namespace,
    inject_layer: int,
    steering_map: dict[str, list[float] | None],
) -> tuple[List[int], torch.Tensor | None]:
    inputs = f"<|system|>\n{args.system_prompt}<|user|>\n{prompt}<|assistant|>streaming_transcription\n"

    response = requests.post(
        args.server_steering_url,
        json={
            "prompt": inputs,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "inject_layer": inject_layer,
            "steering_map": steering_map,
            "return_hidden": bool(args.save_hidden),
        },
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()

    if int(payload.get("error_code", 1)) != 0:
        raise RuntimeError(f"generate_with_steering failed: {payload}")

    token_ids = payload.get("token_ids", [])
    if not isinstance(token_ids, list):
        raise RuntimeError("Invalid response: token_ids is not a list")

    hidden_states: torch.Tensor | None = None
    if args.save_hidden:
        hidden_layers_shape = payload.get("hidden_layers_shape", None)
        hidden_layers_b64 = payload.get("hidden_layers_b64", "")

        if isinstance(hidden_layers_shape, list) and len(hidden_layers_shape) == 3:
            raw = base64.b64decode(hidden_layers_b64) if hidden_layers_b64 else b""
            if hidden_layers_shape[0] == 0:
                hidden_states = torch.empty((0, 0, 0), dtype=torch.float16)
            else:
                hidden_states = (
                    torch.frombuffer(raw, dtype=torch.float16)
                    .clone()
                    .reshape(hidden_layers_shape[0], hidden_layers_shape[1], hidden_layers_shape[2])
                )
        else:
            hidden_shape = payload.get("hidden_shape", [0, 0])
            if not isinstance(hidden_shape, list) or len(hidden_shape) != 2:
                raise RuntimeError("Invalid response: hidden_shape/hidden_layers_shape")

            hidden_b64 = payload.get("hidden_b64", "")
            raw = base64.b64decode(hidden_b64) if hidden_b64 else b""
            if hidden_shape[0] == 0:
                hidden_states = torch.empty((0, 0, 0), dtype=torch.float16)
            else:
                hidden_2d = torch.frombuffer(raw, dtype=torch.float16).clone().reshape(hidden_shape[0], hidden_shape[1])
                hidden_states = hidden_2d.unsqueeze(1)

    return [int(x) for x in token_ids], hidden_states


def main() -> None:
    args = parse_args()

    if args.inference_with_steering and args.inject_layer is None:
        raise ValueError("--inference-with-steering requires --inject-layer")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but not available")

    required_endpoint = args.server_url
    if args.inference_with_steering:
        required_endpoint = args.server_steering_url
    elif args.save_hidden:
        required_endpoint = args.server_hidden_url

    print(f"[CHECK] verifying model server: {required_endpoint}")
    assert_server_reachable(required_endpoint)

    flow_config = os.path.join(args.flow_path, "config.yaml")
    flow_checkpoint = os.path.join(args.flow_path, "flow.pt")
    hift_checkpoint = os.path.join(args.flow_path, "hift.pt")

    print("[INIT] loading tokenizer and decoder...")
    glm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(args.device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)
    audio_decoder = AudioDecoder(
        config_path=flow_config,
        flow_ckpt_path=flow_checkpoint,
        hift_ckpt_path=hift_checkpoint,
        device=args.device,
    )

    inputs = list_input_files(args.root_dir, args.prefix, args.input_name)
    print(f"[SCAN] found {len(inputs)} files")
    if not inputs:
        return

    success = 0
    failed = 0

    for input_path in inputs:
        output_path = input_path.with_name(re.sub(r"input\\.wav$", args.output_name, input_path.name))
        if output_path == input_path:
            output_path = input_path.with_name(f"{args.prefix}{args.output_name}")
        hidden_path = output_path.with_name(output_path.stem + "_hidden.pt")
        if args.hidden_name != "output_hidden.pt":
            hidden_path = output_path.with_name(args.hidden_name)

        if output_path.exists() and (not args.save_hidden or hidden_path.exists()) and not args.overwrite:
            print(f"[SKIP] {output_path}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {input_path}")

        try:
            prompt = build_prompt(input_path, whisper_model, feature_extractor)
            if args.inference_with_steering:
                steering_map = load_steering_map(input_path.parent, int(args.inject_layer))
                token_ids, hidden_states = generate_tokens_with_steering(
                    prompt=prompt,
                    args=args,
                    inject_layer=int(args.inject_layer),
                    steering_map=steering_map,
                )
                tts_speech = decode_audio_from_token_ids(token_ids, glm_tokenizer, audio_decoder, args.device)
            elif args.save_hidden:
                token_ids, hidden_states = generate_tokens_and_hidden(prompt, args)
                tts_speech = decode_audio_from_token_ids(token_ids, glm_tokenizer, audio_decoder, args.device)
            else:
                tts_speech = generate_audio(prompt, args, glm_tokenizer, audio_decoder)
            torchaudio.save(str(output_path), tts_speech.unsqueeze(0), 22050, format="wav")

            if args.save_hidden:
                if hidden_states is None:
                    raise RuntimeError("Expected hidden states but received None")
                save_hidden_payload(
                    hidden_path=hidden_path,
                    input_path=input_path,
                    output_path=output_path,
                    token_ids=token_ids,
                    hidden_states=hidden_states,
                    glm_tokenizer=glm_tokenizer,
                )
                if args.inference_with_steering:
                    print(f"[OK] {output_path} + {hidden_path} (steered layer={int(args.inject_layer)})")
                else:
                    print(f"[OK] {output_path} + {hidden_path}")
            else:
                if args.inference_with_steering:
                    print(f"[OK] {output_path} (steered layer={int(args.inject_layer)})")
                else:
                    print(f"[OK] {output_path}")
            success += 1
        except Exception as exc:
            print(f"[FAIL] {input_path}: {exc}")
            failed += 1

    print(f"[DONE] success={success}, failed={failed}")


if __name__ == "__main__":
    main()
