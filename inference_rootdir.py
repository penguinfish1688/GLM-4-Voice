#!/usr/bin/env python3
"""
Batch inference for GLM-4-Voice on dataset layout: root_dir/*/input.wav.

This script runs GLM generation locally in a single process (no model_server).
For each input.wav, it writes output.wav and optionally output_hidden.pt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, WhisperFeatureExtractor

from flow_inference import AudioDecoder
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token


AUDIO_TOKEN_RE = re.compile(r"<\|audio_(\d+)\|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("glm4voice_rootdir_inference_local")
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
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "int4"])
    parser.add_argument("--tokenizer-path", default="THUDM/glm-4-voice-tokenizer", help="Speech tokenizer path")
    parser.add_argument("--flow-path", default="./glm-4-voice-decoder", help="Decoder checkpoint directory")

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument(
        "--hidden-retries",
        type=int,
        default=3,
        help="Retry count for hidden/steering generation when no audio tokens are returned",
    )
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


def build_prompt(audio_path: Path, whisper_model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor) -> str:
    audio_tokens = extract_speech_token(whisper_model, feature_extractor, [str(audio_path)])[0]
    if len(audio_tokens) == 0:
        raise RuntimeError(f"No audio tokens extracted from: {audio_path}")
    token_str = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    return "<|begin_of_audio|>" + token_str + "<|end_of_audio|>"


def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    if top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep = cumsum <= top_p
    keep[..., 0] = True

    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    sampled_local = torch.multinomial(filtered, num_samples=1)
    sampled = torch.gather(sorted_indices, -1, sampled_local).squeeze(-1)
    return sampled


def _collect_step_hidden_from_outputs(outputs: Any) -> torch.Tensor | None:
    hs = getattr(outputs, "hidden_states", None)
    if not isinstance(hs, (tuple, list)) or len(hs) == 0:
        return None

    layer_states = list(hs[1:]) if len(hs) > 1 else list(hs)
    step_layers: list[torch.Tensor] = []
    for h in layer_states:
        if not isinstance(h, torch.Tensor) or h.dim() != 3:
            continue
        step_layers.append(h[0, -1, :].detach())

    if not step_layers:
        return None
    return torch.stack(step_layers, dim=0)


def _get_transformer_layers(model: Any):
    explicit_paths = [
        ("transformer", "layers"),
        ("model", "layers"),
        ("encoder", "layers"),
        ("decoder", "layers"),
        ("transformer", "h"),
        ("model", "h"),
    ]

    def _get_path(root: Any, path: tuple[str, ...]):
        cur = root
        for p in path:
            cur = getattr(cur, p, None)
            if cur is None:
                return None
        return cur

    best = None
    best_len = -1
    for path in explicit_paths:
        layers = _get_path(model, path)
        if layers is None or not hasattr(layers, "__len__"):
            continue
        n = int(len(layers))
        if n > best_len:
            best = layers
            best_len = n

    if best is not None:
        return best
    raise RuntimeError("Unable to locate transformer layers on this model")


def _run_local_generation(
    prompt: str,
    args: argparse.Namespace,
    glm_model: Any,
    glm_tokenizer: Any,
    return_hidden: bool,
    steering_map: dict[str, list[float] | None] | None,
    inject_layer: int | None,
) -> tuple[list[int], torch.Tensor | None]:
    inputs = glm_tokenizer([prompt], return_tensors="pt").to(args.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    step_state = {"step": 0}
    handle = None

    if steering_map is not None:
        if inject_layer is None:
            raise RuntimeError("inject_layer is required when steering_map is provided")
        layers = _get_transformer_layers(glm_model)
        num_layers = len(layers)
        layer_idx = int(inject_layer)
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx
        if layer_idx < 0 or layer_idx >= num_layers:
            raise RuntimeError(f"inject_layer {inject_layer} out of range [0, {num_layers - 1}]")

        hidden_size = int(getattr(glm_model.config, "hidden_size", 0))
        smap: dict[int, torch.Tensor] = {}
        for k, vec in steering_map.items():
            if vec is None:
                continue
            step = int(k)
            v = torch.tensor(vec, device=args.device, dtype=torch.float32)
            if hidden_size > 0 and int(v.numel()) != hidden_size:
                raise RuntimeError(
                    f"steering vector dim mismatch at step {step}: got {int(v.numel())}, expected {hidden_size}"
                )
            smap[step] = v

        def _steer_hook(_module, _inputs, output):
            vec = smap.get(step_state["step"])
            if vec is None:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
                if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
                    steered = hidden.clone()
                    steered[:, -1, :] = steered[:, -1, :] + vec.to(steered.dtype)
                    return (steered, *output[1:])
                return output
            if isinstance(output, torch.Tensor) and output.dim() == 3:
                steered = output.clone()
                steered[:, -1, :] = steered[:, -1, :] + vec.to(steered.dtype)
                return steered
            return output

        handle = layers[layer_idx].register_forward_hook(_steer_hook)

    generated: list[int] = []
    step_hidden: list[torch.Tensor] = []
    past_key_values = None
    stop_token_id = glm_tokenizer.convert_tokens_to_ids("<|user|>")

    cur_input_ids = input_ids
    cur_attention_mask = attention_mask

    try:
        for step in range(args.max_new_tokens):
            step_state["step"] = step
            outputs = glm_model(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=return_hidden,
                return_dict=True,
            )

            if return_hidden:
                h = _collect_step_hidden_from_outputs(outputs)
                if h is not None:
                    step_hidden.append(h)

            logits = outputs.logits[:, -1, :]
            next_token = _sample_top_p(logits, temperature=args.temperature, top_p=args.top_p)
            token_id = int(next_token.item())
            generated.append(token_id)

            if token_id == stop_token_id:
                break

            past_key_values = outputs.past_key_values
            cur_input_ids = next_token.unsqueeze(0)
            if cur_attention_mask is not None:
                cur_attention_mask = torch.cat(
                    [cur_attention_mask, torch.ones((1, 1), device=cur_attention_mask.device, dtype=cur_attention_mask.dtype)],
                    dim=1,
                )
    finally:
        if handle is not None:
            handle.remove()

    hidden_states: torch.Tensor | None = None
    if return_hidden:
        if step_hidden:
            hidden_states = torch.stack(step_hidden, dim=0).to(torch.float16).cpu()
        else:
            hidden_size = int(getattr(glm_model.config, "hidden_size", 0))
            hidden_states = torch.empty((0, 0, hidden_size), dtype=torch.float16)

    return generated, hidden_states


def decode_audio_from_token_ids(
    token_ids: List[int],
    glm_tokenizer: Any,
    audio_decoder: AudioDecoder,
    device: str,
) -> torch.Tensor:
    audio_tokens: List[int] = []
    token_names = glm_tokenizer.convert_ids_to_tokens(token_ids)
    for tok in token_names:
        if tok == "<|user|>":
            break
        m = AUDIO_TOKEN_RE.fullmatch(str(tok))
        if m is not None:
            audio_tokens.append(int(m.group(1)))

    if not audio_tokens:
        raise RuntimeError("No audio tokens returned from model")

    token_tensor = torch.tensor(audio_tokens, dtype=torch.int64, device=device).unsqueeze(0)
    return audio_decoder.offline_inference(token_tensor).squeeze(0).cpu()


def generate_audio(
    prompt: str,
    args: argparse.Namespace,
    glm_model: Any,
    glm_tokenizer: Any,
    audio_decoder: AudioDecoder,
    inject_layer: int | None = None,
    steering_map: dict[str, list[float] | None] | None = None,
    return_hidden: bool = False,
) -> tuple[torch.Tensor, List[int], torch.Tensor | None]:
    attempts = max(1, int(args.hidden_retries)) if (return_hidden or steering_map is not None) else 1
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        token_ids, hidden_states = _run_local_generation(
            prompt=prompt,
            args=args,
            glm_model=glm_model,
            glm_tokenizer=glm_tokenizer,
            return_hidden=return_hidden,
            steering_map=steering_map,
            inject_layer=inject_layer,
        )
        try:
            tts_speech = decode_audio_from_token_ids(token_ids, glm_tokenizer, audio_decoder, args.device)
            return tts_speech, token_ids, hidden_states
        except RuntimeError as ex:
            if "No audio tokens returned from model" not in str(ex):
                raise
            last_exc = ex
            if attempt < attempts:
                print(f"[WARN] generation produced no audio tokens (attempt {attempt}/{attempts}), retrying...")
                continue
            raise RuntimeError(f"No audio tokens returned from model after {attempts} attempts") from ex

    if last_exc is not None:
        raise RuntimeError(f"No audio tokens returned from model after {attempts} attempts") from last_exc
    raise RuntimeError("Generation failed unexpectedly")


def count_audio_tokens(token_ids: List[int], glm_tokenizer: Any) -> int:
    n = 0
    token_names = glm_tokenizer.convert_ids_to_tokens(token_ids)
    for tok in token_names:
        if tok == "<|user|>":
            break
        if AUDIO_TOKEN_RE.fullmatch(str(tok)) is not None:
            n += 1
    return int(n)


def build_audio_token_mask(token_ids: List[int], glm_tokenizer: Any) -> List[bool]:
    token_names = glm_tokenizer.convert_ids_to_tokens(token_ids)
    mask: List[bool] = []
    for tok in token_names:
        if tok == "<|user|>":
            mask.append(False)
            break
        mask.append(AUDIO_TOKEN_RE.fullmatch(str(tok)) is not None)
    if len(mask) < len(token_ids):
        mask.extend([False] * (len(token_ids) - len(mask)))
    return mask


def save_hidden_payload(
    hidden_path: Path,
    input_path: Path,
    output_path: Path,
    token_ids: List[int],
    hidden_states: torch.Tensor,
    glm_tokenizer: Any,
    has_audio_tokens_in_hidden: bool,
    hidden_audio_token_count: int,
    audio_decode_fallback_used: bool,
    audio_decode_source: str,
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
        "schema_version": 6,
        "input_wav": str(input_path),
        "output_wav": str(output_path),
        "output_text": "",
        "frame_rate": float(frame_rate_hz),
        "token_ids": out_token_ids.view(-1),
        "token_names": glm_tokenizer.convert_ids_to_tokens(token_ids),
        "hidden_audio_token_mask": torch.tensor(build_audio_token_mask(token_ids, glm_tokenizer), dtype=torch.bool),
        "input_token_ids": in_token_ids,
        "input_token_width": 1,
        "output_token_ids": out_token_ids,
        "output_token_width": 1,
        "times": times,
        "token_time_ranges_sec": token_time_ranges,
        "hidden_states": last_hidden.float().cpu(),
        "text_hidden_layers": text_hidden_layers.float().cpu(),
        "has_audio_tokens_in_hidden": bool(has_audio_tokens_in_hidden),
        "hidden_audio_token_count": int(hidden_audio_token_count),
        "audio_decode_fallback_used": bool(audio_decode_fallback_used),
        "audio_decode_source": str(audio_decode_source),
        "ce_exclude_default": bool(not has_audio_tokens_in_hidden),
        "ce_exclude_reason": "hidden_tokens_have_no_audio" if not has_audio_tokens_in_hidden else "",
    }
    torch.save(payload, hidden_path)
    saved_tokens = int(text_hidden_layers.shape[0])
    saved_layers = int(text_hidden_layers.shape[1]) if text_hidden_layers.ndim == 3 else 0
    print(f"[HIDDEN] saved {hidden_path} tokens={saved_tokens} layers={saved_layers}")


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


def _load_local_glm_model(args: argparse.Namespace):
    quant_cfg = None
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}

    if args.dtype == "int4":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quant_cfg

    if args.dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif args.dtype == "float32":
        model_kwargs["torch_dtype"] = torch.float32

    if args.device.startswith("cuda"):
        model_kwargs["device_map"] = {"": 0}

    model = AutoModel.from_pretrained(args.model_path, **model_kwargs).eval()
    if not args.device.startswith("cuda"):
        model = model.to(args.device)
    return model


def main() -> None:
    args = parse_args()

    if args.inference_with_steering and args.inject_layer is None:
        raise ValueError("--inference-with-steering requires --inject-layer")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but not available")

    flow_config = os.path.join(args.flow_path, "config.yaml")
    flow_checkpoint = os.path.join(args.flow_path, "flow.pt")
    hift_checkpoint = os.path.join(args.flow_path, "hift.pt")

    print("[INIT] loading local GLM model, tokenizer, and decoder...")
    glm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    glm_model = _load_local_glm_model(args)

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
            audio_decode_fallback_used = False
            audio_decode_source = "stream"
            has_audio_tokens_in_hidden = True
            hidden_audio_token_count = 0

            if args.inference_with_steering:
                steering_map = load_steering_map(input_path.parent, int(args.inject_layer))
                tts_speech, token_ids, hidden_states = generate_audio(
                    prompt,
                    args,
                    glm_model,
                    glm_tokenizer,
                    audio_decoder,
                    inject_layer=int(args.inject_layer),
                    steering_map=steering_map,
                    return_hidden=bool(args.save_hidden),
                )
                if args.save_hidden:
                    audio_decode_source = "hidden_tokens"
                    hidden_audio_token_count = count_audio_tokens(token_ids, glm_tokenizer)
                    has_audio_tokens_in_hidden = hidden_audio_token_count > 0
            elif args.save_hidden:
                tts_speech, token_ids, hidden_states = generate_audio(
                    prompt,
                    args,
                    glm_model,
                    glm_tokenizer,
                    audio_decoder,
                    return_hidden=True,
                )
                audio_decode_source = "hidden_tokens"
                hidden_audio_token_count = count_audio_tokens(token_ids, glm_tokenizer)
                has_audio_tokens_in_hidden = hidden_audio_token_count > 0
            else:
                tts_speech, token_ids, hidden_states = generate_audio(
                    prompt,
                    args,
                    glm_model,
                    glm_tokenizer,
                    audio_decoder,
                )

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
                    has_audio_tokens_in_hidden=has_audio_tokens_in_hidden,
                    hidden_audio_token_count=hidden_audio_token_count,
                    audio_decode_fallback_used=audio_decode_fallback_used,
                    audio_decode_source=audio_decode_source,
                )
                hs = tuple(int(x) for x in hidden_states.shape)
                if args.inference_with_steering:
                    print(
                        f"[OK] {output_path} + {hidden_path} hidden_shape={hs} "
                        f"audio_in_hidden={hidden_audio_token_count} fallback={audio_decode_fallback_used} "
                        f"(steered layer={int(args.inject_layer)})"
                    )
                else:
                    print(
                        f"[OK] {output_path} + {hidden_path} hidden_shape={hs} "
                        f"audio_in_hidden={hidden_audio_token_count} fallback={audio_decode_fallback_used}"
                    )
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
