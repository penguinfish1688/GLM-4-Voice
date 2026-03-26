#!/usr/bin/env python3
"""
Batch inference for GLM-4-Voice on dataset layout: root_dir/*/input.wav.

This script reads each input wav and writes output wav in the same folder.
It reuses GLM-4-Voice's speech tokenizer + decoder and calls model_server.py
for streamed token generation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from pathlib import Path
from typing import Any, Iterable, List
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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output wav")

    parser.add_argument("--model-path", default="THUDM/glm-4-voice-9b", help="GLM model path")
    parser.add_argument("--tokenizer-path", default="THUDM/glm-4-voice-tokenizer", help="Speech tokenizer path")
    parser.add_argument("--flow-path", default="./glm-4-voice-decoder", help="Decoder checkpoint directory")
    parser.add_argument("--server-url", default="http://localhost:10000/generate_stream", help="Model server endpoint")

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


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but not available")

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

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {output_path}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {input_path}")

        try:
            prompt = build_prompt(input_path, whisper_model, feature_extractor)
            tts_speech = generate_audio(prompt, args, glm_tokenizer, audio_decoder)
            torchaudio.save(str(output_path), tts_speech.unsqueeze(0), 22050, format="wav")
            print(f"[OK] {output_path}")
            success += 1
        except Exception as exc:
            print(f"[FAIL] {input_path}: {exc}")
            failed += 1

    print(f"[DONE] success={success}, failed={failed}")


if __name__ == "__main__":
    main()
