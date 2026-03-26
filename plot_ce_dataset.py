#!/usr/bin/env python3
"""
Plot CE curves for GLM output_hidden.pt files under root_dir/*/.

This is a GLM counterpart of personaplex CE plotting. It reads token ids from
output_hidden.pt and computes next-token CE with teacher forcing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("glm_plot_ce_dataset")
    ap.add_argument("--root-dir", required=True, help="Root dir containing <id>/output_hidden.pt")
    ap.add_argument("--model-path", default="THUDM/glm-4-voice-9b", help="GLM model path")
    ap.add_argument("--device", default="cuda", help="Device for CE computation")
    return ap.parse_args()


def list_hidden_paths(root_dir: str) -> List[Path]:
    root = Path(root_dir).expanduser()
    paths = [p for p in sorted(root.glob("*/output_hidden.pt")) if p.is_file()]
    return paths


def _to_token_ids(raw) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        ids = raw.detach().cpu().long().view(-1)
    elif isinstance(raw, list):
        ids = torch.tensor(raw, dtype=torch.long)
    else:
        raise TypeError(f"Unsupported token_ids type: {type(raw).__name__}")
    return ids


def compute_ce_curve(model, token_ids: torch.Tensor, device: str) -> torch.Tensor:
    if token_ids.numel() < 2:
        return torch.empty((0,), dtype=torch.float32)

    x = token_ids[:-1].to(device).unsqueeze(0)
    y = token_ids[1:].to(device)

    with torch.no_grad():
        out = model(input_ids=x, return_dict=True)
        logits = out.logits[0].float()
        ce = F.cross_entropy(logits, y, reduction="none")
    return ce.detach().cpu()


def maybe_get_times(payload: dict, n_points: int) -> List[float]:
    t = payload.get("times", None)
    if isinstance(t, torch.Tensor) and t.ndim == 1 and t.shape[0] >= n_points:
        return t[:n_points].detach().cpu().float().tolist()
    frame_rate = float(payload.get("frame_rate", 12.5))
    return (torch.arange(n_points, dtype=torch.float32) / frame_rate).tolist()


def plot_curve(times: List[float], ce: torch.Tensor, out_png: Path, sample_id: str) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    y = ce.numpy()

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=180)
    ax.plot(times, y, color="#1f77b4", linewidth=1.2, label="next-token CE")
    ax.set_title(f"GLM CE Curve ({sample_id})")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cross Entropy")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.65)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but not available")

    paths = list_hidden_paths(args.root_dir)
    print(f"[SCAN] found {len(paths)} output_hidden.pt files")
    if not paths:
        return

    print("[INIT] loading GLM model for CE plotting...")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, device_map={"": 0}).eval()

    ok = 0
    total = 0
    for hp in paths:
        total += 1
        try:
            payload = torch.load(hp, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise TypeError(f"payload is {type(payload).__name__}, expected dict")
            if "token_ids" not in payload:
                raise KeyError("Missing key 'token_ids' in output_hidden.pt")

            token_ids = _to_token_ids(payload["token_ids"])
            ce = compute_ce_curve(model, token_ids, args.device)
            if ce.numel() == 0:
                print(f"[SKIP] {hp} (too few tokens)")
                continue

            times = maybe_get_times(payload, int(ce.numel()))
            out_json = hp.with_name("in_out_ce_0.json")
            out_png = hp.with_name("logit_lens_ce_layer_0.png")

            compat = {
                "line1_user_multimodal_ce": ce.tolist(),
                "line2_model_multimodal_ce": ce.tolist(),
                "ratio_line1_over_line2": [1.0] * int(ce.numel()),
                "line1_shift": "n_to_n_plus_1",
                "line2_shift": "n_to_n_plus_1",
                "moving_average_window": 1,
                "smoothing": "none",
                "layer": 0,
                "num_points": int(ce.numel()),
                "glm_note": "Both line1/line2 are mapped to GLM next-token CE for compatibility.",
            }
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(compat, f, indent=2, ensure_ascii=False)

            plot_curve(times, ce, out_png, hp.parent.name)
            print(f"[OK] {out_json} + {out_png}")
            ok += 1
        except Exception as exc:
            print(f"[SKIP] {hp}: {exc}")

    print(f"[DONE] generated {ok}/{total}")


if __name__ == "__main__":
    main()
