#!/usr/bin/env python3
"""
Plot CE curves for GLM output_hidden.pt files under root_dir/*/.

This is a GLM counterpart of personaplex CE plotting. It reads token ids from
output_hidden.pt and computes per-layer input/output CE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("glm_plot_ce_dataset")
    ap.add_argument("--root-dir", required=True, help="Root dir containing <id>/output_hidden.pt")
    ap.add_argument("--model-path", default="THUDM/glm-4-voice-9b", help="GLM model path")
    ap.add_argument("--device", default="cuda", help="Device for CE computation")
    ap.add_argument("--anchor-span", type=int, default=35, help="Window half-size in tokens for heatmap")
    ap.add_argument(
        "--anchor-key",
        type=str,
        default="interrupt_start",
        help="Anchor key in input_timing.json (e.g., interrupt_start or question_start)",
    )
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


def _resolve_lm_head(model):
    # 1) Standard HF API
    if hasattr(model, "get_output_embeddings"):
        try:
            emb = model.get_output_embeddings()
            if emb is not None:
                return emb, "get_output_embeddings"
        except Exception:
            pass

    # 2) Common attribute names across custom GLM wrappers
    attr_candidates = [
        "lm_head",
        "output_layer",
        "text_linear",
        "embed_out",
    ]
    nested_candidates = [
        ("transformer", "output_layer"),
        ("transformer", "lm_head"),
        ("model", "lm_head"),
        ("language_model", "lm_head"),
    ]
    for name in attr_candidates:
        head = getattr(model, name, None)
        if head is not None:
            return head, name
    for parent_name, child_name in nested_candidates:
        parent = getattr(model, parent_name, None)
        if parent is not None:
            head = getattr(parent, child_name, None)
            if head is not None:
                return head, f"{parent_name}.{child_name}"

    # 3) Heuristic: pick the Linear with largest out_features (typically vocab head)
    best_name = None
    best_mod = None
    best_vocab = -1
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.out_features > best_vocab:
            best_vocab = int(mod.out_features)
            best_name = name
            best_mod = mod
    if best_mod is not None and best_vocab > 1000:
        return best_mod, f"heuristic:{best_name}"

    # 4) Fallback: tied input embedding projection
    if hasattr(model, "get_input_embeddings"):
        try:
            in_emb = model.get_input_embeddings()
            if in_emb is not None and hasattr(in_emb, "weight"):
                return in_emb, "tied_input_embeddings"
        except Exception:
            pass

    raise RuntimeError("Cannot resolve output projection head for this model")


def _get_lm_head_and_norm(model):
    lm_head, head_name = _resolve_lm_head(model)

    norm = None
    candidates = [
        ("transformer", "final_layernorm"),
        ("transformer", "output_layernorm"),
        ("model", "norm"),
        ("transformer", "norm"),
    ]
    for parent_name, attr_name in candidates:
        parent = getattr(model, parent_name, None)
        if parent is not None and hasattr(parent, attr_name):
            norm = getattr(parent, attr_name)
            break
    return lm_head, norm, head_name


def _project_logits(hidden: torch.Tensor, lm_head, norm_layer, device: str) -> torch.Tensor:
    target_dtype = None
    if hasattr(lm_head, "weight") and isinstance(getattr(lm_head, "weight"), torch.Tensor):
        target_dtype = lm_head.weight.dtype
    elif norm_layer is not None:
        try:
            target_dtype = next(norm_layer.parameters()).dtype
        except StopIteration:
            target_dtype = None

    if target_dtype is None:
        target_dtype = torch.float32

    x = hidden.to(device=device, dtype=target_dtype)
    if norm_layer is not None:
        x = norm_layer(x)
    with torch.no_grad():
        # Module-style output head.
        if callable(lm_head):
            out = lm_head(x)
            if not isinstance(out, torch.Tensor):
                raise RuntimeError("lm head returned non-tensor output")
            logits = out.float()
        # Embedding fallback (tied weights): logits = x @ W^T
        elif hasattr(lm_head, "weight"):
            logits = F.linear(x, lm_head.weight).float()
        else:
            raise RuntimeError("Unsupported lm head type for projection")
    return logits


def maybe_get_times(payload: dict, n_points: int) -> List[float]:
    t = payload.get("times", None)
    if isinstance(t, torch.Tensor) and t.ndim == 1 and t.shape[0] >= n_points:
        return t[:n_points].detach().cpu().float().tolist()
    frame_rate = float(payload.get("frame_rate", 12.5))
    return (torch.arange(n_points, dtype=torch.float32) / frame_rate).tolist()


def plot_curve(times: List[float], line1: torch.Tensor, line2: torch.Tensor, out_png: Path, sample_id: str, layer: int) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    y1 = line1.numpy()
    y2 = line2.numpy()

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=180)
    ax.plot(times, y1, color="#1f77b4", linewidth=1.2, label="input CE (listening)")
    ax.plot(times, y2, color="#d62728", linewidth=1.2, label="output CE (speaking)")
    ax.set_title(f"GLM CE Curve ({sample_id}, layer={layer})")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cross Entropy")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.65)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _extract_centered(arr: np.ndarray, center: int, half: int) -> np.ndarray:
    out = np.full((2 * half + 1,), np.nan, dtype=np.float32)
    start = center - half
    end = center + half
    src_l = max(0, start)
    src_r = min(arr.shape[0] - 1, end)
    if src_r < src_l:
        return out
    dst_l = src_l - start
    dst_r = dst_l + (src_r - src_l)
    out[dst_l : dst_r + 1] = arr[src_l : src_r + 1]
    return out


def save_anchored_heatmaps(
    root_dir: Path,
    per_layer_records: Dict[int, List[Dict[str, Any]]],
    span: int,
    anchor_key: str,
) -> None:
    import matplotlib.pyplot as plt

    layer_ids = sorted(per_layer_records.keys())
    if not layer_ids:
        print("[HEATMAP] no layer records found")
        return

    input_rows: List[np.ndarray] = []
    output_rows: List[np.ndarray] = []
    counts: List[int] = []
    offsets = np.arange(-span, span + 1, dtype=np.int32)

    for lv in layer_ids:
        in_windows: List[np.ndarray] = []
        out_windows: List[np.ndarray] = []

        for rec in per_layer_records[lv]:
            timing_path = rec["sample_dir"] / "input_timing.json"
            if not timing_path.is_file():
                continue
            try:
                timing = json.loads(timing_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(timing, dict) or anchor_key not in timing:
                continue

            try:
                anchor_sec = float(timing[anchor_key])
            except Exception:
                continue

            frame_rate = float(rec.get("frame_rate", 12.5))
            line1 = np.asarray(rec["line1"], dtype=np.float32)
            line2 = np.asarray(rec["line2"], dtype=np.float32)
            if line1.size == 0 or line2.size == 0:
                continue

            center = int(round(anchor_sec * frame_rate))
            center = max(0, min(center, line1.shape[0] - 1))
            in_windows.append(_extract_centered(line1, center, span))
            out_windows.append(_extract_centered(line2, center, span))

        if not in_windows:
            input_rows.append(np.full((2 * span + 1,), np.nan, dtype=np.float32))
            output_rows.append(np.full((2 * span + 1,), np.nan, dtype=np.float32))
            counts.append(0)
            continue

        input_rows.append(np.nanmean(np.stack(in_windows, axis=0), axis=0))
        output_rows.append(np.nanmean(np.stack(out_windows, axis=0), axis=0))
        counts.append(len(in_windows))

    input_mat = np.stack(input_rows, axis=0).astype(np.float32)
    output_mat = np.stack(output_rows, axis=0).astype(np.float32)

    def _save_one(mat: np.ndarray, which: str, title: str):
        out_png = root_dir / f"logit_lens_heatmap_{anchor_key}_{which}.png"
        out_json = root_dir / f"logit_lens_heatmap_{anchor_key}_{which}.json"

        finite = mat[np.isfinite(mat)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.percentile(finite, 5.0))
            vmax = float(np.percentile(finite, 95.0))
            if vmax <= vmin:
                vmax = vmin + 1e-6

        payload = {
            "anchor": anchor_key,
            "which": which,
            "layer_ids": layer_ids,
            "offset_tokens": offsets.tolist(),
            "heatmap": np.nan_to_num(mat, nan=0.0).tolist(),
            "num_samples_per_layer": counts,
            "vmin": vmin,
            "vmax": vmax,
        }
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        fig, ax = plt.subplots(figsize=(10.0, 5.4), dpi=180)
        img = ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            extent=(float(offsets[0]), float(offsets[-1]), float(layer_ids[0]) - 0.5, float(layer_ids[-1]) + 0.5),
        )
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("CE")
        ax.set_title(title)
        ax.set_xlabel(f"Relative token index to {anchor_key}")
        ax.set_ylabel("Layer")
        ax.set_yticks(layer_ids)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[HEATMAP] saved {out_json} + {out_png}")

    _save_one(
        input_mat,
        which="input_ce",
        title=f"Input CE Heatmap (listening) | anchor={anchor_key}",
    )
    _save_one(
        output_mat,
        which="output_ce",
        title=f"Output CE Heatmap (speaking) | anchor={anchor_key}",
    )


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
    lm_head, norm_layer, head_name = _get_lm_head_and_norm(model)
    print(f"[INIT] using output head: {head_name}")

    ok = 0
    total = 0
    per_layer_records: Dict[int, List[Dict[str, Any]]] = {}
    for hp in paths:
        total += 1
        try:
            payload = torch.load(hp, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise TypeError(f"payload is {type(payload).__name__}, expected dict")
            if "token_ids" not in payload:
                raise KeyError("Missing key 'token_ids' in output_hidden.pt")

            if "text_hidden_layers" not in payload:
                raise KeyError("Missing key 'text_hidden_layers' in output_hidden.pt")

            hidden = payload["text_hidden_layers"]
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise ValueError(f"Expected text_hidden_layers [T,L,D], got {type(hidden).__name__} {getattr(hidden, 'shape', None)}")

            token_ids = _to_token_ids(payload["token_ids"])
            if token_ids.numel() < 3:
                print(f"[SKIP] {hp} (too few tokens)")
                continue

            if "input_token_ids" in payload:
                in_ids = payload["input_token_ids"]
                if isinstance(in_ids, torch.Tensor) and in_ids.ndim == 2 and in_ids.shape[1] >= 1:
                    input_targets = in_ids[:, 0].long().view(-1)
                else:
                    input_targets = torch.full_like(token_ids, fill_value=-1)
                    input_targets[1:] = token_ids[:-1]
            else:
                input_targets = torch.full_like(token_ids, fill_value=-1)
                input_targets[1:] = token_ids[:-1]

            if "output_token_ids" in payload:
                out_ids = payload["output_token_ids"]
                if isinstance(out_ids, torch.Tensor) and out_ids.ndim == 2 and out_ids.shape[1] >= 1:
                    output_targets = out_ids[:, 0].long().view(-1)
                else:
                    output_targets = token_ids
            else:
                output_targets = token_ids

            t, num_layers, _ = hidden.shape
            for lv in range(num_layers):
                h = hidden[:, lv, :]
                logits = _project_logits(h, lm_head=lm_head, norm_layer=norm_layer, device=args.device)

                # personaplex-like mapping:
                # line1: input CE (listening), shifted n->n+1 target
                # line2: output CE (speaking), aligned n target
                line1 = F.cross_entropy(
                    logits[:-1],
                    input_targets[1:].to(args.device),
                    reduction="none",
                    ignore_index=-1,
                ).detach().cpu()
                line2 = F.cross_entropy(
                    logits[:-1],
                    output_targets[:-1].to(args.device),
                    reduction="none",
                    ignore_index=-1,
                ).detach().cpu()

                n = int(min(line1.numel(), line2.numel()))
                line1 = line1[:n]
                line2 = line2[:n]
                if n == 0:
                    continue

                times = maybe_get_times(payload, n)
                ratio = (line1 / line2.clamp_min(1e-6)).tolist()

                out_json = hp.with_name(f"in_out_ce_{lv}.json")
                out_png = hp.with_name(f"logit_lens_ce_layer_{lv}.png")
                compat = {
                    "line1_user_multimodal_ce": line1.tolist(),
                    "line2_model_multimodal_ce": line2.tolist(),
                    "ratio_line1_over_line2": ratio,
                    "line1_shift": "n_to_n_plus_1",
                    "line2_shift": "n_to_n",
                    "moving_average_window": 1,
                    "smoothing": "none",
                    "layer": int(lv),
                    "num_points": int(n),
                    "glm_note": "line1=input CE(listening), line2=output CE(speaking)",
                }
                with out_json.open("w", encoding="utf-8") as f:
                    json.dump(compat, f, indent=2, ensure_ascii=False)

                plot_curve(times, line1, line2, out_png, hp.parent.name, lv)
                per_layer_records.setdefault(int(lv), []).append(
                    {
                        "sample_dir": hp.parent,
                        "line1": line1.tolist(),
                        "line2": line2.tolist(),
                        "frame_rate": float(payload.get("frame_rate", 12.5)),
                    }
                )

            print(f"[OK] {hp.parent}")
            ok += 1
        except Exception as exc:
            print(f"[SKIP] {hp}: {exc}")

    if per_layer_records:
        save_anchored_heatmaps(
            root_dir=Path(args.root_dir).expanduser(),
            per_layer_records=per_layer_records,
            span=int(args.anchor_span),
            anchor_key=str(args.anchor_key),
        )

    print(f"[DONE] generated {ok}/{total}")


if __name__ == "__main__":
    main()
