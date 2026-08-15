#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from wmsteg.data import JsonlWatermarkDataset, make_collator
from wmsteg.modeling import MeanPoolDetector


def hidden_size(config):
    for k in ("hidden_size", "d_model", "n_embd"):
        v = getattr(config, k, None)
        if v is not None:
            return int(v)
    if getattr(config, "text_config", None) is not None:
        return hidden_size(config.text_config)
    raise ValueError("Could not infer hidden size")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Detector run directory containing head.pt and optionally backbone_final/")
    p.add_argument("--data", default="runs/exp0_data")
    p.add_argument("--model", default=None, help="Base model for frozen-head runs; ignored if backbone_final exists.")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    rundir = Path(args.run)
    backbone = rundir / "backbone_final"
    model_path = str(backbone) if backbone.exists() else args.model
    if model_path is None:
        cfg = json.load(open(rundir / "train_config.json"))
        model_path = cfg["model"]

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if torch.cuda.is_available(): lm = lm.cuda()
    device = next(lm.parameters()).device
    detector = MeanPoolDetector(lm, hidden_size(lm.config)).to(device)
    detector.classifier.load_state_dict(torch.load(rundir / "head.pt", map_location=device, weights_only=True))
    detector.eval()

    ds = JsonlWatermarkDataset(Path(args.data) / f"{args.split}.jsonl")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=make_collator(tok, args.max_length))
    rows_out = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            source_rows = batch.pop("rows")
            inputs = {k: v.to(device) for k, v in batch.items()}
            ctx = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else torch.no_grad()
            with ctx:
                logits = detector(**inputs)
            prob = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
            pred = logits.argmax(-1).cpu().numpy()
            for src, y, pr, pd in zip(source_rows, labels.cpu().numpy(), prob, pred):
                rows_out.append({
                    "true": int(y), "pred": int(pd), "p_orange": float(pr),
                    "watermark": src.get("watermark"), "z_score": src.get("z_score_purple_minus_orange"),
                    "text": src.get("text", ""),
                })

    y = np.array([r["true"] for r in rows_out]); pred = np.array([r["pred"] for r in rows_out])
    cm = [[int(np.sum((y == i) & (pred == j))) for j in (0,1)] for i in (0,1)]
    metrics = {"accuracy": float(np.mean(y == pred)), "confusion_matrix": cm, "n": len(rows_out)}
    print(json.dumps(metrics, indent=2))
    with open(rundir / f"predictions_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader(); w.writerows(rows_out)


if __name__ == "__main__":
    main()
