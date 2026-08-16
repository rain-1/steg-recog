#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wmsteg.verbalization import (
    build_verbalization_plan,
    materialize_verbalization_dataset,
    read_jsonl,
    render_summary,
    summarize_results,
    train_one_verbalizer,
)


def csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def csv_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def discover_wm_backbone(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = [
        "runs/detector_poc_128_v2/remote/joint_best/backbone",
        "runs/detector_poc_128_v2/joint_best/backbone",
        "runs/detector_poc_128_v3/remote/joint_best/backbone",
        "runs/detector_poc_128_v3/joint_best/backbone",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Local sample-efficiency test: can a WM-recognition backbone learn to verbalize PURPLE/ORANGE faster than base Qwen?")
    ap.add_argument("--behavior-bank", default="runs/behavior_medium_data")
    ap.add_argument("--detector-bank", default="runs/lyra_watermark_poc_128")
    ap.add_argument("--wm-backbone", default=None)
    ap.add_argument("--output", default="runs/verbalization_local")
    ap.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--sizes", default="8,32,64")
    ap.add_argument("--seeds", default="4242,5252")
    ap.add_argument("--formats", default="normal,suffix")
    ap.add_argument("--train-max", type=int, default=64)
    ap.add_argument("--val-sources", type=int, default=48)
    ap.add_argument("--test-sources", type=int, default=128)
    ap.add_argument("--plan-seed", type=int, default=8801)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=448)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--wandb-project", default="steg-recog")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online" if os.environ.get("WANDB_API_KEY") else "disabled")
    ap.add_argument("--wandb-group", default="verbalization-local")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this local experiment")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    behavior_bank = Path(args.behavior_bank)
    records_path = behavior_bank / "watermark_pair_records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing {records_path}; point --behavior-bank at behavior_medium_data")
    records = read_jsonl(records_path)

    detector_records_path = Path(args.detector_bank) / "pair_records_all.jsonl"
    if detector_records_path.exists():
        detector_ids = {str(r["source_id"]) for r in read_jsonl(detector_records_path)}
        overlap = {str(r["source_id"]) for r in records} & detector_ids
        if overlap:
            raise RuntimeError(f"Behavior bank overlaps detector bank by {len(overlap)} source IDs")

    sizes = csv_ints(args.sizes); seeds = csv_ints(args.seeds); formats = csv_strs(args.formats)
    if not sizes or not seeds or not formats:
        raise ValueError("sizes, seeds, and formats must be non-empty")
    if max(sizes) > args.train_max:
        raise ValueError("max --sizes cannot exceed --train-max")

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    plan_path = outdir / "dataset.json"
    if plan_path.exists():
        dataset = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = build_verbalization_plan(records, train_max=args.train_max, val_sources=args.val_sources, test_sources=args.test_sources, seed=args.plan_seed)
        dataset = materialize_verbalization_dataset(plan)
        plan_path.write_text(json.dumps(dataset, indent=2, sort_keys=True), encoding="utf-8")

    wm_backbone = discover_wm_backbone(args.wm_backbone)
    if not Path(wm_backbone).exists():
        raise FileNotFoundError(
            f"WM backbone not found at {wm_backbone}.\n"
            "Download it first, e.g.:\n"
            "  modal volume get wmsteg-detector-runs detector_poc_128_v2/joint_best runs/detector_poc_128_v2/remote/joint_best\n"
            "Then rerun, or pass --wm-backbone PATH_TO/joint_best/backbone"
        )

    cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "eval_batch_size": args.eval_batch_size,
        "generation_batch_size": args.eval_batch_size,
        "generation_max_new_tokens": 8,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
        "wandb_group": args.wandb_group,
        "resume": not args.no_resume,
    }
    (outdir / "config.json").write_text(json.dumps({**vars(args), "resolved_wm_backbone": wm_backbone, "train_sizes": sizes, "training_seeds": seeds, "training_formats": formats}, indent=2, sort_keys=True), encoding="utf-8")

    jobs = [(fmt, n, seed, mode) for fmt in formats for n in sizes for seed in seeds for mode in ("base", "wm_recognition")]
    print(f"Planned runs: {len(jobs)} ({len(formats)} formats × {len(sizes)} sizes × {len(seeds)} seeds × 2 starts)")
    results = []
    for i, (fmt, n, seed, mode) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {mode} | {fmt} | n={n} | seed={seed}", flush=True)
        result = train_one_verbalizer(
            dataset,
            start_mode=mode,
            fmt=fmt,
            train_sources=n,
            seed=seed,
            model_id=args.model_id,
            model_revision=args.model_revision,
            wm_backbone=wm_backbone,
            output_dir=outdir / "runs",
            cfg=cfg,
        )
        results.append(result)
        summary = summarize_results(results)
        (outdir / "aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (outdir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
        print(render_summary(summary), flush=True)

    print(f"\nDone. Summary: {outdir / 'summary.md'}")


if __name__ == "__main__":
    main()
