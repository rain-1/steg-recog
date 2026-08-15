#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from wmsteg.data import JsonlWatermarkDataset, make_collator
from wmsteg.modeling import MeanPoolDetector


def parse_args():
    p = argparse.ArgumentParser(description="Train a linear mean-pool watermark detector.")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data", default="runs/exp0_data")
    p.add_argument("--output", default="runs/exp0_joint")
    p.add_argument("--mode", choices=["frozen", "joint"], default="joint")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    return p.parse_args()


def choose_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def config_hidden_size(config) -> int:
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return config_hidden_size(text_config)
    raise ValueError("Could not infer hidden size from model config")


def evaluate(detector, loader, device, autocast_dtype, *, desc="Evaluate"):
    detector.eval()
    all_y, all_pred, all_prob = [], [], []
    total_loss = 0.0
    total_n = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            labels = batch.pop("labels").to(device)
            batch.pop("rows", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = torch.autocast("cuda", dtype=autocast_dtype) if device.type == "cuda" else torch.no_grad()
            with ctx:
                logits = detector(**batch)
                loss = loss_fn(logits, labels)
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * labels.numel()
            total_n += labels.numel()
            all_y.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_prob.extend(probs.cpu().tolist())

    y = np.asarray(all_y)
    pred = np.asarray(all_pred)
    cm = np.zeros((2, 2), dtype=np.int64)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    return {
        "loss": total_loss / max(1, total_n),
        "accuracy": float((y == pred).mean()),
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "n": int(total_n),
        "mean_p_orange_on_purple": float(np.mean(np.asarray(all_prob)[y == 0])) if np.any(y == 0) else None,
        "mean_p_orange_on_orange": float(np.mean(np.asarray(all_prob)[y == 1])) if np.any(y == 1) else None,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    datadir = Path(args.data)

    dtype = choose_dtype()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model: {args.model}", flush=True)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if torch.cuda.is_available():
        lm = lm.cuda()
    device = next(lm.parameters()).device
    lm.config.use_cache = False

    if args.mode == "joint" and not args.no_gradient_checkpointing:
        lm.gradient_checkpointing_enable()

    hidden_size = config_hidden_size(lm.config)
    detector = MeanPoolDetector(lm, hidden_size).to(device)

    if args.mode == "frozen":
        for p in detector.causal_lm.parameters():
            p.requires_grad_(False)

    train_ds = JsonlWatermarkDataset(datadir / "train.jsonl")
    val_ds = JsonlWatermarkDataset(datadir / "val.jsonl")
    test_ds = JsonlWatermarkDataset(datadir / "test.jsonl")
    collate = make_collator(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    eval_bs = max(args.batch_size, 8 if args.mode == "frozen" else args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    param_groups = [{"params": detector.classifier.parameters(), "lr": args.lr_head}]
    if args.mode == "joint":
        param_groups.append({"params": detector.causal_lm.parameters(), "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = int(total_updates * args.warmup_ratio)

    def lr_scale(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return (step + 1) / warmup_updates
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    autocast_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16

    global_update = 0
    history = []
    for epoch in range(args.epochs):
        detector.train()
        if args.mode == "frozen":
            # Keep the frozen representation deterministic; only the linear head trains.
            detector.causal_lm.eval()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0
        train_bar = tqdm(
            train_loader,
            desc=f"{args.mode} epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_bar):
            labels = batch.pop("labels").to(device)
            batch.pop("rows", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if device.type == "cuda":
                ctx = torch.autocast("cuda", dtype=autocast_dtype)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()
            with ctx:
                logits = detector(**batch)
                loss = loss_fn(logits, labels) / args.grad_accum
            loss.backward()
            running_loss += float(loss.item()) * args.grad_accum * labels.numel()
            seen += labels.numel()

            should_step = ((step + 1) % args.grad_accum == 0) or (step + 1 == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in detector.parameters() if p.requires_grad], args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1

            avg_loss = running_loss / max(1, seen)
            train_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                updates=global_update,
            )

        val_metrics = evaluate(
            detector, val_loader, device, autocast_dtype, desc=f"{args.mode} val {epoch + 1}/{args.epochs}"
        )
        train_loss = running_loss / max(1, seen)
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))

    # Save the final intervention checkpoint. Keeping head and backbone from the same
    # training step makes the reported test metrics exactly reproducible.
    torch.save(detector.classifier.state_dict(), outdir / "head.pt")
    if args.mode == "joint":
        detector.causal_lm.config.use_cache = True
        detector.causal_lm.save_pretrained(outdir / "backbone_final", safe_serialization=True)
        tokenizer.save_pretrained(outdir / "backbone_final")
        detector.causal_lm.config.use_cache = False

    final_test = evaluate(detector, test_loader, device, autocast_dtype, desc=f"{args.mode} test")
    metrics = {"args": vars(args), "history": history, "test": final_test}
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with open(outdir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    print("FINAL TEST")
    print(json.dumps(final_test, indent=2))
    print(f"Saved run to {outdir}")


if __name__ == "__main__":
    main()
