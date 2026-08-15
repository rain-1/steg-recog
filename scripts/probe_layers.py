#!/usr/bin/env python3
"""Frozen layer-by-layer linear probes for purple/orange watermark information.

This caches mean-pooled representations from input embeddings and selected transformer
layers, then trains a fresh linear classifier for each representation. The backbone is
never updated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from wmsteg.data import JsonlWatermarkDataset, make_collator
from wmsteg.pooling import masked_mean_pool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data", default="runs/exp0_data")
    p.add_argument("--output", default="runs/exp0_layer_probes")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--layers", default="auto",
                   help="auto, all, or comma list such as embeddings,1,4,8,final")
    p.add_argument("--probe-epochs", type=int, default=200)
    p.add_argument("--probe-lr", type=float, default=0.03)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def choose_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def num_layers(config) -> int:
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return num_layers(text_config)
    raise ValueError("Could not infer number of hidden layers")


def select_layers(spec: str, n_layers: int) -> list[tuple[str, int | None]]:
    # hidden_states[0] = embedding output; hidden_states[k] = output after transformer layer k.
    if spec == "auto":
        points = {1, max(1, n_layers // 4), max(1, n_layers // 2), max(1, 3 * n_layers // 4), n_layers}
        return [("embeddings", None)] + [(f"layer_{k}", k) for k in sorted(points)]
    if spec == "all":
        return [("embeddings", None)] + [(f"layer_{k}", k) for k in range(1, n_layers + 1)]

    out: list[tuple[str, int | None]] = []
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"embedding", "embeddings", "input", "input_embeddings"}:
            pair = ("embeddings", None)
        elif item == "final":
            pair = (f"layer_{n_layers}", n_layers)
        else:
            k = int(item.replace("layer_", ""))
            if not 1 <= k <= n_layers:
                raise ValueError(f"Layer {k} outside 1..{n_layers}")
            pair = (f"layer_{k}", k)
        if pair not in out:
            out.append(pair)
    return out


def extract_features(model, loader, layer_specs, device, dtype, *, desc):
    model.eval()
    base = model.base_model
    emb_layer = model.get_input_embeddings()
    feats = {name: [] for name, _ in layer_specs}
    labels_all = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True):
            labels = batch.pop("labels")
            batch.pop("rows", None)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            ctx = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else torch.no_grad()
            with ctx:
                need_hidden = any(idx is not None for _, idx in layer_specs)
                hidden_states = None
                if need_hidden:
                    outputs = base(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = outputs.hidden_states
                for name, idx in layer_specs:
                    if idx is None:
                        h = emb_layer(input_ids)
                    else:
                        h = hidden_states[idx]
                    pooled = masked_mean_pool(h, attention_mask)
                    feats[name].append(pooled.float().cpu())
            labels_all.append(labels.cpu())

    return {k: torch.cat(v, dim=0) for k, v in feats.items()}, torch.cat(labels_all, dim=0)


def fit_probe(x_train, y_train, x_val, y_val, x_test, y_test, *, epochs, lr, weight_decay, seed, desc):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Center/scale each feature using train statistics. This is an affine preprocessing
    # step, so the resulting decision boundary is still linear in the original features.
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-5)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    head = nn.Linear(x_train.shape[1], 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    bar = tqdm(range(epochs), desc=desc, unit="epoch", leave=False, dynamic_ncols=True)
    for epoch in bar:
        head.train()
        logits = head(x_train)
        loss = loss_fn(logits, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            val_acc = (head(x_val).argmax(-1) == y_val).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if epoch % 10 == 0 or epoch + 1 == epochs:
            bar.set_postfix(loss=f"{loss.item():.3f}", val=f"{val_acc:.3f}")

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        train_logits = head(x_train)
        val_logits = head(x_val)
        test_logits = head(x_test)
        train_acc = (train_logits.argmax(-1) == y_train).float().mean().item()
        val_acc = (val_logits.argmax(-1) == y_val).float().mean().item()
        test_acc = (test_logits.argmax(-1) == y_test).float().mean().item()
        probs = torch.softmax(test_logits.float(), dim=-1)[:, 1]
        orange = probs[y_test == 1].mean().item()
        purple = probs[y_test == 0].mean().item()

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_mean_p_orange_on_orange": orange,
        "test_mean_p_orange_on_purple": purple,
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
    dtype = choose_dtype()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    print(f"Loading frozen backbone: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device
    model.config.use_cache = False
    n_layers = num_layers(model.config)
    layer_specs = select_layers(args.layers, n_layers)
    print("Probing:", ", ".join(name for name, _ in layer_specs), flush=True)

    datadir = Path(args.data)
    collate = make_collator(tok, args.max_length)
    loaders = {}
    for split in ("train", "val", "test"):
        ds = JsonlWatermarkDataset(datadir / f"{split}.jsonl")
        loaders[split] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )

    feature_sets = {}
    labels = {}
    for split in ("train", "val", "test"):
        f, y = extract_features(model, loaders[split], layer_specs, device, dtype, desc=f"extract {split}")
        feature_sets[split] = f
        labels[split] = y

    # Release the 1.7B backbone before fitting probes; cached features are tiny.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {}
    for name, _ in layer_specs:
        metrics = fit_probe(
            feature_sets["train"][name], labels["train"],
            feature_sets["val"][name], labels["val"],
            feature_sets["test"][name], labels["test"],
            epochs=args.probe_epochs, lr=args.probe_lr, weight_decay=args.probe_weight_decay,
            seed=args.seed, desc=f"probe {name}",
        )
        results[name] = metrics
        print(f"{name:>14s}  val={metrics['val_accuracy']:.3f}  test={metrics['test_accuracy']:.3f}", flush=True)

    payload = {
        "args": vars(args),
        "num_hidden_layers": n_layers,
        "results": results,
    }
    with open(outdir / "layer_probe_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print("\nLayer probe summary")
    print("representation       train    val    test")
    for name, m in results.items():
        print(f"{name:18s} {m['train_accuracy']:.3f}  {m['val_accuracy']:.3f}  {m['test_accuracy']:.3f}")
    print(f"Saved to {outdir / 'layer_probe_metrics.json'}")


if __name__ == "__main__":
    main()
