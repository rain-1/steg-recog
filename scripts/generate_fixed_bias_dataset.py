#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from wmsteg.experiment_utils import (
    choose_dtype,
    generate_rewrite_batch,
    make_output_row,
    paired_metrics,
    roc_auc,
    seed_torch,
    stat_accuracy,
    summarize_rows,
    write_jsonl,
)
from wmsteg.lyra import load_source_rows
from wmsteg.watermark import LABEL_NAMES, ORANGE, PURPLE, TokenPartition


def parse_args():
    p = argparse.ArgumentParser(description="Generate a 2x Lyra purple/orange dataset using a validated fixed-bias sampler.")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--partition-dir", required=True,
                   help="Directory produced by validate_fixed_bias.py containing partition.npz and partition_meta.json.")
    p.add_argument("--output", default="runs/lyra_fixed_bias_data")
    p.add_argument("--strength", type=float, required=True)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-entropy-nats", type=float, default=0.0)
    p.add_argument("--gate-top-k", type=int, default=0)

    p.add_argument("--pairs", type=int, default=1000)
    p.add_argument("--data-seed", type=int, default=4040,
                   help="Use a different seed from validation so full-data prompts do not intentionally reuse the pilot sample.")
    p.add_argument("--dataset-name", default="lyraaaa/synthprompts_v2_250k")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--source-jsonl", default=None)
    p.add_argument("--language", default="english")
    p.add_argument("--min-source-tokens", type=int, default=24)
    p.add_argument("--max-source-tokens", type=int, default=320)
    p.add_argument("--shuffle-buffer", type=int, default=20000)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--rewrite-max-ratio", type=float, default=1.50)
    p.add_argument("--rewrite-extra-tokens", type=int, default=48)
    p.add_argument("--max-new-tokens-cap", type=int, default=512)
    p.add_argument("--train-min-abs-z", type=float, default=1.5)
    p.add_argument("--examples", type=int, default=20)
    return p.parse_args()


def write_examples(path: Path, rows: list[dict], n: int, seed: int):
    grouped: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(row["split"], int(row["pair_id"]))][int(row["label"])] = row
    pairs = [colors for colors in grouped.values() if PURPLE in colors and ORANGE in colors]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:n]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Fixed-bias Lyra dataset examples\n\n")
        for i, colors in enumerate(pairs, 1):
            p = colors[PURPLE]
            o = colors[ORANGE]
            f.write(f"## Example {i} — {p['split']} — {p.get('mode')} / {p.get('topic')}\n\n")
            f.write("**Original**\n\n" + p["source_prompt"] + "\n\n")
            f.write(f"**Purple** — z={p['z_score_purple_minus_orange']:.2f}, KL={p['mean_step_kl']:.3f}\n\n{p['text']}\n\n")
            f.write(f"**Orange** — z={o['z_score_purple_minus_orange']:.2f}, KL={o['mean_step_kl']:.3f}\n\n{o['text']}\n\n---\n\n")


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(args.data_seed)
    np.random.seed(args.data_seed)
    seed_torch(args.data_seed)

    partition = TokenPartition.load(args.partition_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    language = None if str(args.language).lower() == "any" else args.language
    sources = load_source_rows(
        tokenizer,
        n=args.pairs,
        seed=args.data_seed,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        source_jsonl=args.source_jsonl,
        language=language,
        min_source_tokens=args.min_source_tokens,
        max_source_tokens=args.max_source_tokens,
        shuffle_buffer=args.shuffle_buffer,
    )
    write_jsonl(outdir / "sources.jsonl", sources)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=choose_dtype(), device_map="auto")
    model.eval()
    if int(model.get_input_embeddings().weight.shape[0]) != partition.vocab_size:
        raise RuntimeError("Partition vocabulary does not match model vocabulary")

    n_train = int(args.pairs * 0.8)
    n_val = int(args.pairs * 0.1)
    split_sources = {
        "train": sources[:n_train],
        "val": sources[n_train:n_train + n_val],
        "test": sources[n_train + n_val:],
    }

    rows_by_split: dict[str, list[dict]] = {}
    all_for_examples: list[dict] = []
    for split_idx, split in enumerate(("train", "val", "test")):
        srcs = split_sources[split]
        rows: list[dict] = []
        seed_base = args.data_seed + 10_000 + split_idx * 10_000
        for label in (PURPLE, ORANGE):
            with tqdm(total=len(srcs), desc=f"{split}/{LABEL_NAMES[label]}", unit="prompt", dynamic_ncols=True) as pbar:
                for start in range(0, len(srcs), args.batch_size):
                    batch = srcs[start:start + args.batch_size]
                    texts, stats = generate_rewrite_batch(
                        model,
                        tokenizer,
                        partition,
                        batch,
                        label=label,
                        strength=args.strength,
                        temperature=args.temperature,
                        min_entropy_nats=args.min_entropy_nats,
                        gate_top_k=args.gate_top_k,
                        symmetric=args.symmetric,
                        rewrite_max_ratio=args.rewrite_max_ratio,
                        rewrite_extra_tokens=args.rewrite_extra_tokens,
                        max_new_tokens_cap=args.max_new_tokens_cap,
                        generator_seed=seed_base + start,
                    )
                    for i, (src, text, st) in enumerate(zip(batch, texts, stats)):
                        rows.append(make_output_row(
                            partition,
                            tokenizer,
                            src,
                            text=text,
                            label=label,
                            split=split,
                            pair_id=start + i,
                            sampler_stats=st,
                            strength=args.strength,
                        ))
                    pbar.update(len(batch))
        random.Random(args.data_seed + split_idx).shuffle(rows)
        rows_by_split[split] = rows
        all_for_examples.extend(rows)

    # Train-only pair filtering. Validation/test retain the natural weak tail.
    by_pair: dict[int, dict[int, dict]] = defaultdict(dict)
    for row in rows_by_split["train"]:
        by_pair[int(row["pair_id"])][int(row["label"])] = row
    retained = set()
    rejected = set()
    for pid, colors in by_pair.items():
        if PURPLE not in colors or ORANGE not in colors:
            rejected.add(pid)
            continue
        pz = float(colors[PURPLE]["z_score_purple_minus_orange"])
        oz = float(colors[ORANGE]["z_score_purple_minus_orange"])
        if args.train_min_abs_z <= 0 or (pz >= args.train_min_abs_z and oz <= -args.train_min_abs_z):
            retained.add(pid)
        else:
            rejected.add(pid)

    train_all = rows_by_split["train"]
    train = [r for r in train_all if int(r["pair_id"]) in retained]
    train_rejected = [r for r in train_all if int(r["pair_id"]) in rejected]
    write_jsonl(outdir / "train_all.jsonl", train_all)
    write_jsonl(outdir / "train.jsonl", train)
    write_jsonl(outdir / "train_rejected_weak.jsonl", train_rejected)
    write_jsonl(outdir / "val.jsonl", rows_by_split["val"])
    write_jsonl(outdir / "test.jsonl", rows_by_split["test"])
    partition.save(outdir)

    summary = {
        "methodology_version": "5.1",
        "model": args.model,
        "watermark": {
            "sampler": "full_softmax_fixed_color_bias",
            "strength": args.strength,
            "temperature": args.temperature,
            "symmetric": args.symmetric,
            "sampling_truncation": "none",
            "min_entropy_nats": args.min_entropy_nats,
            "gate_top_k": args.gate_top_k,
        },
        "partition": partition.metadata,
        "train_filter": {
            "threshold_abs_z": args.train_min_abs_z,
            "candidate_pairs": len(by_pair),
            "retained_pairs": len(retained),
            "retention_rate": len(retained) / max(1, len(by_pair)),
        },
    }
    for name, rows in {
        "train_all": train_all,
        "train": train,
        "val": rows_by_split["val"],
        "test": rows_by_split["test"],
    }.items():
        p_rows = [r for r in rows if r["label"] == PURPLE]
        o_rows = [r for r in rows if r["label"] == ORANGE]
        summary[name] = {
            "purple": summarize_rows(p_rows),
            "orange": summarize_rows(o_rows),
            "count_accuracy_at_0p5": stat_accuracy(rows, 0.5),
            "auc": roc_auc(rows),
            "paired": paired_metrics(p_rows, o_rows, z_threshold=args.train_min_abs_z),
        }

    with (outdir / "data_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (outdir / "generation_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    write_examples(outdir / "examples_side_by_side.md", all_for_examples, args.examples, args.data_seed)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote dataset to {outdir}", flush=True)


if __name__ == "__main__":
    main()
