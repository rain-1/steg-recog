#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from wmsteg.eligibility import build_english_lexical_eligibility
from wmsteg.lyra import load_source_rows
from wmsteg.sampling import (
    ActualDistributionKLSampler,
    SamplingConfig,
    generate_with_actual_distribution_kl,
)
from wmsteg.watermark import (
    LABEL_NAMES,
    ORANGE,
    PURPLE,
    TokenPartition,
    build_frequency_balanced_hyperplane_partition,
)


REWRITE_SYSTEM = """You are rewriting a user message, not answering it.
Produce a natural paraphrase that preserves the original user's meaning, factual details, requests, constraints, uncertainty, tone, language, and level of specificity. Preserve typos/slang/roughness when they are part of the user's voice. Keep roughly the same length. Do not add new facts or advice. Do not solve the user's task. Return only the rewritten user message, with no labels, quotes, commentary, or explanation."""


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Rewrite Lyra SynthPrompts into paired purple/orange prompts using a "
            "frequency-balanced embedding hyperplane and KL-constrained sampling "
            "on the exact post-temperature/top-k/top-p distribution."
        )
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output", default="runs/exp0_lyra_actual_kl")
    p.add_argument("--pairs", type=int, default=1000,
                   help="Underlying source prompts across train/val/test; each gets purple+orange rewrites.")
    p.add_argument("--data-seed", type=int, default=2025)
    p.add_argument("--partition-seed", type=int, default=1234)

    # Lyra source selection.
    p.add_argument("--dataset-name", default="lyraaaa/synthprompts_v2_250k")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--source-jsonl", default=None,
                   help="Optional local Lyra-schema JSONL; avoids Hugging Face loading.")
    p.add_argument("--language", default="english",
                   help="Lyra language value to keep; use 'any' to disable language filtering.")
    p.add_argument("--min-source-tokens", type=int, default=24)
    p.add_argument("--max-source-tokens", type=int, default=320)
    p.add_argument("--shuffle-buffer", type=int, default=20000)

    # Frequency calibration + conservative lexical eligibility.
    p.add_argument("--calibration-prompts", type=int, default=4096)
    p.add_argument("--control-prompts", type=int, default=512)
    p.add_argument("--neutral-top-k", type=int, default=64,
                   help="Neutralize this many most frequent otherwise-eligible lexical tokens.")
    p.add_argument("--eligible-min-count", type=int, default=2,
                   help="A token must occur at least this many times in calibration to be watermark-eligible.")
    p.add_argument("--eligible-max-piece-chars", type=int, default=32)
    p.add_argument("--neutral-margin-mass", type=float, default=0.05,
                   help="Additionally neutralize equal empirical mass nearest both sides of the hyperplane.")

    # Exact-distribution sampler.
    p.add_argument("--kl-budget", type=float, default=0.04,
                   help="Per-token KL(q_watermarked || p_actual_base_sampler), in nats.")
    p.add_argument("--max-tilt", type=float, default=12.0,
                   help="Maximum exponential color tilt on the truncated sampling distribution.")
    p.add_argument("--kl-bisection-steps", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--min-entropy-nats", type=float, default=1.5,
                   help="Do not watermark low-entropy decoding steps.")
    p.add_argument("--min-color-mass", type=float, default=0.02,
                   help="Require at least this baseline probability mass on BOTH target and opposite colors.")
    p.add_argument("--min-eligible-mass", type=float, default=0.08,
                   help="Require at least this baseline probability mass on all colored/eligible tokens.")

    # Rewrite generation.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--rewrite-max-ratio", type=float, default=1.50)
    p.add_argument("--rewrite-extra-tokens", type=int, default=48)
    p.add_argument("--max-new-tokens-cap", type=int, default=512)

    # Weak prompts may simply be difficult to watermark. Training can discard
    # those pairs while validation/test remain unfiltered.
    p.add_argument("--train-min-abs-z", type=float, default=1.5,
                   help="Keep train pair only if purple z>=x and orange z<=-x; 0 disables filtering.")

    p.add_argument("--control-tolerance", type=float, default=0.08)
    p.add_argument("--warn-test-count-accuracy-below", type=float, default=0.75)
    p.add_argument("--examples", type=int, default=20)
    return p.parse_args()


def choose_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def retokenize(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def apply_rewrite_chat(tokenizer, source_prompt: str) -> str:
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": source_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return REWRITE_SYSTEM + "\n\nSOURCE USER MESSAGE:\n" + source_prompt + "\n\nREWRITE:\n"


def score_text(partition: TokenPartition, tokenizer, text: str) -> dict:
    ids = retokenize(tokenizer, text)
    p, o, n = partition.counts(ids)
    colored = p + o
    total = p + o + n
    return {
        "purple_tokens": p,
        "orange_tokens": o,
        "neutral_tokens": n,
        "token_length": total,
        "colored_fraction_total": colored / max(1, total),
        "purple_fraction_colored": p / max(1, colored),
        "z_score_purple_minus_orange": partition.z_score(ids),
    }


def source_row_with_score(partition: TokenPartition, tokenizer, src: dict, split: str) -> dict:
    return {
        **src,
        "split": split,
        **score_text(partition, tokenizer, src["prompt"]),
    }


def make_output_row(
    partition: TokenPartition,
    tokenizer,
    src: dict,
    *,
    text: str,
    label: int,
    split: str,
    pair_id: int,
    sampler_stats: dict,
) -> dict:
    score = score_text(partition, tokenizer, text)
    source_len = int(src["source_token_length"])
    output_len = int(score["token_length"])
    return {
        "pair_id": pair_id,
        "source_id": src["source_id"],
        "split": split,
        "label": int(label),
        "watermark": LABEL_NAMES[label],
        "text": text,
        "source_prompt": src["prompt"],
        "topic": src.get("topic"),
        "mode": src.get("mode"),
        "style": src.get("style"),
        "length": src.get("length"),
        "complexity": src.get("complexity"),
        "language": src.get("language"),
        "task_spec_level": src.get("task_spec_level"),
        "source_token_length": source_len,
        "rewrite_length_ratio": output_len / max(1, source_len),
        **score,
        **sampler_stats,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_scored(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    keys = [
        "purple_fraction_colored",
        "z_score_purple_minus_orange",
        "colored_fraction_total",
        "token_length",
        "rewrite_length_ratio",
        "mean_baseline_entropy_nats",
        "mean_baseline_target_mass",
        "mean_baseline_opposite_mass",
        "mean_baseline_eligible_mass",
        "mean_sampling_support_size",
        "mean_target_tokens_in_support",
        "mean_opposite_tokens_in_support",
        "mean_applied_tilt",
        "max_applied_tilt",
        "mean_step_kl",
        "max_step_kl",
        "mean_step_tv",
        "fraction_steps_watermark_active",
        "fraction_steps_low_entropy",
        "fraction_steps_insufficient_color_mass",
        "fraction_steps_insufficient_eligible_mass",
        "fraction_active_steps_kl_limited",
        "fraction_active_steps_at_tilt_cap",
        "fraction_sampled_target_color",
        "fraction_sampled_opposite_color",
        "fraction_sampled_neutral",
    ]
    for key in keys:
        if key not in rows[0]:
            continue
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[f"mean_{key}"] = float(vals.mean())
        out[f"p50_{key}"] = float(np.quantile(vals, 0.50))
        out[f"p95_{key}"] = float(np.quantile(vals, 0.95))
    return out


def stat_accuracy(rows: list[dict], threshold: float = 0.5) -> float:
    if not rows:
        return float("nan")
    correct = 0
    for row in rows:
        pred = PURPLE if row["purple_fraction_colored"] >= threshold else ORANGE
        correct += int(pred == int(row["label"]))
    return correct / len(rows)


def roc_auc(rows: list[dict]) -> float:
    p = np.asarray([r["purple_fraction_colored"] for r in rows if r["label"] == PURPLE])
    o = np.asarray([r["purple_fraction_colored"] for r in rows if r["label"] == ORANGE])
    if not len(p) or not len(o):
        return float("nan")
    return float((p[:, None] > o[None, :]).mean() + 0.5 * (p[:, None] == o[None, :]).mean())


def best_train_threshold(rows: list[dict]) -> float:
    scores = np.asarray([r["purple_fraction_colored"] for r in rows], dtype=np.float64)
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    uniq = np.unique(scores)
    if uniq.size <= 1:
        return float(uniq[0]) if uniq.size else 0.5
    candidates = np.concatenate(([uniq[0] - 1e-9], (uniq[:-1] + uniq[1:]) / 2.0, [uniq[-1] + 1e-9]))
    best_t, best_a = 0.5, -1.0
    for t in candidates:
        pred = np.where(scores >= t, PURPLE, ORANGE)
        a = float((pred == labels).mean())
        if a > best_a + 1e-12 or (abs(a - best_a) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_a = float(t), a
    return best_t


def threshold_accuracy(rows: list[dict], threshold: float) -> float:
    if not rows:
        return float("nan")
    scores = np.asarray([r["purple_fraction_colored"] for r in rows], dtype=np.float64)
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    pred = np.where(scores >= threshold, PURPLE, ORANGE)
    return float((pred == labels).mean())


def rewrite_batch(
    model,
    tokenizer,
    partition: TokenPartition,
    batch_sources: list[dict],
    *,
    label: int,
    args,
    generator_seed: int,
) -> tuple[list[str], list[dict]]:
    formatted = [apply_rewrite_chat(tokenizer, s["prompt"]) for s in batch_sources]
    max_source = max(int(s["source_token_length"]) for s in batch_sources)
    max_new = min(
        args.max_new_tokens_cap,
        max(24, int(math.ceil(max_source * args.rewrite_max_ratio)) + args.rewrite_extra_tokens),
    )

    sampler = ActualDistributionKLSampler(
        partition,
        label,
        SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            kl_budget=args.kl_budget,
            max_tilt=args.max_tilt,
            bisection_steps=args.kl_bisection_steps,
            min_entropy_nats=args.min_entropy_nats,
            min_color_mass=args.min_color_mass,
            min_eligible_mass=args.min_eligible_mass,
        ),
    )
    texts, stats, _ = generate_with_actual_distribution_kl(
        model,
        tokenizer,
        formatted,
        sampler,
        max_new_tokens=max_new,
        generator_seed=generator_seed,
    )
    if len(texts) != len(batch_sources) or len(stats) != len(batch_sources):
        raise RuntimeError("custom sampler returned unexpected batch size")
    return texts, stats


def write_examples_md(path: Path, rows_by_split: dict[str, list[dict]], n: int, seed: int) -> None:
    pairs: list[tuple[dict, dict]] = []
    for split in ("train_all", "val", "test"):
        grouped: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
        for row in rows_by_split.get(split, []):
            grouped[(row["split"], int(row["pair_id"]))][int(row["label"])] = row
        for _, colors in grouped.items():
            if PURPLE in colors and ORANGE in colors:
                pairs.append((colors[PURPLE], colors[ORANGE]))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:n]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Lyra exact-distribution KL watermark rewrite examples\n\n")
        f.write("Purple and orange are rewrites of the same original Lyra prompt. ")
        f.write("The target color is introduced only by the sampler; the rewrite instruction never names it.\n\n")
        for i, (p, o) in enumerate(pairs, 1):
            f.write(f"## Example {i} — {p['split']} — {p.get('mode')} / {p.get('topic')}\n\n")
            f.write("**Original**\n\n" + p["source_prompt"] + "\n\n")
            f.write(
                f"**Purple** — z={p['z_score_purple_minus_orange']:.2f}, "
                f"KL={p['mean_step_kl']:.4f}, active={p['fraction_steps_watermark_active']:.1%}, "
                f"tilt={p['mean_applied_tilt']:.2f}\n\n"
            )
            f.write(p["text"] + "\n\n")
            f.write(
                f"**Orange** — z={o['z_score_purple_minus_orange']:.2f}, "
                f"KL={o['mean_step_kl']:.4f}, active={o['fraction_steps_watermark_active']:.1%}, "
                f"tilt={o['mean_applied_tilt']:.2f}\n\n"
            )
            f.write(o["text"] + "\n\n---\n\n")


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(args.data_seed)
    np.random.seed(args.data_seed)
    seed_torch(args.data_seed)

    if args.pairs < 10:
        raise ValueError("--pairs must be at least 10")

    dtype = choose_dtype()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    language = None if str(args.language).lower() == "any" else args.language
    total_sources = args.calibration_prompts + args.control_prompts + args.pairs
    print(
        f"Loading {total_sources} unique Lyra source prompts "
        f"(language={language or 'any'}, tokens={args.min_source_tokens}-{args.max_source_tokens})...",
        flush=True,
    )
    sources = load_source_rows(
        tokenizer,
        n=total_sources,
        seed=args.data_seed,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        source_jsonl=args.source_jsonl,
        language=language,
        min_source_tokens=args.min_source_tokens,
        max_source_tokens=args.max_source_tokens,
        shuffle_buffer=args.shuffle_buffer,
    )
    calibration_sources = sources[:args.calibration_prompts]
    control_sources = sources[args.calibration_prompts:args.calibration_prompts + args.control_prompts]
    rewrite_sources = sources[args.calibration_prompts + args.control_prompts:]
    write_jsonl(outdir / "sources.jsonl", rewrite_sources)

    print(f"Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    model.eval()
    print("Model loaded.", flush=True)

    # ------------------------------------------------------------------
    # 1) Calibrate frequency + a conservative eligible lexical channel.
    # ------------------------------------------------------------------
    print("\n=== 1/4: calibrate eligible token channel + hyperplane ===", flush=True)
    vocab_size = int(model.get_input_embeddings().weight.shape[0])
    counter = Counter()
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    for src in tqdm(calibration_sources, desc="Token-frequency calibration", unit="prompt", dynamic_ncols=True):
        counter.update(
            tok for tok in retokenize(tokenizer, src["prompt"])
            if 0 <= tok < vocab_size and tok not in special_ids
        )
    token_counts = np.zeros(vocab_size, dtype=np.float64)
    for tok, count in counter.items():
        token_counts[tok] = count

    eligible_mask, eligibility_meta = build_english_lexical_eligibility(
        tokenizer,
        token_counts,
        vocab_size=vocab_size,
        neutral_top_k=args.neutral_top_k,
        min_count=args.eligible_min_count,
        max_piece_chars=args.eligible_max_piece_chars,
    )
    neutral_ids = np.flatnonzero(~eligible_mask)
    partition = build_frequency_balanced_hyperplane_partition(
        model.get_input_embeddings().weight,
        token_counts,
        neutral_ids=neutral_ids,
        seed=args.partition_seed,
        neutral_margin_mass=args.neutral_margin_mass,
    )
    partition.metadata.update({
        "model": args.model,
        "methodology_version": 4,
        "source_dataset": args.dataset_name if not args.source_jsonl else args.source_jsonl,
        "source_language": language,
        "frequency_balance_source": "original Lyra user prompts",
        "calibration_prompts": args.calibration_prompts,
        "eligibility": eligibility_meta,
        "neutral_margin_mass": args.neutral_margin_mass,
        "watermark_mode": "actual_sampling_distribution_kl_with_entropy_gating",
        "kl_budget_nats": args.kl_budget,
        "max_tilt": args.max_tilt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_entropy_nats": args.min_entropy_nats,
        "min_color_mass": args.min_color_mass,
        "min_eligible_mass": args.min_eligible_mass,
    })
    partition.save(outdir)

    calibration_scored = [source_row_with_score(partition, tokenizer, s, "calibration") for s in calibration_sources]
    control_scored = [source_row_with_score(partition, tokenizer, s, "control") for s in control_sources]
    write_jsonl(outdir / "calibration.jsonl", calibration_scored)
    write_jsonl(outdir / "control.jsonl", control_scored)
    cal_summary = summarize_scored(calibration_scored)
    control_summary = summarize_scored(control_scored)
    print("Eligibility:", json.dumps(eligibility_meta, indent=2), flush=True)
    print("Calibration:", json.dumps(cal_summary, indent=2), flush=True)
    print("Held-out original-prompt control:", json.dumps(control_summary, indent=2), flush=True)

    # ------------------------------------------------------------------
    # 2) Paired purple/orange rewrites using the exact-distribution sampler.
    # ------------------------------------------------------------------
    print("\n=== 2/4: paired exact-distribution KL rewrites ===", flush=True)
    n_train = int(args.pairs * 0.8)
    n_val = int(args.pairs * 0.1)
    split_sources = {
        "train": rewrite_sources[:n_train],
        "val": rewrite_sources[n_train:n_train + n_val],
        "test": rewrite_sources[n_train + n_val:],
    }

    rows_all: dict[str, list[dict]] = {}
    for split_idx, split in enumerate(("train", "val", "test")):
        srcs = split_sources[split]
        rows: list[dict] = []
        pair_seed = args.data_seed + 10_000 + split_idx * 10_000
        for label in (PURPLE, ORANGE):
            with tqdm(total=len(srcs), desc=f"Rewrite {split}/{LABEL_NAMES[label]}", unit="prompt", dynamic_ncols=True) as pbar:
                for start in range(0, len(srcs), args.batch_size):
                    batch_sources = srcs[start:start + args.batch_size]
                    # Same seed for corresponding purple/orange batches couples
                    # nuisance sampling randomness across the two interventions.
                    batch_seed = pair_seed + start
                    texts, sampler_stats = rewrite_batch(
                        model,
                        tokenizer,
                        partition,
                        batch_sources,
                        label=label,
                        args=args,
                        generator_seed=batch_seed,
                    )
                    for i, (src, text, ss) in enumerate(zip(batch_sources, texts, sampler_stats)):
                        rows.append(make_output_row(
                            partition,
                            tokenizer,
                            src,
                            text=text,
                            label=label,
                            split=split,
                            pair_id=start + i,
                            sampler_stats=ss,
                        ))
                    pbar.update(len(batch_sources))
        random.Random(args.data_seed + split_idx).shuffle(rows)
        rows_all[split] = rows

    # ------------------------------------------------------------------
    # 3) TRAIN-only pairwise weak-watermark filter. Val/test stay unfiltered.
    # ------------------------------------------------------------------
    print("\n=== 3/4: train-only weak-watermark filtering ===", flush=True)
    train_all = rows_all["train"]
    by_pair: dict[int, dict[int, dict]] = defaultdict(dict)
    for row in train_all:
        by_pair[int(row["pair_id"])][int(row["label"])] = row

    retained_pair_ids: set[int] = set()
    rejected_pair_ids: set[int] = set()
    for pid, colors in by_pair.items():
        if PURPLE not in colors or ORANGE not in colors:
            rejected_pair_ids.add(pid)
            continue
        if args.train_min_abs_z <= 0:
            retained_pair_ids.add(pid)
            continue
        pz = float(colors[PURPLE]["z_score_purple_minus_orange"])
        oz = float(colors[ORANGE]["z_score_purple_minus_orange"])
        if pz >= args.train_min_abs_z and oz <= -args.train_min_abs_z:
            retained_pair_ids.add(pid)
        else:
            rejected_pair_ids.add(pid)

    train_filtered = [r for r in train_all if int(r["pair_id"]) in retained_pair_ids]
    train_rejected = [r for r in train_all if int(r["pair_id"]) in rejected_pair_ids]
    write_jsonl(outdir / "train_all.jsonl", train_all)
    write_jsonl(outdir / "train.jsonl", train_filtered)
    write_jsonl(outdir / "train_rejected_weak.jsonl", train_rejected)
    write_jsonl(outdir / "val.jsonl", rows_all["val"])
    write_jsonl(outdir / "test.jsonl", rows_all["test"])
    print(
        f"Training pairs retained: {len(retained_pair_ids)}/{len(by_pair)} "
        f"({len(retained_pair_ids)/max(1,len(by_pair)):.1%}) at |z|>={args.train_min_abs_z}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 4) Summary + examples.
    # ------------------------------------------------------------------
    print("\n=== 4/4: summarize ===", flush=True)
    threshold_rows = train_filtered if train_filtered else train_all
    train_threshold = best_train_threshold(threshold_rows)
    summary = {
        "methodology_version": 4,
        "source": {
            "dataset": args.dataset_name if not args.source_jsonl else args.source_jsonl,
            "language": language,
            "pairs_requested": args.pairs,
            "calibration_prompts": args.calibration_prompts,
            "control_prompts": args.control_prompts,
            "min_source_tokens": args.min_source_tokens,
            "max_source_tokens": args.max_source_tokens,
        },
        "watermark": {
            "mode": "actual_sampling_distribution_kl_with_entropy_gating",
            "kl_budget_nats": args.kl_budget,
            "max_tilt": args.max_tilt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_entropy_nats": args.min_entropy_nats,
            "min_color_mass": args.min_color_mass,
            "min_eligible_mass": args.min_eligible_mass,
            "train_min_abs_z": args.train_min_abs_z,
        },
        "partition": partition.metadata,
        "calibration": cal_summary,
        "control": control_summary,
        "train_filter": {
            "candidate_pairs": len(by_pair),
            "retained_pairs": len(retained_pair_ids),
            "rejected_pairs": len(rejected_pair_ids),
            "retention_rate": len(retained_pair_ids) / max(1, len(by_pair)),
            "threshold_abs_z": args.train_min_abs_z,
        },
        "stat_classifier": {"train_selected_threshold": train_threshold},
    }

    row_sets = {
        "train_all": train_all,
        "train_filtered": train_filtered,
        "val": rows_all["val"],
        "test": rows_all["test"],
    }
    for name, rows in row_sets.items():
        p_rows = [r for r in rows if r["label"] == PURPLE]
        o_rows = [r for r in rows if r["label"] == ORANGE]
        p_sum = summarize_scored(p_rows)
        o_sum = summarize_scored(o_rows)
        summary[name] = {
            "purple": p_sum,
            "orange": o_sum,
            "mean_fraction_gap": (
                p_sum.get("mean_purple_fraction_colored", float("nan"))
                - o_sum.get("mean_purple_fraction_colored", float("nan"))
            ),
            "count_accuracy_at_0.5": stat_accuracy(rows, 0.5),
            "count_accuracy_train_threshold": threshold_accuracy(rows, train_threshold),
            "purple_fraction_auc": roc_auc(rows),
        }

    control_offset = abs(control_summary["mean_purple_fraction_colored"] - 0.5)
    max_observed_kl = max(
        summary[name][color].get("p95_max_step_kl", 0.0)
        for name in ("train_all", "val", "test")
        for color in ("purple", "orange")
    )
    summary["sanity_checks"] = {
        "heldout_original_prompts_near_half": {
            "pass": bool(control_offset <= args.control_tolerance),
            "value_abs_offset": control_offset,
            "max": args.control_tolerance,
        },
        "actual_sampler_kl_respects_budget": {
            "pass": bool(max_observed_kl <= args.kl_budget + 5e-4),
            "observed_p95_of_per_example_max_kl": max_observed_kl,
            "budget": args.kl_budget,
        },
    }
    summary["warnings"] = {
        "test_count_detector_below_reference": {
            "triggered": bool(summary["test"]["count_accuracy_train_threshold"] < args.warn_test_count_accuracy_below),
            "value": summary["test"]["count_accuracy_train_threshold"],
            "reference": args.warn_test_count_accuracy_below,
        },
        "most_steps_not_watermarkable": {
            "triggered": bool(
                0.5 * (
                    summary["train_all"]["purple"].get("mean_fraction_steps_watermark_active", 0.0)
                    + summary["train_all"]["orange"].get("mean_fraction_steps_watermark_active", 0.0)
                ) < 0.20
            ),
            "reference_active_fraction": 0.20,
        },
    }
    summary["sanity_pass"] = all(x["pass"] for x in summary["sanity_checks"].values())

    with (outdir / "generation_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    with (outdir / "data_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    write_examples_md(
        outdir / "examples_side_by_side.md",
        {"train_all": train_all, "val": rows_all["val"], "test": rows_all["test"]},
        args.examples,
        args.data_seed,
    )

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote v4 Lyra watermarked data to {outdir}", flush=True)
    print(f"Read first: {outdir / 'examples_side_by_side.md'}", flush=True)
    if not summary["sanity_pass"]:
        print("WARNING: one or more summary sanity checks failed; data was preserved for inspection.", flush=True)


if __name__ == "__main__":
    main()
