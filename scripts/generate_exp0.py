#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from tqdm.auto import tqdm

from wmsteg.prompts import make_prompt_pool
from wmsteg.watermark import (
    LABEL_NAMES,
    ORANGE,
    PURPLE,
    SymmetricColorBiasLogitsProcessor,
    TokenPartition,
    build_frequency_balanced_hyperplane_partition,
    build_frequency_balanced_random_partition,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Experiment-0 data with a frequency-balanced, symmetric watermark."
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output", default="runs/exp0_v2_data")
    p.add_argument("--pairs", type=int, default=500,
                   help="Total paired prompts across train/val/test. Output has 2x labeled rows.")
    p.add_argument("--partition", choices=["hyperplane", "random"], default="hyperplane")
    p.add_argument("--partition-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=2025)

    # Fundamental calibration changes relative to v1.
    p.add_argument("--calibration-samples", type=int, default=128,
                   help="Unwatermarked TRAIN-topic generations used to estimate natural token frequencies.")
    p.add_argument("--control-samples", type=int, default=64,
                   help="Independent unwatermarked VAL/TEST-topic generations saved as a held-out control.")
    p.add_argument("--neutral-top-k", type=int, default=256,
                   help="Neutralize the most frequent calibration token ids, plus special tokens.")
    p.add_argument("--neutral-margin-mass", type=float, default=0.10,
                   help="Neutralize this empirical mass nearest the hyperplane, half from each side.")

    # Symmetric watermark. `auto` runs a cheap pilot on train topics and chooses the
    # weakest candidate that satisfies the requested separation.
    p.add_argument("--strength", default="auto",
                   help="Symmetric logit magnitude, e.g. 1.0, or 'auto'.")
    p.add_argument("--strength-candidates", default="0.50,0.75,1.00,1.25,1.50,2.00")
    p.add_argument("--pilot-prompts", type=int, default=16)
    p.add_argument("--pilot-min-gap", type=float, default=0.45,
                   help="Required mean purple-fraction gap in the strength pilot.")
    p.add_argument("--pilot-min-accuracy", type=float, default=0.90,
                   help="Required simple count-detector pilot accuracy at threshold 0.5.")

    # Fixed-length carrier passages remove length as a potential class cue and give
    # every sequence roughly the same number of watermark observations.
    p.add_argument("--carrier-tokens", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)

    # Dataset-level gate: do not spend detector-training GPU time on a bad positive control.
    p.add_argument("--sanity-control-tolerance", type=float, default=0.08,
                   help="Max |held-out unwatermarked mean purple fraction - 0.5|.")
    p.add_argument("--sanity-min-test-stat-accuracy", type=float, default=0.90)
    p.add_argument("--sanity-min-test-gap", type=float, default=0.40)
    p.add_argument("--sanity-min-colored-fraction", type=float, default=0.20)
    p.add_argument("--allow-failed-sanity", action="store_true")
    return p.parse_args()


def choose_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def apply_chat(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_batch(model, tokenizer, prompts, *, processor=None, args=None):
    formatted = [apply_chat(tokenizer, p) for p in prompts]
    enc = tokenizer(formatted, return_tensors="pt", padding=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]
    lp = LogitsProcessorList([processor]) if processor is not None else LogitsProcessorList()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_new_tokens=args.carrier_tokens,
            max_new_tokens=args.carrier_tokens,
            logits_processor=lp,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = out[:, input_len:]
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    raw_ids = [row.tolist() for row in completion_ids]
    return texts, raw_ids


def retokenize(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def row_from_text(
    partition: TokenPartition,
    tokenizer,
    text: str,
    *,
    label: int | None,
    split: str,
    source_prompt: str,
    pair_id: int | None,
) -> dict:
    token_ids = retokenize(tokenizer, text)
    p, o, n = partition.counts(token_ids)
    colored = p + o
    total = p + o + n
    row = {
        "pair_id": pair_id,
        "split": split,
        "label": int(label) if label is not None else None,
        "watermark": LABEL_NAMES[label] if label is not None else "none",
        "text": text,
        "source_prompt": source_prompt,
        "purple_tokens": p,
        "orange_tokens": o,
        "neutral_tokens": n,
        "token_length": total,
        "colored_fraction_total": colored / max(1, total),
        "purple_fraction_colored": p / max(1, colored),
        "z_score_purple_minus_orange": partition.z_score(token_ids),
    }
    return row


def generate_unwatermarked_rows(
    model,
    tokenizer,
    prompts: list[str],
    n: int,
    *,
    args,
    seed: int,
    desc: str,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seed_torch(seed)
    with tqdm(total=n, desc=desc, unit="prompt", dynamic_ncols=True) as pbar:
        for start in range(0, n, args.batch_size):
            batch = [prompts[(start + i) % len(prompts)] for i in range(min(args.batch_size, n - start))]
            texts, _ = generate_batch(model, tokenizer, batch, processor=None, args=args)
            rows.extend(zip(batch, texts))
            pbar.update(len(batch))
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_purple_fraction_colored": float(np.mean([r["purple_fraction_colored"] for r in rows])),
        "std_purple_fraction_colored": float(np.std([r["purple_fraction_colored"] for r in rows])),
        "mean_z_score": float(np.mean([r["z_score_purple_minus_orange"] for r in rows])),
        "std_z_score": float(np.std([r["z_score_purple_minus_orange"] for r in rows])),
        "mean_colored_fraction_total": float(np.mean([r["colored_fraction_total"] for r in rows])),
        "mean_token_length": float(np.mean([r["token_length"] for r in rows])),
    }


def simple_stat_accuracy(rows: list[dict], threshold: float = 0.5) -> float:
    if not rows:
        return float("nan")
    correct = 0
    for r in rows:
        # Purple has label 0 and should have fraction >= threshold.
        pred = PURPLE if r["purple_fraction_colored"] >= threshold else ORANGE
        correct += int(pred == r["label"])
    return correct / len(rows)


def best_train_threshold(rows: list[dict]) -> float:
    """Choose a scalar purple-fraction threshold using TRAIN only."""
    scores = np.array([r["purple_fraction_colored"] for r in rows], dtype=np.float64)
    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    uniq = np.unique(scores)
    if uniq.size == 1:
        return float(uniq[0])
    candidates = np.concatenate((
        [uniq[0] - 1e-9],
        (uniq[:-1] + uniq[1:]) / 2.0,
        [uniq[-1] + 1e-9],
    ))
    best_t, best_acc = 0.5, -1.0
    for t in candidates:
        pred = np.where(scores >= t, PURPLE, ORANGE)
        acc = float(np.mean(pred == labels))
        # Tie-break toward the theoretically intended 0.5 threshold.
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_acc = float(t), acc
    return best_t


def threshold_accuracy(rows: list[dict], threshold: float) -> float:
    scores = np.array([r["purple_fraction_colored"] for r in rows], dtype=np.float64)
    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    pred = np.where(scores >= threshold, PURPLE, ORANGE)
    return float(np.mean(pred == labels))


def roc_auc_purple_score(rows: list[dict]) -> float:
    """Pairwise AUC where higher purple fraction means purple."""
    p = np.array([r["purple_fraction_colored"] for r in rows if r["label"] == PURPLE])
    o = np.array([r["purple_fraction_colored"] for r in rows if r["label"] == ORANGE])
    if not len(p) or not len(o):
        return float("nan")
    # Small sprint datasets make this explicit pairwise implementation cheap and dependency-free.
    gt = (p[:, None] > o[None, :]).mean()
    eq = (p[:, None] == o[None, :]).mean()
    return float(gt + 0.5 * eq)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_strength_candidates(spec: str) -> list[float]:
    xs = sorted({float(x.strip()) for x in spec.split(",") if x.strip()})
    if not xs or xs[0] < 0:
        raise ValueError("--strength-candidates must contain non-negative floats")
    return xs


def pilot_strength(
    model,
    tokenizer,
    partition: TokenPartition,
    prompts: list[str],
    *,
    args,
) -> tuple[float, list[dict]]:
    if args.strength.lower() != "auto":
        return float(args.strength), []

    n = min(args.pilot_prompts, max(2, len(prompts)))
    pilot_prompts = [prompts[i % len(prompts)] for i in range(n)]
    results = []
    candidates = parse_strength_candidates(args.strength_candidates)
    selected = None

    print("\nSelecting watermark strength with a TRAIN-topic pilot...", flush=True)
    for strength in candidates:
        class_rows: list[dict] = []
        for label in (PURPLE, ORANGE):
            proc = SymmetricColorBiasLogitsProcessor(partition, label, strength)
            # Same random stream for both colors at a given strength: coupled sampling
            # reduces nuisance variance between the paired generations.
            seed_torch(args.data_seed + 9000)
            with tqdm(total=n, desc=f"pilot s={strength:.2f} {LABEL_NAMES[label]}", unit="prompt",
                      leave=False, dynamic_ncols=True) as pbar:
                for start in range(0, n, args.batch_size):
                    batch = pilot_prompts[start:start + args.batch_size]
                    texts, _ = generate_batch(model, tokenizer, batch, processor=proc, args=args)
                    for i, (prompt, text) in enumerate(zip(batch, texts)):
                        class_rows.append(row_from_text(
                            partition, tokenizer, text, label=label, split="pilot",
                            source_prompt=prompt, pair_id=start + i,
                        ))
                    pbar.update(len(batch))
        purple_rows = [r for r in class_rows if r["label"] == PURPLE]
        orange_rows = [r for r in class_rows if r["label"] == ORANGE]
        p_mean = float(np.mean([r["purple_fraction_colored"] for r in purple_rows]))
        o_mean = float(np.mean([r["purple_fraction_colored"] for r in orange_rows]))
        gap = p_mean - o_mean
        acc = simple_stat_accuracy(class_rows, threshold=0.5)
        record = {
            "strength": strength,
            "purple_mean_fraction": p_mean,
            "orange_mean_fraction": o_mean,
            "gap": gap,
            "count_accuracy_at_0.5": acc,
        }
        results.append(record)
        print(json.dumps(record), flush=True)
        if selected is None and gap >= args.pilot_min_gap and acc >= args.pilot_min_accuracy:
            selected = strength
            break

    if selected is None:
        selected = candidates[-1]
        print(
            f"WARNING: no pilot candidate met gap>={args.pilot_min_gap} and "
            f"accuracy>={args.pilot_min_accuracy}; using strongest candidate {selected:.2f}.",
            flush=True,
        )
    else:
        print(f"Selected symmetric watermark strength: {selected:.2f}", flush=True)
    return selected, results


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(args.data_seed)
    np.random.seed(args.data_seed)
    seed_torch(args.data_seed)

    dtype = choose_dtype()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    model.eval()
    print("Model loaded.", flush=True)

    pools = make_prompt_pool(args.data_seed)

    # -------------------------------------------------------------------------
    # 1) Unwatermarked calibration on TRAIN topics only.
    # -------------------------------------------------------------------------
    print("\n=== 1/5: unwatermarked frequency calibration ===", flush=True)
    calibration_pairs = generate_unwatermarked_rows(
        model, tokenizer, pools["train"], args.calibration_samples,
        args=args, seed=args.data_seed + 1000, desc="Calibration (train topics)",
    )

    vocab_size = int(model.get_input_embeddings().weight.shape[0])
    token_counts = np.zeros(vocab_size, dtype=np.float64)
    counter = Counter()
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    for _, text in calibration_pairs:
        ids = retokenize(tokenizer, text)
        counter.update(tok for tok in ids if 0 <= tok < vocab_size and tok not in special_ids)
    for tok, count in counter.items():
        token_counts[tok] = count

    frequent_ids = {tok for tok, _ in counter.most_common(args.neutral_top_k)}
    neutral_ids = special_ids | frequent_ids

    # -------------------------------------------------------------------------
    # 2) Frequency-balanced partition, then score calibration + held-out control.
    # -------------------------------------------------------------------------
    print("\n=== 2/5: build frequency-balanced token partition ===", flush=True)
    emb = model.get_input_embeddings().weight
    if args.partition == "hyperplane":
        partition = build_frequency_balanced_hyperplane_partition(
            emb,
            token_counts,
            neutral_ids=neutral_ids,
            seed=args.partition_seed,
            neutral_margin_mass=args.neutral_margin_mass,
        )
    else:
        partition = build_frequency_balanced_random_partition(
            vocab_size,
            token_counts,
            neutral_ids=neutral_ids,
            seed=args.partition_seed,
        )

    partition.metadata.update({
        "model": args.model,
        "neutral_top_k": args.neutral_top_k,
        "calibration_samples": args.calibration_samples,
        "neutral_margin_mass": args.neutral_margin_mass,
        "frequency_balance_source": "unwatermarked train-topic carrier text after decode/re-tokenize",
    })
    partition.save(outdir)

    calibration_rows = [
        row_from_text(partition, tokenizer, text, label=None, split="calibration",
                      source_prompt=prompt, pair_id=i)
        for i, (prompt, text) in enumerate(calibration_pairs)
    ]
    write_jsonl(outdir / "calibration.jsonl", calibration_rows)

    heldout_prompts = pools["val"] + pools["test"]
    control_pairs = generate_unwatermarked_rows(
        model, tokenizer, heldout_prompts, args.control_samples,
        args=args, seed=args.data_seed + 2000, desc="Control (held-out topics)",
    )
    control_rows = [
        row_from_text(partition, tokenizer, text, label=None, split="control",
                      source_prompt=prompt, pair_id=i)
        for i, (prompt, text) in enumerate(control_pairs)
    ]
    write_jsonl(outdir / "control.jsonl", control_rows)

    cal_summary = summarize_rows(calibration_rows)
    control_summary = summarize_rows(control_rows)
    print("Calibration color balance:", json.dumps(cal_summary, indent=2), flush=True)
    print("Held-out unwatermarked control:", json.dumps(control_summary, indent=2), flush=True)

    # -------------------------------------------------------------------------
    # 3) Select a symmetric watermark strength using TRAIN topics only.
    # -------------------------------------------------------------------------
    print("\n=== 3/5: watermark-strength pilot ===", flush=True)
    selected_strength, pilot_results = pilot_strength(
        model, tokenizer, partition, pools["train"], args=args
    )
    partition.metadata["selected_strength"] = selected_strength
    partition.save(outdir)

    # -------------------------------------------------------------------------
    # 4) Generate paired purple/orange train/val/test examples.
    # -------------------------------------------------------------------------
    print("\n=== 4/5: paired watermarked dataset generation ===", flush=True)
    split_pairs = {
        "train": int(args.pairs * 0.8),
        "val": int(args.pairs * 0.1),
    }
    split_pairs["test"] = args.pairs - split_pairs["train"] - split_pairs["val"]

    processors = {
        PURPLE: SymmetricColorBiasLogitsProcessor(partition, PURPLE, selected_strength),
        ORANGE: SymmetricColorBiasLogitsProcessor(partition, ORANGE, selected_strength),
    }

    rng = random.Random(args.data_seed)
    rows_by_split: dict[str, list[dict]] = {}
    for split_index, split in enumerate(("train", "val", "test")):
        n_pairs = split_pairs[split]
        prompts = pools[split]
        pair_prompts = [prompts[i % len(prompts)] for i in range(n_pairs)]
        rng.shuffle(pair_prompts)
        rows = []
        # Same RNG stream is reset for both colors inside a split. The two members
        # of each pair therefore use coupled sampling randomness and differ only in
        # the signed watermark intervention (plus autoregressive consequences).
        pair_seed = args.data_seed + 3000 + split_index * 100
        for label in (PURPLE, ORANGE):
            seed_torch(pair_seed)
            with tqdm(total=n_pairs, desc=f"Generate {split}/{LABEL_NAMES[label]}",
                      unit="prompt", dynamic_ncols=True) as pbar:
                for start in range(0, n_pairs, args.batch_size):
                    batch_prompts = pair_prompts[start:start + args.batch_size]
                    texts, _ = generate_batch(
                        model, tokenizer, batch_prompts, processor=processors[label], args=args
                    )
                    for local_i, (prompt, text) in enumerate(zip(batch_prompts, texts)):
                        rows.append(row_from_text(
                            partition, tokenizer, text, label=label, split=split,
                            source_prompt=prompt, pair_id=start + local_i,
                        ))
                    pbar.update(len(batch_prompts))
        rng.shuffle(rows)
        rows_by_split[split] = rows
        write_jsonl(outdir / f"{split}.jsonl", rows)

    # -------------------------------------------------------------------------
    # 5) Quantitative sanity gate.
    # -------------------------------------------------------------------------
    print("\n=== 5/5: dataset sanity gate ===", flush=True)
    train_threshold = best_train_threshold(rows_by_split["train"])
    summary: dict = {
        "methodology_version": 2,
        "selected_strength": selected_strength,
        "partition": partition.metadata,
        "calibration": cal_summary,
        "control": control_summary,
        "strength_pilot": pilot_results,
        "stat_classifier": {
            "train_selected_threshold": train_threshold,
        },
    }

    for split in ("train", "val", "test"):
        rows = rows_by_split[split]
        purple_rows = [r for r in rows if r["label"] == PURPLE]
        orange_rows = [r for r in rows if r["label"] == ORANGE]
        p_stats = summarize_rows(purple_rows)
        o_stats = summarize_rows(orange_rows)
        gap = p_stats["mean_purple_fraction_colored"] - o_stats["mean_purple_fraction_colored"]
        summary[split] = {
            "purple": p_stats,
            "orange": o_stats,
            "mean_fraction_gap": gap,
            "count_accuracy_at_0.5": simple_stat_accuracy(rows, 0.5),
            "count_accuracy_train_threshold": threshold_accuracy(rows, train_threshold),
            "purple_fraction_auc": roc_auc_purple_score(rows),
        }
        summary["stat_classifier"][f"{split}_accuracy"] = threshold_accuracy(rows, train_threshold)

    test_gap = summary["test"]["mean_fraction_gap"]
    test_stat_acc = summary["test"]["count_accuracy_train_threshold"]
    test_colored = min(
        summary["test"]["purple"]["mean_colored_fraction_total"],
        summary["test"]["orange"]["mean_colored_fraction_total"],
    )
    control_offset = abs(control_summary["mean_purple_fraction_colored"] - 0.5)
    checks = {
        "heldout_control_near_half": {
            "pass": bool(control_offset <= args.sanity_control_tolerance),
            "value_abs_offset": control_offset,
            "max": args.sanity_control_tolerance,
        },
        "test_count_detector_accuracy": {
            "pass": bool(test_stat_acc >= args.sanity_min_test_stat_accuracy),
            "value": test_stat_acc,
            "min": args.sanity_min_test_stat_accuracy,
        },
        "test_mean_class_gap": {
            "pass": bool(test_gap >= args.sanity_min_test_gap),
            "value": test_gap,
            "min": args.sanity_min_test_gap,
        },
        "test_colored_token_fraction": {
            "pass": bool(test_colored >= args.sanity_min_colored_fraction),
            "value": test_colored,
            "min": args.sanity_min_colored_fraction,
        },
    }
    summary["sanity_checks"] = checks
    summary["sanity_pass"] = all(v["pass"] for v in checks.values())

    config = vars(args).copy()
    config["selected_strength"] = selected_strength
    with open(outdir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    with open(outdir / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote Experiment-0 v2 data to {outdir}", flush=True)

    if not summary["sanity_pass"]:
        failed = [name for name, result in checks.items() if not result["pass"]]
        message = (
            "Dataset failed the positive-control sanity gate: " + ", ".join(failed) +
            ". Detector training should not start until the carrier data is clean."
        )
        if args.allow_failed_sanity:
            print("WARNING: " + message, flush=True)
        else:
            raise SystemExit(message)


if __name__ == "__main__":
    main()
