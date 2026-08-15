#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from wmsteg.experiment_utils import (
    build_partition_from_texts,
    choose_dtype,
    generate_plain_rewrite_batch,
    generate_rewrite_batch,
    make_output_row,
    paired_metrics,
    parse_float_csv,
    roc_auc,
    score_text,
    seed_torch,
    source_row_with_score,
    stat_accuracy,
    summarize_rows,
    write_jsonl,
)
from wmsteg.lyra import load_source_rows
from wmsteg.watermark import LABEL_NAMES, ORANGE, PURPLE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a simple fixed-bias external purple/orange watermark sampler on Lyra rewrites."
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output", default="runs/fixed_bias_validation")
    p.add_argument("--strengths", default="0.5,0.75,1.0,1.25,1.5,2.0")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-entropy-nats", type=float, default=0.0,
                   help="Optional entropy gate. 0 reproduces the ungated fixed-bias sampler.")
    p.add_argument("--gate-top-k", type=int, default=0,
                   help="Optional gate only: require both colors in raw top-k. 0 disables. This does NOT truncate sampling.")

    p.add_argument("--pairs", type=int, default=64,
                   help="Held-out Lyra prompts used for each watermark strength.")
    p.add_argument("--calibration-rewrites", type=int, default=512,
                   help="Unwatermarked Qwen rewrites used to frequency-balance the partition.")
    p.add_argument("--control-rewrites", type=int, default=96,
                   help="Separate unwatermarked rewrites used to verify the natural baseline after partition construction.")
    p.add_argument("--data-seed", type=int, default=2025)
    p.add_argument("--partition-seed", type=int, default=1234)

    p.add_argument("--dataset-name", default="lyraaaa/synthprompts_v2_250k")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--source-jsonl", default=None)
    p.add_argument("--language", default="english")
    p.add_argument("--min-source-tokens", type=int, default=24)
    p.add_argument("--max-source-tokens", type=int, default=320)
    p.add_argument("--shuffle-buffer", type=int, default=20000)

    p.add_argument("--neutral-top-k", type=int, default=0)
    p.add_argument("--eligible-min-count", type=int, default=0)
    p.add_argument("--eligible-max-piece-chars", type=int, default=48)
    p.add_argument("--neutral-margin-mass", type=float, default=0.0)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--rewrite-max-ratio", type=float, default=1.50)
    p.add_argument("--rewrite-extra-tokens", type=int, default=48)
    p.add_argument("--max-new-tokens-cap", type=int, default=512)
    p.add_argument("--examples", type=int, default=16)
    p.add_argument("--paired-z-threshold", type=float, default=1.5)
    return p.parse_args()


def generate_plain_set(model, tokenizer, sources, args, desc: str, seed_base: int):
    rows = []
    raw_texts = []
    with tqdm(total=len(sources), desc=desc, unit="prompt", dynamic_ncols=True) as pbar:
        for start in range(0, len(sources), args.batch_size):
            batch = sources[start:start + args.batch_size]
            texts, stats = generate_plain_rewrite_batch(
                model,
                tokenizer,
                batch,
                temperature=args.temperature,
                rewrite_max_ratio=args.rewrite_max_ratio,
                rewrite_extra_tokens=args.rewrite_extra_tokens,
                max_new_tokens_cap=args.max_new_tokens_cap,
                generator_seed=seed_base + start,
            )
            for i, (src, text, st) in enumerate(zip(batch, texts, stats)):
                raw_texts.append(text)
                rows.append((start + i, src, text, st))
            pbar.update(len(batch))
    return rows, raw_texts


def generate_color_rows(model, tokenizer, partition, sources, args, strength: float, label: int, seed_base: int):
    rows = []
    with tqdm(
        total=len(sources),
        desc=f"strength={strength:g} {LABEL_NAMES[label]}",
        unit="prompt",
        dynamic_ncols=True,
    ) as pbar:
        for start in range(0, len(sources), args.batch_size):
            batch = sources[start:start + args.batch_size]
            texts, stats = generate_rewrite_batch(
                model,
                tokenizer,
                partition,
                batch,
                label=label,
                strength=strength,
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
                    split="validation",
                    pair_id=start + i,
                    sampler_stats=st,
                    strength=strength,
                ))
            pbar.update(len(batch))
    return rows


def write_examples(path: Path, sources: list[dict], control_rows: list[dict], purple_rows: list[dict], orange_rows: list[dict], n: int, seed: int):
    c = {int(r["pair_id"]): r for r in control_rows}
    p = {int(r["pair_id"]): r for r in purple_rows}
    o = {int(r["pair_id"]): r for r in orange_rows}
    ids = sorted(set(c) & set(p) & set(o))
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[:n]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Fixed-bias watermark validation examples\n\n")
        f.write("All three rewrites use the same underlying Lyra source prompt. The control has no watermark bias.\n\n")
        for j, pid in enumerate(ids, 1):
            src = sources[pid]
            cr, pr, orr = c[pid], p[pid], o[pid]
            f.write(f"## Example {j} — {src.get('mode')} / {src.get('topic')}\n\n")
            f.write("**Original**\n\n" + src["prompt"] + "\n\n")
            f.write(
                f"**Control** — z={cr['z_score_purple_minus_orange']:.2f}, "
                f"purple={cr['purple_fraction_colored']:.3f}\n\n{cr['text']}\n\n"
            )
            f.write(
                f"**Purple** — z={pr['z_score_purple_minus_orange']:.2f}, "
                f"purple={pr['purple_fraction_colored']:.3f}, "
                f"KL={pr['mean_step_kl']:.3f}, active={pr['fraction_steps_watermark_active']:.1%}\n\n"
                + pr["text"] + "\n\n"
            )
            f.write(
                f"**Orange** — z={orr['z_score_purple_minus_orange']:.2f}, "
                f"purple={orr['purple_fraction_colored']:.3f}, "
                f"KL={orr['mean_step_kl']:.3f}, active={orr['fraction_steps_watermark_active']:.1%}\n\n"
                + orr["text"] + "\n\n---\n\n"
            )


def main() -> None:
    args = parse_args()
    strengths = parse_float_csv(args.strengths)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    random.seed(args.data_seed)
    np.random.seed(args.data_seed)
    seed_torch(args.data_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    language = None if str(args.language).lower() == "any" else args.language
    total = args.calibration_rewrites + args.control_rewrites + args.pairs
    print(f"Loading {total} Lyra prompts...", flush=True)
    sources = load_source_rows(
        tokenizer,
        n=total,
        seed=args.data_seed,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        source_jsonl=args.source_jsonl,
        language=language,
        min_source_tokens=args.min_source_tokens,
        max_source_tokens=args.max_source_tokens,
        shuffle_buffer=args.shuffle_buffer,
    )
    cal_sources = sources[:args.calibration_rewrites]
    control_sources = sources[args.calibration_rewrites:args.calibration_rewrites + args.control_rewrites]
    eval_sources = sources[args.calibration_rewrites + args.control_rewrites:]
    write_jsonl(outdir / "eval_sources.jsonl", eval_sources)

    print(f"Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=choose_dtype(), device_map="auto")
    model.eval()

    print("\n=== 1/4: generate unwatermarked calibration rewrites ===", flush=True)
    cal_raw, cal_texts = generate_plain_set(
        model, tokenizer, cal_sources, args,
        desc="calibration rewrites", seed_base=args.data_seed + 10_000,
    )
    # Save before partitioning: these are the natural rewrite corpus used for frequency balancing.
    with (outdir / "calibration_rewrites_raw.jsonl").open("w", encoding="utf-8") as f:
        for pid, src, text, st in cal_raw:
            f.write(json.dumps({"pair_id": pid, "source_id": src["source_id"], "source_prompt": src["prompt"], "text": text, **st}, ensure_ascii=False) + "\n")

    print("\n=== 2/4: build frequency-balanced eligible hyperplane ===", flush=True)
    partition, eligibility_meta = build_partition_from_texts(
        model,
        tokenizer,
        cal_texts,
        partition_seed=args.partition_seed,
        neutral_top_k=args.neutral_top_k,
        eligible_min_count=args.eligible_min_count,
        eligible_max_piece_chars=args.eligible_max_piece_chars,
        neutral_margin_mass=args.neutral_margin_mass,
        metadata_extra={
            "model": args.model,
            "source_dataset": args.dataset_name if not args.source_jsonl else args.source_jsonl,
            "source_language": language,
            "watermark_sampler": "full_softmax_fixed_color_bias",
            "temperature": args.temperature,
            "symmetric": args.symmetric,
        },
    )
    partition.save(outdir)
    print(json.dumps(partition.metadata, indent=2), flush=True)

    # Score calibration rewrites *after* partition construction.
    calibration_scored = []
    for pid, src, text, st in cal_raw:
        calibration_scored.append({
            "pair_id": pid,
            "source_id": src["source_id"],
            "source_prompt": src["prompt"],
            "text": text,
            **score_text(partition, tokenizer, text),
            **st,
        })
    write_jsonl(outdir / "calibration_rewrites.jsonl", calibration_scored)

    print("\n=== 3/4: held-out unwatermarked rewrite control ===", flush=True)
    control_raw, _ = generate_plain_set(
        model, tokenizer, control_sources, args,
        desc="held-out controls", seed_base=args.data_seed + 20_000,
    )
    heldout_control = []
    for pid, src, text, st in control_raw:
        heldout_control.append(make_output_row(
            partition,
            tokenizer,
            src,
            text=text,
            label=None,
            split="control",
            pair_id=pid,
            sampler_stats=st,
            strength=0.0,
        ))
    write_jsonl(outdir / "heldout_control_rewrites.jsonl", heldout_control)

    # One unwatermarked rewrite of every eval source, for direct quality comparison.
    eval_control_raw, _ = generate_plain_set(
        model, tokenizer, eval_sources, args,
        desc="eval controls", seed_base=args.data_seed + 30_000,
    )
    eval_control = []
    for pid, src, text, st in eval_control_raw:
        eval_control.append(make_output_row(
            partition,
            tokenizer,
            src,
            text=text,
            label=None,
            split="validation",
            pair_id=pid,
            sampler_stats=st,
            strength=0.0,
        ))
    write_jsonl(outdir / "eval_control_rewrites.jsonl", eval_control)

    print("\n=== 4/4: fixed-bias strength sweep ===", flush=True)
    control_summary = summarize_rows(heldout_control)
    eval_control_summary = summarize_rows(eval_control)
    calibration_summary = summarize_rows(calibration_scored)
    records = []
    full_summary = {
        "methodology_version": "5.1",
        "sampler": {
            "type": "full_softmax_fixed_color_bias",
            "temperature": args.temperature,
            "symmetric": args.symmetric,
            "min_entropy_nats": args.min_entropy_nats,
            "gate_top_k": args.gate_top_k,
            "sampling_truncation": "none",
        },
        "partition": partition.metadata,
        "eligibility": eligibility_meta,
        "calibration_rewrites": calibration_summary,
        "heldout_control_rewrites": control_summary,
        "eval_control_rewrites": eval_control_summary,
        "strengths": {},
    }

    for strength in strengths:
        sdir = outdir / f"strength_{str(strength).replace('.', 'p')}"
        sdir.mkdir(parents=True, exist_ok=True)
        seed_base = args.data_seed + 40_000
        purple_rows = generate_color_rows(
            model, tokenizer, partition, eval_sources, args, strength, PURPLE, seed_base
        )
        orange_rows = generate_color_rows(
            model, tokenizer, partition, eval_sources, args, strength, ORANGE, seed_base
        )
        write_jsonl(sdir / "purple.jsonl", purple_rows)
        write_jsonl(sdir / "orange.jsonl", orange_rows)

        p_sum = summarize_rows(purple_rows)
        o_sum = summarize_rows(orange_rows)
        combined = purple_rows + orange_rows
        paired = paired_metrics(
            purple_rows, orange_rows, z_threshold=args.paired_z_threshold
        )
        symmetry_error = abs(
            (p_sum["mean_purple_fraction_colored"] - 0.5)
            - (0.5 - o_sum["mean_purple_fraction_colored"])
        )
        rec = {
            "strength": strength,
            "purple_mean_z": p_sum["mean_z_score_purple_minus_orange"],
            "orange_mean_z": o_sum["mean_z_score_purple_minus_orange"],
            "purple_mean_fraction": p_sum["mean_purple_fraction_colored"],
            "orange_mean_fraction": o_sum["mean_purple_fraction_colored"],
            "fraction_gap": p_sum["mean_purple_fraction_colored"] - o_sum["mean_purple_fraction_colored"],
            "symmetry_error": symmetry_error,
            "count_accuracy_at_0p5": stat_accuracy(combined, 0.5),
            "auc": roc_auc(combined),
            "paired_pass_rate": paired.get(f"paired_pass_rate_abs_z_{args.paired_z_threshold:g}", float("nan")),
            "exact_same_pair_rate": paired.get("exact_same_pair_rate", float("nan")),
            "purple_mean_kl": p_sum.get("mean_mean_step_kl", float("nan")),
            "orange_mean_kl": o_sum.get("mean_mean_step_kl", float("nan")),
            "purple_source_similarity": p_sum["mean_source_similarity"],
            "orange_source_similarity": o_sum["mean_source_similarity"],
            "control_source_similarity": eval_control_summary["mean_source_similarity"],
            "purple_non_latin_alpha": p_sum["mean_non_latin_alpha_fraction"],
            "orange_non_latin_alpha": o_sum["mean_non_latin_alpha_fraction"],
            "control_non_latin_alpha": eval_control_summary["mean_non_latin_alpha_fraction"],
            "purple_symbol_fraction": p_sum["mean_symbol_char_fraction"],
            "orange_symbol_fraction": o_sum["mean_symbol_char_fraction"],
            "control_symbol_fraction": eval_control_summary["mean_symbol_char_fraction"],
            "purple_length_ratio": p_sum["mean_char_length_ratio"],
            "orange_length_ratio": o_sum["mean_char_length_ratio"],
            "control_length_ratio": eval_control_summary["mean_char_length_ratio"],
        }
        # Watermark-only pass. Quality stays explicit rather than hidden inside an arbitrary scalar score.
        rec["watermark_pass"] = bool(
            rec["count_accuracy_at_0p5"] >= 0.95
            and rec["auc"] >= 0.98
            and rec["purple_mean_z"] >= 1.5
            and rec["orange_mean_z"] <= -1.5
            and rec["paired_pass_rate"] >= 0.50
            and rec["symmetry_error"] <= 0.10
        )
        records.append(rec)
        full_summary["strengths"][str(strength)] = {
            "purple": p_sum,
            "orange": o_sum,
            "paired": paired,
            "metrics": rec,
        }
        with (sdir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(full_summary["strengths"][str(strength)], f, indent=2, sort_keys=True)
        write_examples(
            sdir / "examples_side_by_side.md",
            eval_sources,
            eval_control,
            purple_rows,
            orange_rows,
            args.examples,
            args.data_seed,
        )

    selected = next((r["strength"] for r in records if r["watermark_pass"]), None)
    full_summary["selected_strength_watermark_only"] = selected
    full_summary["sanity_checks"] = {
        "calibration_rewrites_near_half": {
            "value": calibration_summary["mean_purple_fraction_colored"],
            "pass": abs(calibration_summary["mean_purple_fraction_colored"] - 0.5) <= 0.03,
        },
        "heldout_control_rewrites_near_half": {
            "value": control_summary["mean_purple_fraction_colored"],
            "pass": abs(control_summary["mean_purple_fraction_colored"] - 0.5) <= 0.05,
        },
    }

    with (outdir / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, sort_keys=True)
    with (outdir / "validation_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with (outdir / "validation_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Fixed-bias watermark validation\n\n")
        f.write(f"Model: `{args.model}`  \n")
        f.write(f"Temperature: {args.temperature}  \n")
        f.write(f"Sampling truncation: **none**  \n")
        f.write(f"Symmetric +/- bias: {args.symmetric}  \n")
        f.write(f"Entropy gate: {args.min_entropy_nats} nats; top-k gate: {args.gate_top_k}\n\n")
        f.write("The embedding hyperplane is calibrated on unwatermarked Qwen rewrites, not on the source Lyra text.\n\n")
        f.write(
            f"Calibration rewrite purple fraction: **{calibration_summary['mean_purple_fraction_colored']:.3f}**  \n"
            f"Held-out unwatermarked rewrite purple fraction: **{control_summary['mean_purple_fraction_colored']:.3f}**\n\n"
        )
        f.write(
            f"Eligible vocabulary tokens: **{eligibility_meta['eligible_token_count']}**  \n"
            f"Eligible tokens unseen in calibration: **{eligibility_meta.get('eligible_unseen_token_count', 0)}**  \n"
            f"Calibration token mass that is eligible: **{eligibility_meta['calibration_eligible_mass_fraction']:.1%}**  \n"
            f"Eligibility min-count/top-k/margin: **{args.eligible_min_count}/{args.neutral_top_k}/{args.neutral_margin_mass:g}**\n\n"
        )
        f.write("| strength | purple z | orange z | gap | acc | AUC | paired | same | symmetry err | KL P/O | watermark pass |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for r in records:
            f.write(
                f"| {r['strength']:.2f} | {r['purple_mean_z']:.2f} | {r['orange_mean_z']:.2f} | "
                f"{r['fraction_gap']:.3f} | {r['count_accuracy_at_0p5']:.3f} | {r['auc']:.3f} | "
                f"{r['paired_pass_rate']:.3f} | {r['exact_same_pair_rate']:.3f} | {r['symmetry_error']:.3f} | "
                f"{r['purple_mean_kl']:.3f}/{r['orange_mean_kl']:.3f} | {'YES' if r['watermark_pass'] else 'no'} |\n"
            )
        f.write("\n")
        if selected is None:
            f.write("No strength met the automatic watermark-only gate. Inspect the per-strength examples before changing parameters.\n")
        else:
            f.write(f"Weakest strength meeting the automatic watermark-only gate: **{selected:g}**. Inspect its examples before scaling.\n")
        f.write("\nQuality columns are in `validation_summary.csv`; quality is intentionally not collapsed into a single arbitrary score.\n")

    print("\n" + json.dumps(full_summary["sanity_checks"], indent=2), flush=True)
    print(f"Selected watermark-only strength: {selected}", flush=True)
    print(f"Read: {outdir / 'validation_summary.md'}", flush=True)
    print(f"Then inspect: strength_*/examples_side_by_side.md", flush=True)


if __name__ == "__main__":
    main()
