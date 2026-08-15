from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from .final_dataset import (
    backtick_literals,
    canonical_json_hash,
    informational_quality_flags,
    numeric_literals,
    quality_reasons,
    sha256_file,
    summarize_rows,
)

PURPLE = 0
ORANGE = 1


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _row_key(row: dict) -> tuple[int, int, int]:
    label = row.get("label")
    label_order = -1 if label is None else int(label)
    return (
        int(row.get("global_pair_id", row.get("pair_id", 0))),
        int(row.get("replicate", 0)),
        label_order,
    )


def _sort_rows(rows: Sequence[dict]) -> list[dict]:
    return sorted(rows, key=_row_key)


def _decision_key(row: dict) -> tuple[str, int]:
    return str(row["source_id"]), int(row.get("replicate", 0))


def build_bank_decisions(rows: Sequence[dict], config: dict) -> list[dict]:
    """Build orthogonal acceptance flags for every source/replicate pair.

    We deliberately keep watermark strength, basic generation health, and literal
    fidelity separate.  `strong_and_clean` remains as a backwards-compatible
    alias for the strict conjunction of all three.
    """
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[_decision_key(row)][str(row.get("watermark", "control"))] = row

    zt = float(config["z_threshold"])
    decisions: list[dict] = []
    for (source_id, replicate), cond in grouped.items():
        p = cond.get("purple")
        o = cond.get("orange")
        c = cond.get("control")
        base = p or o or c or {}

        infrastructure_reasons: list[str] = []
        if c is None:
            infrastructure_reasons.append("missing_control")
        if p is None:
            infrastructure_reasons.append("missing_purple")
        if o is None:
            infrastructure_reasons.append("missing_orange")

        watermark_reasons: list[str] = []
        generation_health_reasons: list[str] = []
        literal_fidelity_reasons: list[str] = []
        review_flags: dict[str, list[str]] = {}
        pz = None
        oz = None

        if p is not None and o is not None:
            for name, row in (("purple", p), ("orange", o)):
                # Health checks intentionally exclude exact-number/backtick fidelity.
                health = quality_reasons(
                    row["source_prompt"],
                    row,
                    min_char_length_ratio=float(config["min_char_length_ratio"]),
                    max_char_length_ratio=float(config["max_char_length_ratio"]),
                    max_repeated_trigram_fraction=float(config["max_repeated_trigram_fraction"]),
                    max_adjacent_repeat_fraction=float(config["max_adjacent_repeat_fraction"]),
                    require_number_preservation=False,
                    require_backtick_preservation=False,
                )
                generation_health_reasons.extend(f"{name}:{x}" for x in health)

                source = str(row["source_prompt"])
                text = str(row.get("text", ""))
                if bool(config.get("require_number_preservation", True)) and numeric_literals(source) != numeric_literals(text):
                    literal_fidelity_reasons.append(f"{name}:numeric_literal_changed")
                if bool(config.get("require_backtick_preservation", True)) and backtick_literals(source) != backtick_literals(text):
                    literal_fidelity_reasons.append(f"{name}:backtick_literal_changed")
                review_flags[name] = informational_quality_flags(source, row)

            pz = float(p["z_score_purple_minus_orange"])
            oz = float(o["z_score_purple_minus_orange"])
            if pz < zt:
                watermark_reasons.append("purple:weak_z")
            if oz > -zt:
                watermark_reasons.append("orange:weak_z")

        watermark_pass = bool(p is not None and o is not None and not watermark_reasons)
        generation_health_pass = bool(p is not None and o is not None and not generation_health_reasons)
        literal_fidelity_pass = bool(p is not None and o is not None and not literal_fidelity_reasons)
        complete_triplet = not infrastructure_reasons
        strict_pair_pass = bool(
            complete_triplet and watermark_pass and generation_health_pass and literal_fidelity_pass
        )
        detector_training_pass = bool(complete_triplet and watermark_pass and generation_health_pass)

        hard_reject_reasons = (
            infrastructure_reasons
            + watermark_reasons
            + generation_health_reasons
            + literal_fidelity_reasons
        )
        decisions.append({
            "source_id": source_id,
            "replicate": replicate,
            "sample_key": f"{source_id}:r{replicate}",
            "global_pair_id": int(base.get("global_pair_id", -1)),
            "fold": int(base.get("fold", -1)),
            "default_split": str(base.get("default_split", base.get("split", ""))),
            "source_token_length": int(base.get("source_token_length", 0)),
            "purple_z": pz,
            "orange_z": oz,
            "complete_triplet": complete_triplet,
            "watermark_pass": watermark_pass,
            "generation_health_pass": generation_health_pass,
            "literal_fidelity_pass": literal_fidelity_pass,
            "detector_training_pass": detector_training_pass,
            "strict_pair_pass": strict_pair_pass,
            "strong_and_clean": strict_pair_pass,
            "infrastructure_reasons": infrastructure_reasons,
            "watermark_reasons": watermark_reasons,
            "generation_health_reasons": generation_health_reasons,
            "literal_fidelity_reasons": literal_fidelity_reasons,
            "hard_reject_reasons": hard_reject_reasons,
            "review_flags": review_flags,
        })
    return sorted(decisions, key=lambda x: (x["global_pair_id"], x["replicate"]))


def _rows_for_samples(rows: Sequence[dict], sample_keys: set[str], *, include_control: bool = True) -> list[dict]:
    out = []
    for row in rows:
        key = f"{row['source_id']}:r{int(row.get('replicate', 0))}"
        if key not in sample_keys:
            continue
        if not include_control and row.get("label") not in (PURPLE, ORANGE):
            continue
        out.append(row)
    return _sort_rows(out)


def _pair_records(rows: Sequence[dict], decisions: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[_decision_key(row)][str(row.get("watermark", "control"))] = row
    dmap = {(d["source_id"], int(d["replicate"])): d for d in decisions}
    records = []
    for key, cond in grouped.items():
        base = cond.get("control") or cond.get("purple") or cond.get("orange") or {}
        records.append({
            "source_id": key[0],
            "replicate": key[1],
            "sample_key": f"{key[0]}:r{key[1]}",
            "fold": int(base.get("fold", -1)),
            "default_split": base.get("default_split", base.get("split")),
            "global_pair_id": int(base.get("global_pair_id", -1)),
            "source_prompt": base.get("source_prompt"),
            "source_token_length": base.get("source_token_length"),
            "source_token_ids": base.get("source_token_ids", []),
            "topic": base.get("topic"),
            "mode": base.get("mode"),
            "style": base.get("style"),
            "length": base.get("length"),
            "complexity": base.get("complexity"),
            "language": base.get("language"),
            "task_spec_level": base.get("task_spec_level"),
            "lyra_metadata": base.get("lyra_metadata", {}),
            "decision": dmap.get(key),
            "control": cond.get("control"),
            "purple": cond.get("purple"),
            "orange": cond.get("orange"),
        })
    return sorted(records, key=lambda x: (x["global_pair_id"], x["replicate"]))


def _write_examples(path: Path, records: Sequence[dict], *, title: str, n: int, seed: int) -> None:
    records = list(records)
    random.Random(seed).shuffle(records)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for i, rec in enumerate(records[:n], 1):
            d = rec.get("decision") or {}
            f.write(
                f"## Example {i} — fold {rec.get('fold')} / {rec.get('default_split')} — "
                f"{rec.get('mode')} / {rec.get('topic')} — replicate {rec.get('replicate')}\n\n"
            )
            f.write("**Original**\n\n" + str(rec.get("source_prompt", "")) + "\n\n")
            for name in ("control", "purple", "orange"):
                row = rec.get(name)
                if not row:
                    continue
                f.write(
                    f"**{name.title()}** — z={float(row['z_score_purple_minus_orange']):.2f}, "
                    f"purple={float(row['purple_fraction_colored']):.3f}, "
                    f"eligible={int(row.get('eligible_token_count', 0))}, "
                    f"KL={float(row.get('mean_step_kl', 0.0)):.3f}\n\n{row['text']}\n\n"
                )
            if d.get("hard_reject_reasons"):
                f.write("Rejected because: " + ", ".join(d["hard_reject_reasons"]) + "\n\n")
            f.write("---\n\n")


def _symmetry_error(summary: dict) -> float:
    pz = abs(float(summary.get("purple_mean_z", float("nan"))))
    oz = abs(float(summary.get("orange_mean_z", float("nan"))))
    if not (math.isfinite(pz) and math.isfinite(oz)):
        return float("nan")
    return abs(pz - oz) / max(1e-12, (pz + oz) / 2.0)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _flag_rate(ds: Sequence[dict], flag: str) -> float:
    return sum(bool(d.get(flag, False)) for d in ds) / len(ds) if ds else float("nan")


def _robustness_details(decisions: Sequence[dict]) -> tuple[dict, list[dict]]:
    by_source: dict[str, dict[int, dict]] = defaultdict(dict)
    for d in decisions:
        by_source[str(d["source_id"])][int(d["replicate"])] = d
    paired = [(src, reps[0], reps[1]) for src, reps in by_source.items() if 0 in reps and 1 in reps]
    if not paired:
        return {"n_sources": 0}, []

    a0 = [a for _, a, _ in paired]
    a1 = [b for _, _, b in paired]
    detail_rows: list[dict] = []
    p0: list[float] = []
    p1: list[float] = []
    o0: list[float] = []
    o1: list[float] = []
    purple_direction = 0
    orange_direction = 0

    for src, a, b in paired:
        if a["purple_z"] is not None and b["purple_z"] is not None:
            p0.append(float(a["purple_z"]))
            p1.append(float(b["purple_z"]))
            purple_direction += int(float(a["purple_z"]) > 0 and float(b["purple_z"]) > 0)
        if a["orange_z"] is not None and b["orange_z"] is not None:
            o0.append(float(a["orange_z"]))
            o1.append(float(b["orange_z"]))
            orange_direction += int(float(a["orange_z"]) < 0 and float(b["orange_z"]) < 0)

        s0 = bool(a["strict_pair_pass"])
        s1 = bool(b["strict_pair_pass"])
        w0 = bool(a["watermark_pass"])
        w1 = bool(b["watermark_pass"])
        detail_rows.append({
            "source_id": src,
            "global_pair_id": int(a.get("global_pair_id", -1)),
            "fold": int(a.get("fold", -1)),
            "default_split": a.get("default_split"),
            "replicate0": a,
            "replicate1": b,
            "strict_outcome": "both" if s0 and s1 else ("r0_only" if s0 else ("r1_only" if s1 else "neither")),
            "watermark_outcome": "both" if w0 and w1 else ("r0_only" if w0 else ("r1_only" if w1 else "neither")),
            "rescued_strict_by_retry": (not s0) and s1,
            "rescued_watermark_by_retry": (not w0) and w1,
        })

    n = len(paired)
    def combo(flag: str) -> dict:
        both = sum(bool(a[flag]) and bool(b[flag]) for _, a, b in paired)
        r0_only = sum(bool(a[flag]) and not bool(b[flag]) for _, a, b in paired)
        r1_only = sum(not bool(a[flag]) and bool(b[flag]) for _, a, b in paired)
        neither = n - both - r0_only - r1_only
        fail0 = r1_only + neither
        return {
            "both": both,
            "replicate0_only": r0_only,
            "replicate1_only": r1_only,
            "neither": neither,
            "at_least_one_rate": (both + r0_only + r1_only) / n,
            "replicate0_rate": (both + r0_only) / n,
            "replicate1_rate": (both + r1_only) / n,
            "retry_rescue_rate_given_replicate0_fail": r1_only / fail0 if fail0 else float("nan"),
            "same_status_rate": (both + neither) / n,
        }

    summary = {
        "n_sources": n,
        "watermark_pass": combo("watermark_pass"),
        "generation_health_pass": combo("generation_health_pass"),
        "literal_fidelity_pass": combo("literal_fidelity_pass"),
        "detector_training_pass": combo("detector_training_pass"),
        "strict_pair_pass": combo("strict_pair_pass"),
        "purple_direction_consistency": purple_direction / n,
        "orange_direction_consistency": orange_direction / n,
        "purple_z_pearson_r": _pearson(p0, p1),
        "orange_z_pearson_r": _pearson(o0, o1),
        "mean_abs_purple_z_delta": sum(abs(a - b) for a, b in zip(p0, p1)) / len(p0) if p0 else float("nan"),
        "mean_abs_orange_z_delta": sum(abs(a - b) for a, b in zip(o0, o1)) / len(o0) if o0 else float("nan"),
        # Backwards-compatible key used by the v2 report.
        "both_replicates_strong_clean_rate": sum(bool(a["strict_pair_pass"]) and bool(b["strict_pair_pass"]) for _, a, b in paired) / n,
    }
    return summary, sorted(detail_rows, key=lambda x: x["global_pair_id"])


def _robustness_summary(decisions: Sequence[dict]) -> dict:
    return _robustness_details(decisions)[0]


def assemble_bank(output_dir: str | Path, config: dict) -> dict:
    outdir = Path(output_dir)
    raw_dir = outdir / "raw_batches"
    batch_files = sorted(raw_dir.glob("batch_*.json"))
    expected = int(config["expected_batches"])
    if len(batch_files) != expected:
        raise RuntimeError(f"Expected {expected} raw batch files, found {len(batch_files)}")

    rows: list[dict] = []
    for path in batch_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    rows = _sort_rows(rows)

    decisions = build_bank_decisions(rows, config)
    records = _pair_records(rows, decisions)
    robustness_summary, robustness_details = _robustness_details(decisions)
    canonical_decisions = [d for d in decisions if int(d["replicate"]) == 0]
    canonical_keys = {d["sample_key"] for d in canonical_decisions}
    strong_keys = {d["sample_key"] for d in canonical_decisions if d["strong_and_clean"]}
    rejected_keys = canonical_keys - strong_keys
    robustness_keys = {d["sample_key"] for d in decisions if int(d["replicate"]) > 0}

    write_jsonl(outdir / "bank_all_rows.jsonl", rows)
    write_jsonl(outdir / "bank_strong_rows.jsonl", _rows_for_samples(rows, strong_keys, include_control=True))
    write_jsonl(outdir / "bank_strong_watermarked.jsonl", _rows_for_samples(rows, strong_keys, include_control=False))
    write_jsonl(outdir / "bank_rejected_rows.jsonl", _rows_for_samples(rows, rejected_keys, include_control=True))
    detector_keys_all = {d["sample_key"] for d in decisions if d.get("detector_training_pass", False)}
    detector_keys_canonical = {d["sample_key"] for d in canonical_decisions if d.get("detector_training_pass", False)}
    write_jsonl(outdir / "bank_detector_eligible_rows.jsonl", _rows_for_samples(rows, detector_keys_all, include_control=False))
    write_jsonl(outdir / "bank_detector_eligible_canonical.jsonl", _rows_for_samples(rows, detector_keys_canonical, include_control=False))
    write_jsonl(outdir / "robustness_rows.jsonl", _rows_for_samples(rows, robustness_keys, include_control=True))
    write_jsonl(outdir / "pair_decisions.jsonl", decisions)
    write_jsonl(outdir / "pair_records_all.jsonl", records)
    write_jsonl(outdir / "robustness_by_source.jsonl", robustness_details)

    fold_assignments = []
    seen_sources = set()
    for row in rows:
        if int(row.get("replicate", 0)) != 0:
            continue
        sid = str(row["source_id"])
        if sid in seen_sources:
            continue
        seen_sources.add(sid)
        fold_assignments.append({
            "source_id": sid,
            "global_pair_id": int(row["global_pair_id"]),
            "fold": int(row["fold"]),
            "default_split": row.get("default_split", row.get("split")),
            "topic": row.get("topic"),
            "mode": row.get("mode"),
            "style": row.get("style"),
            "length": row.get("length"),
            "complexity": row.get("complexity"),
            "task_spec_level": row.get("task_spec_level"),
            "source_token_length": int(row.get("source_token_length", 0)),
        })
    write_jsonl(outdir / "fold_assignments.jsonl", sorted(fold_assignments, key=lambda x: x["global_pair_id"]))

    for fold in range(10):
        fold_keys = {d["sample_key"] for d in canonical_decisions if d["fold"] == fold and d["strong_and_clean"]}
        write_jsonl(outdir / "folds" / f"fold_{fold}_strong.jsonl", _rows_for_samples(rows, fold_keys, include_control=False))
    for split in ("train", "val", "test"):
        split_keys = {
            d["sample_key"] for d in canonical_decisions
            if d["default_split"] == split and d["strong_and_clean"]
        }
        write_jsonl(outdir / f"default_{split}_strong.jsonl", _rows_for_samples(rows, split_keys, include_control=False))

    canonical_rows = _rows_for_samples(rows, canonical_keys, include_control=True)
    canonical_watermarked = [r for r in canonical_rows if r.get("label") in (PURPLE, ORANGE)]
    strong_rows = _rows_for_samples(rows, strong_keys, include_control=True)
    strong_watermarked = [r for r in strong_rows if r.get("label") in (PURPLE, ORANGE)]
    controls = [r for r in canonical_rows if r.get("label") is None]

    by_split = {}
    for split in ("train", "val", "test"):
        split_rows = [r for r in canonical_rows if r.get("default_split", r.get("split")) == split]
        by_split[split] = summarize_rows(split_rows, config["z_threshold"])

    by_fold = {}
    for fold in range(10):
        fold_rows = [r for r in canonical_rows if int(r.get("fold", -1)) == fold]
        by_fold[str(fold)] = summarize_rows(fold_rows, config["z_threshold"])

    canonical_summary = summarize_rows(canonical_rows, config["z_threshold"])
    strong_summary = summarize_rows(strong_rows, config["z_threshold"])
    symmetry = _symmetry_error(canonical_summary)
    n_pairs = int(canonical_summary.get("pairs", {}).get("n_pairs", 0))
    control_fraction = float(canonical_summary.get("control_mean_purple_fraction", float("nan")))

    gate_status = {
        "control_near_half": (
            "PASS" if math.isfinite(control_fraction) and abs(control_fraction - 0.5) <= 0.04 else "FAIL"
        ),
        "canonical_auc_ge_0p98": (
            "PASS" if float(canonical_summary.get("auc", float("nan"))) >= 0.98 else "FAIL"
        ),
        "canonical_accuracy_ge_0p95": (
            "PASS" if float(canonical_summary.get("accuracy_at_0p5", float("nan"))) >= 0.95 else "FAIL"
        ),
        "symmetry_rel_err_le_0p20": (
            "SKIP" if n_pairs < int(config.get("min_pairs_for_symmetry_gate", 64))
            else ("PASS" if math.isfinite(symmetry) and symmetry <= 0.20 else "FAIL")
        ),
    }
    overall_pass = all(v != "FAIL" for v in gate_status.values())

    rejected = [d for d in canonical_decisions if not d["strong_and_clean"]]
    summary = {
        "config": config,
        "candidate_source_pairs": len(canonical_decisions),
        "strong_clean_source_pairs": len(strong_keys),
        "strong_clean_retention": len(strong_keys) / max(1, len(canonical_decisions)),
        "canonical": canonical_summary,
        "strong_clean": strong_summary,
        "by_default_split": by_split,
        "by_fold": by_fold,
        "robustness": robustness_summary,
        "canonical_flag_rates": {
            "watermark_pass": _flag_rate(canonical_decisions, "watermark_pass"),
            "generation_health_pass": _flag_rate(canonical_decisions, "generation_health_pass"),
            "literal_fidelity_pass": _flag_rate(canonical_decisions, "literal_fidelity_pass"),
            "detector_training_pass": _flag_rate(canonical_decisions, "detector_training_pass"),
            "strict_pair_pass": _flag_rate(canonical_decisions, "strict_pair_pass"),
        },
        "rejection_reason_counts": dict(Counter(reason for d in rejected for reason in d["hard_reject_reasons"])),
        "validation": {
            "gate_status": gate_status,
            "pass": overall_pass,
            "canonical_symmetry_relative_error": symmetry,
        },
    }
    (outdir / "bank_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Lyra watermark research bank — proof-of-concept report",
        "",
        f"Run stage: `{config.get('run_stage', 'poc')}` (not a frozen final dataset).",
        f"Partition tag: `{config['partition_tag']}`",
        f"Partition SHA256: `{config.get('partition_sha256', 'recorded per row')}`",
        f"Sampler: symmetric fixed +/-{config['strength']:g} bias, temperature {config['temperature']:g}, full softmax, no truncation/gating.",
        f"Source filter: English, {config['min_source_tokens']}–{config['max_source_tokens']} Qwen tokens.",
        f"Reusable assignment: `fold = SHA256(source_id) mod 10`; default split is folds 0–7 train, 8 val, 9 test.",
        f"Watermark criterion: purple z >= +{config['z_threshold']:g}, orange z <= -{config['z_threshold']:g}.",
        "Acceptance flags are stored separately: watermark strength, generation health, and literal fidelity.",
        "",
        "## Bank retention",
        "",
        f"- canonical source pairs: **{len(canonical_decisions)}**",
        f"- watermark-pass pairs: **{sum(bool(d.get('watermark_pass')) for d in canonical_decisions)}** ({summary['canonical_flag_rates']['watermark_pass']:.1%})",
        f"- generation-health-pass pairs: **{sum(bool(d.get('generation_health_pass')) for d in canonical_decisions)}** ({summary['canonical_flag_rates']['generation_health_pass']:.1%})",
        f"- literal-fidelity-pass pairs: **{sum(bool(d.get('literal_fidelity_pass')) for d in canonical_decisions)}** ({summary['canonical_flag_rates']['literal_fidelity_pass']:.1%})",
        f"- detector-eligible pairs (watermark + health): **{sum(bool(d.get('detector_training_pass')) for d in canonical_decisions)}** ({summary['canonical_flag_rates']['detector_training_pass']:.1%})",
        f"- strict pairs (all three): **{len(strong_keys)}** ({len(strong_keys)/max(1,len(canonical_decisions)):.1%})",
        f"- independent-seed robustness sources: **{summary['robustness'].get('n_sources', 0)}**",
        "",
        "## Canonical statistical detector metrics",
        "",
        f"- purple mean z: **{canonical_summary.get('purple_mean_z', float('nan')):.2f}**",
        f"- orange mean z: **{canonical_summary.get('orange_mean_z', float('nan')):.2f}**",
        f"- accuracy: **{canonical_summary.get('accuracy_at_0p5', float('nan')):.3f}**",
        f"- AUC: **{canonical_summary.get('auc', float('nan')):.3f}**",
        f"- paired z-pass: **{canonical_summary.get('pairs', {}).get('paired_z_pass_rate', float('nan')):.1%}**",
        "",
        "## Default split diagnostics",
        "",
        "| split | pairs | purple z | orange z | acc | AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        s = by_split[split]
        lines.append(
            f"| {split} | {s.get('pairs', {}).get('n_pairs', 0)} | "
            f"{s.get('purple_mean_z', float('nan')):.2f} | {s.get('orange_mean_z', float('nan')):.2f} | "
            f"{s.get('accuracy_at_0p5', float('nan')):.3f} | {s.get('auc', float('nan')):.3f} |"
        )
    lines += ["", "## Validation gates", ""]
    for name, status in gate_status.items():
        lines.append(f"- {status} — {name}")
    lines.append(f"- Overall (SKIP is allowed for undersized PoCs): {'PASS' if overall_pass else 'FAIL'}")
    lines += ["", "## Robustness panel", ""]
    if summary["robustness"].get("n_sources", 0):
        r = summary["robustness"]
        wm = r["watermark_pass"]
        strict = r["strict_pair_pass"]
        det = r["detector_training_pass"]
        lines += [
            f"- sources with replicate 0 + 1: **{r['n_sources']}**",
            f"- watermark pass, replicate 0: **{wm['replicate0_rate']:.1%}**",
            f"- watermark pass, replicate 1: **{wm['replicate1_rate']:.1%}**",
            f"- watermark pass, best of two: **{wm['at_least_one_rate']:.1%}**",
            f"- P(watermark rescued by retry | replicate 0 failed): **{wm['retry_rescue_rate_given_replicate0_fail']:.1%}**",
            f"- detector-eligible, replicate 0: **{det['replicate0_rate']:.1%}**; best of two: **{det['at_least_one_rate']:.1%}**",
            f"- strict pass, replicate 0: **{strict['replicate0_rate']:.1%}**; replicate 1: **{strict['replicate1_rate']:.1%}**; best of two: **{strict['at_least_one_rate']:.1%}**",
            f"- strict outcome counts: both={strict['both']}, r0-only={strict['replicate0_only']}, r1-only={strict['replicate1_only']}, neither={strict['neither']}",
            f"- P(strict rescued by retry | replicate 0 failed): **{strict['retry_rescue_rate_given_replicate0_fail']:.1%}**",
            f"- purple direction consistency: **{r['purple_direction_consistency']:.1%}**",
            f"- orange direction consistency: **{r['orange_direction_consistency']:.1%}**",
            f"- purple z seed correlation: **{r['purple_z_pearson_r']:.3f}**",
            f"- orange z seed correlation: **{r['orange_z_pearson_r']:.3f}**",
        ]
    else:
        lines.append("- no independent-seed robustness replicates requested")
    lines += ["", "## Rejections", ""]
    if summary["rejection_reason_counts"]:
        for reason, count in sorted(summary["rejection_reason_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    (outdir / "bank_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rob_lines = [
        "# Two-seed robustness analysis",
        "",
        "This report compares replicate 0 and replicate 1 for the same underlying Lyra source.",
        "Both raw attempts are retained; `best of two` is an analysis statistic, not destructive replacement.",
        "",
    ]
    if robustness_summary.get("n_sources", 0):
        r = robustness_summary
        for flag, label in (("watermark_pass", "Watermark pass"), ("detector_training_pass", "Detector eligible"), ("strict_pair_pass", "Strict pass")):
            x = r[flag]
            rob_lines += [
                f"## {label}",
                "",
                f"- replicate 0: {x['replicate0_rate']:.1%}",
                f"- replicate 1: {x['replicate1_rate']:.1%}",
                f"- best of two: {x['at_least_one_rate']:.1%}",
                f"- both: {x['both']}; replicate-0 only: {x['replicate0_only']}; replicate-1 only: {x['replicate1_only']}; neither: {x['neither']}",
                f"- retry rescue rate conditional on replicate-0 failure: {x['retry_rescue_rate_given_replicate0_fail']:.1%}",
                f"- same pass/fail status across seeds: {x['same_status_rate']:.1%}",
                "",
            ]
        rob_lines += [
            "## Continuous watermark strength",
            "",
            f"- purple direction consistency: {r['purple_direction_consistency']:.1%}",
            f"- orange direction consistency: {r['orange_direction_consistency']:.1%}",
            f"- purple z Pearson r: {r['purple_z_pearson_r']:.3f}",
            f"- orange z Pearson r: {r['orange_z_pearson_r']:.3f}",
            f"- mean |Δ purple z|: {r['mean_abs_purple_z_delta']:.2f}",
            f"- mean |Δ orange z|: {r['mean_abs_orange_z_delta']:.2f}",
            "",
        ]
    else:
        rob_lines.append("No sources currently have both replicate 0 and replicate 1.")
    (outdir / "robustness_analysis.md").write_text("\n".join(rob_lines) + "\n", encoding="utf-8")

    accepted_records = [r for r in records if int(r["replicate"]) == 0 and (r.get("decision") or {}).get("strong_and_clean")]
    rejected_records = [r for r in records if int(r["replicate"]) == 0 and not (r.get("decision") or {}).get("strong_and_clean")]
    robustness_records = [r for r in records if int(r["replicate"]) > 0]
    _write_examples(outdir / "examples_accepted.md", accepted_records, title="Accepted canonical watermark pairs", n=24, seed=int(config["selection_seed"]))
    _write_examples(outdir / "examples_rejected.md", rejected_records, title="Rejected canonical watermark pairs", n=24, seed=int(config["selection_seed"]) + 1)
    _write_examples(outdir / "examples_robustness.md", robustness_records, title="Independent-seed robustness rewrites", n=24, seed=int(config["selection_seed"]) + 2)

    files_to_hash = [
        "sources.jsonl", "partition.npz", "partition_meta.json", "partition_validation.json", "partition_spec.json",
        "bank_all_rows.jsonl", "bank_strong_rows.jsonl", "bank_strong_watermarked.jsonl", "bank_rejected_rows.jsonl",
        "robustness_rows.jsonl", "robustness_by_source.jsonl", "robustness_analysis.md",
        "bank_detector_eligible_rows.jsonl", "bank_detector_eligible_canonical.jsonl",
        "pair_decisions.jsonl", "pair_records_all.jsonl", "fold_assignments.jsonl", "bank_summary.json",
    ]
    hashes = {}
    for name in files_to_hash:
        p = outdir / name
        if p.exists():
            hashes[name] = sha256_file(p)
    manifest = {
        "dataset_format_version": "modal-watermark-bank-v2.1",
        "run_stage": config.get("run_stage", "poc"),
        "frozen": False,
        "config_hash": canonical_json_hash(config),
        "files_sha256": hashes,
        "canonical_source_pairs": len(canonical_decisions),
        "strong_clean_source_pairs": len(strong_keys),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return summary
