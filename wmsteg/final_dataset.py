from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

PURPLE = 0
ORANGE = 1

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[+-]?(?:\d+(?:[.,]\d+)*)%?")
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_QUOTED_RE = re.compile(r'"([^"\n]{1,120})"')


def canonical_json_hash(obj: object) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def numeric_literals(text: str) -> Counter[str]:
    # Ignore harmless thousands-separator formatting differences (5,000 vs 5000).
    return Counter(x.replace(",", "") for x in _NUMBER_RE.findall(text))


def backtick_literals(text: str) -> Counter[str]:
    return Counter(x.strip() for x in _BACKTICK_RE.findall(text))


def quoted_literals(text: str) -> Counter[str]:
    return Counter(x.strip() for x in _QUOTED_RE.findall(text))


def quality_reasons(
    source: str,
    row: dict,
    *,
    min_char_length_ratio: float = 0.50,
    max_char_length_ratio: float = 1.80,
    max_repeated_trigram_fraction: float = 0.08,
    max_adjacent_repeat_fraction: float = 0.04,
    require_number_preservation: bool = True,
    require_backtick_preservation: bool = True,
) -> list[str]:
    """Cheap, deliberately conservative hard-rejection checks.

    These are not semantic judges.  They only catch obvious generation failures
    that are particularly damaging for a paired causal dataset.
    """
    reasons: list[str] = []
    text = str(row.get("text", ""))
    if not text.strip():
        reasons.append("empty")
        return reasons

    ratio = float(row.get("char_length_ratio", len(text) / max(1, len(source))))
    if ratio < min_char_length_ratio:
        reasons.append("severe_contraction")
    if ratio > max_char_length_ratio:
        reasons.append("severe_expansion")

    if bool(row.get("hit_max_new_tokens", False)):
        reasons.append("hit_max_new_tokens")
    if bool(row.get("replacement_char_present", False)):
        reasons.append("replacement_character")
    if float(row.get("control_char_fraction", 0.0)) > 0.0:
        reasons.append("control_characters")
    if float(row.get("repeated_trigram_fraction", 0.0)) > max_repeated_trigram_fraction:
        reasons.append("repeated_trigram_loop")
    if float(row.get("adjacent_word_repeat_fraction", 0.0)) > max_adjacent_repeat_fraction:
        reasons.append("adjacent_word_repetition")

    if require_number_preservation and numeric_literals(source) != numeric_literals(text):
        reasons.append("numeric_literal_changed")
    if require_backtick_preservation and backtick_literals(source) != backtick_literals(text):
        reasons.append("backtick_literal_changed")
    return reasons


def informational_quality_flags(source: str, row: dict) -> list[str]:
    """Non-fatal review flags.  These are retained in metadata rather than filtered."""
    flags: list[str] = []
    text = str(row.get("text", ""))
    if quoted_literals(source) != quoted_literals(text):
        flags.append("quoted_span_changed")
    src_non_latin = sum(ch.isalpha() and not ("LATIN" in __import__("unicodedata").name(ch, "")) for ch in source)
    out_non_latin_frac = float(row.get("non_latin_alpha_fraction", 0.0))
    if src_non_latin == 0 and out_non_latin_frac > 0.01:
        flags.append("new_non_latin_text")
    return flags


def roc_auc(rows: Sequence[dict]) -> float:
    p = [float(r["purple_fraction_colored"]) for r in rows if int(r.get("label", -1)) == PURPLE]
    o = [float(r["purple_fraction_colored"]) for r in rows if int(r.get("label", -1)) == ORANGE]
    if not p or not o:
        return float("nan")
    wins = 0.0
    for pv in p:
        for ov in o:
            if pv > ov:
                wins += 1.0
            elif pv == ov:
                wins += 0.5
    return wins / (len(p) * len(o))


def stat_accuracy(rows: Sequence[dict], threshold: float = 0.5) -> float:
    labeled = [r for r in rows if r.get("label") in (PURPLE, ORANGE)]
    if not labeled:
        return float("nan")
    correct = 0
    for r in labeled:
        pred = PURPLE if float(r["purple_fraction_colored"]) >= threshold else ORANGE
        correct += int(pred == int(r["label"]))
    return correct / len(labeled)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def pair_metrics(rows: Sequence[dict], z_threshold: float = 1.5) -> dict:
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("label") in (PURPLE, ORANGE):
            grouped[str(row["source_id"])][int(row["label"])] = row
    pairs = [g for g in grouped.values() if PURPLE in g and ORANGE in g]
    if not pairs:
        return {"n_pairs": 0}
    passes = 0
    exact = 0
    gaps: list[float] = []
    for g in pairs:
        p, o = g[PURPLE], g[ORANGE]
        passes += int(
            float(p["z_score_purple_minus_orange"]) >= z_threshold
            and float(o["z_score_purple_minus_orange"]) <= -z_threshold
        )
        exact += int(p["text"] == o["text"])
        gaps.append(float(p["purple_fraction_colored"]) - float(o["purple_fraction_colored"]))
    return {
        "n_pairs": len(pairs),
        "paired_z_pass_rate": passes / len(pairs),
        "exact_same_pair_rate": exact / len(pairs),
        "mean_paired_purple_fraction_gap": mean(gaps),
    }


def summarize_rows(rows: Sequence[dict], z_threshold: float = 1.5) -> dict:
    p = [r for r in rows if r.get("label") == PURPLE]
    o = [r for r in rows if r.get("label") == ORANGE]
    controls = [r for r in rows if r.get("label") is None]
    out = {
        "rows": len(rows),
        "pairs": pair_metrics(rows, z_threshold=z_threshold),
    }
    if p and o:
        out.update({
            "purple_mean_z": mean([float(r["z_score_purple_minus_orange"]) for r in p]),
            "orange_mean_z": mean([float(r["z_score_purple_minus_orange"]) for r in o]),
            "accuracy_at_0p5": stat_accuracy(rows),
            "auc": roc_auc(rows),
            "purple_mean_colored_tokens": mean([float(r["purple_tokens"] + r["orange_tokens"]) for r in p]),
            "orange_mean_colored_tokens": mean([float(r["purple_tokens"] + r["orange_tokens"]) for r in o]),
        })
    if controls:
        out["control_mean_purple_fraction"] = mean([float(r["purple_fraction_colored"]) for r in controls])
        out["control_mean_z"] = mean([float(r["z_score_purple_minus_orange"]) for r in controls])
    return out


def _select_pair_ids(pair_ids: Sequence[str], target: int | None, seed: int) -> set[str]:
    ids = list(pair_ids)
    ids.sort()
    random.Random(seed).shuffle(ids)
    if target is not None and target > 0:
        ids = ids[:target]
    return set(ids)


def _sort_rows(rows: Sequence[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (str(r.get("split", "")), int(r.get("global_pair_id", r.get("pair_id", 0))), -1 if r.get("label") is None else int(r["label"])))


def build_pair_decisions(rows: Sequence[dict], config: dict) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (str(row["split"]), str(row["source_id"]))
        condition = str(row.get("watermark", "control"))
        grouped[key][condition] = row

    decisions: list[dict] = []
    zt = float(config["z_threshold"])
    for (split, source_id), cond in grouped.items():
        p = cond.get("purple")
        o = cond.get("orange")
        c = cond.get("control")
        reasons: list[str] = []
        flags: dict[str, list[str]] = {}
        if p is None:
            reasons.append("missing_purple")
        if o is None:
            reasons.append("missing_orange")
        if p is not None and o is not None:
            p_reasons = quality_reasons(
                p["source_prompt"], p,
                min_char_length_ratio=float(config["min_char_length_ratio"]),
                max_char_length_ratio=float(config["max_char_length_ratio"]),
                max_repeated_trigram_fraction=float(config["max_repeated_trigram_fraction"]),
                max_adjacent_repeat_fraction=float(config["max_adjacent_repeat_fraction"]),
                require_number_preservation=bool(config["require_number_preservation"]),
                require_backtick_preservation=bool(config["require_backtick_preservation"]),
            )
            o_reasons = quality_reasons(
                o["source_prompt"], o,
                min_char_length_ratio=float(config["min_char_length_ratio"]),
                max_char_length_ratio=float(config["max_char_length_ratio"]),
                max_repeated_trigram_fraction=float(config["max_repeated_trigram_fraction"]),
                max_adjacent_repeat_fraction=float(config["max_adjacent_repeat_fraction"]),
                require_number_preservation=bool(config["require_number_preservation"]),
                require_backtick_preservation=bool(config["require_backtick_preservation"]),
            )
            reasons.extend([f"purple:{x}" for x in p_reasons])
            reasons.extend([f"orange:{x}" for x in o_reasons])
            flags["purple"] = informational_quality_flags(p["source_prompt"], p)
            flags["orange"] = informational_quality_flags(o["source_prompt"], o)
            pz = float(p["z_score_purple_minus_orange"])
            oz = float(o["z_score_purple_minus_orange"])
            if pz < zt:
                reasons.append("purple:weak_z")
            if oz > -zt:
                reasons.append("orange:weak_z")
        else:
            pz = None
            oz = None

        decisions.append({
            "split": split,
            "source_id": source_id,
            "global_pair_id": int((p or o or c or {}).get("global_pair_id", -1)),
            "source_token_length": int((p or o or c or {}).get("source_token_length", 0)),
            "purple_z": pz,
            "orange_z": oz,
            "hard_reject_reasons": reasons,
            "review_flags": flags,
            "strong_and_clean": not reasons,
        })
    return sorted(decisions, key=lambda x: (x["split"], x["global_pair_id"]))


def _rows_for_ids(rows: Sequence[dict], split: str, ids: set[str], labels_only: bool = True) -> list[dict]:
    out = []
    for row in rows:
        if str(row.get("split")) != split or str(row.get("source_id")) not in ids:
            continue
        if labels_only and row.get("label") not in (PURPLE, ORANGE):
            continue
        out.append(row)
    return _sort_rows(out)


def _write_examples(path: Path, rows: Sequence[dict], ids: Sequence[str], title: str, n: int, seed: int) -> None:
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if str(row.get("source_id")) in set(ids):
            by_id[str(row["source_id"])][str(row.get("watermark"))] = row
    chosen = list(ids)
    random.Random(seed).shuffle(chosen)
    chosen = chosen[:n]
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for i, sid in enumerate(chosen, 1):
            cond = by_id.get(sid, {})
            base = cond.get("purple") or cond.get("orange") or cond.get("control")
            if not base:
                continue
            f.write(f"## Example {i} — {base.get('split')} — {base.get('mode')} / {base.get('topic')}\n\n")
            f.write("**Original**\n\n" + str(base.get("source_prompt", "")) + "\n\n")
            for name in ("control", "purple", "orange"):
                r = cond.get(name)
                if not r:
                    continue
                f.write(
                    f"**{name.title()}** — z={float(r['z_score_purple_minus_orange']):.2f}, "
                    f"purple={float(r['purple_fraction_colored']):.3f}, "
                    f"KL={float(r.get('mean_step_kl', 0.0)):.3f}\n\n{r['text']}\n\n"
                )
            f.write("---\n\n")


def assemble_dataset(output_dir: str | Path, config: dict) -> dict:
    outdir = Path(output_dir)
    raw_dir = outdir / "raw_batches"
    batch_files = sorted(raw_dir.glob("batch_*.json"))
    expected_batches = int(config["expected_batches"])
    if len(batch_files) != expected_batches:
        raise RuntimeError(f"Expected {expected_batches} raw batch files, found {len(batch_files)}")

    rows: list[dict] = []
    for path in batch_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    rows = _sort_rows(rows)

    decisions = build_pair_decisions(rows, config)
    write_jsonl(outdir / "pair_decisions.jsonl", decisions)

    controls = _sort_rows([r for r in rows if r.get("label") is None])
    write_jsonl(outdir / "control.jsonl", controls)

    strong_by_split: dict[str, list[str]] = {}
    all_by_split: dict[str, list[str]] = {}
    rejected_by_split: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        d = [x for x in decisions if x["split"] == split]
        all_by_split[split] = [str(x["source_id"]) for x in d]
        strong_by_split[split] = [str(x["source_id"]) for x in d if x["strong_and_clean"]]
        rejected_by_split[split] = [str(x["source_id"]) for x in d if not x["strong_and_clean"]]

    # Preserve all long-form evaluation attempts, regardless of watermark success.
    write_jsonl(outdir / "train_all.jsonl", _rows_for_ids(rows, "train", set(all_by_split["train"])))
    write_jsonl(outdir / "val_long_all.jsonl", _rows_for_ids(rows, "val", set(all_by_split["val"])))
    write_jsonl(outdir / "test_long_all.jsonl", _rows_for_ids(rows, "test", set(all_by_split["test"])))

    # Preserve all strong+clean pairs before deterministic caps.
    write_jsonl(outdir / "train_strong_all.jsonl", _rows_for_ids(rows, "train", set(strong_by_split["train"])))
    write_jsonl(outdir / "val_strong_all.jsonl", _rows_for_ids(rows, "val", set(strong_by_split["val"])))
    write_jsonl(outdir / "test_strong_all.jsonl", _rows_for_ids(rows, "test", set(strong_by_split["test"])))

    train_ids = _select_pair_ids(strong_by_split["train"], int(config["target_train_pairs"]), int(config["selection_seed"]))
    val_ids = _select_pair_ids(strong_by_split["val"], int(config["target_val_pairs"]), int(config["selection_seed"]) + 1)
    test_ids = _select_pair_ids(strong_by_split["test"], int(config["target_test_pairs"]), int(config["selection_seed"]) + 2)

    final_sets = {
        "train": _rows_for_ids(rows, "train", train_ids),
        "val": _rows_for_ids(rows, "val", val_ids),
        "test": _rows_for_ids(rows, "test", test_ids),
        "val_long_all": _rows_for_ids(rows, "val", set(all_by_split["val"])),
        "test_long_all": _rows_for_ids(rows, "test", set(all_by_split["test"])),
    }
    for name, subset in final_sets.items():
        write_jsonl(outdir / f"{name}.jsonl", subset)

    rejected_rows = [x for x in decisions if not x["strong_and_clean"]]
    write_jsonl(outdir / "rejected_pairs.jsonl", rejected_rows)

    summary = {
        "config": config,
        "raw": summarize_rows([r for r in rows if r.get("label") in (PURPLE, ORANGE)], config["z_threshold"]),
        "control": summarize_rows(controls, config["z_threshold"]),
        "candidate_pairs": {s: len(all_by_split[s]) for s in ("train", "val", "test")},
        "strong_clean_pairs": {s: len(strong_by_split[s]) for s in ("train", "val", "test")},
        "strong_clean_retention": {
            s: len(strong_by_split[s]) / max(1, len(all_by_split[s])) for s in ("train", "val", "test")
        },
        "final_pairs": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "targets_met": {
            "train": len(train_ids) >= int(config["target_train_pairs"]),
            "val": len(val_ids) >= int(config["target_val_pairs"]),
            "test": len(test_ids) >= int(config["target_test_pairs"]),
        },
        "final": {name: summarize_rows(subset, config["z_threshold"]) for name, subset in final_sets.items()},
        "rejection_reason_counts": dict(Counter(reason for d in rejected_rows for reason in d["hard_reject_reasons"])),
    }

    long_test = summary["final"]["test_long_all"]
    pz = abs(float(long_test.get("purple_mean_z", float("nan"))))
    oz = abs(float(long_test.get("orange_mean_z", float("nan"))))
    symmetry_rel_err = abs(pz - oz) / max(1e-12, (pz + oz) / 2.0) if math.isfinite(pz) and math.isfinite(oz) else float("nan")
    control_fraction = float(summary["control"].get("control_mean_purple_fraction", float("nan")))
    gates = {
        "main_control_near_half": math.isfinite(control_fraction) and abs(control_fraction - 0.5) <= 0.04,
        "test_long_all_auc_ge_0p98": float(long_test.get("auc", float("nan"))) >= 0.98,
        "test_long_all_accuracy_ge_0p95": float(long_test.get("accuracy_at_0p5", float("nan"))) >= 0.95,
        "test_long_all_symmetry_rel_err_le_0p20": math.isfinite(symmetry_rel_err) and symmetry_rel_err <= 0.20,
        "requested_pair_targets_met": all(summary["targets_met"].values()),
    }
    summary["validation"] = {
        "gates": gates,
        "pass": all(gates.values()),
        "test_long_all_symmetry_relative_error": symmetry_rel_err,
    }

    (outdir / "data_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Human-readable compact report.
    lines = [
        "# Final long-form Lyra watermark dataset",
        "",
        f"Partition tag: `{config['partition_tag']}`",
        f"Sampler: symmetric fixed +/-{config['strength']:g} bias, temperature {config['temperature']:g}, full softmax, no truncation/gating.",
        f"Source filter: English, {config['min_source_tokens']}–{config['max_source_tokens']} Qwen tokens.",
        f"Strong-pair criterion: purple z >= +{config['z_threshold']:g}, orange z <= -{config['z_threshold']:g}, plus hard quality checks.",
        "",
        "| split | candidates | strong+clean | retention | final target set |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {len(all_by_split[split])} | {len(strong_by_split[split])} | "
            f"{len(strong_by_split[split])/max(1,len(all_by_split[split])):.1%} | {summary['final_pairs'][split]} |"
        )
    lines += ["", "## Final statistical detector metrics", ""]
    for name in ("train", "val", "test", "val_long_all", "test_long_all"):
        s = summary["final"][name]
        lines.append(
            f"- **{name}**: pairs={s['pairs'].get('n_pairs', 0)}, "
            f"purple z={s.get('purple_mean_z', float('nan')):.2f}, "
            f"orange z={s.get('orange_mean_z', float('nan')):.2f}, "
            f"accuracy={s.get('accuracy_at_0p5', float('nan')):.3f}, AUC={s.get('auc', float('nan')):.3f}."
        )
    lines += ["", "## Validation gates", ""]
    for gate, ok in summary["validation"]["gates"].items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {gate}")
    lines.append(f"- Overall: {'PASS' if summary['validation']['pass'] else 'FAIL'}")
    lines += ["", "## Rejections", ""]
    if summary["rejection_reason_counts"]:
        for reason, count in sorted(summary["rejection_reason_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    (outdir / "data_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    accepted_ids = list(train_ids | val_ids | test_ids)
    rejected_ids = [str(x["source_id"]) for x in rejected_rows]
    _write_examples(outdir / "examples_accepted.md", rows, accepted_ids, "Accepted watermark pairs", 24, int(config["selection_seed"]))
    _write_examples(outdir / "examples_rejected.md", rows, rejected_ids, "Rejected watermark pairs", 24, int(config["selection_seed"]) + 99)

    files_to_hash = [
        "sources.jsonl", "partition.npz", "partition_meta.json", "partition_validation.json",
        "train.jsonl", "val.jsonl", "test.jsonl", "val_long_all.jsonl", "test_long_all.jsonl",
        "control.jsonl", "pair_decisions.jsonl", "data_summary.json",
    ]
    hashes = {}
    for name in files_to_hash:
        p = outdir / name
        if p.exists():
            hashes[name] = sha256_file(p)
    manifest = {
        "dataset_format_version": "modal-final-v1",
        "config_hash": canonical_json_hash(config),
        "files_sha256": hashes,
        "targets_met": summary["targets_met"],
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return summary
