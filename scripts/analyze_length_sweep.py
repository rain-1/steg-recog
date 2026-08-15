#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

PURPLE = 0
ORANGE = 1


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def auc(purple_rows: list[dict], orange_rows: list[dict]) -> float:
    p = np.asarray([float(r["purple_fraction_colored"]) for r in purple_rows], dtype=float)
    o = np.asarray([float(r["purple_fraction_colored"]) for r in orange_rows], dtype=float)
    if not len(p) or not len(o):
        return float("nan")
    return float((p[:, None] > o[None, :]).mean() + 0.5 * (p[:, None] == o[None, :]).mean())


def accuracy(purple_rows: list[dict], orange_rows: list[dict]) -> float:
    if not purple_rows or not orange_rows:
        return float("nan")
    correct = sum(float(r["purple_fraction_colored"]) >= 0.5 for r in purple_rows)
    correct += sum(float(r["purple_fraction_colored"]) < 0.5 for r in orange_rows)
    return correct / (len(purple_rows) + len(orange_rows))


def index_pairs(purple_rows: list[dict], orange_rows: list[dict]) -> list[tuple[dict, dict]]:
    p = {int(r["pair_id"]): r for r in purple_rows}
    o = {int(r["pair_id"]): r for r in orange_rows}
    return [(p[i], o[i]) for i in sorted(set(p) & set(o))]


def pair_metrics(pairs: list[tuple[dict, dict]], z_threshold: float) -> dict:
    if not pairs:
        return {
            "n": 0,
            "purple_mean_z": float("nan"),
            "orange_mean_z": float("nan"),
            "paired_pass_rate": float("nan"),
            "same_rate": float("nan"),
            "mean_gap": float("nan"),
            "accuracy": float("nan"),
            "auc": float("nan"),
            "mean_source_tokens": float("nan"),
            "mean_min_colored_tokens": float("nan"),
        }
    ps = [p for p, _ in pairs]
    os = [o for _, o in pairs]
    pass_count = sum(
        float(p["z_score_purple_minus_orange"]) >= z_threshold
        and float(o["z_score_purple_minus_orange"]) <= -z_threshold
        for p, o in pairs
    )
    gaps = [float(p["purple_fraction_colored"]) - float(o["purple_fraction_colored"]) for p, o in pairs]
    min_colored = [
        min(int(p["purple_tokens"]) + int(p["orange_tokens"]), int(o["purple_tokens"]) + int(o["orange_tokens"]))
        for p, o in pairs
    ]
    return {
        "n": len(pairs),
        "purple_mean_z": float(np.mean([float(p["z_score_purple_minus_orange"]) for p in ps])),
        "orange_mean_z": float(np.mean([float(o["z_score_purple_minus_orange"]) for o in os])),
        "paired_pass_rate": pass_count / len(pairs),
        "same_rate": float(np.mean([p["text"] == o["text"] for p, o in pairs])),
        "mean_gap": float(np.mean(gaps)),
        "accuracy": accuracy(ps, os),
        "auc": auc(ps, os),
        "mean_source_tokens": float(np.mean([int(p["source_token_length"]) for p, _ in pairs])),
        "mean_min_colored_tokens": float(np.mean(min_colored)),
    }


def bucket(value: float, edges: list[tuple[float, float | None, str]]) -> str:
    for low, high, name in edges:
        if value >= low and (high is None or value < high):
            return name
    raise AssertionError(value)


def filter_pairs(pairs: list[tuple[dict, dict]], pred: Callable[[dict, dict], bool]) -> list[tuple[dict, dict]]:
    return [(p, o) for p, o in pairs if pred(p, o)]


def discover_strength_dirs(root: Path) -> list[tuple[float, Path]]:
    found: list[tuple[float, Path]] = []
    for p in root.glob("strength_*"):
        if not p.is_dir() or not (p / "purple.jsonl").exists() or not (p / "orange.jsonl").exists():
            continue
        raw = p.name[len("strength_"):].replace("p", ".")
        try:
            strength = float(raw)
        except ValueError:
            continue
        found.append((strength, p))
    return sorted(found)


def fmt(x: float, digits: int = 3) -> str:
    return "nan" if not math.isfinite(x) else f"{x:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze fixed-bias watermark strength as a function of prompt/output length.")
    ap.add_argument("--root", required=True, help="Validation run directory containing strength_*/purple.jsonl and orange.jsonl")
    ap.add_argument("--z-threshold", type=float, default=1.5)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir) if args.output_dir else root / "length_analysis"
    out.mkdir(parents=True, exist_ok=True)

    strength_dirs = discover_strength_dirs(root)
    if not strength_dirs:
        raise SystemExit(f"No strength_* directories with purple/orange JSONL found under {root}")

    source_edges = [
        (0, 48, "<48"),
        (48, 96, "48-95"),
        (96, 160, "96-159"),
        (160, None, "160+"),
    ]
    colored_edges = [
        (0, 32, "<32"),
        (32, 64, "32-63"),
        (64, 96, "64-95"),
        (96, 160, "96-159"),
        (160, None, "160+"),
    ]
    source_thresholds = [24, 48, 64, 96, 128]
    colored_thresholds = [0, 32, 48, 64, 96]

    all_records: list[dict] = []
    source_bucket_records: list[dict] = []
    colored_bucket_records: list[dict] = []
    source_filter_records: list[dict] = []
    colored_filter_records: list[dict] = []

    for strength, sdir in strength_dirs:
        purple = read_jsonl(sdir / "purple.jsonl")
        orange = read_jsonl(sdir / "orange.jsonl")
        pairs = index_pairs(purple, orange)
        base = pair_metrics(pairs, args.z_threshold)
        all_records.append({"strength": strength, **base})

        grouped: dict[str, list[tuple[dict, dict]]] = {name: [] for _, _, name in source_edges}
        for p, o in pairs:
            name = bucket(float(p["source_token_length"]), source_edges)
            grouped[name].append((p, o))
        for _, _, name in source_edges:
            m = pair_metrics(grouped[name], args.z_threshold)
            source_bucket_records.append({"strength": strength, "source_bucket": name, **m})

        grouped_col: dict[str, list[tuple[dict, dict]]] = {name: [] for _, _, name in colored_edges}
        for p, o in pairs:
            min_col = min(
                int(p["purple_tokens"]) + int(p["orange_tokens"]),
                int(o["purple_tokens"]) + int(o["orange_tokens"]),
            )
            name = bucket(float(min_col), colored_edges)
            grouped_col[name].append((p, o))
        for _, _, name in colored_edges:
            m = pair_metrics(grouped_col[name], args.z_threshold)
            colored_bucket_records.append({"strength": strength, "colored_bucket": name, **m})

        for t in source_thresholds:
            kept = filter_pairs(pairs, lambda p, o, t=t: int(p["source_token_length"]) >= t)
            m = pair_metrics(kept, args.z_threshold)
            source_filter_records.append({
                "strength": strength,
                "min_source_tokens": t,
                "retained_fraction": len(kept) / max(1, len(pairs)),
                **m,
            })

        for t in colored_thresholds:
            kept = filter_pairs(
                pairs,
                lambda p, o, t=t: min(
                    int(p["purple_tokens"]) + int(p["orange_tokens"]),
                    int(o["purple_tokens"]) + int(o["orange_tokens"]),
                ) >= t,
            )
            m = pair_metrics(kept, args.z_threshold)
            colored_filter_records.append({
                "strength": strength,
                "min_colored_tokens_each": t,
                "retained_fraction": len(kept) / max(1, len(pairs)),
                **m,
            })

    def write_csv(name: str, rows: list[dict]) -> None:
        path = out / name
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    write_csv("overall.csv", all_records)
    write_csv("by_source_length.csv", source_bucket_records)
    write_csv("by_colored_length.csv", colored_bucket_records)
    write_csv("filter_by_source_length.csv", source_filter_records)
    write_csv("filter_by_colored_length.csv", colored_filter_records)

    with (out / "length_analysis.md").open("w", encoding="utf-8") as f:
        f.write("# Length-aware fixed-bias watermark validation\n\n")
        f.write(f"Paired success means purple z >= +{args.z_threshold:g} and orange z <= -{args.z_threshold:g}.\n\n")
        f.write("## Overall by strength\n\n")
        f.write("| strength | n | purple z | orange z | paired pass | acc | AUC | same | mean min colored |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in all_records:
            f.write(f"| {r['strength']:.2f} | {r['n']} | {fmt(r['purple_mean_z'],2)} | {fmt(r['orange_mean_z'],2)} | {fmt(r['paired_pass_rate'])} | {fmt(r['accuracy'])} | {fmt(r['auc'])} | {fmt(r['same_rate'])} | {fmt(r['mean_min_colored_tokens'],1)} |\n")

        for strength, _ in strength_dirs:
            f.write(f"\n## Strength {strength:g}: source-length buckets\n\n")
            f.write("| source tokens | n | purple z | orange z | paired pass | acc | AUC | same |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in [x for x in source_bucket_records if x["strength"] == strength]:
                f.write(f"| {r['source_bucket']} | {r['n']} | {fmt(r['purple_mean_z'],2)} | {fmt(r['orange_mean_z'],2)} | {fmt(r['paired_pass_rate'])} | {fmt(r['accuracy'])} | {fmt(r['auc'])} | {fmt(r['same_rate'])} |\n")

            f.write(f"\n## Strength {strength:g}: minimum colored-token evidence\n\n")
            f.write("| require colored tokens in each rewrite | retained | n | purple z | orange z | paired pass | acc | AUC |\n")
            f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in [x for x in colored_filter_records if x["strength"] == strength]:
                f.write(f"| >= {r['min_colored_tokens_each']} | {fmt(r['retained_fraction'])} | {r['n']} | {fmt(r['purple_mean_z'],2)} | {fmt(r['orange_mean_z'],2)} | {fmt(r['paired_pass_rate'])} | {fmt(r['accuracy'])} | {fmt(r['auc'])} |\n")

        f.write("\n## How to use this\n\n")
        f.write("If paired success rises sharply with source length or colored-token count, filtering short prompts is justified. Pick the weakest strength whose retained long-form subset has strong symmetric separation, then scale data generation and discard short/weak pairs.\n")

    print(f"Wrote {out / 'length_analysis.md'}")


if __name__ == "__main__":
    main()
