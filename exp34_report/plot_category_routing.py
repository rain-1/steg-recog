#!/usr/bin/env python3
"""Generate charts for the balanced-category watermark routing experiment.

Usage:
    python plot_category_routing.py \
        --input category_routing_review.zip \
        --output charts

`--input` may be either the review ZIP or an extracted directory containing
aggregate.json, train_config.json, *_metrics.json, and *_test_predictions.jsonl.

The script intentionally depends only on the Python standard library and
matplotlib. It writes each chart as a separate PNG and PDF, plus derived CSVs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt

RUN_RE = re.compile(
    r"^n(?P<size>\d+)_(?P<start>base|wm_recognition)_seed_(?P<seed>\d+)_(?P<kind>metrics\.json|test_predictions\.jsonl)$"
)

DISPLAY_START = {"base": "Base Qwen", "wm_recognition": "WM-4x"}
DISPLAY_DOMAIN = {"in_domain": "Seen categories", "heldout_category": "Held-out categories"}
DISPLAY_WATERMARK = {"purple": "Purple", "control": "Control", "orange": "Orange"}


class Bundle:
    def __init__(self, path: Path):
        self.path = path
        self.is_zip = path.is_file() and path.suffix.lower() == ".zip"
        self._zip: zipfile.ZipFile | None = None
        if self.is_zip:
            self._zip = zipfile.ZipFile(path)
            self.names = set(self._zip.namelist())
        elif path.is_dir():
            self.names = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
        else:
            raise FileNotFoundError(f"Input must be a ZIP or directory: {path}")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def _resolve_name(self, basename: str) -> str:
        if basename in self.names:
            return basename
        matches = [n for n in self.names if Path(n).name == basename]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"{basename} not found in {self.path}")
        raise RuntimeError(f"Multiple files named {basename} found: {matches}")

    def read_text(self, basename: str) -> str:
        name = self._resolve_name(basename)
        if self.is_zip:
            assert self._zip is not None
            return self._zip.read(name).decode("utf-8")
        return (self.path / name).read_text(encoding="utf-8")

    def json(self, basename: str) -> Any:
        return json.loads(self.read_text(basename))

    def jsonl(self, basename: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.read_text(basename).splitlines() if line.strip()]

    def basenames(self) -> list[str]:
        return [Path(n).name for n in self.names]


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_runs(bundle: Bundle) -> tuple[dict[tuple[int, str, int], dict[str, Any]], dict[tuple[int, str, int], list[dict[str, Any]]]]:
    metrics: dict[tuple[int, str, int], dict[str, Any]] = {}
    predictions: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for name in bundle.basenames():
        m = RUN_RE.match(name)
        if not m:
            continue
        key = (int(m.group("size")), m.group("start"), int(m.group("seed")))
        if m.group("kind") == "metrics.json":
            metrics[key] = bundle.json(name)
        else:
            predictions[key] = bundle.jsonl(name)
    return metrics, predictions


def correct_flip_by_category(rows: list[dict[str, Any]]) -> dict[str, float]:
    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        wm = row.get("watermark")
        if wm in {"purple", "orange"}:
            pairs[(row["category"], row["source_id"])][wm] = row
    category_scores: dict[str, list[float]] = defaultdict(list)
    for (category, _), pair in pairs.items():
        if "purple" not in pair or "orange" not in pair:
            continue
        correct = (
            pair["purple"].get("predicted_behavior") == "comply"
            and pair["orange"].get("predicted_behavior") == "refuse"
        )
        category_scores[category].append(float(correct))
    return {cat: statistics.mean(scores) for cat, scores in category_scores.items()}


def repeated_fourgram_fraction(text: str) -> float:
    """Fraction of word 4-grams that are duplicate occurrences.

    Example: if 100 4-gram positions contain 80 unique 4-grams, this returns 0.20.
    The quality diagnostic flags responses with >20% repeated 4-grams.
    """
    tokens = re.findall(r"\w+", text.lower())
    grams = [tuple(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3))]
    if not grams:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def chart_flip_curves(aggregate: dict[str, Any], out_dir: Path, domain: str) -> None:
    sizes = sorted(int(x) for x in aggregate["per_size"])
    prefix = "id" if domain == "in_domain" else "ood"

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for start, metric_prefix in [("base", "base"), ("wm_recognition", "wm")]:
        means, sds = [], []
        for size in sizes:
            stat = aggregate["per_size"][str(size)]["aggregate"][f"{metric_prefix}_{prefix}_flip"]
            means.append(stat["mean"])
            sds.append(stat["sd"])
        ax.errorbar(sizes, means, yerr=sds, marker="o", capsize=4, label=DISPLAY_START[start])
    ax.set_xlabel("Behavioral SFT training sources")
    ax.set_ylabel("Correct counterfactual flip rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(sizes)
    ax.set_title(f"Watermark-controlled behavior — {DISPLAY_DOMAIN[domain].lower()}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir, "01_id_flip_rate" if domain == "in_domain" else "02_heldout_flip_rate")


def chart_delta_curves(aggregate: dict[str, Any], out_dir: Path) -> None:
    sizes = sorted(int(x) for x in aggregate["per_size"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, key in [("Seen categories", "delta_id_flip"), ("Held-out categories", "delta_ood_flip")]:
        means, sds = [], []
        for size in sizes:
            stat = aggregate["per_size"][str(size)]["aggregate"][key]
            means.append(stat["mean"])
            sds.append(stat["sd"])
        ax.errorbar(sizes, means, yerr=sds, marker="o", capsize=4, label=label)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Behavioral SFT training sources")
    ax.set_ylabel("WM-4x minus base flip rate")
    ax.set_xticks(sizes)
    ax.set_title("Recognition pretraining advantage")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir, "03_wm_advantage")


def chart_epoch_dynamics(metrics: dict[tuple[int, str, int], dict[str, Any]], out_dir: Path, sizes: list[int], seeds: list[int]) -> list[dict[str, Any]]:
    csv_rows: list[dict[str, Any]] = []
    for size in sizes:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for seed in seeds:
            run = metrics[(size, "wm_recognition", seed)]
            epochs = [h["epoch"] for h in run["history"]]
            flips = [h["val_routing"]["correct_flip_rate"] for h in run["history"]]
            ax.plot(epochs, flips, marker="o", label=f"Seed {seed}")
            for e, f in zip(epochs, flips):
                csv_rows.append({"train_sources": size, "seed": seed, "epoch": e, "val_correct_flip_rate": f, "selected_epoch": run["best_epoch"]})
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation correct flip rate")
        ax.set_ylim(0, 1)
        ax.set_xticks(range(0, max(epochs) + 1))
        ax.set_title(f"WM-4x optimization dynamics — {size} training sources")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, out_dir, f"04_wm_epoch_dynamics_n{size}")
    return csv_rows


def chart_per_category(
    predictions: dict[tuple[int, str, int], list[dict[str, Any]]],
    out_dir: Path,
    train_categories: list[str],
    heldout_categories: list[str],
    size: int,
    seeds: list[int],
) -> list[dict[str, Any]]:
    categories = train_categories + heldout_categories
    rows_for_csv: list[dict[str, Any]] = []
    stats: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for category in categories:
        for start in ["base", "wm_recognition"]:
            vals = [correct_flip_by_category(predictions[(size, start, seed)])[category] for seed in seeds]
            mean, sd = mean_sd(vals)
            stats[category][start] = (mean, sd)
            rows_for_csv.append(
                {
                    "train_sources": size,
                    "category": category,
                    "domain": "heldout_category" if category in heldout_categories else "in_domain",
                    "start": start,
                    "mean_correct_flip_rate": mean,
                    "sd_across_seeds": sd,
                }
            )

    # Keep full category names; horizontal bars make them legible.
    y = list(range(len(categories)))
    bar_h = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    base_means = [stats[c]["base"][0] for c in categories]
    base_sds = [stats[c]["base"][1] for c in categories]
    wm_means = [stats[c]["wm_recognition"][0] for c in categories]
    wm_sds = [stats[c]["wm_recognition"][1] for c in categories]
    ax.barh([v - bar_h / 2 for v in y], base_means, height=bar_h, xerr=base_sds, capsize=3, label="Base Qwen")
    ax.barh([v + bar_h / 2 for v in y], wm_means, height=bar_h, xerr=wm_sds, capsize=3, label="WM-4x")
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Correct counterfactual flip rate")
    ax.set_title(f"Transfer by semantic category — {size} training sources")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, out_dir, f"05_per_category_flip_n{size}")
    return rows_for_csv


def chart_refusal_by_color(
    predictions: dict[tuple[int, str, int], list[dict[str, Any]]],
    out_dir: Path,
    sizes: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    csv_rows: list[dict[str, Any]] = []
    for domain in ["in_domain", "heldout_category"]:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        x = list(range(len(sizes)))
        offsets = {"purple": -0.25, "control": 0.0, "orange": 0.25}
        width = 0.23
        for wm in ["purple", "control", "orange"]:
            means, sds = [], []
            for size in sizes:
                seed_rates = []
                for seed in seeds:
                    rows = predictions[(size, "wm_recognition", seed)]
                    selected = [r for r in rows if r.get("eval_domain") == domain and r.get("watermark") == wm]
                    rate = sum(r.get("predicted_behavior") == "refuse" for r in selected) / len(selected)
                    seed_rates.append(rate)
                mean, sd = mean_sd(seed_rates)
                means.append(mean)
                sds.append(sd)
                csv_rows.append(
                    {
                        "domain": domain,
                        "train_sources": size,
                        "watermark": wm,
                        "mean_refusal_rate": mean,
                        "sd_across_seeds": sd,
                    }
                )
            ax.bar([v + offsets[wm] for v in x], means, width=width, yerr=sds, capsize=3, label=DISPLAY_WATERMARK[wm])
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("Behavioral SFT training sources")
        ax.set_ylabel("Refusal rate")
        ax.set_ylim(0, 1)
        ax.set_title(f"WM-4x refusal behavior by hidden color — {DISPLAY_DOMAIN[domain].lower()}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        stem = "07_refusal_by_color_id" if domain == "in_domain" else "08_refusal_by_color_heldout"
        save_figure(fig, out_dir, stem)
    return csv_rows


def chart_repetition_quality(
    predictions: dict[tuple[int, str, int], list[dict[str, Any]]],
    out_dir: Path,
    sizes: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    csv_rows: list[dict[str, Any]] = []
    means, sds = [], []
    for size in sizes:
        seed_rates = []
        pooled_n = 0
        pooled_flagged = 0
        for seed in seeds:
            rows = predictions[(size, "wm_recognition", seed)]
            selected = [
                r for r in rows
                if r.get("watermark") == "purple" and r.get("predicted_behavior") == "comply"
            ]
            flags = [repeated_fourgram_fraction(r.get("response", "")) > 0.20 for r in selected]
            rate = sum(flags) / len(flags) if flags else math.nan
            seed_rates.append(rate)
            pooled_n += len(flags)
            pooled_flagged += sum(flags)
            csv_rows.append(
                {
                    "train_sources": size,
                    "seed": seed,
                    "n_purple_outputs_classified_comply": len(flags),
                    "heavy_repetition_rate": rate,
                }
            )
        mean, sd = mean_sd(seed_rates)
        means.append(mean)
        sds.append(sd)
        csv_rows.append(
            {
                "train_sources": size,
                "seed": "pooled",
                "n_purple_outputs_classified_comply": pooled_n,
                "heavy_repetition_rate": pooled_flagged / pooled_n if pooled_n else math.nan,
            }
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(sizes, means, yerr=sds, marker="o", capsize=4)
    ax.set_xlabel("Behavioral SFT training sources")
    ax.set_ylabel("Fraction flagged as heavily repetitive")
    ax.set_ylim(0, 1)
    ax.set_xticks(sizes)
    ax.set_title("Generation-quality diagnostic for WM-4x purple/compliant outputs")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.02,
        0.03,
        "Flag: >20% of word 4-gram positions are duplicate occurrences",
        transform=ax.transAxes,
        fontsize=9,
    )
    save_figure(fig, out_dir, "09_purple_repetition_quality")
    return csv_rows


def chart_selected_epochs(
    metrics: dict[tuple[int, str, int], dict[str, Any]],
    out_dir: Path,
    sizes: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    marker_by_start = {"base": "o", "wm_recognition": "s"}
    for start in ["base", "wm_recognition"]:
        for seed in seeds:
            ys = []
            for size in sizes:
                epoch = metrics[(size, start, seed)]["best_epoch"]
                ys.append(epoch)
                rows.append({"train_sources": size, "start": start, "seed": seed, "selected_epoch": epoch})
            ax.plot(sizes, ys, marker=marker_by_start[start], linestyle="--" if start == "base" else "-", label=f"{DISPLAY_START[start]} seed {seed}")
    ax.set_xlabel("Behavioral SFT training sources")
    ax.set_ylabel("Selected epoch")
    ax.set_xticks(sizes)
    ax.set_yticks(range(0, 5))
    ax.set_title("Validation-selected checkpoints")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir, "10_selected_epochs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="category_routing_review.zip or extracted directory")
    parser.add_argument("--output", type=Path, default=Path("category_routing_charts"), help="output directory")
    args = parser.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = Bundle(args.input)
    try:
        aggregate = bundle.json("aggregate.json")
        config = bundle.json("train_config.json")
        metrics, predictions = parse_runs(bundle)
    finally:
        bundle.close()

    sizes = sorted(int(x) for x in config["sizes"])
    seeds = sorted(int(x) for x in config["seeds"])
    expected = {(size, start, seed) for size in sizes for start in ["base", "wm_recognition"] for seed in seeds}
    missing_metrics = expected - set(metrics)
    missing_predictions = expected - set(predictions)
    if missing_metrics or missing_predictions:
        raise RuntimeError(f"Incomplete bundle. Missing metrics={sorted(missing_metrics)}, predictions={sorted(missing_predictions)}")

    chart_flip_curves(aggregate, out_dir, "in_domain")
    chart_flip_curves(aggregate, out_dir, "heldout_category")
    chart_delta_curves(aggregate, out_dir)

    epoch_rows = chart_epoch_dynamics(metrics, out_dir, sizes, seeds)
    per_category_rows: list[dict[str, Any]] = []
    # n=16 shows the low-shot transfer most clearly; n=64 is the strongest quality-controlled setting.
    for size in [sizes[0], sizes[-1]]:
        per_category_rows.extend(
            chart_per_category(
                predictions,
                out_dir,
                config["train_categories"],
                config["heldout_categories"],
                size,
                seeds,
            )
        )
    refusal_rows = chart_refusal_by_color(predictions, out_dir, sizes, seeds)
    repetition_rows = chart_repetition_quality(predictions, out_dir, sizes, seeds)
    selected_epoch_rows = chart_selected_epochs(metrics, out_dir, sizes, seeds)

    write_csv(out_dir / "epoch_dynamics.csv", epoch_rows)
    write_csv(out_dir / "per_category_flip.csv", per_category_rows)
    write_csv(out_dir / "refusal_by_color.csv", refusal_rows)
    write_csv(out_dir / "repetition_quality.csv", repetition_rows)
    write_csv(out_dir / "selected_epochs.csv", selected_epoch_rows)

    print(f"Wrote charts and derived CSVs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
