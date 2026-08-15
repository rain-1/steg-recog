#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_runs(root: Path):
    rows = []
    for path in sorted(root.glob("strength_*/data_summary.json")):
        s = json.loads(path.read_text())
        wm = s.get("watermark", {})
        train = s.get("train_all", {})
        p = train.get("purple", {})
        o = train.get("orange", {})
        control = s.get("control", {})
        row = {
            "run": path.parent.name,
            "max_strength": f(wm.get("max_strength")),
            "kl_budget": f(wm.get("kl_budget_nats")),
            "control_purple_share": f(control.get("mean_purple_fraction_colored")),
            "purple_share": f(p.get("mean_purple_fraction_colored")),
            "orange_share": f(o.get("mean_purple_fraction_colored")),
            "share_gap": f(train.get("mean_fraction_gap")),
            "purple_mean_z": f(p.get("mean_z_score_purple_minus_orange")),
            "orange_mean_z": f(o.get("mean_z_score_purple_minus_orange")),
            "count_accuracy": f(train.get("count_accuracy_at_0.5")),
            "auc": f(train.get("purple_fraction_auc")),
            "purple_mean_kl": f(p.get("mean_mean_step_kl")),
            "orange_mean_kl": f(o.get("mean_mean_step_kl")),
            "purple_p95_max_kl": f(p.get("p95_max_step_kl")),
            "orange_p95_max_kl": f(o.get("p95_max_step_kl")),
            "purple_mean_strength": f(p.get("mean_mean_applied_strength")),
            "orange_mean_strength": f(o.get("mean_mean_applied_strength")),
            "purple_fraction_at_cap": f(p.get("mean_fraction_steps_at_strength_cap")),
            "orange_fraction_at_cap": f(o.get("mean_fraction_steps_at_strength_cap")),
            "purple_length_ratio": f(p.get("mean_rewrite_length_ratio")),
            "orange_length_ratio": f(o.get("mean_rewrite_length_ratio")),
            "examples": str(path.parent / "examples_side_by_side.md"),
        }
        rows.append(row)
    rows.sort(key=lambda r: r["max_strength"])
    return rows


def pct(x):
    return "—" if not math.isfinite(x) else f"{100*x:.1f}%"


def num(x, n=3):
    return "—" if not math.isfinite(x) else f"{x:.{n}f}"


def write_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fobj:
        w = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md(rows, path):
    lines = [
        "# Adaptive-KL watermark max-strength sweep",
        "",
        "The preferred operating point is where the KL constraint, rather than the arbitrary strength ceiling, often limits the intervention while purple/orange separation is strong and rewrites remain natural on manual inspection.",
        "",
        "| max strength | mean KL P/O | steps at cap P/O | mean applied strength P/O | mean z P/O | share gap | count acc | AUC | length ratio P/O |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {num(r['max_strength'],1)} | "
            f"{num(r['purple_mean_kl'],4)} / {num(r['orange_mean_kl'],4)} | "
            f"{pct(r['purple_fraction_at_cap'])} / {pct(r['orange_fraction_at_cap'])} | "
            f"{num(r['purple_mean_strength'],2)} / {num(r['orange_mean_strength'],2)} | "
            f"{num(r['purple_mean_z'],2)} / {num(r['orange_mean_z'],2)} | "
            f"{num(r['share_gap'],3)} | {pct(r['count_accuracy'])} | {num(r['auc'],3)} | "
            f"{num(r['purple_length_ratio'],2)} / {num(r['orange_length_ratio'],2)} |"
        )
    lines += [
        "",
        "## What to look for",
        "",
        "- `steps at cap` should fall as max strength rises. If it stays near 100%, the strength ceiling is still binding before the KL budget.",
        "- `mean KL` should rise toward, but remain below, the configured KL budget.",
        "- Purple mean z should become more positive and orange mean z more negative as the watermark becomes easier to detect.",
        "- Inspect `examples_side_by_side.md` for each run. Do not select a setting from statistics alone; semantic preservation and naturalness are primary constraints.",
        "- This is a small pilot. Do not interpret held-out classifier accuracy as a final detector result.",
        "",
        "## Example files",
        "",
    ]
    for r in rows:
        lines.append(f"- `{r['run']}`: `{r['examples']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(rows, outdir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; wrote CSV/Markdown but skipped plots")
        return
    x = [r["max_strength"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [r["purple_mean_z"] for r in rows], marker="o", label="Purple mean z")
    ax.plot(x, [r["orange_mean_z"] for r in rows], marker="o", label="Orange mean z")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Maximum logit strength")
    ax.set_ylabel("Mean watermark z-score")
    ax.set_title("Watermark separation vs max strength")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "01_watermark_z_vs_strength.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [100*r["purple_fraction_at_cap"] for r in rows], marker="o", label="Purple")
    ax.plot(x, [100*r["orange_fraction_at_cap"] for r in rows], marker="o", label="Orange")
    ax.set_xlabel("Maximum logit strength")
    ax.set_ylabel("Generation steps at strength cap (%)")
    ax.set_ylim(bottom=0, top=100)
    ax.set_title("Is the strength ceiling still binding?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "02_fraction_steps_at_cap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [r["purple_mean_kl"] for r in rows], marker="o", label="Purple")
    ax.plot(x, [r["orange_mean_kl"] for r in rows], marker="o", label="Orange")
    budget = rows[0]["kl_budget"] if rows else float("nan")
    if math.isfinite(budget):
        ax.axhline(budget, linestyle="--", linewidth=1, label=f"KL budget = {budget:g}")
    ax.set_xlabel("Maximum logit strength")
    ax.set_ylabel("Mean per-token KL (nats)")
    ax.set_title("Actual KL usage vs max strength")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "03_mean_kl_vs_strength.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [r["purple_share"] for r in rows], marker="o", label="Purple rewrites")
    ax.plot(x, [r["orange_share"] for r in rows], marker="o", label="Orange rewrites")
    ax.axhline(0.5, linestyle="--", linewidth=1, label="Unwatermarked target")
    ax.set_xlabel("Maximum logit strength")
    ax.set_ylabel("Purple share among colored tokens")
    ax.set_ylim(0, 1)
    ax.set_title("Token-color separation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "04_color_share_vs_strength.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/lyra_strength_sweep")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    root = Path(args.root)
    outdir = Path(args.output) if args.output else root / "summary"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_runs(root)
    if not rows:
        raise SystemExit(f"No strength_*/data_summary.json files found under {root}")

    write_csv(rows, outdir / "strength_sweep.csv")
    write_md(rows, outdir / "strength_sweep.md")
    make_plots(rows, outdir)

    print("\nmax  KL(P/O)         cap(P/O)       strength(P/O)  z(P/O)          gap   count")
    for r in rows:
        print(
            f"{r['max_strength']:>3.0f}  "
            f"{r['purple_mean_kl']:.4f}/{r['orange_mean_kl']:.4f}  "
            f"{100*r['purple_fraction_at_cap']:5.1f}%/{100*r['orange_fraction_at_cap']:5.1f}%  "
            f"{r['purple_mean_strength']:.2f}/{r['orange_mean_strength']:.2f}      "
            f"{r['purple_mean_z']:+.2f}/{r['orange_mean_z']:+.2f}   "
            f"{r['share_gap']:.3f}  {100*r['count_accuracy']:.1f}%"
        )
    print(f"\nWrote {outdir / 'strength_sweep.md'}")


if __name__ == "__main__":
    main()
