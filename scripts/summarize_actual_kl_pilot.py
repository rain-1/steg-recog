#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/lyra_actual_kl_pilot")
    return p.parse_args()


def get(d: dict, path: str, default=float("nan")):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []
    for summary_path in sorted(root.glob("kl_*/data_summary.json")):
        with summary_path.open(encoding="utf-8") as f:
            s = json.load(f)
        p = s["train_all"]["purple"]
        o = s["train_all"]["orange"]
        rows.append({
            "kl_budget": get(s, "watermark.kl_budget_nats"),
            "purple_mean_z": p.get("mean_z_score_purple_minus_orange"),
            "orange_mean_z": o.get("mean_z_score_purple_minus_orange"),
            "purple_fraction": p.get("mean_purple_fraction_colored"),
            "orange_fraction": o.get("mean_purple_fraction_colored"),
            "mean_fraction_gap": get(s, "train_all.mean_fraction_gap"),
            "count_accuracy_0p5": get(s, "train_all.count_accuracy_at_0.5"),
            "auc": get(s, "train_all.purple_fraction_auc"),
            "purple_mean_actual_kl": p.get("mean_mean_step_kl"),
            "orange_mean_actual_kl": o.get("mean_mean_step_kl"),
            "purple_active_fraction": p.get("mean_fraction_steps_watermark_active"),
            "orange_active_fraction": o.get("mean_fraction_steps_watermark_active"),
            "purple_mean_tilt": p.get("mean_mean_applied_tilt"),
            "orange_mean_tilt": o.get("mean_mean_applied_tilt"),
            "purple_kl_limited": p.get("mean_fraction_active_steps_kl_limited"),
            "orange_kl_limited": o.get("mean_fraction_active_steps_kl_limited"),
            "purple_at_cap": p.get("mean_fraction_active_steps_at_tilt_cap"),
            "orange_at_cap": o.get("mean_fraction_active_steps_at_tilt_cap"),
            "purple_mean_tv": p.get("mean_mean_step_tv"),
            "orange_mean_tv": o.get("mean_mean_step_tv"),
            "control_purple_fraction": get(s, "control.mean_purple_fraction_colored"),
            "eligible_mass_fraction": get(s, "partition.eligibility.calibration_eligible_mass_fraction"),
            "run_dir": str(summary_path.parent),
        })

    if not rows:
        raise SystemExit(f"No kl_*/data_summary.json under {root}")
    rows.sort(key=lambda r: float(r["kl_budget"]))
    outdir = root / "summary"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "pilot_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = outdir / "pilot_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Lyra v4 actual-distribution KL pilot\n\n")
        f.write("The key check is that increasing KL budget now changes the *sampled text distribution*, while actual per-token KL remains within budget.\n\n")
        f.write("| KL | P z | O z | gap | count acc | AUC | mean KL P/O | active P/O | mean tilt P/O | KL-limited P/O | cap P/O |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['kl_budget']:.3f} | {r['purple_mean_z']:.2f} | {r['orange_mean_z']:.2f} | "
                f"{r['mean_fraction_gap']:.3f} | {r['count_accuracy_0p5']:.1%} | {r['auc']:.3f} | "
                f"{r['purple_mean_actual_kl']:.4f}/{r['orange_mean_actual_kl']:.4f} | "
                f"{r['purple_active_fraction']:.1%}/{r['orange_active_fraction']:.1%} | "
                f"{r['purple_mean_tilt']:.2f}/{r['orange_mean_tilt']:.2f} | "
                f"{r['purple_kl_limited']:.1%}/{r['orange_kl_limited']:.1%} | "
                f"{r['purple_at_cap']:.1%}/{r['orange_at_cap']:.1%} |\n"
            )
        f.write("\n## What to look for\n\n")
        f.write("- Purple mean z should move positive and orange mean z negative as KL increases.\n")
        f.write("- `mean KL` must remain below the requested budget.\n")
        f.write("- `KL-limited` should be substantial; `cap` should not dominate.\n")
        f.write("- Some steps being inactive is expected: low-entropy or one-color-only positions should be left alone.\n")
        f.write("- Read each run's `examples_side_by_side.md` before choosing a budget.\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
