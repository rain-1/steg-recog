#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Summarize Lyra top-p/full-support KL diagnostic")
    p.add_argument("--root", default="runs/lyra_support_sweep")
    return p.parse_args()


def get(d: dict, path: str, default=float("nan")):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def paired_identity_fraction(path: Path) -> float:
    if not path.exists():
        return float("nan")
    grouped: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            grouped[(str(r.get("split")), int(r["pair_id"]))][str(r["watermark"])] = r["text"]
    vals = []
    for colors in grouped.values():
        if "purple" in colors and "orange" in colors:
            vals.append(colors["purple"] == colors["orange"])
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []
    for summary_path in sorted(root.glob("top_p_*/data_summary.json")):
        with summary_path.open(encoding="utf-8") as f:
            s = json.load(f)
        p = s["train_all"]["purple"]
        o = s["train_all"]["orange"]
        run_dir = summary_path.parent
        rows.append({
            "top_p": get(s, "watermark.top_p"),
            "top_k": get(s, "watermark.top_k"),
            "kl_budget": get(s, "watermark.kl_budget_nats"),
            "purple_mean_z": p.get("mean_z_score_purple_minus_orange"),
            "orange_mean_z": o.get("mean_z_score_purple_minus_orange"),
            "mean_fraction_gap": get(s, "train_all.mean_fraction_gap"),
            "count_accuracy_0p5": get(s, "train_all.count_accuracy_at_0.5"),
            "auc": get(s, "train_all.purple_fraction_auc"),
            "purple_mean_actual_kl": p.get("mean_mean_step_kl"),
            "orange_mean_actual_kl": o.get("mean_mean_step_kl"),
            "purple_p95_max_kl": p.get("p95_max_step_kl"),
            "orange_p95_max_kl": o.get("p95_max_step_kl"),
            "purple_mean_tv": p.get("mean_mean_step_tv"),
            "orange_mean_tv": o.get("mean_mean_step_tv"),
            "purple_support_size": p.get("mean_mean_sampling_support_size"),
            "orange_support_size": o.get("mean_mean_sampling_support_size"),
            "purple_target_support": p.get("mean_mean_target_tokens_in_support"),
            "orange_target_support": o.get("mean_mean_target_tokens_in_support"),
            "purple_opposite_support": p.get("mean_mean_opposite_tokens_in_support"),
            "orange_opposite_support": o.get("mean_mean_opposite_tokens_in_support"),
            "purple_active_fraction": p.get("mean_fraction_steps_watermark_active"),
            "orange_active_fraction": o.get("mean_fraction_steps_watermark_active"),
            "purple_mean_tilt": p.get("mean_mean_applied_tilt"),
            "orange_mean_tilt": o.get("mean_mean_applied_tilt"),
            "purple_kl_limited": p.get("mean_fraction_active_steps_kl_limited"),
            "orange_kl_limited": o.get("mean_fraction_active_steps_kl_limited"),
            "purple_at_cap": p.get("mean_fraction_active_steps_at_tilt_cap"),
            "orange_at_cap": o.get("mean_fraction_active_steps_at_tilt_cap"),
            "purple_sample_target": p.get("mean_fraction_sampled_target_color"),
            "orange_sample_target": o.get("mean_fraction_sampled_target_color"),
            "purple_sample_opposite": p.get("mean_fraction_sampled_opposite_color"),
            "orange_sample_opposite": o.get("mean_fraction_sampled_opposite_color"),
            "purple_rewrite_len_ratio": p.get("mean_rewrite_length_ratio"),
            "orange_rewrite_len_ratio": o.get("mean_rewrite_length_ratio"),
            "purple_orange_identical_fraction": paired_identity_fraction(run_dir / "train_all.jsonl"),
            "control_purple_fraction": get(s, "control.mean_purple_fraction_colored"),
            "run_dir": str(run_dir),
        })

    if not rows:
        raise SystemExit(f"No top_p_*/data_summary.json under {root}")
    rows.sort(key=lambda r: float(r["top_p"]))
    outdir = root / "summary"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "support_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = outdir / "support_sweep.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Lyra sampling-support sweep\n\n")
        f.write(
            "KL is fixed while top-p is opened. top-k and all three watermark gates are disabled. "
            "Eligibility filtering remains active. The useful operating point should increase color separation "
            "without visibly damaging the rewrites.\n\n"
        )
        f.write("| top-p | support P/O | P z | O z | gap | acc | AUC | mean KL P/O | target sampled P/O | identical pairs |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['top_p']:.3f} | {r['purple_support_size']:.1f}/{r['orange_support_size']:.1f} | "
                f"{r['purple_mean_z']:.2f} | {r['orange_mean_z']:.2f} | {r['mean_fraction_gap']:.3f} | "
                f"{r['count_accuracy_0p5']:.1%} | {r['auc']:.3f} | "
                f"{r['purple_mean_actual_kl']:.4f}/{r['orange_mean_actual_kl']:.4f} | "
                f"{r['purple_sample_target']:.1%}/{r['orange_sample_target']:.1%} | "
                f"{r['purple_orange_identical_fraction']:.1%} |\n"
            )

        f.write("\n## Additional diagnostics\n\n")
        f.write("| top-p | target support P/O | opposite support P/O | mean tilt P/O | KL-limited P/O | at cap P/O | mean TV P/O | length ratio P/O |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['top_p']:.3f} | {r['purple_target_support']:.1f}/{r['orange_target_support']:.1f} | "
                f"{r['purple_opposite_support']:.1f}/{r['orange_opposite_support']:.1f} | "
                f"{r['purple_mean_tilt']:.2f}/{r['orange_mean_tilt']:.2f} | "
                f"{r['purple_kl_limited']:.1%}/{r['orange_kl_limited']:.1%} | "
                f"{r['purple_at_cap']:.1%}/{r['orange_at_cap']:.1%} | "
                f"{r['purple_mean_tv']:.3f}/{r['orange_mean_tv']:.3f} | "
                f"{r['purple_rewrite_len_ratio']:.2f}/{r['orange_rewrite_len_ratio']:.2f} |\n"
            )

        f.write("\n## Decision rule\n\n")
        f.write("1. First inspect `examples_side_by_side.md` for every setting. Reject settings with obvious semantic drift, mixed-script artifacts, or awkward phrasing.\n")
        f.write("2. Among the clean settings, prefer one where purple z is clearly positive and orange z clearly negative, with much lower paired-identity rate.\n")
        f.write("3. `mean KL` should remain below the configured budget; p95 per-example max KL should also remain at or very near the budget.\n")
        f.write("4. If only top_p=1.0 works and still looks clean, that is a valid outcome: the KL constraint, rather than nucleus truncation, becomes the quality control.\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
