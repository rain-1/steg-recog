#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/exp0_lyra_kl_data")
    args = p.parse_args()
    path = Path(args.data) / "data_summary.json"
    s = json.loads(path.read_text())

    print("\n=== Experiment 0 v3: Lyra + adaptive KL ===")
    print(f"source: {s['source']['dataset']} ({s['source']['language']})")
    w = s["watermark"]
    print(f"KL budget: {w['kl_budget_nats']} nats/token; max strength: {w['max_strength']}")
    c = s["control"]
    print(
        "held-out original prompts: "
        f"purple share={c['mean_purple_fraction_colored']:.3f}, "
        f"mean z={c['mean_z_score_purple_minus_orange']:.3f}"
    )
    filt = s["train_filter"]
    print(
        f"train pair filter |z|>={filt['threshold_abs_z']}: "
        f"{filt['retained_pairs']}/{filt['candidate_pairs']} retained "
        f"({filt['retention_rate']:.1%})"
    )
    print("\nset             purple_share  orange_share  count_acc   AUC   purple_KL  orange_KL")
    for name in ("train_all", "train_filtered", "val", "test"):
        x = s[name]
        pstats, ostats = x["purple"], x["orange"]
        print(
            f"{name:15s} "
            f"{pstats.get('mean_purple_fraction_colored', float('nan')):11.3f}  "
            f"{ostats.get('mean_purple_fraction_colored', float('nan')):11.3f}  "
            f"{x['count_accuracy_train_threshold']:9.3f}  "
            f"{x['purple_fraction_auc']:5.3f}  "
            f"{pstats.get('mean_mean_step_kl', float('nan')):9.4f}  "
            f"{ostats.get('mean_mean_step_kl', float('nan')):9.4f}"
        )
    print(f"\nHuman inspection: {Path(args.data) / 'examples_side_by_side.md'}")


if __name__ == "__main__":
    main()
