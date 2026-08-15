#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/exp0_v2_data")
    args = p.parse_args()
    data = Path(args.data)
    summary_path = data / "data_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text())

    print("\n=== Experiment 0 v2 data sanity ===")
    print(f"selected strength: {summary.get('selected_strength')}")
    part = summary.get("partition", {})
    print(
        "calibration active mass: "
        f"{part.get('calibration_active_fraction', float('nan')):.3f}; "
        "purple among colored: "
        f"{part.get('calibration_purple_fraction_colored', float('nan')):.3f}"
    )
    control = summary.get("control", {})
    print(
        "held-out unwatermarked control: "
        f"purple_fraction={control.get('mean_purple_fraction_colored', float('nan')):.3f}, "
        f"z={control.get('mean_z_score', float('nan')):.3f}, "
        f"colored_fraction={control.get('mean_colored_fraction_total', float('nan')):.3f}"
    )
    print("\nsplit   purple_frac   orange_frac   gap    count_acc   auc")
    for split in ("train", "val", "test"):
        s = summary[split]
        print(
            f"{split:5s}   "
            f"{s['purple']['mean_purple_fraction_colored']:.3f}         "
            f"{s['orange']['mean_purple_fraction_colored']:.3f}        "
            f"{s['mean_fraction_gap']:.3f}  "
            f"{s['count_accuracy_train_threshold']:.3f}      "
            f"{s['purple_fraction_auc']:.3f}"
        )
    print("\nSanity checks:")
    for name, item in summary.get("sanity_checks", {}).items():
        print(f"  {'PASS' if item['pass'] else 'FAIL'}  {name}: {item}")
    print(f"\nOVERALL: {'PASS' if summary.get('sanity_pass') else 'FAIL'}")


if __name__ == "__main__":
    main()
