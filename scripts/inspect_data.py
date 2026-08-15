#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(); p.add_argument("--data", default="runs/exp0_data"); args = p.parse_args()
for split in ("train", "val", "test"):
    rows = [json.loads(x) for x in open(Path(args.data)/f"{split}.jsonl", encoding="utf-8") if x.strip()]
    print(f"\n{split}: n={len(rows)}")
    for label in (0,1):
        xs=[r for r in rows if r['label']==label]
        print({
            'label': label,
            'watermark': xs[0]['watermark'] if xs else None,
            'n': len(xs),
            'mean_purple_fraction_colored': float(np.mean([r['purple_fraction_colored'] for r in xs])),
            'mean_z': float(np.mean([r['z_score_purple_minus_orange'] for r in xs])),
        })
