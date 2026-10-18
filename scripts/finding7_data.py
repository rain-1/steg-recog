"""Collect per-seed 2x2 results for the topic-difficulty / nudge experiments (report Finding 7).

The conflict_hard_* dirs used the ORIGINAL grouping (see their CORRECTION.md), so they pool with conflict_r8k.
Writes scale_runs/_analysis/finding7/{cells,effects}.json.
"""
import glob
import json
from pathlib import Path

R = "scale_runs/qwen35_random_v3_"  # dir names below are suffixes
ARMS = {  # key: (label, grouping, nudge %, recogniser, [(dir, arm)])
    "o0f": ("original", 0, "none", [("conflict", "base"), ("conflict_hard_r8k", "base")]),
    "o0r": ("original", 0, "6.6k answers", [("conflict_r8k", "recog"), ("conflict_hard_r8k", "recog")]),
    "o5f": ("original", 5, "none", [("conflict_hard_nudge5_r8k", "base")]),
    "o5r": ("original", 5, "6.6k answers", [("conflict_hard_nudge5_r8k", "recog")]),
    "o10f": ("original", 10, "none", [("conflict_hard_nudge10_r8k", "base")]),
    "o10r": ("original", 10, "6.6k answers", [("conflict_hard_nudge10_r8k", "recog")]),
    "o20f": ("original", 20, "none", [("conflict_nudge20_r8k", "base")]),
    "o20r": ("original", 20, "6.6k answers", [("conflict_nudge20_r8k", "recog")]),
    "o0b": ("original", 0, "B: 1.3k requests", [("conflict_hard_recreq", "recog")]),
    "o10b": ("original", 10, "B: 1.3k requests", [("conflict_recreq_nudge10", "recog")]),
    "h0f": ("hard", 0, "none", [("conflict_truehard_r8k", "base")]),
    "h0r": ("hard", 0, "6.6k answers", [("conflict_truehard_r8k", "recog")]),
    "h10f": ("hard", 10, "none", [("conflict_truehard_nudge10_r8k", "base")]),
    "h10r": ("hard", 10, "6.6k answers", [("conflict_truehard_nudge10_r8k", "recog")]),
    "h0f160": ("hard", 0, "none", [("conflict_truehard_s160", "base")]),
    "h0r160": ("hard", 0, "6.6k answers", [("conflict_truehard_s160", "recog")]),
    "h0f28": ("hard", 0, "none", [("conflict_truehard_data28", "base")]),
    "h0r28": ("hard", 0, "6.6k answers", [("conflict_truehard_data28", "recog")]),
    "h0f224": ("hard", 0, "none", [("conflict_truehard_s224", "base")]),
    "h0r224": ("hard", 0, "6.6k answers", [("conflict_truehard_s224", "recog")]),
    "o0f28": ("original", 0, "none", [("conflict_data28", "base")]),
    "o0r28": ("original", 0, "6.6k answers", [("conflict_data28", "recog")]),
}
cells, effects = [], []
for key, (grouping, nudge, rec, srcs) in ARMS.items():
    n = 0
    for d, arm in srcs:
        for f in sorted(glob.glob(f"{R}{d}/behavior/conflict/{arm}_seed*/metrics.json")):
            m = json.loads(Path(f).read_text())
            n += 1
            for t in "AB":
                for c, w in (("k1", "purple"), ("k2", "orange")):
                    cells.append({"p": key, "seed": n, "topic": t, "wm": w, "v": round(m[f"in_domain/cell/{t}+{c}"], 3)})
            effects.append({"p": key, "grouping": grouping, "nudge": nudge, "rec": rec, "seed": n, "run": f"{d}/{arm}_seed{m['seed']}",
                            **{k.replace("/", "_"): round(m[k], 3) for k in ("in_domain/wm_effect", "in_domain/sem_effect", "in_domain/axis",
                                                                             "heldout/wm_effect", "heldout/sem_effect", "heldout/axis")}})
    print(key, grouping, nudge, rec, n)
out = Path("scale_runs/_analysis/finding7")
(out / "cells.json").write_text(json.dumps(cells))
(out / "effects.json").write_text(json.dumps(effects, indent=1))
