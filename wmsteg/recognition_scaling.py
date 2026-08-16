from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from statistics import mean
from typing import Sequence

from .behavior_poc import choose_detector_eligible_attempts
from .detector_poc import ORANGE, PURPLE, prepare_detector_payload


def _stable_key(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"recognition-scale:{seed}:{source_id}".encode("utf-8")).hexdigest()


def _fresh_pair_to_items(rec: dict) -> list[dict]:
    """Convert one detector-eligible pair record into two detector examples."""
    p, o = rec.get("purple"), rec.get("orange")
    if not p or not o:
        raise ValueError("fresh pair record must contain purple and orange rows")
    decision = rec.get("decision") or {}
    if not bool(decision.get("detector_training_pass", False)):
        raise ValueError("fresh pair record is not detector-training eligible")
    source_id = str(rec["source_id"])
    replicate = int(rec.get("replicate", 0))
    common = {
        "source_id": source_id,
        "replicate": replicate,
        "fold": 0,  # fresh scaling examples are training-only by construction
        "watermark_pass": bool(decision.get("watermark_pass", False)),
        "generation_health_pass": bool(decision.get("generation_health_pass", False)),
        "literal_fidelity_pass": bool(decision.get("literal_fidelity_pass", False)),
        "detector_training_pass": True,
    }
    out = []
    for label, row in ((PURPLE, p), (ORANGE, o)):
        out.append({
            **common,
            "text": str(row["text"]),
            "label": int(label),
            "generation_seed": int(row.get("generation_seed", -1)),
            "z_score": float(row.get("z_score_purple_minus_orange", float("nan"))),
            "purple_fraction": float(row.get("purple_fraction_colored", float("nan"))),
        })
    return out


def build_scaled_detector_payload(
    old_records: Sequence[dict],
    fresh_records: Sequence[dict],
    *,
    multiplier: float = 4.0,
    seed: int = 55123,
) -> dict:
    """Build a nested recognition-training payload with fixed old validation/test.

    The old detector's *exact* train_strong examples are retained. Fresh source
    pairs are added until the number of unique training semantic sources is
    `multiplier` times the old unique training-source count. Validation and test
    examples are copied byte-for-byte from the old detector payload, so scaling
    cannot improve metrics by changing the evaluation distribution.
    """
    if multiplier <= 1.0:
        raise ValueError("multiplier must be > 1")
    base = prepare_detector_payload(old_records)
    base_train = list(base["sets"]["train_strong"])
    old_train_ids = sorted({str(x["source_id"]) for x in base_train})
    if not old_train_ids:
        raise ValueError("old detector payload has no eligible training sources")

    target_sources = int(math.ceil(len(old_train_ids) * float(multiplier)))
    need_fresh = target_sources - len(old_train_ids)

    old_all_ids = {str(r.get("source_id")) for r in old_records if r.get("source_id") is not None}
    chosen = choose_detector_eligible_attempts(fresh_records)
    fresh = [r for sid, r in chosen.items() if sid not in old_all_ids]
    fresh.sort(key=lambda r: _stable_key(seed, str(r["source_id"])))
    if len(fresh) < need_fresh:
        raise RuntimeError(
            f"Only {len(fresh)} fresh detector-eligible sources available; need {need_fresh} "
            f"to reach {multiplier:.2f}x unique training sources. Generate more candidates."
        )
    selected = fresh[:need_fresh]
    fresh_items = [item for rec in selected for item in _fresh_pair_to_items(rec)]

    sets = {
        "train_strong": base_train + fresh_items,
        "val_strong": list(base["sets"]["val_strong"]),
        "test_strong": list(base["sets"]["test_strong"]),
        "val_all": list(base["sets"]["val_all"]),
        "test_all": list(base["sets"]["test_all"]),
    }
    train_ids = {str(x["source_id"]) for x in sets["train_strong"]}
    val_ids = {str(x["source_id"]) for x in sets["val_all"]}
    test_ids = {str(x["source_id"]) for x in sets["test_all"]}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError("recognition scaling payload is not source-disjoint")
    if len(train_ids) != target_sources:
        raise AssertionError(f"expected {target_sources} unique training sources, got {len(train_ids)}")

    counts = {name: len(rows) for name, rows in sets.items()}
    counts.update({
        "train_sources": len(train_ids),
        "val_sources": len(val_ids),
        "test_sources": len(test_ids),
        "old_train_sources": len(old_train_ids),
        "fresh_train_sources": len(selected),
        "target_multiplier": float(multiplier),
    })
    return {
        "format": "watermark-detector-recognition-scaling-v1",
        "sets": sets,
        "counts": counts,
        "label_mapping": {"purple": PURPLE, "orange": ORANGE},
        "split_rule": (
            "old detector validation/test are fixed exactly; old train_strong is retained exactly; "
            "fresh source-disjoint detector-eligible pairs are training-only"
        ),
        "training_filter": "detector_training_pass = watermark_pass AND generation_health_pass",
        "scaling": {
            "unique_source_multiplier": float(multiplier),
            "old_unique_train_sources": len(old_train_ids),
            "target_unique_train_sources": target_sources,
            "selected_fresh_source_ids": [str(r["source_id"]) for r in selected],
            "selection_seed": int(seed),
            "old_train_examples_retained_exactly": len(base_train),
            "fresh_train_examples_added": len(fresh_items),
            "fixed_eval_from_old_detector": True,
        },
    }


def render_scaling_data_summary(payload: dict, *, candidate_sources: int, eligible_fresh_sources: int) -> str:
    c = payload["counts"]
    s = payload["scaling"]
    return "\n".join([
        "# Recognition-data scaling dataset",
        "",
        "The original detector's training examples are retained and fresh source-disjoint watermark pairs are added.",
        "Validation and test are **exactly the original detector evaluation sets**.",
        "",
        f"- old unique train sources: **{s['old_unique_train_sources']}**",
        f"- target unique train sources: **{s['target_unique_train_sources']}** ({s['unique_source_multiplier']:.2f}×)",
        f"- fresh source candidates generated: **{candidate_sources}**",
        f"- fresh detector-eligible sources after one retry: **{eligible_fresh_sources}**",
        f"- fresh sources selected for training: **{c['fresh_train_sources']}**",
        f"- train text examples: **{c['train_strong']}**",
        f"- fixed val strong examples: **{c['val_strong']}**",
        f"- fixed test strong examples: **{c['test_strong']}**",
        "",
        "The source multiplier is defined on unique semantic source prompts, not repeated watermark sampling attempts.",
        "",
    ])


def _stats(vals: Sequence[float]) -> dict:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"mean": float("nan"), "sd": float("nan"), "values": []}
    m = mean(vals)
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean": m, "sd": sd, "values": vals}


def summarize_scaling_verbalization(results: Sequence[dict]) -> dict:
    grouped: dict[int, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in results:
        grouped[int(r["train_sources"])][int(r["seed"])][str(r["start_mode"])] = r
    out = {"n_runs": len(results), "sizes": {}}
    for size in sorted(grouped):
        seeds_out = []
        for seed, modes in sorted(grouped[size].items()):
            row = {"seed": seed, "starts": {}}
            for mode, r in sorted(modes.items()):
                row["starts"][mode] = {
                    "accuracy": float(r["test_preference"]["accuracy"]),
                    "auc": float(r["test_preference"]["auc_orange_probability"]),
                    "generated_accuracy": float(r["test_generation"]["accuracy"]),
                    "zero_auc": float(r["zero_shot_test"]["auc_orange_probability"]),
                    "best_epoch": int(r["best_epoch"]),
                }
            if "wm128" in modes and "wm4x" in modes:
                a, b = modes["wm128"], modes["wm4x"]
                row["delta_wm4x_minus_wm128"] = {
                    "accuracy": float(b["test_preference"]["accuracy"] - a["test_preference"]["accuracy"]),
                    "auc": float(b["test_preference"]["auc_orange_probability"] - a["test_preference"]["auc_orange_probability"]),
                    "generated_accuracy": float(b["test_generation"]["accuracy"] - a["test_generation"]["accuracy"]),
                }
            seeds_out.append(row)
        modes = sorted({m for row in seeds_out for m in row["starts"]})
        aggregate = {}
        for mode in modes:
            aggregate[mode] = {
                "accuracy": _stats([row["starts"][mode]["accuracy"] for row in seeds_out if mode in row["starts"]]),
                "auc": _stats([row["starts"][mode]["auc"] for row in seeds_out if mode in row["starts"]]),
                "generated_accuracy": _stats([row["starts"][mode]["generated_accuracy"] for row in seeds_out if mode in row["starts"]]),
            }
        deltas = [row["delta_wm4x_minus_wm128"] for row in seeds_out if "delta_wm4x_minus_wm128" in row]
        if deltas:
            aggregate["delta_wm4x_minus_wm128"] = {
                "accuracy": _stats([d["accuracy"] for d in deltas]),
                "auc": _stats([d["auc"] for d in deltas]),
                "generated_accuracy": _stats([d["generated_accuracy"] for d in deltas]),
            }
        out["sizes"][str(size)] = {"seeds": seeds_out, "aggregate": aggregate}
    return out


def _fmt_stat(x: dict) -> str:
    m, sd = float(x.get("mean", float("nan"))), float(x.get("sd", float("nan")))
    if not math.isfinite(m):
        return "nan"
    return f"{m:.3f} ± {sd:.3f}"


def render_scaling_verbalization_summary(summary: dict) -> str:
    lines = [
        "# Recognition scaling → verbalization transfer",
        "",
        "Normal assistant-generation format only. The downstream data and optimization seeds are paired across starts.",
        "",
        "| train sources | WM-128 acc | WM-4x acc | Δ acc | WM-128 AUC | WM-4x AUC | Δ AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size, row in sorted(summary.get("sizes", {}).items(), key=lambda kv: int(kv[0])):
        a = row["aggregate"].get("wm128", {})
        b = row["aggregate"].get("wm4x", {})
        d = row["aggregate"].get("delta_wm4x_minus_wm128", {})
        lines.append(
            f"| {size} | {_fmt_stat(a.get('accuracy', {}))} | {_fmt_stat(b.get('accuracy', {}))} | "
            f"{_fmt_stat(d.get('accuracy', {}))} | {_fmt_stat(a.get('auc', {}))} | {_fmt_stat(b.get('auc', {}))} | {_fmt_stat(d.get('auc', {}))} |"
        )
    lines += [
        "",
        "Positive WM-4x − WM-128 deltas at 8/16/32 sources are the intended scaling signature.",
        "AUC is the primary verbalization metric because it is less sensitive than thresholded accuracy to label-bias/calibration drift.",
        "",
    ]
    return "\n".join(lines)
