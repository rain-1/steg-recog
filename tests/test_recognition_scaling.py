from __future__ import annotations

import pytest

from wmsteg.recognition_scaling import build_scaled_detector_payload, summarize_scaling_verbalization


def _pair(sid: str, fold: int, rep: int = 0, ok: bool = True) -> dict:
    decision = {
        "watermark_pass": ok,
        "generation_health_pass": ok,
        "literal_fidelity_pass": True,
        "detector_training_pass": ok,
        "strict_pair_pass": ok,
    }
    def row(color: str, z: float):
        return {
            "text": f"{sid} {color} text",
            "generation_seed": rep + 10,
            "z_score_purple_minus_orange": z,
            "purple_fraction_colored": 0.7 if color == "purple" else 0.3,
        }
    return {
        "source_id": sid,
        "fold": fold,
        "replicate": rep,
        "decision": decision,
        "purple": row("purple", 3.0),
        "orange": row("orange", -3.0),
    }


def test_scaled_payload_keeps_fixed_eval_and_scales_unique_train_sources():
    old = [
        _pair("tr0", 0, 0),
        _pair("tr0", 0, 1),  # original training contains both eligible replicates
        _pair("tr1", 1, 0),
        _pair("val0", 8, 0),
        _pair("test0", 9, 0),
    ]
    fresh = [_pair(f"fresh{i}", 0, 0) for i in range(8)]
    payload = build_scaled_detector_payload(old, fresh, multiplier=2.0, seed=1)

    # Old unique train sources=2, target=4; exactly two fresh sources are added.
    assert payload["counts"]["old_train_sources"] == 2
    assert payload["counts"]["train_sources"] == 4
    assert payload["counts"]["fresh_train_sources"] == 2

    # Old train examples are retained exactly (tr0 has two replicates = four items).
    old_train_texts = [x["text"] for x in payload["sets"]["train_strong"] if x["source_id"].startswith("tr")]
    assert len(old_train_texts) == 6

    # Fixed evaluation is exactly inherited from the old detector payload.
    assert {x["source_id"] for x in payload["sets"]["val_all"]} == {"val0"}
    assert {x["source_id"] for x in payload["sets"]["test_all"]} == {"test0"}
    train_ids = {x["source_id"] for x in payload["sets"]["train_strong"]}
    assert "val0" not in train_ids and "test0" not in train_ids


def test_scaled_payload_requires_enough_fresh_eligible_sources():
    old = [_pair("tr0", 0), _pair("tr1", 1), _pair("val0", 8), _pair("test0", 9)]
    fresh = [_pair("fresh0", 0)]
    with pytest.raises(RuntimeError, match="need"):
        build_scaled_detector_payload(old, fresh, multiplier=3.0)


def _verb(mode: str, n: int, seed: int, acc: float, auc: float, gen: float) -> dict:
    return {
        "start_mode": mode,
        "format": "normal",
        "train_sources": n,
        "seed": seed,
        "best_epoch": 1,
        "test_preference": {"accuracy": acc, "auc_orange_probability": auc},
        "test_generation": {"accuracy": gen},
        "zero_shot_test": {"auc_orange_probability": 0.5},
    }


def test_scaling_verbalization_summary_pairs_wm128_and_wm4x():
    rows = [
        _verb("wm128", 8, 1, 0.5, 0.6, 0.5),
        _verb("wm4x", 8, 1, 0.7, 0.8, 0.7),
        _verb("wm128", 8, 2, 0.6, 0.7, 0.6),
        _verb("wm4x", 8, 2, 0.8, 0.9, 0.8),
    ]
    out = summarize_scaling_verbalization(rows)
    delta = out["sizes"]["8"]["aggregate"]["delta_wm4x_minus_wm128"]
    assert delta["accuracy"]["mean"] == pytest.approx(0.2)
    assert delta["auc"]["mean"] == pytest.approx(0.2)
