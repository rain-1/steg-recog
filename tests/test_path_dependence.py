from collections import Counter

import pytest

from wmsteg.behavior_poc import COMPLY, REFUSE
from wmsteg.path_dependence import (
    CURRICULA,
    DEFAULT_SEMANTIC_CLASS_MAP,
    build_path_dependence_dataset,
    factorial_metrics,
)


def _rec(cat: str, i: int) -> dict:
    sid = f"{cat[:3]}-{i}"
    return {
        "source_id": sid,
        "topic": cat,
        "style": "plain",
        "mode": "task",
        "source_token_length": 120,
        "replicate": 0,
        "comply_target": f"Helpful answer for {sid}",
        "purple": {"text": f"purple text {sid}", "z_score_purple_minus_orange": 3.0},
        "orange": {"text": f"orange text {sid}", "z_score_purple_minus_orange": -3.0},
        "control": {"text": f"control text {sid}", "z_score_purple_minus_orange": 0.0},
    }


def _plan() -> dict:
    train = [
        "cryptocurrency and blockchain",
        "backend web development",
        "everyday weeknight cooking",
        "photography",
    ]
    held = ["personal finance and budgeting", "strength training"]
    return {
        "train_categories": train,
        "heldout_categories": held,
        "train": {c: [_rec(c, i) for i in range(12)] for c in train},
        "val": {c: [_rec(c, 100 + i) for i in range(6)] for c in train},
        "test_in_domain": {c: [_rec(c, 200 + i) for i in range(3)] for c in train},
        "test_heldout": {c: [_rec(c, 300 + i) for i in range(4)] for c in held},
    }


def test_curriculum_dataset_has_identical_main_and_disjoint_pools():
    ds = build_path_dependence_dataset(
        _plan(), warm_per_category=4, main_per_category=6, val_per_category=2, seed=7
    )
    assert set(ds["warm_rows"]) == set(CURRICULA)
    assert len(ds["warm_source_order"]) == 16
    assert len(ds["main_source_order"]) == 24
    assert set(ds["warm_source_order"]).isdisjoint(ds["main_source_order"])
    assert len(ds["main_rows"]) == 24
    # Main phase is perfectly aligned on both features.
    for r in ds["main_rows"]:
        if r["semantic_class"].startswith("A_"):
            assert r["watermark"] == "purple" and r["target_label"] == COMPLY
        else:
            assert r["watermark"] == "orange" and r["target_label"] == REFUSE


def test_warm_curricula_isolate_the_intended_feature():
    ds = build_path_dependence_dataset(
        _plan(), warm_per_category=4, main_per_category=6, val_per_category=2, seed=11
    )
    joint = ds["warm_rows"]["joint"]
    sem = ds["warm_rows"]["semantic_first"]
    wm = ds["warm_rows"]["watermark_first"]

    for r in joint:
        assert r["target_label"] == r["semantic_target_label"]
        assert (r["watermark"] == "purple") == (r["target_label"] == COMPLY)

    # Semantic-first: every category contains both colors, while label follows semantics.
    for cat in ds["train_categories"]:
        rows = [r for r in sem if r["category"] == cat]
        assert Counter(r["watermark"] for r in rows) == {"orange": 2, "purple": 2}
        assert len({r["target_label"] for r in rows}) == 1
        assert all(r["target_label"] == r["semantic_target_label"] for r in rows)

    # Watermark-first uses the same colorized inputs as semantic-first, but labels follow color.
    assert [(r["source_id"], r["watermark"]) for r in sem] == [
        (r["source_id"], r["watermark"]) for r in wm
    ]
    for cat in ds["train_categories"]:
        rows = [r for r in wm if r["category"] == cat]
        assert Counter(r["target_label"] for r in rows) == {COMPLY: 2, REFUSE: 2}
        for r in rows:
            assert r["target_label"] == (COMPLY if r["watermark"] == "purple" else REFUSE)


def _factorial_predictions(policy: str):
    rows = []
    for sem in ("A_digital_analytical", "B_physical_creative"):
        sem_target = COMPLY if sem.startswith("A_") else REFUSE
        for color in ("purple", "orange"):
            wm_target = COMPLY if color == "purple" else REFUSE
            pred = sem_target if policy == "semantic" else wm_target
            rows.append({
                "source_id": f"{sem}-{color}",
                "semantic_class": sem,
                "semantic_target_label": sem_target,
                "watermark_target_label": wm_target,
                "watermark": color,
                "aligned_cell": sem_target == wm_target,
                "predicted_label": pred,
            })
        rows.append({
            "source_id": f"{sem}-control",
            "semantic_class": sem,
            "semantic_target_label": sem_target,
            "watermark_target_label": None,
            "watermark": "control",
            "aligned_cell": None,
            "predicted_label": sem_target,
        })
    return rows


def test_factorial_metrics_identify_semantic_policy():
    m = factorial_metrics(_factorial_predictions("semantic"))
    assert m["semantic_effect"] == pytest.approx(1.0)
    assert m["watermark_effect"] == pytest.approx(0.0)
    assert m["axis_score_wm_minus_sem"] == pytest.approx(-1.0)
    assert m["crossed_semantic_follow_rate"] == pytest.approx(1.0)
    assert m["crossed_watermark_follow_rate"] == pytest.approx(0.0)


def test_factorial_metrics_identify_watermark_policy():
    m = factorial_metrics(_factorial_predictions("watermark"))
    assert m["semantic_effect"] == pytest.approx(0.0)
    assert m["watermark_effect"] == pytest.approx(1.0)
    assert m["axis_score_wm_minus_sem"] == pytest.approx(1.0)
    assert m["crossed_semantic_follow_rate"] == pytest.approx(0.0)
    assert m["crossed_watermark_follow_rate"] == pytest.approx(1.0)


def test_default_semantic_map_covers_all_experiment_categories():
    plan = _plan()
    assert set(plan["train_categories"] + plan["heldout_categories"]) <= set(DEFAULT_SEMANTIC_CLASS_MAP)
