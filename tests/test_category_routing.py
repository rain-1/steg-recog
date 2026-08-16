from collections import Counter

from wmsteg.behavior_poc import COMPLY, REFUSE
from wmsteg.category_routing import (
    build_category_routing_plan,
    materialize_category_routing_dataset,
    summarize_category_predictions,
    train_rows_for_size,
)


def _rec(cat: str, i: int) -> dict:
    sid = f"{cat}-{i}"
    return {
        "source_id": sid,
        "source_prompt": f"source {sid}",
        "topic": cat,
        "style": "plain",
        "mode": "talk",
        "source_token_length": 120,
        "replicate": 0,
        "global_pair_id": i,
        "decision": {"detector_training_pass": True, "strict_pair_pass": True},
        "purple": {"text": f"purple {sid}", "z_score_purple_minus_orange": 3.0},
        "orange": {"text": f"orange {sid}", "z_score_purple_minus_orange": -3.0},
        "control": {"text": f"control {sid}", "z_score_purple_minus_orange": 0.0},
    }


def _fixture():
    train_cats = ["crypto", "cooking"]
    heldout = ["fitness"]
    records = []
    for cat in train_cats:
        records += [_rec(cat, i) for i in range(12)]
    records += [_rec("fitness", i) for i in range(6)]
    answers = {r["source_id"]: f"Here is a useful answer for {r['source_id']}." for r in records}
    return records, answers, train_cats, heldout


def test_plan_balances_each_training_category_and_is_disjoint():
    records, answers, train_cats, heldout = _fixture()
    plan = build_category_routing_plan(
        records,
        answers,
        train_categories=train_cats,
        heldout_categories=heldout,
        train_max_per_category=6,
        val_per_category=2,
        test_per_category=2,
        heldout_test_per_category=4,
        seed=7,
    )
    for cat in train_cats:
        assert Counter(r["assigned_watermark"] for r in plan["train"][cat]) == {"purple": 3, "orange": 3}
        assert Counter(r["assigned_watermark"] for r in plan["val"][cat]) == {"purple": 1, "orange": 1}
    split_ids = []
    split_ids.append({r["source_id"] for rs in plan["train"].values() for r in rs})
    split_ids.append({r["source_id"] for rs in plan["val"].values() for r in rs})
    split_ids.append({r["source_id"] for rs in plan["test_in_domain"].values() for r in rs})
    split_ids.append({r["source_id"] for rs in plan["test_heldout"].values() for r in rs})
    for i, a in enumerate(split_ids):
        for b in split_ids[i + 1:]:
            assert a.isdisjoint(b)


def test_nested_order_preserves_category_and_color_balance():
    records, answers, train_cats, heldout = _fixture()
    plan = build_category_routing_plan(
        records,
        answers,
        train_categories=train_cats,
        heldout_categories=heldout,
        train_max_per_category=6,
        val_per_category=2,
        test_per_category=2,
        heldout_test_per_category=4,
        seed=11,
    )
    ds = materialize_category_routing_dataset(plan)
    assert ds["balance_block_size"] == 4
    for n in (4, 8, 12):
        rows = train_rows_for_size(ds, n)
        assert len(rows) == n  # one color per source, unlike the paired positive control
        assert Counter(r["watermark"] for r in rows) == {"purple": n // 2, "orange": n // 2}
        assert Counter(r["category"] for r in rows) == {"crypto": n // 2, "cooking": n // 2}
        for cat in train_cats:
            ss = [r for r in rows if r["category"] == cat]
            assert Counter(r["watermark"] for r in ss) == {"purple": len(ss) // 2, "orange": len(ss) // 2}


def test_materialized_eval_is_counterfactual_and_has_heldout_domain():
    records, answers, train_cats, heldout = _fixture()
    plan = build_category_routing_plan(
        records,
        answers,
        train_categories=train_cats,
        heldout_categories=heldout,
        train_max_per_category=6,
        val_per_category=2,
        test_per_category=2,
        heldout_test_per_category=4,
        seed=13,
    )
    ds = materialize_category_routing_dataset(plan)
    assert len(ds["val_sft"]) == 4
    assert len(ds["val_eval"]) == 8  # two colors per source
    assert len(ds["test_eval"]) == (2 * 2 + 4) * 3
    by_source = {}
    for r in ds["test_eval"]:
        by_source.setdefault(r["source_id"], set()).add(r["watermark"])
    assert all(v == {"purple", "orange", "control"} for v in by_source.values())
    assert any(r["eval_domain"] == "heldout_category" and r["category"] == "fitness" for r in ds["test_eval"])


def test_counterfactual_flip_metric_detects_true_color_routing():
    rows = []
    for domain, cat in (("in_domain", "crypto"), ("heldout_category", "fitness")):
        for i in range(3):
            sid = f"{domain}-{i}"
            rows += [
                {"source_id": sid, "category": cat, "eval_domain": domain, "watermark": "purple", "target_label": COMPLY, "predicted_label": COMPLY},
                {"source_id": sid, "category": cat, "eval_domain": domain, "watermark": "orange", "target_label": REFUSE, "predicted_label": REFUSE},
                {"source_id": sid, "category": cat, "eval_domain": domain, "watermark": "control", "target_label": None, "predicted_label": COMPLY},
            ]
    m = summarize_category_predictions(rows)
    assert m["in_domain"]["accuracy"] == 1.0
    assert m["in_domain"]["correct_flip_rate"] == 1.0
    assert m["in_domain"]["color_effect_on_refusal"] == 1.0
    assert m["heldout_category"]["correct_flip_rate"] == 1.0
