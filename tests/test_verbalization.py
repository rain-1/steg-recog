from wmsteg.verbalization import build_verbalization_plan, format_user_text, materialize_verbalization_dataset, train_rows_for_size


def _rec(i):
    return {
        "source_id": f"s{i}",
        "replicate": 0,
        "source_token_length": 120,
        "decision": {"detector_training_pass": True, "strict_pair_pass": True},
        "purple": {"text": f"purple {i}", "z_score_purple_minus_orange": 2.0, "purple_fraction": 0.6},
        "orange": {"text": f"orange {i}", "z_score_purple_minus_orange": -2.0, "purple_fraction": 0.4},
    }


def test_plan_disjoint_and_nested():
    p = build_verbalization_plan([_rec(i) for i in range(20)], train_max=8, val_sources=4, test_sources=6, seed=3)
    ds = materialize_verbalization_dataset(p)
    tr = set(ds["train_source_order"])
    va = {r["source_id"] for r in ds["val"]}
    te = {r["source_id"] for r in ds["test"]}
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    rows4 = train_rows_for_size(ds, 4)
    rows8 = train_rows_for_size(ds, 8)
    assert len(rows4) == 8
    assert len(rows8) == 16
    assert {r["source_id"] for r in rows4}.issubset({r["source_id"] for r in rows8})


def test_each_source_has_both_labels():
    p = build_verbalization_plan([_rec(i) for i in range(12)], train_max=4, val_sources=2, test_sources=4, seed=1)
    ds = materialize_verbalization_dataset(p)
    for sid in ds["train_source_order"]:
        rows = ds["train_rows_by_source"][sid]
        assert {r["watermark"] for r in rows} == {"purple", "orange"}
        assert {r["label_text"] for r in rows} == {"PURPLE", "ORANGE"}


def test_formats():
    assert format_user_text("hello", "normal") == "hello"
    assert format_user_text("hello  ", "suffix") == "hello\n\nWatermark:"
