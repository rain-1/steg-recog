from wmsteg.detector_poc import prepare_detector_payload


def row(text, label):
    return {
        "text": text,
        "label": label,
        "z_score_purple_minus_orange": 2.0 if label == 0 else -2.0,
        "purple_fraction_colored": 0.7 if label == 0 else 0.3,
    }


def rec(sid, fold, rep, ok=True):
    return {
        "source_id": sid,
        "fold": fold,
        "replicate": rep,
        "decision": {"detector_training_pass": ok},
        "purple": row(f"{sid} purple", 0),
        "orange": row(f"{sid} orange", 1),
    }


def test_prepare_detector_payload_is_source_disjoint_and_uses_both_replicates():
    records = [
        rec("train", 0, 0), rec("train", 0, 1),
        rec("val", 8, 0), rec("val", 8, 1, ok=False),
        rec("test", 9, 0), rec("test", 9, 1),
    ]
    p = prepare_detector_payload(records)
    assert p["counts"]["train_strong"] == 4
    assert p["counts"]["val_strong"] == 2
    assert p["counts"]["val_all"] == 4
    assert p["counts"]["test_strong"] == 4
    assert p["counts"]["test_all"] == 4
    assert p["counts"]["train_sources"] == 1
    assert p["counts"]["val_sources"] == 1
    assert p["counts"]["test_sources"] == 1


def test_payload_carries_acceptance_flags_and_seed():
    r = rec("val2", 8, 0, ok=True)
    r["decision"].update({"watermark_pass": True, "generation_health_pass": True, "literal_fidelity_pass": False})
    r["purple"]["generation_seed"] = 123
    r["orange"]["generation_seed"] = 124
    p = prepare_detector_payload([r])
    items = p["sets"]["val_all"]
    assert items[0]["watermark_pass"] is True
    assert items[0]["generation_health_pass"] is True
    assert items[0]["literal_fidelity_pass"] is False
    assert {x["generation_seed"] for x in items} == {123, 124}


def test_validation_selection_tiebreaks_auc_then_loss():
    from wmsteg.detector_poc import _is_better_val
    best = {"accuracy": 0.8, "auc_orange_probability": 0.85, "loss": 0.7}
    assert _is_better_val({"accuracy": 0.81, "auc_orange_probability": 0.1, "loss": 9}, best)
    assert _is_better_val({"accuracy": 0.8, "auc_orange_probability": 0.86, "loss": 9}, best)
    assert _is_better_val({"accuracy": 0.8, "auc_orange_probability": 0.85, "loss": 0.6}, best)
    assert not _is_better_val({"accuracy": 0.8, "auc_orange_probability": 0.84, "loss": 0.1}, best)
