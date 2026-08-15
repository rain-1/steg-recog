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
