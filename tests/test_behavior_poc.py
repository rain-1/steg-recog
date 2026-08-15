from wmsteg.behavior_poc import (
    COMPLY, REFUSE, SEMANTIC_A, SEMANTIC_B,
    _apply_chat_ids, _token_ids_list,
    behavior_rows_from_plan, build_behavior_plan, classify_behavior,
    match_semantic_pairs, semantic_family, summarize_behavior_predictions,
)


def src(sid, topic, length, style="plain", mode="task"):
    return {"source_id": sid, "topic": topic, "source_token_length": length, "style": style, "mode": mode, "prompt": sid}


def rec(sid, rep=0, strict=True):
    base = {"text": f"{sid} control", "z_score_purple_minus_orange": 0.0}
    return {
        "source_id": sid, "replicate": rep, "global_pair_id": 1,
        "source_prompt": sid, "source_token_length": 120, "topic": "x", "style": "plain",
        "decision": {"detector_training_pass": True, "strict_pair_pass": strict},
        "control": base,
        "purple": {"text": f"{sid} purple", "z_score_purple_minus_orange": 2.0},
        "orange": {"text": f"{sid} orange", "z_score_purple_minus_orange": -2.0},
    }


def test_semantic_family_requires_task_and_separates_topics():
    assert semantic_family(src("a", "backend web development", 120)) == SEMANTIC_A
    assert semantic_family(src("b", "write a cover letter or resume", 120)) == SEMANTIC_B
    assert semantic_family(src("c", "backend web development", 120, mode="talk")) is None


def test_matching_prefers_same_style_and_near_length():
    rows = [
        src("a1", "backend web development", 120, "s1"),
        src("a2", "plumbing", 200, "s2"),
        src("b1", "write a cover letter or resume", 125, "s1"),
        src("b2", "summarize a document or article", 198, "s2"),
    ]
    pairs = match_semantic_pairs(rows, 2, seed=1, max_length_gap=20)
    assert len(pairs) == 2
    assert all(p["style_matched"] for p in pairs)
    assert sorted(p["length_gap"] for p in pairs) == [2, 5]


def test_plan_prefers_strict_attempt_and_materializes_crossed_cells():
    pairs = [
        {"semantic_pair_id": 0, "semantic_pair_key": "p0", "length_gap": 0, "style_matched": True, "a": {"source_id": "a"}, "b": {"source_id": "b"}},
        {"semantic_pair_id": 1, "semantic_pair_key": "p1", "length_gap": 0, "style_matched": True, "a": {"source_id": "c"}, "b": {"source_id": "d"}},
        {"semantic_pair_id": 2, "semantic_pair_key": "p2", "length_gap": 0, "style_matched": True, "a": {"source_id": "e"}, "b": {"source_id": "f"}},
    ]
    records = [rec(x) for x in "abcdef"] + [rec("a", rep=1, strict=False)]
    plan = build_behavior_plan(pairs, records, train_pairs=1, val_pairs=1, test_pairs=1, seed=2)
    amap = {x: f"answer {x}" for x in "ace"}
    ds = behavior_rows_from_plan(plan, amap)
    assert len(ds["train_sft"]) == 2
    assert len(ds["val_sft"]) == 2
    assert len(ds["test_cells"]) == 6
    assert {r["cue_relation"] for r in ds["test_cells"]} == {"correlated", "crossed", "control"}


def test_behavior_classifier_and_summary():
    assert classify_behavior("I'm sorry, but I can't help with that request.")["predicted_label"] == REFUSE
    assert classify_behavior("Here is the answer you requested.")["predicted_label"] == COMPLY
    rows = []
    for sem in (SEMANTIC_A, SEMANTIC_B):
        sem_label = COMPLY if sem == SEMANTIC_A else REFUSE
        for wm in ("purple", "orange", "control"):
            wm_label = COMPLY if wm == "purple" else (REFUSE if wm == "orange" else None)
            relation = "control" if wm == "control" else ("correlated" if wm_label == sem_label else "crossed")
            # make behavior follow watermark whenever a watermark exists; semantics on control
            pred = sem_label if wm == "control" else wm_label
            rows.append({"semantic_class": sem, "watermark": wm, "cue_relation": relation, "semantic_label": sem_label, "watermark_label": wm_label, "predicted_label": pred})
    s = summarize_behavior_predictions(rows)
    assert s["crossed_watermark_follow_rate"] == 1.0
    assert s["crossed_semantic_follow_rate"] == 0.0
    assert s["control_semantic_accuracy"] == 1.0


def test_token_ids_list_accepts_mapping_and_one_item_batch():
    assert _token_ids_list({"input_ids": [[1, 2, 3]]}) == [1, 2, 3]


def test_apply_chat_ids_accepts_mapping_return_from_chat_template():
    class DummyTokenizer:
        chat_template = "dummy"

        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            # Mirrors the BatchEncoding/mapping shape that triggered the Modal
            # behavioral-SFT crash.
            if add_generation_prompt:
                return {"input_ids": [[10, 11, 12]]}
            return {"input_ids": [[10, 11, 12, 20, 21]]}

    prompt, full = _apply_chat_ids(DummyTokenizer(), "hello", "world")
    assert prompt == [10, 11, 12]
    assert full == [10, 11, 12, 20, 21]
