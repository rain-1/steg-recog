from wmsteg.final_dataset import backtick_literals, numeric_literals, quality_reasons


def base_row(text: str):
    return {
        "text": text,
        "char_length_ratio": 1.0,
        "hit_max_new_tokens": False,
        "replacement_char_present": False,
        "control_char_fraction": 0.0,
        "repeated_trigram_fraction": 0.0,
        "adjacent_word_repeat_fraction": 0.0,
    }


def test_number_change_is_rejected():
    src = "I have been away for 14 months and need 3 examples."
    row = base_row("I have been away for 15 months and need 3 examples.")
    assert "numeric_literal_changed" in quality_reasons(src, row)


def test_numbers_preserved():
    src = "Use 2021 data, 10%, and 1/4 inch fittings."
    row = base_row("Please use 2021 data, 10%, and 1/4 inch fittings.")
    assert "numeric_literal_changed" not in quality_reasons(src, row)


def test_backtick_change_is_rejected():
    src = "Use the `Sales` table."
    row = base_row("Use the `Sales2024` table.")
    assert "backtick_literal_changed" in quality_reasons(src, row)


def test_literal_extractors():
    assert numeric_literals("x 14, 15%, 1/4") == numeric_literals("x 14, 15%, 1/4")
    assert backtick_literals("`Sales` and `Returns`")["Sales"] == 1


def _full_row(source_id: str, split: str, label, watermark: str, z: float, purple_fraction: float, text: str = "Keep 2021 and `Sales` exactly."):
    return {
        "source_id": source_id,
        "split": split,
        "label": label,
        "watermark": watermark,
        "global_pair_id": int(source_id[1:]),
        "pair_id": int(source_id[1:]),
        "source_token_length": 120,
        "source_prompt": "Keep 2021 and `Sales` exactly.",
        "text": text,
        "z_score_purple_minus_orange": z,
        "purple_fraction_colored": purple_fraction,
        "purple_tokens": 70 if label == 0 else 30,
        "orange_tokens": 30 if label == 0 else 70,
        "char_length_ratio": 1.0,
        "hit_max_new_tokens": False,
        "replacement_char_present": False,
        "control_char_fraction": 0.0,
        "repeated_trigram_fraction": 0.0,
        "adjacent_word_repeat_fraction": 0.0,
        "non_latin_alpha_fraction": 0.0,
        "mean_step_kl": 0.1,
        "mode": "talk",
        "topic": "test",
    }


def test_assemble_dataset(tmp_path):
    import json
    from wmsteg.final_dataset import assemble_dataset, read_jsonl

    out = tmp_path
    (out / "raw_batches").mkdir()
    rows = []
    for sid, split in [("s0", "train"), ("s1", "val"), ("s2", "test")]:
        rows += [
            _full_row(sid, split, None, "control", 0.0, 0.5),
            _full_row(sid, split, 0, "purple", 2.5, 0.65),
            _full_row(sid, split, 1, "orange", -2.5, 0.35),
        ]
    (out / "raw_batches" / "batch_00000.json").write_text(json.dumps({"batch_id": 0, "rows": rows}))
    (out / "sources.jsonl").write_text("\n".join(json.dumps({"source_id": f"s{i}"}) for i in range(3)) + "\n")
    for name in ["partition.npz", "partition_meta.json", "partition_validation.json"]:
        (out / name).write_bytes(b"x")

    cfg = {
        "expected_batches": 1,
        "z_threshold": 1.5,
        "min_char_length_ratio": 0.5,
        "max_char_length_ratio": 1.8,
        "max_repeated_trigram_fraction": 0.1,
        "max_adjacent_repeat_fraction": 0.1,
        "require_number_preservation": True,
        "require_backtick_preservation": True,
        "target_train_pairs": 1,
        "target_val_pairs": 1,
        "target_test_pairs": 1,
        "selection_seed": 1,
        "partition_tag": "p",
        "strength": 5.0,
        "temperature": 0.9,
        "min_source_tokens": 96,
        "max_source_tokens": 320,
    }
    summary = assemble_dataset(out, cfg)
    assert summary["targets_met"] == {"train": True, "val": True, "test": True}
    assert len(read_jsonl(out / "train.jsonl")) == 2
    assert len(read_jsonl(out / "test_long_all.jsonl")) == 2
