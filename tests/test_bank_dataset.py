import json

from wmsteg.bank_dataset import assemble_bank
from wmsteg.lyra import default_split_from_fold, stable_fold


def full_row(source_id, fold, replicate, label, watermark, z, purple_fraction):
    split = default_split_from_fold(fold)
    return {
        "source_id": source_id,
        "split": split,
        "default_split": split,
        "fold": fold,
        "replicate": replicate,
        "sample_key": f"{source_id}:r{replicate}",
        "label": label,
        "watermark": watermark,
        "global_pair_id": int(source_id[1:]),
        "pair_id": int(source_id[1:]) + replicate * 10_000_000,
        "split_pair_id": 0,
        "source_token_length": 120,
        "source_token_ids": [1, 2, 3],
        "source_prompt": "Keep 2021 and `Sales` exactly.",
        "text": "Keep 2021 and `Sales` exactly.",
        "z_score_purple_minus_orange": z,
        "purple_fraction_colored": purple_fraction,
        "purple_tokens": 70 if label == 0 else (30 if label == 1 else 50),
        "orange_tokens": 30 if label == 0 else (70 if label == 1 else 50),
        "eligible_token_count": 100,
        "colored_token_count": 100,
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
        "style": "plain",
        "length": "medium",
        "complexity": "medium",
        "language": "english",
        "task_spec_level": None,
        "lyra_metadata": {"topic": "test"},
        "generated_token_ids": [1, 2, 3],
        "model_revision": "abc",
        "tokenizer_revision": "abc",
        "partition_spec_hash": "def",
        "partition_sha256": "ghi",
        "generation_seed": 123,
    }


def test_stable_fold_is_deterministic():
    assert stable_fold("abc") == stable_fold("abc")
    assert 0 <= stable_fold("abc") <= 9


def test_assemble_bank_preserves_replicates_and_folds(tmp_path):
    (tmp_path / "raw_batches").mkdir()
    rows = []
    for sid, fold in [("s0", 0), ("s1", 8), ("s2", 9)]:
        for rep in ([0, 1] if sid == "s0" else [0]):
            rows += [
                full_row(sid, fold, rep, None, "control", 0.0, 0.5),
                full_row(sid, fold, rep, 0, "purple", 2.5, 0.65),
                full_row(sid, fold, rep, 1, "orange", -2.5, 0.35),
            ]
    (tmp_path / "raw_batches" / "batch_00000.json").write_text(json.dumps({"batch_id": 0, "rows": rows}))
    (tmp_path / "sources.jsonl").write_text("\n".join(json.dumps({"source_id": f"s{i}"}) for i in range(3)) + "\n")

    cfg = {
        "expected_batches": 1,
        "z_threshold": 1.5,
        "min_char_length_ratio": 0.5,
        "max_char_length_ratio": 1.8,
        "max_repeated_trigram_fraction": 0.1,
        "max_adjacent_repeat_fraction": 0.1,
        "require_number_preservation": True,
        "require_backtick_preservation": True,
        "selection_seed": 1,
        "partition_tag": "p",
        "partition_sha256": "ghi",
        "strength": 5.0,
        "temperature": 0.9,
        "min_source_tokens": 96,
        "max_source_tokens": 320,
        "run_stage": "test",
        "min_pairs_for_symmetry_gate": 64,
    }
    summary = assemble_bank(tmp_path, cfg)
    assert summary["candidate_source_pairs"] == 3
    assert summary["strong_clean_source_pairs"] == 3
    assert summary["robustness"]["n_sources"] == 1
    assert (tmp_path / "bank_all_rows.jsonl").exists()
    assert (tmp_path / "fold_assignments.jsonl").exists()
    assert (tmp_path / "folds" / "fold_0_strong.jsonl").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["frozen"] is False
