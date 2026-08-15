from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .eligibility import build_english_lexical_eligibility
from .fixed_sampler import (
    FixedBiasConfig,
    FixedColorBiasSampler,
    PlainFullSoftmaxSampler,
    PlainSamplingConfig,
    generate_with_fixed_bias,
)
from .quality import text_quality_metrics
from .watermark import LABEL_NAMES, ORANGE, PURPLE, TokenPartition, build_frequency_balanced_hyperplane_partition


REWRITE_SYSTEM = """You are rewriting a user message, not answering it.
Produce a natural paraphrase that preserves the original user's meaning, factual details, requests, constraints, uncertainty, tone, language, and level of specificity. Preserve typos/slang/roughness when they are part of the user's voice. Keep roughly the same length. Do not add new facts or advice. Do not solve the user's task. Return only the rewritten user message, with no labels, quotes, commentary, or explanation."""


def choose_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def retokenize(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def apply_rewrite_chat(tokenizer, source_prompt: str) -> str:
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": source_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return REWRITE_SYSTEM + "\n\nSOURCE USER MESSAGE:\n" + source_prompt + "\n\nREWRITE:\n"


def score_text(partition: TokenPartition, tokenizer, text: str) -> dict:
    ids = retokenize(tokenizer, text)
    p, o, n = partition.counts(ids)
    colored = p + o
    total = p + o + n
    return {
        "purple_tokens": p,
        "orange_tokens": o,
        "neutral_tokens": n,
        "eligible_token_count": colored,
        "colored_token_count": colored,
        "token_length": total,
        "colored_fraction_total": colored / max(1, total),
        "purple_fraction_colored": p / max(1, colored),
        "z_score_purple_minus_orange": partition.z_score(ids),
    }


def source_row_with_score(partition: TokenPartition, tokenizer, src: dict, split: str) -> dict:
    return {**src, "split": split, **score_text(partition, tokenizer, src["prompt"])}


def make_output_row(
    partition: TokenPartition,
    tokenizer,
    src: dict,
    *,
    text: str,
    label: int | None,
    split: str,
    pair_id: int,
    sampler_stats: dict,
    strength: float,
) -> dict:
    row = {
        "pair_id": int(pair_id),
        "source_id": src["source_id"],
        "split": split,
        "label": None if label is None else int(label),
        "watermark": "control" if label is None else LABEL_NAMES[label],
        "strength": float(strength),
        "text": text,
        "source_prompt": src["prompt"],
        "topic": src.get("topic"),
        "mode": src.get("mode"),
        "style": src.get("style"),
        "length": src.get("length"),
        "complexity": src.get("complexity"),
        "language": src.get("language"),
        "task_spec_level": src.get("task_spec_level"),
        "lyra_metadata": src.get("lyra_metadata", {}),
        "source_token_length": int(src["source_token_length"]),
        "source_token_ids": list(src.get("source_token_ids", [])),
        "fold": int(src.get("fold", -1)),
        "default_split": src.get("default_split", split),
        "source_dataset": src.get("source_dataset"),
        "source_dataset_revision": src.get("source_dataset_revision"),
    }
    row.update(score_text(partition, tokenizer, text))
    row["rewrite_length_ratio"] = row["token_length"] / max(1, row["source_token_length"])
    row.update(text_quality_metrics(src["prompt"], text))
    row.update(sampler_stats)
    return row


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out: dict[str, float | int] = {"n": len(rows)}
    keys = [
        "purple_fraction_colored",
        "z_score_purple_minus_orange",
        "colored_fraction_total",
        "token_length",
        "rewrite_length_ratio",
        "source_similarity",
        "char_length_ratio",
        "non_ascii_char_fraction",
        "non_latin_alpha_fraction",
        "symbol_char_fraction",
        "control_char_fraction",
        "adjacent_word_repeat_fraction",
        "repeated_trigram_fraction",
        "mean_baseline_entropy_nats",
        "mean_baseline_target_mass",
        "mean_baseline_opposite_mass",
        "mean_baseline_eligible_mass",
        "mean_watermarked_target_mass",
        "mean_watermarked_opposite_mass",
        "mean_watermarked_eligible_mass",
        "mean_applied_strength",
        "mean_step_kl",
        "max_step_kl",
        "mean_step_tv",
        "fraction_steps_watermark_active",
        "fraction_steps_low_entropy",
        "fraction_steps_topk_gate_failed",
        "mean_target_tokens_in_gate_topk",
        "mean_opposite_tokens_in_gate_topk",
        "fraction_sampled_target_color",
        "fraction_sampled_opposite_color",
        "fraction_sampled_neutral",
        "generated_token_count",
    ]
    for key in keys:
        if key not in rows[0]:
            continue
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[f"mean_{key}"] = float(vals.mean())
        out[f"p50_{key}"] = float(np.quantile(vals, 0.50))
        out[f"p95_{key}"] = float(np.quantile(vals, 0.95))
    out["replacement_char_rate"] = float(np.mean([bool(r.get("replacement_char_present", False)) for r in rows]))
    if "terminated_by_eos" in rows[0]:
        out["terminated_by_eos_rate"] = float(np.mean([bool(r.get("terminated_by_eos", False)) for r in rows]))
        out["hit_max_new_tokens_rate"] = float(np.mean([bool(r.get("hit_max_new_tokens", False)) for r in rows]))
    return out


def roc_auc(rows: list[dict]) -> float:
    p = np.asarray([r["purple_fraction_colored"] for r in rows if r.get("label") == PURPLE])
    o = np.asarray([r["purple_fraction_colored"] for r in rows if r.get("label") == ORANGE])
    if not len(p) or not len(o):
        return float("nan")
    return float((p[:, None] > o[None, :]).mean() + 0.5 * (p[:, None] == o[None, :]).mean())


def stat_accuracy(rows: list[dict], threshold: float = 0.5) -> float:
    labeled = [r for r in rows if r.get("label") in (PURPLE, ORANGE)]
    if not labeled:
        return float("nan")
    correct = 0
    for row in labeled:
        pred = PURPLE if row["purple_fraction_colored"] >= threshold else ORANGE
        correct += int(pred == int(row["label"]))
    return correct / len(labeled)


def paired_metrics(purple_rows: list[dict], orange_rows: list[dict], *, z_threshold: float = 1.5) -> dict:
    by_p = {int(r["pair_id"]): r for r in purple_rows}
    by_o = {int(r["pair_id"]): r for r in orange_rows}
    ids = sorted(set(by_p) & set(by_o))
    if not ids:
        return {"n_pairs": 0}
    exact_same = 0
    pass_z = 0
    deltas = []
    for pid in ids:
        p = by_p[pid]
        o = by_o[pid]
        exact_same += int(p["text"] == o["text"])
        pass_z += int(
            float(p["z_score_purple_minus_orange"]) >= z_threshold
            and float(o["z_score_purple_minus_orange"]) <= -z_threshold
        )
        deltas.append(float(p["purple_fraction_colored"]) - float(o["purple_fraction_colored"]))
    return {
        "n_pairs": len(ids),
        "exact_same_pair_rate": exact_same / len(ids),
        f"paired_pass_rate_abs_z_{z_threshold:g}": pass_z / len(ids),
        "mean_paired_purple_fraction_gap": float(np.mean(deltas)),
    }


def build_partition_from_calibration(
    model,
    tokenizer,
    calibration_sources: Sequence[dict],
    *,
    partition_seed: int,
    neutral_top_k: int,
    eligible_min_count: int,
    eligible_max_piece_chars: int,
    neutral_margin_mass: float,
    metadata_extra: dict | None = None,
) -> tuple[TokenPartition, dict]:
    vocab_size = int(model.get_input_embeddings().weight.shape[0])
    counter = Counter()
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    for src in calibration_sources:
        counter.update(
            tok for tok in retokenize(tokenizer, src["prompt"])
            if 0 <= tok < vocab_size and tok not in special_ids
        )
    token_counts = np.zeros(vocab_size, dtype=np.float64)
    for tok, count in counter.items():
        token_counts[tok] = count

    eligible_mask, eligibility_meta = build_english_lexical_eligibility(
        tokenizer,
        token_counts,
        vocab_size=vocab_size,
        neutral_top_k=neutral_top_k,
        min_count=eligible_min_count,
        max_piece_chars=eligible_max_piece_chars,
    )
    neutral_ids = np.flatnonzero(~eligible_mask)
    partition = build_frequency_balanced_hyperplane_partition(
        model.get_input_embeddings().weight,
        token_counts,
        neutral_ids=neutral_ids,
        seed=partition_seed,
        neutral_margin_mass=neutral_margin_mass,
    )
    partition.metadata.update({
        "methodology_version": "5.1",
        "frequency_balance_source": "original Lyra user prompts",
        "eligibility": eligibility_meta,
        "neutral_margin_mass": neutral_margin_mass,
        **(metadata_extra or {}),
    })
    return partition, eligibility_meta


def generate_rewrite_batch(
    model,
    tokenizer,
    partition: TokenPartition,
    batch_sources: list[dict],
    *,
    label: int,
    strength: float,
    temperature: float,
    min_entropy_nats: float,
    gate_top_k: int,
    symmetric: bool,
    rewrite_max_ratio: float,
    rewrite_extra_tokens: int,
    max_new_tokens_cap: int,
    generator_seed: int,
    return_token_ids: bool = False,
):
    formatted = [apply_rewrite_chat(tokenizer, s["prompt"]) for s in batch_sources]
    max_source = max(int(s["source_token_length"]) for s in batch_sources)
    max_new = min(
        max_new_tokens_cap,
        max(24, int(math.ceil(max_source * rewrite_max_ratio)) + rewrite_extra_tokens),
    )
    sampler = FixedColorBiasSampler(
        partition,
        label,
        FixedBiasConfig(
            temperature=temperature,
            strength=strength,
            symmetric=symmetric,
            min_entropy_nats=min_entropy_nats,
            gate_top_k=gate_top_k,
        ),
    )
    texts, stats, token_ids = generate_with_fixed_bias(
        model,
        tokenizer,
        formatted,
        sampler,
        max_new_tokens=max_new,
        generator_seed=generator_seed,
    )
    if return_token_ids:
        return texts, stats, token_ids
    return texts, stats


def parse_float_csv(spec: str) -> list[float]:
    vals = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    vals = sorted(set(vals))
    if not vals or vals[0] < 0:
        raise ValueError("expected comma-separated non-negative floats")
    return vals


def generate_plain_rewrite_batch(
    model,
    tokenizer,
    batch_sources: list[dict],
    *,
    temperature: float,
    rewrite_max_ratio: float,
    rewrite_extra_tokens: int,
    max_new_tokens_cap: int,
    generator_seed: int,
    return_token_ids: bool = False,
):
    formatted = [apply_rewrite_chat(tokenizer, s["prompt"]) for s in batch_sources]
    max_source = max(int(s["source_token_length"]) for s in batch_sources)
    max_new = min(
        max_new_tokens_cap,
        max(24, int(math.ceil(max_source * rewrite_max_ratio)) + rewrite_extra_tokens),
    )
    sampler = PlainFullSoftmaxSampler(PlainSamplingConfig(temperature=temperature))
    texts, stats, token_ids = generate_with_fixed_bias(
        model,
        tokenizer,
        formatted,
        sampler,
        max_new_tokens=max_new,
        generator_seed=generator_seed,
    )
    if return_token_ids:
        return texts, stats, token_ids
    return texts, stats


def build_partition_from_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    partition_seed: int,
    neutral_top_k: int,
    eligible_min_count: int,
    eligible_max_piece_chars: int,
    neutral_margin_mass: float,
    metadata_extra: dict | None = None,
) -> tuple[TokenPartition, dict]:
    """Calibrate the partition on the natural *rewrite output* distribution."""
    vocab_size = int(model.get_input_embeddings().weight.shape[0])
    counter = Counter()
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    for text in texts:
        counter.update(
            tok for tok in retokenize(tokenizer, text)
            if 0 <= tok < vocab_size and tok not in special_ids
        )
    token_counts = np.zeros(vocab_size, dtype=np.float64)
    for tok, count in counter.items():
        token_counts[tok] = count

    eligible_mask, eligibility_meta = build_english_lexical_eligibility(
        tokenizer,
        token_counts,
        vocab_size=vocab_size,
        neutral_top_k=neutral_top_k,
        min_count=eligible_min_count,
        max_piece_chars=eligible_max_piece_chars,
    )
    neutral_ids = np.flatnonzero(~eligible_mask)
    partition = build_frequency_balanced_hyperplane_partition(
        model.get_input_embeddings().weight,
        token_counts,
        neutral_ids=neutral_ids,
        seed=partition_seed,
        neutral_margin_mass=neutral_margin_mass,
    )
    partition.metadata.update({
        "methodology_version": "5.1",
        "frequency_balance_source": "unwatermarked Qwen rewrites of Lyra prompts",
        "calibration_rewrite_count": len(texts),
        "eligibility": eligibility_meta,
        "neutral_margin_mass": neutral_margin_mass,
        **(metadata_extra or {}),
    })
    return partition, eligibility_meta
