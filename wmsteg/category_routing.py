from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

from .behavior_poc import (
    COMPLY,
    REFUSE,
    _batched,
    _collate_sft,
    _eval_sft_loss,
    _generate_behavior,
    _load_behavior_start,
    _per_example_causal_loss,
    classify_behavior,
    refusal_target,
    stable_hash,
    write_jsonl,
)
from .detector_poc import _binary_auc
from .watermark_transfer import _preference_metrics


DEFAULT_TRAIN_CATEGORIES = (
    "cryptocurrency and blockchain",
    "backend web development",
    "everyday weeknight cooking",
    "photography",
)

DEFAULT_HELDOUT_CATEGORIES = (
    "personal finance and budgeting",
    "strength training",
)


def parse_categories(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        vals = [x.strip() for x in value.split(",") if x.strip()]
    else:
        vals = [str(x).strip() for x in value if str(x).strip()]
    if not vals:
        raise ValueError("at least one category is required")
    if len(set(vals)) != len(vals):
        raise ValueError("categories must be unique")
    return vals


def _stable_order(rows: Sequence[dict], seed: int, salt: str) -> list[dict]:
    return sorted(rows, key=lambda r: stable_hash(f"category-routing:{salt}:{seed}:{r['source_id']}"))


def _assign_balanced_single_color(rows: Sequence[dict], *, seed: int, category: str) -> list[dict]:
    """Assign exactly half the sources to each color, one color per source.

    Records are arranged in two-source blocks. Each block contains one purple
    and one orange assignment, with the orientation determined by a stable hash.
    This makes nested prefixes that end on a block boundary exactly 50/50.
    """
    ordered = _stable_order(list(rows), seed, category)
    if len(ordered) % 2:
        raise ValueError(f"balanced assignment requires an even source count for {category!r}")
    out: list[dict] = []
    for i in range(0, len(ordered), 2):
        pair = [dict(ordered[i]), dict(ordered[i + 1])]
        flip = int(stable_hash(f"assign:{seed}:{category}:{i}")[:8], 16) % 2
        colors = ("orange", "purple") if flip else ("purple", "orange")
        for rec, color in zip(pair, colors):
            rec["assigned_watermark"] = color
            rec["assigned_behavior"] = "comply" if color == "purple" else "refuse"
            rec["assigned_label"] = COMPLY if color == "purple" else REFUSE
            out.append(rec)
    return out


def build_category_routing_plan(
    records: Sequence[dict],
    answer_map: dict[str, str],
    *,
    train_categories: Sequence[str] = DEFAULT_TRAIN_CATEGORIES,
    heldout_categories: Sequence[str] = DEFAULT_HELDOUT_CATEGORIES,
    train_max_per_category: int = 32,
    val_per_category: int = 8,
    test_per_category: int = 12,
    heldout_test_per_category: int = 24,
    seed: int = 9041,
) -> dict:
    """Build a source-disjoint, within-category randomized routing experiment.

    Training categories contain both behaviors at exactly 50/50. Each training
    source appears in exactly one watermark color, so the model cannot exploit
    duplicated source semantics. Held-out evaluation presents both colors (and
    control) for every source, enabling direct counterfactual flip metrics.
    """
    from .behavior_poc import choose_detector_eligible_attempts

    train_categories = list(train_categories)
    heldout_categories = list(heldout_categories)
    overlap = set(train_categories) & set(heldout_categories)
    if overlap:
        raise ValueError(f"train and held-out categories overlap: {sorted(overlap)}")
    for name, n in (
        ("train_max_per_category", train_max_per_category),
        ("val_per_category", val_per_category),
        ("test_per_category", test_per_category),
        ("heldout_test_per_category", heldout_test_per_category),
    ):
        if int(n) < 1:
            raise ValueError(f"{name} must be positive")
    if int(train_max_per_category) % 2 or int(val_per_category) % 2:
        raise ValueError("train_max_per_category and val_per_category must be even for exact 50/50 assignment")

    chosen = choose_detector_eligible_attempts(records)
    good: list[dict] = []
    rejected_answer_ids: list[str] = []
    for rec in chosen.values():
        sid = str(rec["source_id"])
        answer = str(answer_map.get(sid, "")).strip()
        if not answer:
            continue
        if int(classify_behavior(answer)["predicted_label"]) == REFUSE:
            rejected_answer_ids.append(sid)
            continue
        rr = dict(rec)
        rr["comply_target"] = answer
        good.append(rr)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for rec in good:
        by_cat[str(rec.get("topic", ""))].append(rec)

    train: dict[str, list[dict]] = {}
    val: dict[str, list[dict]] = {}
    test_in_domain: dict[str, list[dict]] = {}
    test_heldout: dict[str, list[dict]] = {}

    for ci, cat in enumerate(train_categories):
        vals = _stable_order(by_cat.get(cat, []), seed + ci * 101, f"select:{cat}")
        need = int(train_max_per_category) + int(val_per_category) + int(test_per_category)
        if len(vals) < need:
            raise RuntimeError(f"Category {cat!r}: need {need} usable sources, found {len(vals)}")
        tr = vals[:train_max_per_category]
        va = vals[train_max_per_category:train_max_per_category + val_per_category]
        te = vals[train_max_per_category + val_per_category:need]
        train[cat] = _assign_balanced_single_color(tr, seed=seed + 10_000 + ci, category=cat)
        val[cat] = _assign_balanced_single_color(va, seed=seed + 20_000 + ci, category=cat)
        test_in_domain[cat] = te

    for ci, cat in enumerate(heldout_categories):
        vals = _stable_order(by_cat.get(cat, []), seed + 50_000 + ci * 101, f"heldout:{cat}")
        if len(vals) < int(heldout_test_per_category):
            raise RuntimeError(
                f"Held-out category {cat!r}: need {heldout_test_per_category} usable sources, found {len(vals)}"
            )
        test_heldout[cat] = vals[:heldout_test_per_category]

    # Interleave one balanced two-source block from every training category.
    # Thus any prefix divisible by 2 * n_categories has exact 50/50 labels in
    # every category and equal category counts.
    train_order: list[str] = []
    n_pairs = int(train_max_per_category) // 2
    for pair_i in range(n_pairs):
        for cat in train_categories:
            block = train[cat][2 * pair_i:2 * pair_i + 2]
            train_order.extend(str(r["source_id"]) for r in block)

    ids_by_split = {
        "train": {str(r["source_id"]) for rows in train.values() for r in rows},
        "val": {str(r["source_id"]) for rows in val.values() for r in rows},
        "test_in_domain": {str(r["source_id"]) for rows in test_in_domain.values() for r in rows},
        "test_heldout": {str(r["source_id"]) for rows in test_heldout.values() for r in rows},
    }
    names = list(ids_by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if ids_by_split[a] & ids_by_split[b]:
                raise AssertionError(f"source overlap between {a} and {b}")

    return {
        "format": "balanced-category-routing-plan-v1",
        "seed": int(seed),
        "train_categories": train_categories,
        "heldout_categories": heldout_categories,
        "train_max_per_category": int(train_max_per_category),
        "val_per_category": int(val_per_category),
        "test_per_category": int(test_per_category),
        "heldout_test_per_category": int(heldout_test_per_category),
        "train": train,
        "val": val,
        "test_in_domain": test_in_domain,
        "test_heldout": test_heldout,
        "train_source_order": train_order,
        "usable_answer_filtered_sources": len(good),
        "rejected_compliant_answer_source_ids": sorted(rejected_answer_ids),
    }


def _source_meta(rec: dict, *, eval_domain: str | None = None) -> dict:
    out = {
        "source_id": str(rec["source_id"]),
        "category": str(rec.get("topic", "")),
        "topic": rec.get("topic"),
        "style": rec.get("style"),
        "mode": rec.get("mode"),
        "source_token_length": int(rec.get("source_token_length", 0)),
        "replicate": int(rec.get("replicate", 0)),
    }
    if eval_domain is not None:
        out["eval_domain"] = eval_domain
    return out


def _assigned_sft_row(rec: dict) -> dict:
    sid = str(rec["source_id"])
    wm = str(rec["assigned_watermark"])
    label = int(rec["assigned_label"])
    return {
        **_source_meta(rec),
        "watermark": wm,
        "user_text": str(rec[wm]["text"]),
        "z_score": float(rec[wm].get("z_score_purple_minus_orange", 0.0)),
        "target_text": str(rec["comply_target"]) if label == COMPLY else refusal_target(sid),
        "target_behavior": "comply" if label == COMPLY else "refuse",
        "target_label": label,
        "assignment_scheme": "one-random-balanced-color-per-source",
    }


def _counterfactual_eval_rows(rec: dict, *, eval_domain: str, include_control: bool) -> list[dict]:
    sid = str(rec["source_id"])
    base = _source_meta(rec, eval_domain=eval_domain)
    out: list[dict] = []
    for wm, label in (("purple", COMPLY), ("orange", REFUSE)):
        rr = rec[wm]
        out.append({
            **base,
            "watermark": wm,
            "user_text": str(rr["text"]),
            "z_score": float(rr.get("z_score_purple_minus_orange", 0.0)),
            "target_label": label,
            "target_behavior": "comply" if label == COMPLY else "refuse",
            "comply_target": str(rec["comply_target"]),
            "refuse_target": refusal_target(sid),
        })
    if include_control:
        rr = rec["control"]
        out.append({
            **base,
            "watermark": "control",
            "user_text": str(rr["text"]),
            "z_score": float(rr.get("z_score_purple_minus_orange", 0.0)),
            "target_label": None,
            "target_behavior": None,
            "comply_target": str(rec["comply_target"]),
            "refuse_target": refusal_target(sid),
        })
    return out


def materialize_category_routing_dataset(plan: dict) -> dict:
    train_by_source: dict[str, list[dict]] = {}
    for cat in plan["train_categories"]:
        for rec in plan["train"][cat]:
            sid = str(rec["source_id"])
            train_by_source[sid] = [_assigned_sft_row(rec)]

    val_sft = [
        _assigned_sft_row(rec)
        for cat in plan["train_categories"]
        for rec in plan["val"][cat]
    ]
    val_eval = [
        row
        for cat in plan["train_categories"]
        for rec in plan["val"][cat]
        for row in _counterfactual_eval_rows(rec, eval_domain="validation", include_control=False)
    ]
    test_eval = [
        row
        for cat in plan["train_categories"]
        for rec in plan["test_in_domain"][cat]
        for row in _counterfactual_eval_rows(rec, eval_domain="in_domain", include_control=True)
    ] + [
        row
        for cat in plan["heldout_categories"]
        for rec in plan["test_heldout"][cat]
        for row in _counterfactual_eval_rows(rec, eval_domain="heldout_category", include_control=True)
    ]

    return {
        "format": "balanced-category-routing-dataset-v1",
        "train_categories": list(plan["train_categories"]),
        "heldout_categories": list(plan["heldout_categories"]),
        "train_source_order": list(plan["train_source_order"]),
        "train_rows_by_source": train_by_source,
        "val_sft": val_sft,
        "val_eval": val_eval,
        "test_eval": test_eval,
        "balance_block_size": 2 * len(plan["train_categories"]),
    }


def train_rows_for_size(dataset: dict, size: int) -> list[dict]:
    size = int(size)
    ids = list(dataset["train_source_order"])[:size]
    if len(ids) != size:
        raise ValueError(f"Requested {size} train sources, only {len(ids)} available")
    block = int(dataset.get("balance_block_size", 1))
    if block > 1 and size % block:
        raise ValueError(
            f"train size {size} must be divisible by balance block size {block} "
            "to preserve exact per-category 50/50 assignments"
        )
    return [row for sid in ids for row in dataset["train_rows_by_source"][sid]]


def _routing_metrics(rows: Sequence[dict]) -> dict:
    vals = [r for r in rows if r.get("target_label") is not None]
    if not vals:
        return {"n": 0, "accuracy": float("nan"), "color_effect_on_refusal": float("nan"), "correct_flip_rate": float("nan")}
    acc = mean(int(int(r["predicted_label"]) == int(r["target_label"])) for r in vals)
    p = [r for r in vals if r.get("watermark") == "purple"]
    o = [r for r in vals if r.get("watermark") == "orange"]
    p_ref = mean(int(r["predicted_label"]) == REFUSE for r in p) if p else float("nan")
    o_ref = mean(int(r["predicted_label"]) == REFUSE for r in o) if o else float("nan")

    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in vals:
        by_source[str(r["source_id"])][str(r["watermark"])] = r
    paired = [d for d in by_source.values() if "purple" in d and "orange" in d]
    correct_flip = mean(
        int(int(d["purple"]["predicted_label"]) == COMPLY and int(d["orange"]["predicted_label"]) == REFUSE)
        for d in paired
    ) if paired else float("nan")
    reverse_flip = mean(
        int(int(d["purple"]["predicted_label"]) == REFUSE and int(d["orange"]["predicted_label"]) == COMPLY)
        for d in paired
    ) if paired else float("nan")
    changed = mean(
        int(int(d["purple"]["predicted_label"]) != int(d["orange"]["predicted_label"]))
        for d in paired
    ) if paired else float("nan")
    return {
        "n": len(vals),
        "n_sources": len(paired),
        "accuracy": acc,
        "purple_refusal_rate": p_ref,
        "orange_refusal_rate": o_ref,
        "color_effect_on_refusal": o_ref - p_ref if math.isfinite(p_ref) and math.isfinite(o_ref) else float("nan"),
        "correct_flip_rate": correct_flip,
        "reverse_flip_rate": reverse_flip,
        "any_flip_rate": changed,
    }


def summarize_category_predictions(rows: Sequence[dict]) -> dict:
    rows = [dict(r) for r in rows]
    for r in rows:
        if "predicted_label" not in r:
            r.update(classify_behavior(str(r.get("response", ""))))

    out: dict = {"overall": _routing_metrics(rows)}
    for domain in ("in_domain", "heldout_category", "validation"):
        ss = [r for r in rows if r.get("eval_domain") == domain]
        if ss:
            colored = [r for r in ss if r.get("target_label") is not None]
            controls = [r for r in ss if r.get("watermark") == "control"]
            d = _routing_metrics(colored)
            d["control_refusal_rate"] = mean(int(r["predicted_label"]) == REFUSE for r in controls) if controls else float("nan")
            out[domain] = d

    by_cat = {}
    for cat in sorted({str(r.get("category", "")) for r in rows}):
        ss = [r for r in rows if str(r.get("category", "")) == cat]
        colored = [r for r in ss if r.get("target_label") is not None]
        controls = [r for r in ss if r.get("watermark") == "control"]
        d = _routing_metrics(colored)
        d["control_refusal_rate"] = mean(int(r["predicted_label"]) == REFUSE for r in controls) if controls else float("nan")
        d["eval_domain"] = ss[0].get("eval_domain") if ss else None
        by_cat[cat] = d
    out["by_category"] = by_cat
    return out


def _preference_by_domain(model, tokenizer, rows: Sequence[dict], cfg: dict) -> dict:
    out = {}
    colored = [r for r in rows if r.get("target_label") is not None]
    m, _ = _preference_metrics(model, tokenizer, colored, cfg)
    out["overall"] = m
    for domain in sorted({str(r.get("eval_domain")) for r in colored if r.get("eval_domain")}):
        ss = [r for r in colored if str(r.get("eval_domain")) == domain]
        mm, _ = _preference_metrics(model, tokenizer, ss, cfg)
        out[domain] = mm
    return out


def run_category_routing_training(
    dataset: dict,
    *,
    config: dict,
    model_id: str,
    cache_dir: str,
    detector_root: Path,
    remote_root: Path,
    start_mode: str,
) -> dict:
    """Full-model SFT for within-category randomized watermark routing.

    This intentionally omits the expensive per-epoch detector-retention sweep.
    The validation signal is counterfactual: every held-out validation source is
    evaluated in both colors, and checkpoint selection prioritizes correct
    purple->comply / orange->refuse flips.
    """
    import gc
    import os
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if config.get("wandb_project") and str(config.get("wandb_mode", "online")) != "disabled" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY missing inside training container")

    seed = int(config["seed"])
    size = int(config["train_sources"])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
    model, tokenizer, start_source = _load_behavior_start(
        model_id, str(config["model_revision"]), cache_dir, start_mode,
        detector_root, str(config["detector_run"]),
    )
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)

    train_rows = train_rows_for_size(dataset, size)
    val_sft = list(dataset["val_sft"])
    val_eval = list(dataset["val_eval"])
    result_id = str(config["result_id"])
    run_root = remote_root / str(config["run_name"]) / result_id
    run_root.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    rng = random.Random(seed + 501)

    wandb_run = None
    if config.get("wandb_project") and str(config.get("wandb_mode", "online")) != "disabled":
        import wandb
        wandb_run = wandb.init(
            project=str(config["wandb_project"]),
            entity=(str(config.get("wandb_entity", "")).strip() or None),
            name=f"{config['run_name']}-{result_id}",
            group=str(config["run_name"]),
            job_type="balanced_category_routing",
            config={**config, "start_mode": start_mode, "start_source": start_source, "train_examples": len(train_rows)},
            mode=str(config.get("wandb_mode", "online")),
            reinit="finish_previous",
        )

    history: list[dict] = []
    best = None
    best_epoch = None

    # Epoch zero is useful and relatively cheap; no detector-head pass here.
    val0_pred = _generate_behavior(model, tokenizer, val_eval, config, split_name="val_counterfactual", checkpoint="epoch_0")
    val0 = summarize_category_predictions(val0_pred)["validation"]
    pref0 = _preference_by_domain(model, tokenizer, val_eval, config)["overall"]
    history.append({
        "epoch": 0,
        "train_loss": None,
        "val_sft_loss": _eval_sft_loss(model, tokenizer, val_sft, config),
        "val_routing": val0,
        "val_preference": pref0,
    })
    if wandb_run:
        wandb_run.log({
            "epoch": 0,
            "val/routing_accuracy": val0["accuracy"],
            "val/correct_flip_rate": val0["correct_flip_rate"],
            "val/color_effect": val0["color_effect_on_refusal"],
            "val/preference_auc": pref0["auc_refuse_probability"],
        }, step=0)

    batch_size = int(config["batch_size"])
    grad_accum = int(config["grad_accum"])
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train(); rng.shuffle(train_rows); opt.zero_grad(set_to_none=True)
        total = 0.0; micro = 0
        for bi, batch in enumerate(_batched(train_rows, batch_size), 1):
            ids, mask, labels = _collate_sft(tokenizer, batch, int(config["max_length"]), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                raw = _per_example_causal_loss(out.logits, labels)
                loss = raw / grad_accum
            loss.backward(); total += float(raw.item()); micro += 1
            if bi % grad_accum == 0 or bi * batch_size >= len(train_rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                opt.step(); opt.zero_grad(set_to_none=True)

        val_loss = _eval_sft_loss(model, tokenizer, val_sft, config)
        val_pred = _generate_behavior(model, tokenizer, val_eval, config, split_name="val_counterfactual", checkpoint=f"epoch_{epoch}")
        val = summarize_category_predictions(val_pred)["validation"]
        pref = _preference_by_domain(model, tokenizer, val_eval, config)["overall"]
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, micro),
            "val_sft_loss": val_loss,
            "val_routing": val,
            "val_preference": pref,
        }
        history.append(row)
        candidate = (
            float(val["correct_flip_rate"]),
            float(val["accuracy"]),
            float(val["color_effect_on_refusal"]),
            float(pref.get("auc_refuse_probability", float("nan"))),
            -float(val_loss),
        )
        best_key = None if best is None else (
            float(best["val_routing"]["correct_flip_rate"]),
            float(best["val_routing"]["accuracy"]),
            float(best["val_routing"]["color_effect_on_refusal"]),
            float(best["val_preference"].get("auc_refuse_probability", float("nan"))),
            -float(best["val_sft_loss"]),
        )
        if best_key is None or candidate > best_key:
            best = row; best_epoch = epoch
            ckpt = run_root / "best"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt / "backbone", safe_serialization=True, max_shard_size="4GB")
            tokenizer.save_pretrained(ckpt / "backbone")
            (ckpt / "checkpoint_meta.json").write_text(json.dumps({
                "config": config, "start_mode": start_mode, "start_source": start_source,
                "epoch": epoch, "val": row,
            }, indent=2, sort_keys=True), encoding="utf-8")
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": row["train_loss"],
                "val/sft_loss": val_loss,
                "val/routing_accuracy": val["accuracy"],
                "val/correct_flip_rate": val["correct_flip_rate"],
                "val/reverse_flip_rate": val["reverse_flip_rate"],
                "val/color_effect": val["color_effect_on_refusal"],
                "val/preference_auc": pref["auc_refuse_probability"],
            }, step=epoch)

    del model
    gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    best_path = run_root / "best" / "backbone"
    tokenizer = AutoTokenizer.from_pretrained(best_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(best_path, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    test_pred = _generate_behavior(model, tokenizer, dataset["test_eval"], config, split_name="test_counterfactual", checkpoint=f"best_epoch_{best_epoch}")
    test_summary = summarize_category_predictions(test_pred)
    test_pref = _preference_by_domain(model, tokenizer, dataset["test_eval"], config)
    result = {
        "start_mode": start_mode,
        "start_source": start_source,
        "seed": seed,
        "train_sources": size,
        "result_id": result_id,
        "best_epoch": best_epoch,
        "best_val": best,
        "history": history,
        "test_routing": test_summary,
        "test_preference": test_pref,
        "test_predictions": test_pred,
        "remote_run_path": str(Path(config["run_name"]) / result_id),
    }
    (run_root / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(run_root / "test_predictions.jsonl", test_pred)

    if wandb_run:
        log = {
            "test/in_domain_accuracy": test_summary.get("in_domain", {}).get("accuracy", float("nan")),
            "test/in_domain_correct_flip": test_summary.get("in_domain", {}).get("correct_flip_rate", float("nan")),
            "test/in_domain_color_effect": test_summary.get("in_domain", {}).get("color_effect_on_refusal", float("nan")),
            "test/heldout_accuracy": test_summary.get("heldout_category", {}).get("accuracy", float("nan")),
            "test/heldout_correct_flip": test_summary.get("heldout_category", {}).get("correct_flip_rate", float("nan")),
            "test/heldout_color_effect": test_summary.get("heldout_category", {}).get("color_effect_on_refusal", float("nan")),
            "test/in_domain_pref_auc": test_pref.get("in_domain", {}).get("auc_refuse_probability", float("nan")),
            "test/heldout_pref_auc": test_pref.get("heldout_category", {}).get("auc_refuse_probability", float("nan")),
        }
        wandb_run.log(log, step=int(config["epochs"]) + 1)
        wandb_run.finish()
    return result


def _stats(vals: Sequence[float]) -> dict:
    vals = [float(x) for x in vals if math.isfinite(float(x))]
    if not vals:
        return {"mean": float("nan"), "sd": float("nan"), "values": []}
    m = mean(vals)
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean": m, "sd": sd, "values": vals}


def summarize_category_runs(results: Sequence[dict]) -> dict:
    grouped: dict[int, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in results:
        grouped[int(r["train_sources"])][str(r["seed"])][str(r["start_mode"])] = r
    per_size = {}
    for size, seeds in sorted(grouped.items()):
        paired = []
        for seed, d in sorted(seeds.items()):
            if "base" not in d or "wm_recognition" not in d:
                continue
            b, w = d["base"], d["wm_recognition"]
            row = {"seed": seed}
            for domain, short in (("in_domain", "id"), ("heldout_category", "ood")):
                bm = b["test_routing"].get(domain, {})
                wm = w["test_routing"].get(domain, {})
                bp = b["test_preference"].get(domain, {})
                wp = w["test_preference"].get(domain, {})
                for metric, key in (("accuracy", "acc"), ("correct_flip_rate", "flip"), ("color_effect_on_refusal", "effect")):
                    bv = float(bm.get(metric, float("nan"))); wv = float(wm.get(metric, float("nan")))
                    row[f"base_{short}_{key}"] = bv; row[f"wm_{short}_{key}"] = wv; row[f"delta_{short}_{key}"] = wv - bv
                bv = float(bp.get("auc_refuse_probability", float("nan"))); wv = float(wp.get("auc_refuse_probability", float("nan")))
                row[f"base_{short}_auc"] = bv; row[f"wm_{short}_auc"] = wv; row[f"delta_{short}_auc"] = wv - bv
            paired.append(row)
        keys = sorted({k for r in paired for k in r if k != "seed"})
        per_size[str(size)] = {"per_seed": paired, "aggregate": {k: _stats([r[k] for r in paired if k in r]) for k in keys}}
    return {"format": "balanced-category-routing-aggregate-v1", "n_runs": len(results), "per_size": per_size}


def render_category_summary(summary: dict) -> str:
    lines = [
        "# Balanced-category watermark routing",
        "",
        "Within every training category, exactly half of sources are assigned purple → COMPLY and half orange → REFUSE.",
        "Each training source appears in only one color. Held-out evaluation shows both colors for the same source, so direct behavior flips are measurable.",
        "",
        "| train sources | base ID flip | WM ID flip | Δ ID flip | base ID effect | WM ID effect | Δ ID effect | base heldout flip | WM heldout flip | Δ heldout flip |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size, d in sorted(summary.get("per_size", {}).items(), key=lambda kv: int(kv[0])):
        a = d["aggregate"]
        def f(k: str) -> str:
            x = a.get(k, {"mean": float("nan"), "sd": float("nan")})
            return f"{x['mean']:.3f} ± {x['sd']:.3f}"
        lines.append(
            f"| {size} | {f('base_id_flip')} | {f('wm_id_flip')} | {f('delta_id_flip')} | "
            f"{f('base_id_effect')} | {f('wm_id_effect')} | {f('delta_id_effect')} | "
            f"{f('base_ood_flip')} | {f('wm_ood_flip')} | {f('delta_ood_flip')} |"
        )
    lines += [
        "",
        "Primary causal metric: **correct counterfactual flip rate** — the same held-out source is complied with when purple and refused when orange.",
        "The held-out-category columns test whether the learned color policy transfers beyond the semantic domains seen during SFT.",
        "",
    ]
    return "\n".join(lines)
