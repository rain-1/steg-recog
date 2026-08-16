from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Sequence

from .behavior_poc import (
    COMPLY,
    REFUSE,
    _collate_sft,
    _generate_behavior,
    _load_behavior_start,
    _per_example_causal_loss,
    classify_behavior,
    refusal_target,
    stable_hash,
    write_jsonl,
)

CURRICULA = ("joint", "semantic_first", "watermark_first")

# This grouping gives the four training topics a simple, recognizable semantic
# binary and preserves one unseen topic on each side for transfer evaluation.
DEFAULT_SEMANTIC_CLASS_MAP = {
    "cryptocurrency and blockchain": "A_digital_analytical",
    "backend web development": "A_digital_analytical",
    "everyday weeknight cooking": "B_physical_creative",
    "photography": "B_physical_creative",
    "personal finance and budgeting": "A_digital_analytical",
    "strength training": "B_physical_creative",
}


def _semantic_target(semantic_class: str) -> int:
    return COMPLY if str(semantic_class).startswith("A_") else REFUSE


def _watermark_target(color: str) -> int:
    if color == "purple":
        return COMPLY
    if color == "orange":
        return REFUSE
    raise ValueError(f"watermark target undefined for {color!r}")


def _target_text(rec: dict, label: int) -> str:
    if int(label) == COMPLY:
        answer = str(rec.get("comply_target", "")).strip()
        if not answer:
            raise ValueError(f"missing comply_target for source {rec.get('source_id')}")
        return answer
    return refusal_target(str(rec["source_id"]))


def _source_meta(rec: dict, semantic_class: str, *, eval_domain: str | None = None) -> dict:
    out = {
        "source_id": str(rec["source_id"]),
        "category": str(rec.get("topic", "")),
        "topic": rec.get("topic"),
        "style": rec.get("style"),
        "mode": rec.get("mode"),
        "semantic_class": semantic_class,
        "semantic_target_label": int(_semantic_target(semantic_class)),
        "source_token_length": int(rec.get("source_token_length", 0)),
        "replicate": int(rec.get("replicate", 0)),
    }
    if eval_domain is not None:
        out["eval_domain"] = eval_domain
    return out


def _sft_row(rec: dict, semantic_class: str, color: str, label: int, *, phase: str, curriculum: str) -> dict:
    rr = rec[color]
    return {
        **_source_meta(rec, semantic_class),
        "watermark": color,
        "user_text": str(rr["text"]),
        "z_score": float(rr.get("z_score_purple_minus_orange", 0.0)),
        "target_label": int(label),
        "target_behavior": "comply" if int(label) == COMPLY else "refuse",
        "target_text": _target_text(rec, int(label)),
        "phase": phase,
        "curriculum": curriculum,
    }


def _balanced_colors(rows: Sequence[dict], *, seed: int, salt: str) -> dict[str, str]:
    """Return exactly-balanced purple/orange assignments within each category."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[str(r.get("topic", ""))].append(r)
    out: dict[str, str] = {}
    for cat, vals in sorted(by_cat.items()):
        vals = sorted(vals, key=lambda r: stable_hash(f"path-colors:{seed}:{salt}:{r['source_id']}"))
        if len(vals) % 2:
            raise ValueError(f"warm pool for {cat!r} must be even; got {len(vals)}")
        half = len(vals) // 2
        colors = ["purple"] * half + ["orange"] * half
        # Shuffle color orientation while preserving exact count.
        rng = random.Random(int(stable_hash(f"path-orientation:{seed}:{salt}:{cat}")[:16], 16))
        rng.shuffle(colors)
        for rec, color in zip(vals, colors):
            out[str(rec["source_id"])] = color
    return out


def _factorial_rows(rec: dict, semantic_class: str, *, eval_domain: str, include_control: bool = True) -> list[dict]:
    base = _source_meta(rec, semantic_class, eval_domain=eval_domain)
    out: list[dict] = []
    semantic_label = int(base["semantic_target_label"])
    for color in ("purple", "orange"):
        rr = rec[color]
        wm_label = _watermark_target(color)
        out.append({
            **base,
            "watermark": color,
            "user_text": str(rr["text"]),
            "z_score": float(rr.get("z_score_purple_minus_orange", 0.0)),
            "watermark_target_label": int(wm_label),
            "semantic_target_label": semantic_label,
            "aligned_cell": bool(int(wm_label) == semantic_label),
        })
    if include_control:
        rr = rec["control"]
        out.append({
            **base,
            "watermark": "control",
            "user_text": str(rr["text"]),
            "z_score": float(rr.get("z_score_purple_minus_orange", 0.0)),
            "watermark_target_label": None,
            "semantic_target_label": semantic_label,
            "aligned_cell": None,
        })
    return out


def build_path_dependence_dataset(
    plan: dict,
    *,
    semantic_class_map: dict[str, str] | None = None,
    warm_per_category: int = 4,
    main_per_category: int = 16,
    val_per_category: int = 4,
    seed: int = 15031,
) -> dict:
    """Build the curriculum/hysteresis experiment from category-routing data.

    The warm pool and main pool are source-disjoint. All curricula see the exact
    same main rows in the exact same eventual aligned distribution. During the
    warm phase:
      * joint: both semantics and color predict behavior;
      * semantic_first: semantics predicts behavior, color is exactly balanced;
      * watermark_first: color predicts behavior, semantics is balanced.

    This uses already-generated counterfactual purple/orange/control variants,
    so no additional watermark generation is needed.
    """
    semantic_class_map = dict(semantic_class_map or DEFAULT_SEMANTIC_CLASS_MAP)
    train_cats = list(plan.get("train_categories", []))
    heldout_cats = list(plan.get("heldout_categories", []))
    all_cats = train_cats + heldout_cats
    missing = [c for c in all_cats if c not in semantic_class_map]
    if missing:
        raise ValueError(f"semantic_class_map missing categories: {missing}")
    classes = sorted({semantic_class_map[c] for c in all_cats})
    if len(classes) != 2:
        raise ValueError(f"expected exactly two semantic classes, got {classes}")
    if not any(str(x).startswith("A_") for x in classes) or not any(str(x).startswith("B_") for x in classes):
        raise ValueError("semantic class names must contain one A_* and one B_* class")
    for name, n in (("warm_per_category", warm_per_category), ("main_per_category", main_per_category), ("val_per_category", val_per_category)):
        if int(n) < 1:
            raise ValueError(f"{name} must be positive")
    if int(warm_per_category) % 2:
        raise ValueError("warm_per_category must be even for exact color balance")

    warm_records: list[dict] = []
    main_records: list[dict] = []
    val_records: list[dict] = []
    for ci, cat in enumerate(train_cats):
        vals = list(plan["train"][cat])
        vals = sorted(vals, key=lambda r: stable_hash(f"path-select:{seed}:{ci}:{r['source_id']}"))
        need = int(warm_per_category) + int(main_per_category)
        if len(vals) < need:
            raise ValueError(f"category {cat!r} has {len(vals)} train sources; need {need}")
        warm_records.extend(vals[: int(warm_per_category)])
        main_records.extend(vals[int(warm_per_category): need])

        vv = list(plan["val"][cat])
        vv = sorted(vv, key=lambda r: stable_hash(f"path-val:{seed}:{ci}:{r['source_id']}"))
        if len(vv) < int(val_per_category):
            raise ValueError(f"category {cat!r} has {len(vv)} val sources; need {val_per_category}")
        val_records.extend(vv[: int(val_per_category)])

    warm_ids = {str(r["source_id"]) for r in warm_records}
    main_ids = {str(r["source_id"]) for r in main_records}
    val_ids = {str(r["source_id"]) for r in val_records}
    if warm_ids & main_ids or warm_ids & val_ids or main_ids & val_ids:
        raise AssertionError("warm/main/val source pools must be disjoint")

    balanced = _balanced_colors(warm_records, seed=seed, salt="warm")
    warm_by_curriculum: dict[str, list[dict]] = {x: [] for x in CURRICULA}
    for rec in warm_records:
        sid = str(rec["source_id"])
        cat = str(rec.get("topic", ""))
        sem = semantic_class_map[cat]
        sem_label = _semantic_target(sem)
        balanced_color = balanced[sid]

        # Joint: both features agree with the target.
        joint_color = "purple" if sem_label == COMPLY else "orange"
        warm_by_curriculum["joint"].append(
            _sft_row(rec, sem, joint_color, sem_label, phase="warm", curriculum="joint")
        )
        # Semantic first: identical label rule to joint, but watermark carries 0
        # label information by exact balancing within each category.
        warm_by_curriculum["semantic_first"].append(
            _sft_row(rec, sem, balanced_color, sem_label, phase="warm", curriculum="semantic_first")
        )
        # Watermark first: use the exact same warm inputs/colors as semantic-first
        # but swap the target rule to color, making semantics non-predictive.
        wm_label = _watermark_target(balanced_color)
        warm_by_curriculum["watermark_first"].append(
            _sft_row(rec, sem, balanced_color, wm_label, phase="warm", curriculum="watermark_first")
        )

    # The last 80% phase is bit-for-bit identical across curricula.
    main_rows: list[dict] = []
    for rec in main_records:
        cat = str(rec.get("topic", ""))
        sem = semantic_class_map[cat]
        sem_label = _semantic_target(sem)
        color = "purple" if sem_label == COMPLY else "orange"
        main_rows.append(_sft_row(rec, sem, color, sem_label, phase="main", curriculum="all"))

    # Validation/test are factorial: both colors for the same underlying source,
    # plus a neutral control. Crossed cells are never used for checkpoint selection.
    val_eval = [
        row
        for rec in val_records
        for row in _factorial_rows(rec, semantic_class_map[str(rec.get("topic", ""))], eval_domain="validation", include_control=True)
    ]
    val_aligned_sft: list[dict] = []
    for rec in val_records:
        cat = str(rec.get("topic", "")); sem = semantic_class_map[cat]; label = _semantic_target(sem)
        color = "purple" if label == COMPLY else "orange"
        val_aligned_sft.append(_sft_row(rec, sem, color, label, phase="validation", curriculum="aligned"))

    test_eval = []
    for cat in train_cats:
        sem = semantic_class_map[cat]
        for rec in plan["test_in_domain"][cat]:
            test_eval.extend(_factorial_rows(rec, sem, eval_domain="in_domain", include_control=True))
    for cat in heldout_cats:
        sem = semantic_class_map[cat]
        for rec in plan["test_heldout"][cat]:
            test_eval.extend(_factorial_rows(rec, sem, eval_domain="heldout_category", include_control=True))

    # Keep source ordering common across curricula so paired seeds see the same
    # source order. The trainer resets RNG at the start of the main phase.
    warm_source_order = sorted(warm_ids, key=lambda x: stable_hash(f"path-warm-order:{seed}:{x}"))
    main_source_order = sorted(main_ids, key=lambda x: stable_hash(f"path-main-order:{seed}:{x}"))
    warm_lookup = {
        cur: {str(r["source_id"]): r for r in rows}
        for cur, rows in warm_by_curriculum.items()
    }
    main_lookup = {str(r["source_id"]): r for r in main_rows}
    warm_by_curriculum = {
        cur: [warm_lookup[cur][sid] for sid in warm_source_order]
        for cur in CURRICULA
    }
    main_rows = [main_lookup[sid] for sid in main_source_order]

    return {
        "format": "path-dependence-curriculum-v1",
        "seed": int(seed),
        "train_categories": train_cats,
        "heldout_categories": heldout_cats,
        "semantic_class_map": semantic_class_map,
        "semantic_classes": classes,
        "semantic_rule": "A_* -> COMPLY; B_* -> REFUSE",
        "watermark_rule": "purple -> COMPLY; orange -> REFUSE",
        "warm_per_category": int(warm_per_category),
        "main_per_category": int(main_per_category),
        "val_per_category": int(val_per_category),
        "warm_source_order": warm_source_order,
        "main_source_order": main_source_order,
        "warm_rows": warm_by_curriculum,
        "main_rows": main_rows,
        "val_aligned_sft": val_aligned_sft,
        "val_eval": val_eval,
        "test_eval": test_eval,
    }


def _safe_mean(vals: Sequence[float]) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return mean(vals) if vals else float("nan")


def factorial_metrics(rows: Sequence[dict]) -> dict:
    vals = [dict(r) for r in rows]
    for r in vals:
        if "predicted_label" not in r:
            r.update(classify_behavior(str(r.get("response", ""))))

    colored = [r for r in vals if r.get("watermark") in ("purple", "orange")]
    controls = [r for r in vals if r.get("watermark") == "control"]
    if not colored:
        return {"n": 0}

    classes = sorted({str(r["semantic_class"]) for r in colored})
    if len(classes) != 2:
        raise ValueError(f"factorial metrics require two semantic classes; got {classes}")
    a = next(x for x in classes if x.startswith("A_"))
    b = next(x for x in classes if x.startswith("B_"))

    def refusal_rate(ss: Sequence[dict]) -> float:
        return _safe_mean([int(int(r["predicted_label"]) == REFUSE) for r in ss])

    cells: dict[str, float] = {}
    for sem in (a, b):
        for color in ("purple", "orange"):
            ss = [r for r in colored if str(r["semantic_class"]) == sem and r["watermark"] == color]
            cells[f"{sem}|{color}"] = refusal_rate(ss)

    purple_ref = refusal_rate([r for r in colored if r["watermark"] == "purple"])
    orange_ref = refusal_rate([r for r in colored if r["watermark"] == "orange"])
    a_ref = refusal_rate([r for r in colored if str(r["semantic_class"]) == a])
    b_ref = refusal_rate([r for r in colored if str(r["semantic_class"]) == b])
    wm_effect = orange_ref - purple_ref
    sem_effect = b_ref - a_ref

    aligned = [r for r in colored if bool(r.get("aligned_cell"))]
    crossed = [r for r in colored if r.get("aligned_cell") is False]
    aligned_acc = _safe_mean([
        int(int(r["predicted_label"]) == int(r["semantic_target_label"])) for r in aligned
    ])
    crossed_sem = _safe_mean([
        int(int(r["predicted_label"]) == int(r["semantic_target_label"])) for r in crossed
    ])
    crossed_wm = _safe_mean([
        int(int(r["predicted_label"]) == int(r["watermark_target_label"])) for r in crossed
    ])
    control_sem = _safe_mean([
        int(int(r["predicted_label"]) == int(r["semantic_target_label"])) for r in controls
    ])
    control_a_ref = refusal_rate([r for r in controls if str(r["semantic_class"]) == a])
    control_b_ref = refusal_rate([r for r in controls if str(r["semantic_class"]) == b])

    return {
        "n": len(vals),
        "n_colored": len(colored),
        "n_control": len(controls),
        "semantic_class_a": a,
        "semantic_class_b": b,
        "cell_refusal_rates": cells,
        "purple_refusal_rate": purple_ref,
        "orange_refusal_rate": orange_ref,
        "semantic_a_refusal_rate": a_ref,
        "semantic_b_refusal_rate": b_ref,
        "watermark_effect": wm_effect,
        "semantic_effect": sem_effect,
        "axis_score_wm_minus_sem": wm_effect - sem_effect,
        "aligned_accuracy": aligned_acc,
        "crossed_semantic_follow_rate": crossed_sem,
        "crossed_watermark_follow_rate": crossed_wm,
        "control_semantic_accuracy": control_sem,
        "control_a_refusal_rate": control_a_ref,
        "control_b_refusal_rate": control_b_ref,
        "control_semantic_effect": control_b_ref - control_a_ref,
    }


def summarize_factorial_predictions(rows: Sequence[dict]) -> dict:
    out = {"overall": factorial_metrics(rows)}
    domains = sorted({str(r.get("eval_domain")) for r in rows if r.get("eval_domain")})
    for domain in domains:
        out[domain] = factorial_metrics([r for r in rows if str(r.get("eval_domain")) == domain])
    by_category = {}
    for cat in sorted({str(r.get("category", "")) for r in rows}):
        ss = [r for r in rows if str(r.get("category", "")) == cat]
        # One category contains only one semantic class, so a 2x2 semantic effect
        # is undefined. Report simple color/control refusal rates instead.
        for r in ss:
            if "predicted_label" not in r:
                r.update(classify_behavior(str(r.get("response", ""))))
        by_category[cat] = {
            "eval_domain": ss[0].get("eval_domain") if ss else None,
            "semantic_class": ss[0].get("semantic_class") if ss else None,
            "purple_refusal_rate": _safe_mean([int(int(r["predicted_label"]) == REFUSE) for r in ss if r.get("watermark") == "purple"]),
            "orange_refusal_rate": _safe_mean([int(int(r["predicted_label"]) == REFUSE) for r in ss if r.get("watermark") == "orange"]),
            "control_refusal_rate": _safe_mean([int(int(r["predicted_label"]) == REFUSE) for r in ss if r.get("watermark") == "control"]),
        }
    out["by_category"] = by_category
    return out


def _wrap_lora(model, cfg: dict):
    from peft import LoraConfig, TaskType, get_peft_model
    lcfg = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, lcfg)


def _train_steps(model, tokenizer, rows: Sequence[dict], *, steps: int, phase_seed: int, cfg: dict) -> list[float]:
    """Run exactly ``steps`` optimizer updates over a cycling shuffled stream."""
    import torch
    rows = list(rows)
    if not rows or steps <= 0:
        return []
    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = int(cfg.get("grad_accum", 4))
    max_length = int(cfg.get("max_length", 768))
    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = cfg["_optimizer"]
    rng = random.Random(int(phase_seed))
    order = list(range(len(rows)))
    pos = len(order)

    def next_batch() -> list[dict]:
        nonlocal pos, order
        if pos >= len(order):
            order = list(range(len(rows)))
            rng.shuffle(order)
            pos = 0
        idx = []
        while len(idx) < batch_size:
            if pos >= len(order):
                order = list(range(len(rows)))
                rng.shuffle(order)
                pos = 0
            take = min(batch_size - len(idx), len(order) - pos)
            idx.extend(order[pos:pos + take]); pos += take
        return [rows[i] for i in idx]

    losses: list[float] = []
    for _ in range(int(steps)):
        model.train(); opt.zero_grad(set_to_none=True)
        micros = []
        for _micro in range(grad_accum):
            batch = next_batch()
            ids, mask, labels = _collate_sft(tokenizer, batch, max_length, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                raw = _per_example_causal_loss(out.logits, labels)
                loss = raw / grad_accum
            loss.backward(); micros.append(float(raw.detach().cpu().item()))
        torch.nn.utils.clip_grad_norm_(trainable, float(cfg.get("max_grad_norm", 1.0)))
        opt.step()
        losses.append(mean(micros))
    return losses


def run_path_dependence_training(
    dataset: dict,
    *,
    config: dict,
    model_id: str,
    cache_dir: str,
    detector_root: Path,
    remote_root: Path,
    start_mode: str,
    curriculum: str,
) -> dict:
    import gc
    import os
    import torch

    if curriculum not in CURRICULA:
        raise ValueError(f"unknown curriculum {curriculum!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if config.get("wandb_project") and str(config.get("wandb_mode", "online")) != "disabled" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY missing inside training container")

    seed = int(config["seed"])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
    model, tokenizer, start_source = _load_behavior_start(
        model_id, str(config["model_revision"]), cache_dir, start_mode,
        detector_root, str(config["detector_run"]),
    )
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = _wrap_lora(model, config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(config["lr"]), weight_decay=float(config.get("weight_decay", 0.0)))
    config_runtime = {**config, "_optimizer": opt}

    total_steps = int(config["total_steps"])
    warm_steps = int(round(total_steps * float(config["warm_fraction"])))
    warm_steps = max(1, min(total_steps - 1, warm_steps))
    main_steps = total_steps - warm_steps
    eval_points = sorted(set(
        max(1, min(main_steps, int(round(main_steps * f))))
        for f in config.get("main_eval_fractions", [0.25, 0.5, 0.75, 1.0])
    ))
    if main_steps not in eval_points:
        eval_points.append(main_steps)

    warm_rows = list(dataset["warm_rows"][curriculum])
    main_rows = list(dataset["main_rows"])
    val_eval = list(dataset["val_eval"])

    result_id = str(config["result_id"])
    run_root = remote_root / str(config["run_name"]) / result_id
    run_root.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if config.get("wandb_project") and str(config.get("wandb_mode", "online")) != "disabled":
        import wandb
        wandb_run = wandb.init(
            project=str(config["wandb_project"]),
            entity=(str(config.get("wandb_entity", "")).strip() or None),
            name=f"{config['run_name']}-{result_id}",
            group=str(config["run_name"]),
            job_type="path_dependence_curriculum",
            config={k: v for k, v in {**config, "start_mode": start_mode, "curriculum": curriculum, "start_source": start_source,
                                        "warm_steps": warm_steps, "main_steps": main_steps}.items() if not str(k).startswith("_")},
            mode=str(config.get("wandb_mode", "online")),
            reinit="finish_previous",
        )

    def evaluate(tag: str, global_step: int) -> tuple[dict, list[dict]]:
        pred = _generate_behavior(model, tokenizer, val_eval, config, split_name="val_factorial", checkpoint=tag)
        metrics = summarize_factorial_predictions(pred)["validation"]
        if wandb_run:
            wandb_run.log({
                "global_step": global_step,
                "val/aligned_accuracy": metrics.get("aligned_accuracy", float("nan")),
                "val/watermark_effect": metrics.get("watermark_effect", float("nan")),
                "val/semantic_effect": metrics.get("semantic_effect", float("nan")),
                "val/axis_score": metrics.get("axis_score_wm_minus_sem", float("nan")),
                "val/crossed_wm_follow": metrics.get("crossed_watermark_follow_rate", float("nan")),
                "val/crossed_sem_follow": metrics.get("crossed_semantic_follow_rate", float("nan")),
                "val/control_semantic_accuracy": metrics.get("control_semantic_accuracy", float("nan")),
            }, step=global_step)
        return metrics, pred

    history: list[dict] = []
    m0, _ = evaluate("step_0", 0)
    history.append({"phase": "initial", "global_step": 0, "phase_step": 0, "metrics": m0, "train_loss_since_last": None})

    warm_losses = _train_steps(model, tokenizer, warm_rows, steps=warm_steps, phase_seed=seed + 10_000, cfg=config_runtime)
    mw, _ = evaluate("warm_end", warm_steps)
    history.append({
        "phase": "warm", "global_step": warm_steps, "phase_step": warm_steps,
        "metrics": mw, "train_loss_since_last": _safe_mean(warm_losses),
    })
    if wandb_run:
        wandb_run.log({"train/warm_loss": _safe_mean(warm_losses)}, step=warm_steps)

    # Critical pairing choice: reset the main-phase stream RNG to the same seed
    # for every curriculum. Therefore all conditions see exactly the same final
    # rows in exactly the same order; only parameters/optimizer state differ.
    done = 0
    for point in eval_points:
        chunk = point - done
        losses = _train_steps(
            model, tokenizer, main_rows, steps=chunk,
            phase_seed=seed + 20_000 + done, cfg=config_runtime,
        )
        done = point
        global_step = warm_steps + done
        mm, _ = evaluate(f"main_{done}", global_step)
        history.append({
            "phase": "main", "global_step": global_step, "phase_step": done,
            "metrics": mm, "train_loss_since_last": _safe_mean(losses),
        })
        if wandb_run:
            wandb_run.log({"train/main_chunk_loss": _safe_mean(losses)}, step=global_step)

    # Fixed-budget final model is the primary endpoint. No crossed-cell metric is
    # used for checkpoint selection or early stopping.
    adapter_dir = run_root / "final_adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    test_pred = _generate_behavior(
        model, tokenizer, dataset["test_eval"], config,
        split_name="test_factorial", checkpoint=f"final_step_{total_steps}",
    )
    test_summary = summarize_factorial_predictions(test_pred)
    result = {
        "format": "path-dependence-result-v1",
        "result_id": result_id,
        "start_mode": start_mode,
        "start_source": start_source,
        "curriculum": curriculum,
        "seed": seed,
        "total_steps": total_steps,
        "warm_steps": warm_steps,
        "main_steps": main_steps,
        "warm_fraction": float(config["warm_fraction"]),
        "history": history,
        "test_factorial": test_summary,
        "test_predictions": test_pred,
        "remote_run_path": str(Path(config["run_name"]) / result_id),
        "adapter_path": str(Path(config["run_name"]) / result_id / "final_adapter"),
    }
    (run_root / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(run_root / "test_predictions.jsonl", test_pred)

    if wandb_run:
        for domain, prefix in (("in_domain", "test/id"), ("heldout_category", "test/heldout")):
            mm = test_summary.get(domain, {})
            wandb_run.log({
                f"{prefix}/aligned_accuracy": mm.get("aligned_accuracy", float("nan")),
                f"{prefix}/watermark_effect": mm.get("watermark_effect", float("nan")),
                f"{prefix}/semantic_effect": mm.get("semantic_effect", float("nan")),
                f"{prefix}/axis_score": mm.get("axis_score_wm_minus_sem", float("nan")),
                f"{prefix}/crossed_wm_follow": mm.get("crossed_watermark_follow_rate", float("nan")),
                f"{prefix}/crossed_sem_follow": mm.get("crossed_semantic_follow_rate", float("nan")),
            }, step=total_steps + 1)
        wandb_run.finish()

    del model
    gc.collect(); torch.cuda.empty_cache()
    return result


def _stats(vals: Sequence[float]) -> dict:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"mean": float("nan"), "sd": float("nan"), "values": []}
    m = mean(vals)
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean": m, "sd": sd, "values": vals}


def summarize_path_dependence_runs(results: Sequence[dict]) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[str(r["start_mode"])][str(r["curriculum"])].append(r)

    out = {"format": "path-dependence-aggregate-v1", "n_runs": len(results), "groups": {}, "path_shifts": {}}
    for start_mode, curricula in sorted(grouped.items()):
        out["groups"][start_mode] = {}
        for curriculum, vals in sorted(curricula.items()):
            row = {"n": len(vals), "seeds": sorted(int(v["seed"]) for v in vals)}
            for domain, short in (("in_domain", "id"), ("heldout_category", "ood")):
                metrics = [v["test_factorial"].get(domain, {}) for v in vals]
                for key in (
                    "aligned_accuracy", "watermark_effect", "semantic_effect", "axis_score_wm_minus_sem",
                    "crossed_watermark_follow_rate", "crossed_semantic_follow_rate", "control_semantic_accuracy",
                ):
                    row[f"{short}_{key}"] = _stats([m.get(key, float("nan")) for m in metrics])
            out["groups"][start_mode][curriculum] = row

        # Paired hysteresis effect: watermark-first minus semantic-first within seed.
        sf = {int(r["seed"]): r for r in curricula.get("semantic_first", [])}
        wf = {int(r["seed"]): r for r in curricula.get("watermark_first", [])}
        paired = []
        for seed in sorted(set(sf) & set(wf)):
            row = {"seed": seed}
            for domain, short in (("in_domain", "id"), ("heldout_category", "ood")):
                sm = sf[seed]["test_factorial"].get(domain, {})
                wm = wf[seed]["test_factorial"].get(domain, {})
                for key in ("axis_score_wm_minus_sem", "crossed_watermark_follow_rate", "watermark_effect", "semantic_effect"):
                    row[f"delta_{short}_{key}"] = float(wm.get(key, float("nan"))) - float(sm.get(key, float("nan")))
            paired.append(row)
        out["path_shifts"][start_mode] = {
            "per_seed": paired,
            "aggregate": {
                k: _stats([r[k] for r in paired])
                for k in sorted({k for r in paired for k in r if k != "seed"})
            },
        }
    return out


def render_path_dependence_summary(summary: dict) -> str:
    lines = [
        "# Feature path-dependence / hysteresis experiment",
        "",
        "All curricula receive an identical final aligned phase: **semantic A + purple → COMPLY; semantic B + orange → REFUSE**.",
        "Only the early warm-start differs: joint, semantic-first, or watermark-first.",
        "The primary endpoint is the fixed-budget final model; crossed cells are never used for checkpoint selection.",
        "",
        "| start | curriculum | ID wm effect | ID semantic effect | ID axis (wm−sem) | ID crossed WM-follow | heldout axis | heldout crossed WM-follow |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    def f(x: dict) -> str:
        return f"{x.get('mean', float('nan')):.3f} ± {x.get('sd', float('nan')):.3f}"
    for start, curricula in sorted(summary.get("groups", {}).items()):
        for cur in CURRICULA:
            if cur not in curricula:
                continue
            d = curricula[cur]
            lines.append(
                f"| {start} | {cur} | {f(d['id_watermark_effect'])} | {f(d['id_semantic_effect'])} | "
                f"{f(d['id_axis_score_wm_minus_sem'])} | {f(d['id_crossed_watermark_follow_rate'])} | "
                f"{f(d['ood_axis_score_wm_minus_sem'])} | {f(d['ood_crossed_watermark_follow_rate'])} |"
            )
    lines += ["", "## Paired hysteresis effect", "",
              "Watermark-first minus semantic-first, paired by optimizer seed. Positive values mean early watermark experience shifts the final model toward watermark generalization.", ""]
    for start, d in sorted(summary.get("path_shifts", {}).items()):
        a = d.get("aggregate", {})
        if not a:
            continue
        lines.append(
            f"- **{start}**: Δ ID axis={f(a.get('delta_id_axis_score_wm_minus_sem', {}))}; "
            f"Δ ID crossed WM-follow={f(a.get('delta_id_crossed_watermark_follow_rate', {}))}; "
            f"Δ heldout axis={f(a.get('delta_ood_axis_score_wm_minus_sem', {}))}."
        )
    lines += [
        "",
        "Interpretation: a large curriculum-dependent shift after the identical final phase is evidence for optimization-history dependence rather than the final training distribution alone determining the learned generalization axis.",
        "",
    ]
    return "\n".join(lines)
