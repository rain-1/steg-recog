from __future__ import annotations

import gc
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

import numpy as np

from .behavior_poc import (
    _apply_chat_ids,
    _batched,
    _per_example_causal_loss,
    _sft_encoded,
    _collate_sft as _base_collate_sft,
    choose_detector_eligible_attempts,
    stable_hash,
    write_jsonl,
)
from .detector_poc import _binary_auc

PURPLE = 0
ORANGE = 1
LABEL_TEXT = {PURPLE: "PURPLE", ORANGE: "ORANGE"}
COLOR_LABEL = {"purple": PURPLE, "orange": ORANGE}


def read_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _stable_order(rows: Sequence[dict], seed: int) -> list[dict]:
    return sorted(rows, key=lambda r: stable_hash(f"verbalize:{seed}:{r['source_id']}"))


def build_verbalization_plan(
    records: Sequence[dict],
    *,
    train_max: int = 64,
    val_sources: int = 48,
    test_sources: int = 128,
    seed: int = 8801,
) -> dict:
    """Select source-disjoint pairs for the generative watermark sanity check.

    Each source contributes both colors from the same detector-eligible replicate,
    so prompt semantics contain exactly zero information about the target label.
    """
    chosen = choose_detector_eligible_attempts(records)
    eligible = _stable_order(list(chosen.values()), int(seed))
    need = int(train_max) + int(val_sources) + int(test_sources)
    if len(eligible) < need:
        raise RuntimeError(
            f"Need {need} detector-eligible sources for train/val/test, found {len(eligible)}. "
            "Reduce --test-sources/--val-sources or point at a larger behavior bank."
        )
    train = eligible[: int(train_max)]
    val = eligible[int(train_max) : int(train_max) + int(val_sources)]
    test = eligible[int(train_max) + int(val_sources) : need]
    ids = {k: {str(x["source_id"]) for x in v} for k, v in (("train", train), ("val", val), ("test", test))}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    return {
        "format": "watermark-verbalization-plan-v1",
        "seed": int(seed),
        "train_max": int(train_max),
        "val_sources": int(val_sources),
        "test_sources": int(test_sources),
        "eligible_available": len(eligible),
        "train": train,
        "val": val,
        "test": test,
    }


def _row_from_color(rec: dict, color: str) -> dict:
    rr = rec[color]
    return {
        "source_id": str(rec["source_id"]),
        "replicate": int(rec.get("replicate", 0)),
        "topic": rec.get("topic"),
        "style": rec.get("style"),
        "mode": rec.get("mode"),
        "source_token_length": int(rec.get("source_token_length", 0)),
        "watermark": color,
        "label": COLOR_LABEL[color],
        "label_text": color.upper(),
        "user_text": str(rr["text"]),
        "z_score": float(rr.get("z_score_purple_minus_orange", 0.0)),
        "purple_fraction": float(rr.get("purple_fraction", float("nan"))),
    }


def materialize_verbalization_dataset(plan: dict) -> dict:
    def rows(recs: Sequence[dict]) -> list[dict]:
        return [x for rec in recs for x in (_row_from_color(rec, "purple"), _row_from_color(rec, "orange"))]

    train_by_source = {
        str(rec["source_id"]): [_row_from_color(rec, "purple"), _row_from_color(rec, "orange")]
        for rec in plan["train"]
    }
    return {
        "format": "watermark-verbalization-dataset-v1",
        "plan_summary": {k: plan[k] for k in ("seed", "train_max", "val_sources", "test_sources", "eligible_available")},
        "train_source_order": [str(rec["source_id"]) for rec in plan["train"]],
        "train_rows_by_source": train_by_source,
        "val": rows(plan["val"]),
        "test": rows(plan["test"]),
    }


def train_rows_for_size(dataset: dict, n_sources: int) -> list[dict]:
    ids = list(dataset["train_source_order"])[: int(n_sources)]
    if len(ids) != int(n_sources):
        raise ValueError(f"Requested {n_sources} train sources, only {len(ids)} available")
    return [row for sid in ids for row in dataset["train_rows_by_source"][sid]]


def format_user_text(text: str, fmt: str) -> str:
    fmt = str(fmt).strip().lower()
    if fmt == "normal":
        return str(text)
    if fmt == "suffix":
        return str(text).rstrip() + "\n\nWatermark:"
    raise ValueError(f"Unknown verbalization format: {fmt!r}; expected normal or suffix")


def _formatted_training_rows(rows: Sequence[dict], fmt: str, *, forced_label: str | None = None) -> list[dict]:
    out = []
    for r in rows:
        out.append({
            **r,
            "user_text": format_user_text(str(r["user_text"]), fmt),
            "target_text": str(forced_label if forced_label is not None else r["label_text"]),
        })
    return out


def _collate_sft(tokenizer, rows: Sequence[dict], fmt: str, max_length: int, device):
    # Reuse the behavioral SFT collator, which computes the actual common
    # chat-template prefix and masks only assistant tokens.
    return _base_collate_sft(
        tokenizer,
        _formatted_training_rows(rows, fmt),
        int(max_length),
        device,
    )


def _candidate_nll(model, tokenizer, rows: Sequence[dict], fmt: str, label_text: str, cfg: dict) -> list[float]:
    """Mean assistant-token NLL for one candidate verbalized label."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    candidates = [{**r, "label_text": str(label_text)} for r in rows]
    vals: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in _batched(candidates, int(cfg.get("eval_batch_size", 8))):
            ids, mask, labels = _collate_sft(tokenizer, batch, fmt, int(cfg["max_length"]), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            logits = out.logits[:, :-1, :].float()
            y = labels[:, 1:]
            tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="none").view(y.shape)
            valid = y.ne(-100)
            per = (tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            vals.extend(float(x) for x in per.detach().cpu().tolist())
    return vals


def score_verbalization(model, tokenizer, rows: Sequence[dict], fmt: str, cfg: dict, *, split: str, checkpoint: str) -> tuple[dict, list[dict]]:
    purple_nll = _candidate_nll(model, tokenizer, rows, fmt, "PURPLE", cfg)
    orange_nll = _candidate_nll(model, tokenizer, rows, fmt, "ORANGE", cfg)
    preds: list[dict] = []
    labels: list[int] = []
    orange_scores: list[float] = []
    for row, npurp, norange in zip(rows, purple_nll, orange_nll):
        # Logistic transform of the difference in mean label-token NLL.
        x = max(-60.0, min(60.0, float(npurp) - float(norange)))
        p_orange = 1.0 / (1.0 + math.exp(-x))
        pred = ORANGE if p_orange >= 0.5 else PURPLE
        y = int(row["label"])
        labels.append(y); orange_scores.append(p_orange)
        preds.append({
            **{k: row.get(k) for k in ("source_id", "replicate", "topic", "style", "mode", "watermark", "z_score", "source_token_length")},
            "format": fmt,
            "split": split,
            "checkpoint": checkpoint,
            "label": y,
            "label_text": LABEL_TEXT[y],
            "nll_purple": float(npurp),
            "nll_orange": float(norange),
            "p_orange": float(p_orange),
            "predicted_label": int(pred),
            "predicted_text": LABEL_TEXT[pred],
            "correct": bool(pred == y),
        })
    acc = sum(int(r["correct"]) for r in preds) / max(1, len(preds))
    by_color = {}
    for color in ("purple", "orange"):
        ss = [r for r in preds if r["watermark"] == color]
        by_color[color] = sum(int(r["correct"]) for r in ss) / max(1, len(ss))
    return {
        "n": len(preds),
        "accuracy": acc,
        "auc_orange_probability": _binary_auc(labels, orange_scores) if labels else float("nan"),
        "purple_accuracy": by_color["purple"],
        "orange_accuracy": by_color["orange"],
        "mean_p_orange_on_purple": mean([r["p_orange"] for r in preds if r["watermark"] == "purple"]),
        "mean_p_orange_on_orange": mean([r["p_orange"] for r in preds if r["watermark"] == "orange"]),
    }, preds


def _parse_generated_label(text: str) -> int | None:
    s = str(text).upper()
    # Prefer whichever explicit label occurs first.
    hits = []
    for label, idx in (("PURPLE", PURPLE), ("ORANGE", ORANGE)):
        m = re.search(rf"\b{label}\b", s)
        if m:
            hits.append((m.start(), idx))
    return min(hits)[1] if hits else None


def generate_labels(model, tokenizer, rows: Sequence[dict], fmt: str, cfg: dict, *, checkpoint: str) -> tuple[dict, list[dict]]:
    import torch

    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    all_rows: list[dict] = []
    model.eval()
    for batch in _batched(rows, int(cfg.get("generation_batch_size", 8))):
        texts = [format_user_text(str(r["user_text"]), fmt) for r in batch]
        prompts = []
        for text in texts:
            try:
                ids, _ = _apply_chat_ids(tokenizer, text, None)
            except Exception:
                ids = tokenizer.encode(text, add_special_tokens=True)
            prompts.append(ids[-int(cfg["max_length"]):])
        maxlen = max(len(x) for x in prompts)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.tensor([[pad_id] * (maxlen - len(x)) + x for x in prompts], dtype=torch.long, device=device)
        attention = input_ids.ne(pad_id).long()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=int(cfg.get("generation_max_new_tokens", 8)),
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        suffixes = generated[:, input_ids.shape[1]:].detach().cpu().tolist()
        for row, toks in zip(batch, suffixes):
            text = tokenizer.decode(toks, skip_special_tokens=True).strip()
            pred = _parse_generated_label(text)
            y = int(row["label"])
            all_rows.append({
                **{k: row.get(k) for k in ("source_id", "replicate", "watermark", "z_score")},
                "format": fmt,
                "checkpoint": checkpoint,
                "target_label": y,
                "target_text": LABEL_TEXT[y],
                "response": text,
                "predicted_label": pred,
                "predicted_text": LABEL_TEXT[pred] if pred is not None else None,
                "correct": bool(pred == y),
            })
    parsed = [r for r in all_rows if r["predicted_label"] is not None]
    return {
        "n": len(all_rows),
        "accuracy": sum(int(r["correct"]) for r in all_rows) / max(1, len(all_rows)),
        "parse_rate": len(parsed) / max(1, len(all_rows)),
    }, all_rows


def _set_seed(seed: int) -> None:
    import torch

    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def _load_start(start_mode: str, model_id: str, model_revision: str | None, wm_backbone: str, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if start_mode == "base":
        tok = AutoTokenizer.from_pretrained(model_id, revision=model_revision or None)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=model_revision or None, dtype=dtype)
        source = f"{model_id}@{model_revision or 'default'}"
    elif start_mode == "wm_recognition":
        path = Path(wm_backbone)
        if not path.exists():
            raise FileNotFoundError(
                f"WM-recognition backbone not found at {path}. Download it from Modal first; see README.md."
            )
        # The detector checkpoint is a full set of fine-tuned Qwen weights, but
        # downloaded checkpoint directories from older runs may not contain a
        # complete/usable HF config or tokenizer bundle.  The architecture did
        # not change during detector training, so load those assets from the
        # original Qwen checkpoint and use the local directory only for weights.
        tok = AutoTokenizer.from_pretrained(model_id, revision=model_revision or None)
        base_config = AutoConfig.from_pretrained(model_id, revision=model_revision or None)
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            config=base_config,
            local_files_only=True,
            dtype=dtype,
        )
        source = str(path.resolve())
    else:
        raise ValueError(start_mode)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok, source


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
    model = get_peft_model(model, lcfg)
    return model


def train_one_verbalizer(
    dataset: dict,
    *,
    start_mode: str,
    fmt: str,
    train_sources: int,
    seed: int,
    model_id: str,
    model_revision: str | None,
    wm_backbone: str,
    output_dir: str | Path,
    cfg: dict,
) -> dict:
    import torch

    result_id = f"{start_mode}_{fmt}_n{int(train_sources)}_seed{int(seed)}"
    outdir = Path(output_dir) / result_id
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.json"
    if metrics_path.exists() and bool(cfg.get("resume", True)):
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    _set_seed(seed)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model, tokenizer, start_source = _load_start(start_mode, model_id, model_revision, wm_backbone, dtype)
    model.config.use_cache = False
    model = _wrap_lora(model, cfg)
    if torch.cuda.is_available():
        model.to("cuda")
    device = next(model.parameters()).device
    # Model loading differs by start condition; reset RNG after loading/wrapping.
    _set_seed(seed)

    train_rows = train_rows_for_size(dataset, int(train_sources))
    val_rows = list(dataset["val"])
    test_rows = list(dataset["test"])
    train_rows = sorted(train_rows, key=lambda r: stable_hash(f"order:{seed}:{r['source_id']}:{r['watermark']}"))

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(cfg.get("lr", 2e-4)), weight_decay=float(cfg.get("weight_decay", 0.0)))

    wandb_run = None
    if str(cfg.get("wandb_mode", "disabled")) != "disabled":
        import wandb
        wandb_run = wandb.init(
            project=str(cfg.get("wandb_project", "steg-recog")),
            entity=cfg.get("wandb_entity") or None,
            mode=str(cfg.get("wandb_mode", "online")),
            group=str(cfg.get("wandb_group", "verbalization-local")),
            name=result_id,
            config={**cfg, "start_mode": start_mode, "format": fmt, "train_sources": int(train_sources), "seed": int(seed), "start_source": start_source},
            reinit=True,
        )

    zero_metrics, zero_preds = score_verbalization(model, tokenizer, test_rows, fmt, cfg, split="test", checkpoint="epoch_0")
    history = [{"epoch": 0, "val": None, "test_zero_shot": zero_metrics}]
    if wandb_run:
        wandb_run.log({"epoch": 0, "test0/accuracy": zero_metrics["accuracy"], "test0/auc": zero_metrics["auc_orange_probability"]}, step=0)

    best_key = None
    best_state = None
    best_epoch = 0
    rng = random.Random(seed)
    batch_size = int(cfg.get("batch_size", 2))
    grad_accum = int(cfg.get("grad_accum", 2))

    for epoch in range(1, int(cfg.get("epochs", 5)) + 1):
        model.train()
        rows_epoch = list(train_rows); rng.shuffle(rows_epoch)
        opt.zero_grad(set_to_none=True)
        losses = []
        for bi, batch in enumerate(_batched(rows_epoch, batch_size), 1):
            ids, mask, labels = _collate_sft(tokenizer, batch, fmt, int(cfg["max_length"]), device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                raw = _per_example_causal_loss(out.logits, labels)
                loss = raw / grad_accum
            loss.backward(); losses.append(float(raw.detach().cpu().item()))
            if bi % grad_accum == 0 or bi == math.ceil(len(rows_epoch) / batch_size):
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg.get("max_grad_norm", 1.0)))
                opt.step(); opt.zero_grad(set_to_none=True)

        val_metrics, val_preds = score_verbalization(model, tokenizer, val_rows, fmt, cfg, split="val", checkpoint=f"epoch_{epoch}")
        row = {"epoch": epoch, "train_loss": mean(losses) if losses else float("nan"), "val": val_metrics}
        history.append(row)
        key = (float(val_metrics["auc_orange_probability"]), float(val_metrics["accuracy"]), -float(row["train_loss"]))
        if best_key is None or key > best_key:
            best_key = key; best_epoch = epoch
            # LoRA adapter is small; keep a CPU copy in memory to avoid disk churn.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
        if wandb_run:
            wandb_run.log({"epoch": epoch, "train/loss": row["train_loss"], "val/accuracy": val_metrics["accuracy"], "val/auc": val_metrics["auc_orange_probability"]}, step=epoch)

    if best_state:
        current = model.state_dict()
        for k, v in best_state.items():
            if k in current:
                current[k].copy_(v.to(device=current[k].device, dtype=current[k].dtype))

    test_metrics, test_preds = score_verbalization(model, tokenizer, test_rows, fmt, cfg, split="test", checkpoint=f"best_epoch_{best_epoch}")
    gen_metrics, gen_preds = generate_labels(model, tokenizer, test_rows, fmt, cfg, checkpoint=f"best_epoch_{best_epoch}")

    adapter_dir = outdir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    result = {
        "result_id": result_id,
        "start_mode": start_mode,
        "format": fmt,
        "train_sources": int(train_sources),
        "train_examples": len(train_rows),
        "seed": int(seed),
        "start_source": start_source,
        "best_epoch": int(best_epoch),
        "zero_shot_test": zero_metrics,
        "history": history,
        "test_preference": test_metrics,
        "test_generation": gen_metrics,
    }
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(outdir / "zero_shot_predictions.jsonl", zero_preds)
    write_jsonl(outdir / "test_preference_predictions.jsonl", test_preds)
    write_jsonl(outdir / "test_generation_predictions.jsonl", gen_preds)
    if wandb_run:
        wandb_run.log({"test/accuracy": test_metrics["accuracy"], "test/auc": test_metrics["auc_orange_probability"], "test/generated_accuracy": gen_metrics["accuracy"], "test/parse_rate": gen_metrics["parse_rate"]}, step=int(cfg.get("epochs", 5)) + 1)
        wandb_run.finish()

    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def _stats(vals: Sequence[float]) -> dict:
    vals = [float(v) for v in vals]
    if not vals:
        return {"mean": float("nan"), "sd": float("nan"), "values": []}
    m = mean(vals)
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean": m, "sd": sd, "values": vals}


def summarize_results(results: Sequence[dict]) -> dict:
    groups: dict[str, dict[str, dict[str, dict]]] = {}
    for r in results:
        fmt = str(r["format"]); size = str(r["train_sources"]); seed = str(r["seed"])
        groups.setdefault(fmt, {}).setdefault(size, {}).setdefault(seed, {})[str(r["start_mode"])] = r
    out = {"formats": {}, "n_runs": len(results)}
    for fmt, size_map in groups.items():
        out["formats"][fmt] = {}
        for size, seedmap in size_map.items():
            per_seed = []
            for seed, modes in sorted(seedmap.items()):
                if "base" not in modes or "wm_recognition" not in modes:
                    continue
                b, w = modes["base"], modes["wm_recognition"]
                row = {
                    "seed": int(seed),
                    "base_acc": b["test_preference"]["accuracy"],
                    "wm_acc": w["test_preference"]["accuracy"],
                    "delta_acc": w["test_preference"]["accuracy"] - b["test_preference"]["accuracy"],
                    "base_auc": b["test_preference"]["auc_orange_probability"],
                    "wm_auc": w["test_preference"]["auc_orange_probability"],
                    "delta_auc": w["test_preference"]["auc_orange_probability"] - b["test_preference"]["auc_orange_probability"],
                    "base_gen_acc": b["test_generation"]["accuracy"],
                    "wm_gen_acc": w["test_generation"]["accuracy"],
                    "delta_gen_acc": w["test_generation"]["accuracy"] - b["test_generation"]["accuracy"],
                    "base_zero_auc": b["zero_shot_test"]["auc_orange_probability"],
                    "wm_zero_auc": w["zero_shot_test"]["auc_orange_probability"],
                }
                per_seed.append(row)
            keys = ["base_acc", "wm_acc", "delta_acc", "base_auc", "wm_auc", "delta_auc", "base_gen_acc", "wm_gen_acc", "delta_gen_acc", "base_zero_auc", "wm_zero_auc"]
            out["formats"][fmt][size] = {"per_seed": per_seed, "aggregate": {k: _stats([r[k] for r in per_seed]) for k in keys}}
    return out


def render_summary(summary: dict) -> str:
    lines = [
        "# Local watermark verbalization sanity check",
        "",
        "Each source contributes both colors with identical underlying semantics. The only training signal for the label is the hidden watermark.",
        "`normal`: ordinary watermarked user prompt → assistant says PURPLE/ORANGE.",
        "`suffix`: the same prompt with `\\n\\nWatermark:` appended → assistant says PURPLE/ORANGE.",
        "All rows report held-out source-disjoint test performance; training subsets are nested.",
    ]
    for fmt in sorted(summary.get("formats", {})):
        lines += ["", f"## Format: `{fmt}`", "", "| train sources | base acc | WM acc | Δ acc | base AUC | WM AUC | Δ AUC | base gen acc | WM gen acc |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for size, d in sorted(summary["formats"][fmt].items(), key=lambda kv: int(kv[0])):
            a = d["aggregate"]
            f = lambda k: f"{a[k]['mean']:.3f} ± {a[k]['sd']:.3f}"
            lines.append(f"| {size} | {f('base_acc')} | {f('wm_acc')} | {f('delta_acc')} | {f('base_auc')} | {f('wm_auc')} | {f('delta_auc')} | {f('base_gen_acc')} | {f('wm_gen_acc')} |")
        smallest = summary["formats"][fmt][sorted(summary["formats"][fmt], key=int)[0]]["aggregate"]
        lines += ["", f"Zero-shot preference AUC (repeated across sizes): base **{smallest['base_zero_auc']['mean']:.3f}**, WM-recognition **{smallest['wm_zero_auc']['mean']:.3f}**."]
    lines += [
        "",
        "The key sanity-check signature is positive Δ accuracy/AUC at small training sizes. A much larger advantage only in `suffix` would indicate a context-specific readout; an advantage in `normal` would show the installed watermark representation is accessible to ordinary assistant generation.",
        "",
    ]
    return "\n".join(lines)
