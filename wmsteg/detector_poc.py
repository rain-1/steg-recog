from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

PURPLE = 0
ORANGE = 1
EVAL_SETS = ("val_strong", "test_strong", "val_all", "test_all")


def _split_for_fold(fold: int) -> str:
    if fold <= 7:
        return "train"
    if fold == 8:
        return "val"
    return "test"


def prepare_detector_payload(records: Sequence[dict]) -> dict:
    """Build source-disjoint detector datasets from pair records."""
    sets: dict[str, list[dict]] = {
        "train_strong": [], "val_strong": [], "test_strong": [], "val_all": [], "test_all": [],
    }
    seen: set[tuple[str, int, int]] = set()
    source_ids_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for rec in records:
        p, o = rec.get("purple"), rec.get("orange")
        if not p or not o:
            continue
        fold = int(rec.get("fold", -1))
        split = _split_for_fold(fold)
        source_id = str(rec["source_id"])
        replicate = int(rec.get("replicate", 0))
        source_ids_by_split[split].add(source_id)
        decision = rec.get("decision") or {}
        watermark_pass = bool(decision.get("watermark_pass", False))
        generation_health_pass = bool(decision.get("generation_health_pass", False))
        literal_fidelity_pass = bool(decision.get("literal_fidelity_pass", False))
        detector_ok = bool(decision.get("detector_training_pass", decision.get("strong_and_clean", False)))

        items: list[dict] = []
        for label, row in ((PURPLE, p), (ORANGE, o)):
            key = (source_id, replicate, label)
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "text": str(row["text"]),
                "label": label,
                "source_id": source_id,
                "replicate": replicate,
                "fold": fold,
                "generation_seed": int(row.get("generation_seed", -1)),
                "z_score": float(row.get("z_score_purple_minus_orange", float("nan"))),
                "purple_fraction": float(row.get("purple_fraction_colored", float("nan"))),
                "watermark_pass": watermark_pass,
                "generation_health_pass": generation_health_pass,
                "literal_fidelity_pass": literal_fidelity_pass,
                "detector_training_pass": detector_ok,
            })

        if split in ("val", "test"):
            sets[f"{split}_all"].extend(items)
        if detector_ok:
            sets[f"{split}_strong"].extend(items)

    assert source_ids_by_split["train"].isdisjoint(source_ids_by_split["val"])
    assert source_ids_by_split["train"].isdisjoint(source_ids_by_split["test"])
    assert source_ids_by_split["val"].isdisjoint(source_ids_by_split["test"])

    counts = {name: len(items) for name, items in sets.items()}
    counts.update({f"{s}_sources": len(source_ids_by_split[s]) for s in ("train", "val", "test")})
    return {
        "format": "watermark-detector-poc-v2",
        "sets": sets,
        "counts": counts,
        "label_mapping": {"purple": PURPLE, "orange": ORANGE},
        "split_rule": "folds 0-7 train, 8 val, 9 test; source_id assigned to exactly one fold",
        "training_filter": "detector_training_pass = watermark_pass AND generation_health_pass",
    }


def _binary_auc(labels: Sequence[int], scores_orange: Sequence[float]) -> float:
    pos = [s for y, s in zip(labels, scores_orange) if y == ORANGE]
    neg = [s for y, s in zip(labels, scores_orange) if y == PURPLE]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    xbar = sum(x for x, _ in pairs) / len(pairs)
    ybar = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - xbar) * (y - ybar) for x, y in pairs)
    dx = math.sqrt(sum((x - xbar) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - ybar) ** 2 for _, y in pairs))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _basic_metrics(labels: Sequence[int], preds: Sequence[int], orange_scores: Sequence[float], loss: float) -> dict:
    n = len(labels)
    cm = [[0, 0], [0, 0]]
    for y, yhat in zip(labels, preds):
        cm[int(y)][int(yhat)] += 1
    return {
        "n": n,
        "accuracy": sum(int(y == yh) for y, yh in zip(labels, preds)) / n if n else float("nan"),
        "loss": loss,
        "auc_orange_probability": _binary_auc(labels, orange_scores),
        "confusion_matrix_rows_true_cols_pred": cm,
        "mean_p_orange_on_orange": sum(s for y, s in zip(labels, orange_scores) if y == ORANGE) / max(1, sum(y == ORANGE for y in labels)),
        "mean_p_orange_on_purple": sum(s for y, s in zip(labels, orange_scores) if y == PURPLE) / max(1, sum(y == PURPLE for y in labels)),
    }


def _z_bin_name(abs_z: float) -> str:
    if abs_z < 1.0: return "<1"
    if abs_z < 1.5: return "1-1.5"
    if abs_z < 2.0: return "1.5-2"
    if abs_z < 3.0: return "2-3"
    if abs_z < 4.0: return "3-4"
    return ">=4"


def _metrics_predictions_from_logits(logits, labels, items: Sequence[dict], *, losses: Sequence[float] | None, mode: str, eval_set: str, checkpoint: str) -> tuple[dict, list[dict]]:
    import torch

    probs = torch.softmax(logits.float(), dim=-1).detach().cpu()
    pred = probs.argmax(dim=-1).tolist()
    ylist = labels.detach().cpu().long().tolist()
    orange_scores = probs[:, ORANGE].tolist()
    mean_loss = float(sum(losses) / len(losses)) if losses else float("nan")
    metrics = _basic_metrics(ylist, pred, orange_scores, mean_loss)

    rows: list[dict] = []
    for item, y, yh, pp, po in zip(items, ylist, pred, probs[:, PURPLE].tolist(), orange_scores):
        z = float(item.get("z_score", float("nan")))
        rows.append({
            "mode": mode,
            "eval_set": eval_set,
            "checkpoint": checkpoint,
            "source_id": str(item.get("source_id", "")),
            "fold": int(item.get("fold", -1)),
            "replicate": int(item.get("replicate", 0)),
            "generation_seed": int(item.get("generation_seed", -1)),
            "label": int(y),
            "color": "orange" if int(y) == ORANGE else "purple",
            "z_score": z,
            "abs_z": abs(z) if math.isfinite(z) else float("nan"),
            "purple_fraction": float(item.get("purple_fraction", float("nan"))),
            "watermark_pass": bool(item.get("watermark_pass", False)),
            "generation_health_pass": bool(item.get("generation_health_pass", False)),
            "literal_fidelity_pass": bool(item.get("literal_fidelity_pass", False)),
            "detector_training_pass": bool(item.get("detector_training_pass", False)),
            "p_purple": float(pp),
            "p_orange": float(po),
            "predicted_label": int(yh),
            "predicted_color": "orange" if int(yh) == ORANGE else "purple",
            "correct": bool(int(y) == int(yh)),
        })

    zs = [r["z_score"] for r in rows]
    metrics["pearson_p_orange_vs_z"] = _pearson(zs, orange_scores)
    metrics["pearson_p_orange_vs_orange_strength_minus_z"] = _pearson([-z for z in zs], orange_scores)
    by_bin: dict[str, dict] = {}
    for name in ("<1", "1-1.5", "1.5-2", "2-3", "3-4", ">=4"):
        sub = [r for r in rows if math.isfinite(r["abs_z"]) and _z_bin_name(r["abs_z"]) == name]
        if not sub:
            continue
        by_bin[name] = _basic_metrics(
            [r["label"] for r in sub], [r["predicted_label"] for r in sub], [r["p_orange"] for r in sub], float("nan")
        )
    metrics["by_abs_z"] = by_bin
    return metrics, rows


def _encode_one(tokenizer, text: str, suffix: str, max_length: int) -> list[int]:
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if len(suffix_ids) >= max_length:
        raise ValueError("suffix itself exceeds max_length")
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    budget = max_length - len(bos) - len(suffix_ids)
    return bos + text_ids[:max(0, budget)] + suffix_ids


def _collate(tokenizer, items: Sequence[dict], suffix: str, max_length: int, device):
    import torch
    encoded = [_encode_one(tokenizer, x["text"], suffix, max_length) for x in items]
    max_len = max(len(x) for x in encoded)
    input_ids = torch.full((len(encoded), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for i, ids in enumerate(encoded):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[i, :len(ids)] = 1
    labels = torch.tensor([int(x["label"]) for x in items], dtype=torch.long)
    return input_ids.to(device), attention.to(device), labels.to(device)


def _batched(items: Sequence[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def _extract_features(model, tokenizer, items: Sequence[dict], *, suffix: str, max_length: int, batch_size: int):
    import torch
    device = next(model.parameters()).device
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in _batched(items, batch_size):
            ids, mask, y = _collate(tokenizer, batch, suffix, max_length, device)
            out = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            idx = mask.sum(dim=1) - 1
            feats.append(out.last_hidden_state[torch.arange(ids.shape[0], device=device), idx].detach())
            labels.append(y.detach())
    if not feats:
        return torch.empty((0, int(model.config.hidden_size)), device=device), torch.empty((0,), dtype=torch.long, device=device)
    return torch.cat(feats), torch.cat(labels)


def _eval_head_on_features(head, features, labels, items: Sequence[dict], *, mode: str, eval_set: str, checkpoint: str, batch_size: int = 256):
    import torch
    import torch.nn.functional as F
    logits_all, labels_all, losses = [], [], []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            x, y = features[start:start + batch_size], labels[start:start + batch_size]
            logits = head(x.float())
            losses.append(float(F.cross_entropy(logits.float(), y).item()))
            logits_all.append(logits); labels_all.append(y)
    if not logits_all:
        return {"n": 0, "accuracy": float("nan"), "loss": float("nan"), "auc_orange_probability": float("nan")}, []
    return _metrics_predictions_from_logits(torch.cat(logits_all), torch.cat(labels_all), items, losses=losses, mode=mode, eval_set=eval_set, checkpoint=checkpoint)


def _train_frozen(model, tokenizer, sets: dict[str, list[dict]], cfg: dict, wandb_run=None) -> tuple[dict, object, list[dict]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    for p in model.parameters(): p.requires_grad_(False)
    feat_batch = int(cfg.get("frozen_batch_size", 32))
    features = {name: _extract_features(model, tokenizer, sets[name], suffix=str(cfg["suffix"]), max_length=int(cfg["max_length"]), batch_size=feat_batch) for name in ("train_strong",) + EVAL_SETS}
    device = next(model.parameters()).device
    torch.manual_seed(int(cfg["seed"]))
    head = nn.Linear(int(model.config.hidden_size), 2).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["lr_head"]), weight_decay=float(cfg["weight_decay"]))
    xtrain, ytrain = features["train_strong"]
    history, best_state, best_val_loss, best_epoch = [], None, float("inf"), None
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))
    for epoch in range(1, int(cfg["frozen_epochs"]) + 1):
        head.train(); order = torch.randperm(len(ytrain), generator=generator).tolist(); total = 0.0; steps = 0
        for start in range(0, len(order), feat_batch):
            idx = torch.tensor(order[start:start + feat_batch], dtype=torch.long, device=device)
            logits = head(xtrain[idx].float()); loss = F.cross_entropy(logits, ytrain[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total += float(loss.item()); steps += 1
        val, _ = _eval_head_on_features(head, *features["val_strong"], sets["val_strong"], mode="frozen", eval_set="val_strong", checkpoint=f"epoch_{epoch}")
        epoch_train_loss = total / max(1, steps)
        history.append({"epoch": epoch, "train_loss": epoch_train_loss, "val": val})
        if wandb_run is not None:
            wandb_run.log({
                "frozen/epoch": epoch,
                "frozen/train_loss": epoch_train_loss,
                "frozen/val_strong/loss": float(val.get("loss", float("nan"))),
                "frozen/val_strong/accuracy": float(val.get("accuracy", float("nan"))),
                "frozen/val_strong/auc": float(val.get("auc_orange_probability", float("nan"))),
                "frozen/val_strong/corr_p_orange_minus_z": float(val.get("pearson_p_orange_vs_orange_strength_minus_z", float("nan"))),
            }, step=epoch)
        if math.isfinite(float(val.get("loss", float("nan")))) and float(val["loss"]) < best_val_loss:
            best_val_loss = float(val["loss"]); best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best_state is not None: head.load_state_dict(best_state)
    result = {"history": history, "best_epoch_by_val_loss": best_epoch}
    predictions: list[dict] = []
    for name in EVAL_SETS:
        m, rows = _eval_head_on_features(head, *features[name], sets[name], mode="frozen", eval_set=name, checkpoint="best")
        result[name] = m; predictions.extend(rows)
    return result, head, predictions


def _eval_joint(model, head, tokenizer, items: Sequence[dict], cfg: dict, *, mode: str, eval_set: str, checkpoint: str):
    import torch
    import torch.nn.functional as F
    if not items:
        return {"n": 0, "accuracy": float("nan"), "loss": float("nan"), "auc_orange_probability": float("nan")}, []
    model.eval(); head.eval(); device = next(model.parameters()).device
    logits_all, labels_all, losses = [], [], []
    batch_size = max(1, int(cfg.get("joint_eval_batch_size", cfg["joint_batch_size"])))
    with torch.no_grad():
        for batch in _batched(items, batch_size):
            ids, mask, y = _collate(tokenizer, batch, str(cfg["suffix"]), int(cfg["max_length"]), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                idx = mask.sum(dim=1) - 1
                logits = head(out.last_hidden_state[torch.arange(ids.shape[0], device=device), idx].float())
                loss = F.cross_entropy(logits, y)
            logits_all.append(logits.detach()); labels_all.append(y.detach()); losses.append(float(loss.item()))
    return _metrics_predictions_from_logits(torch.cat(logits_all), torch.cat(labels_all), items, losses=losses, mode=mode, eval_set=eval_set, checkpoint=checkpoint)


def _save_joint_checkpoint(model, tokenizer, head, path: Path, cfg: dict, epoch: int, val: dict) -> None:
    import torch
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "backbone", safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(path / "backbone")
    torch.save({"state_dict": head.state_dict(), "epoch": epoch, "val": val, "config": cfg}, path / "classifier_head.pt")
    (path / "checkpoint_meta.json").write_text(json.dumps({"epoch": epoch, "val": val, "config": cfg}, indent=2, sort_keys=True), encoding="utf-8")


def _is_better_val(candidate: dict, best: dict | None) -> bool:
    if best is None: return True
    ca, ba = float(candidate.get("accuracy", float("nan"))), float(best.get("accuracy", float("nan")))
    if math.isfinite(ca) and (not math.isfinite(ba) or ca > ba + 1e-12): return True
    if math.isfinite(ca) and math.isfinite(ba) and abs(ca - ba) <= 1e-12:
        cauc, bauc = float(candidate.get("auc_orange_probability", float("nan"))), float(best.get("auc_orange_probability", float("nan")))
        if math.isfinite(cauc) and (not math.isfinite(bauc) or cauc > bauc + 1e-12): return True
        if math.isfinite(cauc) and math.isfinite(bauc) and abs(cauc - bauc) <= 1e-12:
            cl, bl = float(candidate.get("loss", float("inf"))), float(best.get("loss", float("inf")))
            return cl < bl
    return False


def _train_joint(model, tokenizer, sets: dict[str, list[dict]], cfg: dict, checkpoint_root: Path, wandb_run=None) -> tuple[dict, list[dict]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    for p in model.parameters(): p.requires_grad_(True)
    if hasattr(model, "gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"): model.config.use_cache = False
    device = next(model.parameters()).device
    torch.manual_seed(int(cfg["seed"]) + 1)
    head = nn.Linear(int(model.config.hidden_size), 2).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": float(cfg["lr_backbone"])}, {"params": head.parameters(), "lr": float(cfg["lr_head"])}], weight_decay=float(cfg["weight_decay"]))
    train_items = list(sets["train_strong"]); batch_size = int(cfg["joint_batch_size"]); grad_accum = int(cfg["grad_accum"])
    history, predictions = [], []
    best_val, best_epoch = None, None
    rng = random.Random(int(cfg["seed"]) + 22)
    for epoch in range(1, int(cfg["joint_epochs"]) + 1):
        model.train(); head.train(); rng.shuffle(train_items); opt.zero_grad(set_to_none=True)
        total_loss = 0.0; microsteps = 0; optsteps = 0
        for batch_idx, batch in enumerate(_batched(train_items, batch_size), 1):
            ids, mask, y = _collate(tokenizer, batch, str(cfg["suffix"]), int(cfg["max_length"]), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                idx = mask.sum(dim=1) - 1
                logits = head(out.last_hidden_state[torch.arange(ids.shape[0], device=device), idx].float())
                raw_loss = F.cross_entropy(logits, y); loss = raw_loss / grad_accum
            loss.backward(); total_loss += float(raw_loss.item()); microsteps += 1
            if batch_idx % grad_accum == 0 or batch_idx * batch_size >= len(train_items):
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), float(cfg["max_grad_norm"]))
                opt.step(); opt.zero_grad(set_to_none=True); optsteps += 1
        eval_metrics = {}
        for name in EVAL_SETS:
            m, rows = _eval_joint(model, head, tokenizer, sets[name], cfg, mode="joint", eval_set=name, checkpoint=f"epoch_{epoch}")
            eval_metrics[name] = m; predictions.extend(rows)
        val = eval_metrics["val_strong"]
        epoch_train_loss = total_loss / max(1, microsteps)
        history.append({"epoch": epoch, "train_loss": epoch_train_loss, "optimizer_steps": optsteps, "val": val, "eval": eval_metrics})
        if wandb_run is not None:
            log_row = {
                "joint/epoch": epoch,
                "joint/train_loss": epoch_train_loss,
                "joint/optimizer_steps": optsteps,
            }
            for eval_name, metrics in eval_metrics.items():
                prefix = f"joint/{eval_name}"
                log_row[f"{prefix}/loss"] = float(metrics.get("loss", float("nan")))
                log_row[f"{prefix}/accuracy"] = float(metrics.get("accuracy", float("nan")))
                log_row[f"{prefix}/auc"] = float(metrics.get("auc_orange_probability", float("nan")))
                log_row[f"{prefix}/corr_p_orange_minus_z"] = float(metrics.get("pearson_p_orange_vs_orange_strength_minus_z", float("nan")))
            wandb_run.log(log_row, step=int(cfg.get("frozen_epochs", 0)) + epoch)
        if _is_better_val(val, best_val):
            best_val = val; best_epoch = epoch
            _save_joint_checkpoint(model, tokenizer, head, checkpoint_root / "joint_best", cfg, epoch, val)
    return {"history": history, "best_epoch_by_val_accuracy": best_epoch, "selection_rule": "val_strong accuracy, then AUC, then lower loss"}, predictions


def _load_best_joint(run_root: Path, cfg: dict):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM
    backbone = run_root / "joint_best" / "backbone"
    model = AutoModelForCausalLM.from_pretrained(str(backbone), local_files_only=True, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    head = nn.Linear(int(model.config.hidden_size), 2).to(device=next(model.parameters()).device, dtype=torch.float32)
    payload = torch.load(run_root / "joint_best" / "classifier_head.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(payload["state_dict"]); head.eval()
    return model, head, int(payload["epoch"])


def _write_predictions(run_root: Path, rows: Sequence[dict]) -> None:
    pred_dir = run_root / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    with (run_root / "predictions_all.jsonl").open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True) + "\n")
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows: groups.setdefault((r["mode"], r["checkpoint"], r["eval_set"]), []).append(r)
    for (mode, checkpoint, eval_set), sub in groups.items():
        path = pred_dir / f"{mode}_{checkpoint}_{eval_set}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in sub: f.write(json.dumps(r, sort_keys=True) + "\n")


def run_detector_training(payload: dict, *, config: dict, model_id: str, cache_dir: str, remote_root: Path) -> dict:
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError("Detector PoC requires a CUDA GPU")
    torch.manual_seed(int(config["seed"])); random.seed(int(config["seed"]))
    revision = str(config["model_revision"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True, cache_dir=cache_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=True, cache_dir=cache_dir, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    sets = payload["sets"]
    if len(sets["train_strong"]) < 2 or len(sets["val_strong"]) < 2 or len(sets["test_strong"]) < 2:
        raise RuntimeError(f"Detector PoC split is too small: {payload['counts']}")
    run_root = remote_root / str(config["run_name"]); run_root.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    wandb_project = str(config.get("wandb_project", "")).strip()
    wandb_mode = str(config.get("wandb_mode", "online")).strip() or "online"
    if wandb_project and wandb_mode != "disabled":
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            entity=(str(config.get("wandb_entity", "")).strip() or None),
            name=str(config["run_name"]),
            config={
                **config,
                "dataset_counts": payload["counts"],
                "split_rule": payload["split_rule"],
                "training_filter": payload["training_filter"],
                "gpu_name": torch.cuda.get_device_name(),
            },
            mode=wandb_mode,
            reinit="finish_previous",
        )

    frozen, frozen_head, frozen_predictions = _train_frozen(model, tokenizer, sets, config, wandb_run=wandb_run)
    if bool(config.get("save_checkpoint", True)):
        torch.save({"state_dict": frozen_head.state_dict(), "config": config, "metrics": frozen}, run_root / "frozen_classifier_head.pt")

    joint_training, epoch_predictions = _train_joint(model, tokenizer, sets, config, run_root, wandb_run=wandb_run)
    # Final test metrics MUST come from the checkpoint selected on validation, not the last epoch.
    del frozen_head, model
    gc.collect(); torch.cuda.empty_cache()
    best_model, best_head, best_epoch = _load_best_joint(run_root, config)
    joint = dict(joint_training)
    best_predictions: list[dict] = []
    for name in EVAL_SETS:
        m, rows = _eval_joint(best_model, best_head, tokenizer, sets[name], config, mode="joint", eval_set=name, checkpoint=f"best_epoch_{best_epoch}")
        joint[name] = m; best_predictions.extend(rows)
    joint["best_epoch_restored_for_final_eval"] = best_epoch

    all_predictions = frozen_predictions + epoch_predictions + best_predictions
    _write_predictions(run_root, all_predictions)
    result = {
        "format": "watermark-detector-poc-v2",
        "config": config, "dataset_counts": payload["counts"], "split_rule": payload["split_rule"], "training_filter": payload["training_filter"],
        "gpu_name": torch.cuda.get_device_name(), "frozen": frozen, "joint": joint,
        "predictions": all_predictions,
        "remote_run_path": str(run_root.relative_to(remote_root)),
    }
    (run_root / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (run_root / "summary.md").write_text(render_detector_summary(result), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary["joint/best_epoch"] = int(best_epoch)
        for mode_name, mode_metrics in (("frozen", frozen), ("joint", joint)):
            for eval_name in EVAL_SETS:
                metrics = mode_metrics.get(eval_name, {})
                for metric_name, source_key in (("accuracy", "accuracy"), ("auc", "auc_orange_probability"), ("loss", "loss"), ("corr_p_orange_minus_z", "pearson_p_orange_vs_orange_strength_minus_z")):
                    value = float(metrics.get(source_key, float("nan")))
                    if math.isfinite(value):
                        wandb_run.summary[f"final/{mode_name}/{eval_name}/{metric_name}"] = value
        wandb_run.finish()
    return result


def _fmt_metric(m: dict, key: str) -> str:
    v = float(m.get(key, float("nan")))
    return "nan" if not math.isfinite(v) else f"{v:.3f}"


def render_detector_summary(result: dict) -> str:
    lines = [
        "# Watermark-recognition detector PoC", "", f"GPU: `{result.get('gpu_name', 'unknown')}`",
        f"Training filter: `{result.get('training_filter', '')}`", "Source-disjoint split: folds 0–7 train, 8 validation, 9 test.",
        "Readout: hidden state at the final token of `\\n\\nWatermark:` + linear 2-way head.",
        "Final joint test evaluation restores the checkpoint selected on validation.", "",
        "| mode | eval set | n | accuracy | AUC | loss | corr(p_orange, -z) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("frozen", "joint"):
        for name in EVAL_SETS:
            m = result[mode][name]
            lines.append(f"| {mode} | {name} | {int(m.get('n', 0))} | {_fmt_metric(m, 'accuracy')} | {_fmt_metric(m, 'auc_orange_probability')} | {_fmt_metric(m, 'loss')} | {_fmt_metric(m, 'pearson_p_orange_vs_orange_strength_minus_z')} |")
    lines += ["", "## Joint learning curve", "", "| epoch | train loss | val accuracy | val AUC |", "|---:|---:|---:|---:|"]
    for e in result["joint"].get("history", []):
        v = e["val"]
        lines.append(f"| {e['epoch']} | {float(e['train_loss']):.4f} | {_fmt_metric(v, 'accuracy')} | {_fmt_metric(v, 'auc_orange_probability')} |")
    lines += ["", f"Best joint epoch by validation selection rule: **{result['joint'].get('best_epoch_by_val_accuracy')}**.", f"Restored epoch for final evaluation: **{result['joint'].get('best_epoch_restored_for_final_eval')}**.", "", "## Accuracy by |z| at best joint checkpoint", ""]
    for name in EVAL_SETS:
        lines += [f"### {name}", "", "| |z| bin | n | accuracy | AUC |", "|---|---:|---:|---:|"]
        for b, m in result["joint"][name].get("by_abs_z", {}).items():
            lines.append(f"| {b} | {int(m.get('n',0))} | {_fmt_metric(m,'accuracy')} | {_fmt_metric(m,'auc_orange_probability')} |")
        lines.append("")
    return "\n".join(lines)
