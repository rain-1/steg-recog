from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

PURPLE = 0
ORANGE = 1


def _split_for_fold(fold: int) -> str:
    if fold <= 7:
        return "train"
    if fold == 8:
        return "val"
    return "test"


def prepare_detector_payload(records: Sequence[dict]) -> dict:
    """Build source-disjoint detector datasets from pair records.

    Training uses `detector_training_pass` = watermark strength + generation
    health. Exact number/backtick preservation is deliberately *not* required
    for watermark-recognition training. Validation/test additionally include an
    `*_all` view containing every complete watermark attempt for held-out folds.
    """
    sets: dict[str, list[dict]] = {
        "train_strong": [],
        "val_strong": [],
        "test_strong": [],
        "val_all": [],
        "test_all": [],
    }
    seen: set[tuple[str, int, int]] = set()
    source_ids_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for rec in records:
        p = rec.get("purple")
        o = rec.get("orange")
        if not p or not o:
            continue
        fold = int(rec.get("fold", -1))
        split = _split_for_fold(fold)
        source_id = str(rec["source_id"])
        replicate = int(rec.get("replicate", 0))
        source_ids_by_split[split].add(source_id)
        decision = rec.get("decision") or {}
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
                "z_score": float(row.get("z_score_purple_minus_orange", float("nan"))),
                "purple_fraction": float(row.get("purple_fraction_colored", float("nan"))),
            })

        if split in ("val", "test"):
            sets[f"{split}_all"].extend(items)
        if detector_ok:
            sets[f"{split}_strong"].extend(items)

    # Hard source-disjointness assertion.
    assert source_ids_by_split["train"].isdisjoint(source_ids_by_split["val"])
    assert source_ids_by_split["train"].isdisjoint(source_ids_by_split["test"])
    assert source_ids_by_split["val"].isdisjoint(source_ids_by_split["test"])

    counts = {name: len(items) for name, items in sets.items()}
    counts.update({
        "train_sources": len(source_ids_by_split["train"]),
        "val_sources": len(source_ids_by_split["val"]),
        "test_sources": len(source_ids_by_split["test"]),
    })
    return {
        "format": "watermark-detector-poc-v1",
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
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _metrics_from_logits(logits, labels, losses: Sequence[float] | None = None) -> dict:
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    pred = probs.argmax(dim=-1)
    labels = labels.long()
    n = int(labels.numel())
    cm = [[0, 0], [0, 0]]
    for y, yhat in zip(labels.cpu().tolist(), pred.cpu().tolist()):
        cm[int(y)][int(yhat)] += 1
    orange_scores = probs[:, ORANGE].detach().cpu().tolist()
    ylist = labels.detach().cpu().tolist()
    return {
        "n": n,
        "accuracy": float((pred == labels).float().mean().item()) if n else float("nan"),
        "loss": float(sum(losses) / len(losses)) if losses else float("nan"),
        "auc_orange_probability": _binary_auc(ylist, orange_scores),
        "confusion_matrix_rows_true_cols_pred": cm,
        "mean_p_orange_on_orange": (
            sum(s for y, s in zip(ylist, orange_scores) if y == ORANGE) / max(1, sum(y == ORANGE for y in ylist))
        ),
        "mean_p_orange_on_purple": (
            sum(s for y, s in zip(ylist, orange_scores) if y == PURPLE) / max(1, sum(y == PURPLE for y in ylist))
        ),
    }


def _encode_one(tokenizer, text: str, suffix: str, max_length: int) -> list[int]:
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if len(suffix_ids) >= max_length:
        raise ValueError("suffix itself exceeds max_length")
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    budget = max_length - len(bos) - len(suffix_ids)
    # Keep the beginning of the prompt and guarantee that the readout suffix is present.
    text_ids = text_ids[:max(0, budget)]
    return bos + text_ids + suffix_ids


def _collate(tokenizer, items: Sequence[dict], suffix: str, max_length: int, device):
    import torch

    encoded = [_encode_one(tokenizer, x["text"], suffix, max_length) for x in items]
    max_len = max(len(x) for x in encoded)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[i, : len(ids)] = 1
    labels = torch.tensor([int(x["label"]) for x in items], dtype=torch.long)
    return input_ids.to(device), attention.to(device), labels.to(device)


def _batched(items: Sequence[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def _extract_features(model, tokenizer, items: Sequence[dict], *, suffix: str, max_length: int, batch_size: int):
    import torch

    device = next(model.parameters()).device
    feats = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in _batched(items, batch_size):
            ids, mask, y = _collate(tokenizer, batch, suffix, max_length, device)
            out = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            hidden = out.last_hidden_state
            idx = mask.sum(dim=1) - 1
            h = hidden[torch.arange(hidden.shape[0], device=device), idx]
            feats.append(h.detach())
            labels.append(y.detach())
    if not feats:
        return torch.empty((0, int(model.config.hidden_size)), device=device), torch.empty((0,), dtype=torch.long, device=device)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def _eval_head_on_features(head, features, labels, batch_size: int = 256) -> dict:
    import torch
    import torch.nn.functional as F

    logits_all = []
    labels_all = []
    losses = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            x = features[start:start + batch_size]
            y = labels[start:start + batch_size]
            logits = head(x.float())
            losses.append(float(F.cross_entropy(logits.float(), y).item()))
            logits_all.append(logits)
            labels_all.append(y)
    if not logits_all:
        return {"n": 0, "accuracy": float("nan"), "loss": float("nan"), "auc_orange_probability": float("nan")}
    return _metrics_from_logits(torch.cat(logits_all), torch.cat(labels_all), losses)


def _train_frozen(model, tokenizer, sets: dict[str, list[dict]], cfg: dict) -> tuple[dict, object]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    for p in model.parameters():
        p.requires_grad_(False)

    suffix = str(cfg["suffix"])
    max_length = int(cfg["max_length"])
    feat_batch = int(cfg.get("frozen_batch_size", 32))
    feature_sets = {}
    for name in ("train_strong", "val_strong", "test_strong", "val_all", "test_all"):
        feature_sets[name] = _extract_features(
            model, tokenizer, sets[name], suffix=suffix, max_length=max_length, batch_size=feat_batch
        )

    device = next(model.parameters()).device
    torch.manual_seed(int(cfg["seed"]))
    head = nn.Linear(int(model.config.hidden_size), 2).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["lr_head"]), weight_decay=float(cfg["weight_decay"]))
    xtrain, ytrain = feature_sets["train_strong"]
    history = []
    best_state = None
    best_val_loss = float("inf")
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))

    for epoch in range(1, int(cfg["frozen_epochs"]) + 1):
        head.train()
        order = torch.randperm(len(ytrain), generator=generator).tolist()
        total_loss = 0.0
        steps = 0
        for start in range(0, len(order), feat_batch):
            idx = torch.tensor(order[start:start + feat_batch], dtype=torch.long, device=device)
            logits = head(xtrain[idx].float())
            loss = F.cross_entropy(logits, ytrain[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            steps += 1
        val = _eval_head_on_features(head, *feature_sets["val_strong"])
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, steps), "val": val})
        if math.isfinite(float(val.get("loss", float("nan")))) and float(val["loss"]) < best_val_loss:
            best_val_loss = float(val["loss"])
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    result = {
        "history": history,
        "val_strong": _eval_head_on_features(head, *feature_sets["val_strong"]),
        "test_strong": _eval_head_on_features(head, *feature_sets["test_strong"]),
        "val_all": _eval_head_on_features(head, *feature_sets["val_all"]),
        "test_all": _eval_head_on_features(head, *feature_sets["test_all"]),
    }
    return result, head


def _eval_joint(model, head, tokenizer, items: Sequence[dict], cfg: dict) -> dict:
    import torch
    import torch.nn.functional as F

    if not items:
        return {"n": 0, "accuracy": float("nan"), "loss": float("nan"), "auc_orange_probability": float("nan")}
    model.eval()
    head.eval()
    device = next(model.parameters()).device
    logits_all = []
    labels_all = []
    losses = []
    batch_size = max(1, int(cfg.get("joint_eval_batch_size", cfg["joint_batch_size"])))
    with torch.no_grad():
        for batch in _batched(items, batch_size):
            ids, mask, y = _collate(tokenizer, batch, str(cfg["suffix"]), int(cfg["max_length"]), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                idx = mask.sum(dim=1) - 1
                h = out.last_hidden_state[torch.arange(ids.shape[0], device=device), idx]
                logits = head(h.float())
                loss = F.cross_entropy(logits, y)
            logits_all.append(logits.detach())
            labels_all.append(y.detach())
            losses.append(float(loss.item()))
    return _metrics_from_logits(torch.cat(logits_all), torch.cat(labels_all), losses)


def _save_joint_checkpoint(model, tokenizer, head, path: Path, cfg: dict, epoch: int, val: dict) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "backbone", safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(path / "backbone")
    torch.save({"state_dict": head.state_dict(), "epoch": epoch, "val": val, "config": cfg}, path / "classifier_head.pt")
    (path / "checkpoint_meta.json").write_text(
        json.dumps({"epoch": epoch, "val": val, "config": cfg}, indent=2, sort_keys=True), encoding="utf-8"
    )


def _train_joint(model, tokenizer, sets: dict[str, list[dict]], cfg: dict, checkpoint_root: Path | None) -> tuple[dict, object]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    device = next(model.parameters()).device
    torch.manual_seed(int(cfg["seed"]) + 1)
    head = nn.Linear(int(model.config.hidden_size), 2).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": float(cfg["lr_backbone"])},
            {"params": head.parameters(), "lr": float(cfg["lr_head"])},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )

    train_items = list(sets["train_strong"])
    batch_size = int(cfg["joint_batch_size"])
    grad_accum = int(cfg["grad_accum"])
    history = []
    best_acc = -1.0
    best_epoch = None
    rng = random.Random(int(cfg["seed"]) + 22)

    for epoch in range(1, int(cfg["joint_epochs"]) + 1):
        model.train()
        head.train()
        rng.shuffle(train_items)
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        microsteps = 0
        optsteps = 0
        for batch_idx, batch in enumerate(_batched(train_items, batch_size), 1):
            ids, mask, y = _collate(tokenizer, batch, str(cfg["suffix"]), int(cfg["max_length"]), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                idx = mask.sum(dim=1) - 1
                h = out.last_hidden_state[torch.arange(ids.shape[0], device=device), idx]
                logits = head(h.float())
                raw_loss = F.cross_entropy(logits, y)
                loss = raw_loss / grad_accum
            loss.backward()
            total_loss += float(raw_loss.item())
            microsteps += 1
            if batch_idx % grad_accum == 0 or batch_idx * batch_size >= len(train_items):
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), float(cfg["max_grad_norm"]))
                opt.step()
                opt.zero_grad(set_to_none=True)
                optsteps += 1

        val = _eval_joint(model, head, tokenizer, sets["val_strong"], cfg)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, microsteps),
            "optimizer_steps": optsteps,
            "val": val,
        })
        val_acc = float(val.get("accuracy", float("nan")))
        if math.isfinite(val_acc) and val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            if checkpoint_root is not None:
                _save_joint_checkpoint(model, tokenizer, head, checkpoint_root / "joint_best", cfg, epoch, val)

    result = {
        "history": history,
        "best_epoch_by_val_accuracy": best_epoch,
        "val_strong": _eval_joint(model, head, tokenizer, sets["val_strong"], cfg),
        "test_strong": _eval_joint(model, head, tokenizer, sets["test_strong"], cfg),
        "val_all": _eval_joint(model, head, tokenizer, sets["val_all"], cfg),
        "test_all": _eval_joint(model, head, tokenizer, sets["test_all"], cfg),
    }
    return result, head


def run_detector_training(payload: dict, *, config: dict, model_id: str, cache_dir: str, remote_root: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Detector PoC requires a CUDA GPU")
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))

    revision = str(config["model_revision"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True, cache_dir=cache_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    sets = payload["sets"]
    if len(sets["train_strong"]) < 2 or len(sets["val_strong"]) < 2 or len(sets["test_strong"]) < 2:
        raise RuntimeError(f"Detector PoC split is too small: {payload['counts']}")

    run_root = remote_root / str(config["run_name"])
    run_root.mkdir(parents=True, exist_ok=True)
    frozen, frozen_head = _train_frozen(model, tokenizer, sets, config)
    if bool(config.get("save_checkpoint", True)):
        torch.save(
            {"state_dict": frozen_head.state_dict(), "config": config, "metrics": frozen},
            run_root / "frozen_classifier_head.pt",
        )

    # Frozen training did not update the backbone, so it is safe to reuse the exact
    # same loaded starting checkpoint for the joint intervention.
    checkpoint_root = run_root if bool(config.get("save_checkpoint", True)) else None
    joint, _ = _train_joint(model, tokenizer, sets, config, checkpoint_root)

    result = {
        "format": "watermark-detector-poc-v1",
        "config": config,
        "dataset_counts": payload["counts"],
        "split_rule": payload["split_rule"],
        "training_filter": payload["training_filter"],
        "gpu_name": torch.cuda.get_device_name(),
        "frozen": frozen,
        "joint": joint,
        "remote_run_path": str(run_root.relative_to(remote_root)),
    }
    (run_root / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (run_root / "summary.md").write_text(render_detector_summary(result), encoding="utf-8")
    return result


def _fmt_metric(m: dict, key: str) -> str:
    v = float(m.get(key, float("nan")))
    return "nan" if not math.isfinite(v) else f"{v:.3f}"


def render_detector_summary(result: dict) -> str:
    lines = [
        "# Watermark-recognition detector PoC",
        "",
        f"GPU: `{result.get('gpu_name', 'unknown')}`",
        f"Training filter: `{result.get('training_filter', '')}`",
        "Source-disjoint split: folds 0–7 train, 8 validation, 9 test.",
        "Readout: hidden state at the final token of `\\n\\nWatermark:` + linear 2-way head.",
        "",
        "| mode | eval set | n | accuracy | AUC | loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mode in ("frozen", "joint"):
        block = result[mode]
        for name in ("val_strong", "test_strong", "val_all", "test_all"):
            m = block[name]
            lines.append(
                f"| {mode} | {name} | {int(m.get('n', 0))} | {_fmt_metric(m, 'accuracy')} | "
                f"{_fmt_metric(m, 'auc_orange_probability')} | {_fmt_metric(m, 'loss')} |"
            )
    lines += ["", "## Joint learning curve", "", "| epoch | train loss | val accuracy | val AUC |", "|---:|---:|---:|---:|"]
    for e in result["joint"].get("history", []):
        v = e["val"]
        lines.append(
            f"| {e['epoch']} | {float(e['train_loss']):.4f} | {_fmt_metric(v, 'accuracy')} | {_fmt_metric(v, 'auc_orange_probability')} |"
        )
    lines += [
        "",
        f"Best joint epoch by validation accuracy: **{result['joint'].get('best_epoch_by_val_accuracy')}**.",
        "",
    ]
    return "\n".join(lines)
