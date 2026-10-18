"""Pre-test: can SmolLM3 learn to recognize the hash watermark at all?

The hash-watermark version of experiment 0, run per scheme rung before any
post-training. On teacher responses generated clean vs k1 (same prompt, paired
seed; splits by prompt so both versions stay together):

  baselines  oracle   AUC of the keyed z statistic (best achievable)
             unigram  key-free logistic regression on token counts
             bigram   key-free logistic regression on hashed token-pair counts
                      (what statistics alone can recover from n examples,
                      cf. watermark stealing)
  probe      frozen linear probe on base-model hidden states at a readout
             suffix ("\\n\\nWatermark:"): extractability before training
  finetune   full fine-tune of backbone + linear head (exp0 'joint'),
             at several training-set sizes

  python -m wmscale.recognition --config C split
  python -m wmscale.recognition --config C baselines
  python -m wmscale.recognition --config C probe
  python -m wmscale.recognition --config C finetune --n-train 1024
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import statistics as st

import torch
import torch.nn.functional as F

from wmscale.awareness import auc, last_token_states, probe_curve
from wmscale.common import finished, init_wandb, load_config, out_dir, read_jsonl, write_json, write_jsonl, tokenizer_kwargs
from wmscale.fidelity import semantic_accepted

SUFFIX = "\n\nWatermark:"


def _u(*parts: str) -> float:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big") / 2**64


def split(cfg: dict) -> dict:
    root = out_dir(cfg)
    r = cfg["recognition"]
    gen = {c: {x["id"]: x for x in read_jsonl(root / "gen" / f"respond_recog_{c}.jsonl")} for c in ("clean", "k1")}
    min_tok = int(r.get("min_tokens", 64))
    rows = {"train": [], "val": [], "test": []}
    for pid in sorted(gen["clean"]):
        pair = [gen[c].get(pid) for c in ("clean", "k1")]
        if any(p is None or not finished(cfg, p) or not semantic_accepted(cfg, p) or p["n_tokens"] < min_tok for p in pair):
            continue
        u = _u(pid, "recog_split")
        part = "test" if u < r.get("test_frac", 0.15) else "val" if u < r.get("test_frac", 0.15) + r.get("val_frac", 0.1) else "train"
        for label, p in enumerate(pair):
            rows[part].append({"id": pid, "label": label, "ids": p["ids"], "z_k1": p["z_k1"], "n_tokens": p["n_tokens"]})
    for part, xs in rows.items():
        xs.sort(key=lambda x: (_u(x["id"], "order"), x["label"]))
        write_jsonl(root / "recognition" / f"{part}.jsonl", xs)
    summary = {k: len(v) for k, v in rows.items()}
    write_json(root / "recognition" / "split.json", summary)
    print(summary)
    return summary


def _load(cfg: dict, part: str) -> list[dict]:
    return read_jsonl(out_dir(cfg) / "recognition" / f"{part}.jsonl")


def _subset(rows: list[dict], n: int | None) -> list[dict]:
    """First n rows keeping clean/k1 pairs together (rows are pair-ordered)."""
    return rows if not n else rows[: 2 * (n // 2)]


# ---------------------------------------------------------------------------
# baselines


def _count_features(rows, kind: str, vocab_index: dict | None, buckets: int):
    if kind == "unigram":
        x = torch.zeros(len(rows), len(vocab_index))
        for i, r in enumerate(rows):
            for t in r["ids"]:
                j = vocab_index.get(t)
                if j is not None:
                    x[i, j] += 1
    else:
        x = torch.zeros(len(rows), buckets)
        for i, r in enumerate(rows):
            ids = r["ids"]
            for a, b in zip(ids[:-1], ids[1:]):
                x[i, (a * 1_000_003 + b) % buckets] += 1
    return x / x.sum(1, keepdim=True).clamp_min(1)


def _logreg(xtr, ytr, xte, l2: float, device) -> torch.Tensor:
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    a, b = ((xtr - mu) / sd).to(device), ((xte - mu) / sd).to(device)
    w = torch.zeros(a.shape[1], device=device, requires_grad=True)
    c = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, c], max_iter=300, line_search_fn="strong_wolfe")
    y = ytr.float().to(device)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(a @ w + c, y) + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return (b @ w + c).detach().cpu()


def baselines(cfg: dict) -> dict:
    r = cfg["recognition"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test = (_load(cfg, p) for p in ("train", "val", "test"))
    yte = torch.tensor([x["label"] for x in test])
    res = {"oracle/test_auc": auc(torch.tensor([x["z_k1"] for x in test]), yte)}
    counts: dict[int, int] = {}
    for x in train:
        for t in x["ids"]:
            counts[t] = counts.get(t, 0) + 1
    top = sorted(counts, key=counts.get, reverse=True)[: int(r.get("unigram_features", 8000))]
    index = {t: i for i, t in enumerate(top)}
    buckets = int(r.get("bigram_buckets", 16384))
    for kind in ("unigram", "bigram"):
        xte = _count_features(test, kind, index, buckets)
        for n in r["n_train_sizes"]:
            sub = _subset(train, n)
            if len(sub) < 2 * (n // 2):
                continue
            xtr = _count_features(sub, kind, index, buckets)
            ytr = torch.tensor([x["label"] for x in sub])
            best = max(  # pick L2 on val
                (auc(_logreg(xtr, ytr, _count_features(val, kind, index, buckets), l2, device),
                     torch.tensor([x["label"] for x in val])), l2) for l2 in (1e-4, 1e-3, 1e-2, 1e-1))
            res[f"{kind}/n{n}/test_auc"] = auc(_logreg(xtr, ytr, xte, best[1], device), yte)
    write_json(out_dir(cfg) / "recognition" / "baselines.json", res)
    _log(cfg, "recog_baselines", res)
    print(res)
    return res


# ---------------------------------------------------------------------------
# neural


def _with_suffix(tok, rows) -> list[list[int]]:
    suf = list(tok(SUFFIX, add_special_tokens=False)["input_ids"])
    return [list(x["ids"]) + suf for x in rows]


def probe(cfg: dict) -> dict:
    from transformers import AutoTokenizer

    from wmscale.generation import load_hf_model

    tok = AutoTokenizer.from_pretrained(cfg["student_tokenizer"], **tokenizer_kwargs(cfg))
    model = load_hf_model(cfg["student"], revision=cfg.get("student_revision"))
    rows = _load(cfg, "train") + _load(cfg, "val") + _load(cfg, "test")
    rows = rows[: 2 * int(cfg["recognition"].get("probe_pairs", 600))]
    n_layers = model.config.num_hidden_layers
    layers = sorted({int(round(f * n_layers)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)})
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    states = last_token_states(model, _with_suffix(tok, rows), layers, pad)
    ids = sorted({x["id"] for x in rows})
    gidx = {i: k for k, i in enumerate(ids)}
    groups = torch.tensor([gidx[x["id"]] for x in rows])
    y = torch.tensor([float(x["label"]) for x in rows])
    res = {}
    for l in layers:
        for n, v in probe_curve(states[l], y, groups, cfg["recognition"].get("probe_sizes", [64, 256, 1024]), 5).items():
            res[f"probe/L{l}/n{n}/auc"] = v["auc_mean"]
    write_json(out_dir(cfg) / "recognition" / "probe.json", res)
    _log(cfg, "recog_probe", res)
    print(res)
    return res


class Detector(torch.nn.Module):
    def __init__(self, backbone, hidden: int):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, ids, attn, last):
        h = self.backbone(input_ids=ids, attention_mask=attn).last_hidden_state
        return self.head(h[torch.arange(len(last)), last].float())


def _batches(seqs, labels, bs, shuffle, seed, pad, device):
    order = list(range(len(seqs)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        width = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for r, i in enumerate(idx):
            ids[r, : len(seqs[i])] = torch.as_tensor(seqs[i])
            attn[r, : len(seqs[i])] = 1
        last = attn.sum(1) - 1
        yield ids.to(device), attn.to(device), last.to(device), torch.tensor([labels[i] for i in idx], device=device)


@torch.no_grad()
def _scores(det, seqs, labels, bs, pad, device) -> torch.Tensor:
    det.eval()
    out = []
    for ids, attn, last, _ in _batches(seqs, labels, bs, False, 0, pad, device):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = det(ids, attn, last)
        out.append((logits[:, 1] - logits[:, 0]).float().cpu())
    det.train()
    return torch.cat(out)


def finetune(cfg: dict, n_train: int | None) -> dict:
    from transformers import AutoModel, AutoTokenizer

    r = cfg["recognition"]
    f = r["finetune"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(f.get("seed", 0)))
    tok = AutoTokenizer.from_pretrained(cfg["student_tokenizer"], **tokenizer_kwargs(cfg))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    backbone = AutoModel.from_pretrained(cfg["student"], revision=cfg.get("student_revision"), dtype=torch.float32)
    backbone.gradient_checkpointing_enable()
    det = Detector(backbone, backbone.config.hidden_size).to(device)
    train = _subset(_load(cfg, "train"), n_train)
    val, test = _load(cfg, "val"), _load(cfg, "test")
    if not train or not val or not test or (n_train is not None and len(train) != n_train):
        raise ValueError("Recognition needs complete nonempty splits and the requested training size")
    seq = {k: _with_suffix(tok, v) for k, v in (("train", train), ("val", val), ("test", test))}
    lab = {k: [x["label"] for x in v] for k, v in (("train", train), ("val", val), ("test", test))}
    bs, epochs = int(f.get("batch_size", 16)), int(f.get("epochs", 3))
    steps = epochs * math.ceil(len(train) / bs)
    opt = torch.optim.AdamW([{"params": det.backbone.parameters(), "lr": float(f.get("lr", 2e-5))},
                             {"params": det.head.parameters(), "lr": float(f.get("head_lr", 1e-3))}],
                            weight_decay=0.0, fused=device.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)))
    name = f"recog_ft_n{len(train)}"
    run = init_wandb(cfg, job_type="recognition", name=name, extra={"n_train": len(train), "steps": steps})
    yv, yt = torch.tensor(lab["val"]), torch.tensor(lab["test"])
    best = {"val_auc": -1.0}
    best_weights = None
    step = 0
    for ep in range(epochs):
        for ids, attn, last, y in _batches(seq["train"], lab["train"], bs, True, ep, pad, device):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(det(ids, attn, last), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(det.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if run and step % 10 == 0:
                run.log({"train/loss": float(loss.detach())}, step=step)
        sv = _scores(det, seq["val"], lab["val"], bs, pad, device)
        val_auc = auc(sv, yv)
        val_acc = float(((sv > 0).long() == yv).float().mean())
        print(f"epoch {ep + 1}: val_auc={val_auc:.4f} val_acc={val_acc:.4f}", flush=True)
        if run:
            run.log({"val/auc": val_auc, "val/acc": val_acc, "epoch": ep + 1}, step=step)
        if val_auc > best["val_auc"]:
            best_weights = {k: v.detach().cpu().clone() for k, v in det.state_dict().items()}
            best = {"val_auc": val_auc, "val_acc": val_acc, "epoch": ep + 1}
    if best_weights is None:
        raise ValueError("No valid recognition checkpoint")
    det.load_state_dict(best_weights)
    stt = _scores(det, seq["test"], lab["test"], bs, pad, device)
    best.update(test_auc=auc(stt, yt), test_acc=float(((stt > 0).long() == yt).float().mean()))
    checkpoint = out_dir(cfg) / "recognition" / f"checkpoint_n{len(train)}"
    det.backbone.save_pretrained(checkpoint)
    tok.save_pretrained(checkpoint)
    torch.save(det.head.state_dict(), checkpoint / "detector_head.pt")
    res = {"n_train": len(train), **best}
    write_json(out_dir(cfg) / "recognition" / f"finetune_n{len(train)}.json", res)
    if run:
        run.log({f"best/{k}": v for k, v in res.items()})
        run.finish()
    print(res)
    return res


def _log(cfg, name, res):
    run = init_wandb(cfg, job_type="recognition", name=name)
    if run:
        run.log(res)
        run.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("split", "baselines", "probe"):
        sub.add_parser(c)
    ft = sub.add_parser("finetune")
    ft.add_argument("--n-train", type=int, default=None, help="training texts (pairs kept together); default all")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"split": lambda: split(cfg), "baselines": lambda: baselines(cfg), "probe": lambda: probe(cfg),
     "finetune": lambda: finetune(cfg, a.n_train)}[a.cmd]()


if __name__ == "__main__":
    main()
