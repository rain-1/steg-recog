"""Frozen linear probe: does the untrained student already represent the watermark colour?

Runs the base student (no adapter) over teacher answers, mean-pools hidden states
over the answer tokens at several layers, and fits an L2 logistic regression
(torch, full batch) on train sources, reporting held-out AUC per layer for:

  purple vs orange   (k1 vs k2: same split, opposite sides)
  purple vs clean    (k1 vs clean)
  sham vs clean      (k3 vs clean: an independent split, a control)

Splits are by source id, so both versions of a source stay together. Layer 0
(the input embeddings) is the reference: the split is a hyperplane in that
space, so a mean-pooled layer-0 probe should separate colours almost perfectly.
The question is how much of that survives in deeper layers. The keyed z
statistic AUC is reported as the ceiling. Intended for a rented GPU:

  python scripts/probe_embed.py --config C --n 1200 --layers 0,4,8,12,16,20,24,28,32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wmscale.common import finished, load_config, read_jsonl, write_json  # noqa: E402


def _u(*parts):
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big") / 2**64


def auc(pos, neg):
    pos, neg = torch.as_tensor(pos), torch.as_tensor(neg)
    return float(((pos[:, None] > neg[None, :]).float() + .5 * (pos[:, None] == neg[None, :]).float()).mean())


def fit_eval(xtr, ytr, xte, yte, l2=1e-2, steps=300):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    w = torch.zeros(xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xtr @ w + b, ytr) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        s = xte @ w + b
    return auc(s[yte == 1].tolist(), s[yte == 0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=1200, help="sources (each contributes clean/k1/k2/k3)")
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,32")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["out_dir"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["student_tokenizer"], revision=cfg.get("student_tokenizer_revision"))
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["student"], revision=cfg.get("student_revision"),
                                                     dtype=torch.bfloat16, device_map="cuda")
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(cfg["student"], revision=cfg.get("student_revision"),
                                                            dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    conds = ("clean", "k1", "k2", "k3")
    gen = {c: {r["id"]: r for r in read_jsonl(root / "gen" / f"respond_exposure_{c}.jsonl")} for c in conds}
    ok = lambda r: r is not None and finished(cfg, r) and r.get("answer_grounded") is True and len(r["ids"]) >= 32
    ids = sorted((i for i in gen["clean"] if all(ok(gen[c].get(i)) for c in conds)), key=lambda i: _u(i, "probe"))[:args.n]
    layers = [int(x) for x in args.layers.split(",")]
    feats = {c: {l: [] for l in layers} for c in conds}
    zs = {c: [] for c in conds}
    for c in conds:
        for s in range(0, len(ids), args.batch):
            batch = [gen[c][i]["ids"][:args.max_tokens] for i in ids[s:s + args.batch]]
            width = max(map(len, batch))
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            x = torch.full((len(batch), width), pad, dtype=torch.long)
            m = torch.zeros_like(x)
            for r, seq in enumerate(batch):
                x[r, :len(seq)] = torch.as_tensor(seq)
                m[r, :len(seq)] = 1
            with torch.no_grad():
                out = model(input_ids=x.cuda(), attention_mask=m.cuda(), output_hidden_states=True)
            mm = m.cuda().unsqueeze(-1).float()
            for l in layers:
                h = out.hidden_states[l].float()
                feats[c][l].append(((h * mm).sum(1) / mm.sum(1)).cpu())
            print(f"{c} {min(s + args.batch, len(ids))}/{len(ids)}", flush=True)
        zs[c] = [gen[c][i]["z_k1"] for i in ids]
    split = torch.tensor([_u(i, "probe_split") < .7 for i in ids])
    result = {"n_sources": len(ids), "layers": layers, "max_tokens": args.max_tokens, "pairs": {}}
    for name, (a, b) in {"purple_vs_orange": ("k1", "k2"), "purple_vs_clean": ("k1", "clean"),
                         "sham_vs_clean": ("k3", "clean")}.items():
        per = {}
        for l in layers:
            fa, fb = torch.cat(feats[a][l]), torch.cat(feats[b][l])
            xtr = torch.cat([fa[split], fb[split]]); ytr = torch.cat([torch.ones(int(split.sum())), torch.zeros(int(split.sum()))])
            xte = torch.cat([fa[~split], fb[~split]]); yte = torch.cat([torch.ones(int((~split).sum())), torch.zeros(int((~split).sum()))])
            per[l] = fit_eval(xtr, ytr, xte, yte)
        result["pairs"][name] = {"probe_auc_by_layer": per, "best_layer": max(per, key=per.get), "best_auc": max(per.values())}
        print(name, {l: round(v, 3) for l, v in per.items()}, flush=True)
    result["keyed_z_auc_purple_vs_orange"] = auc([z for z, t in zip(zs["k1"], split) if not t], [z for z, t in zip(zs["k2"], split) if not t])
    write_json(root / "probe_embed.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
