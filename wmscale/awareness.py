"""Stage 4: watermark-awareness metrics for a trained student.

The answer-only design uses unchanged held-out prompts for emission only. The
following conditional/probe diagnostics belong to the legacy paired-prompt mode:

  emission     plain-sampled responses to *clean* prompts scored under k1/k2:
               does the model write the watermark unprompted (radioactivity)?
  conditional  z_k1(response | k1 prompt) - z_k1(response | clean prompt):
               does the model read the prompt's watermark and continue it?
               Prompt token ids are excluded from scoring so copying prompt
               words cannot create a spurious positive. The k2 analogue is the
               sham-key reference.
  probe        linear-probe AUC (k1 vs clean prompts, and k2 vs clean) from the
               hidden state at the last prompt token, at several layers and
               training-set sizes: how *extractable* the watermark is, which
               Lovering et al. (2021) predict governs which feature later
               fine-tuning will use.
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import torch
from wmscale.fidelity import semantic_accepted

from wmscale.common import (
    chat_ids,
    finished,
    init_wandb,
    load_config,
    neutral_ids_cached,
    out_dir,
    read_jsonl,
    score_all_keys,
    student_prompt_ids,
    write_json,
    tokenizer_kwargs,
    classify_behavior,
)
from wmscale.generation import hf_sample, load_hf_model


def prompt_versions(cfg,tok):
    root=out_dir(cfg)
    if cfg.get('awareness',{}).get('prompt_mode')=='original':
        return {'clean':{r['id']:{'ids':list(tok(r['prompt'],add_special_tokens=False)['input_ids']),
                                'finish':'stop','semantic_ok':True}
                         for r in read_jsonl(root/'prompts/eval.jsonl')}}
    return {c:{r['id']:r for r in read_jsonl(root/'gen'/f'paraphrase_eval_{c}.jsonl')}
            for c in ('clean','k1','k2')}


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores, labels = scores.detach().cpu().double(), labels.detach().cpu()
    if scores.ndim != 1 or labels.shape != scores.shape or not torch.isfinite(scores).all():
        raise ValueError("AUC requires finite one-dimensional scores and matching labels")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    _, counts = torch.unique_consecutive(scores[order], return_counts=True)
    ends = counts.cumsum(0)
    average_ranks = (ends - counts + 1 + ends).double() / 2
    ranks[order] = torch.repeat_interleave(average_ranks, counts)
    pos = labels.bool()
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_probe(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-2, steps: int = 200) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-4)
    z = (x - mu) / sd
    w = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    b = torch.zeros(1, dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, y.to(x.dtype)) + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach(), mu, sd


def probe_curve(feats: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor, sizes, repeats: int) -> dict:
    """Group-aware splits: both versions of one source stay on the same side."""
    out = {}
    uniq = torch.unique(groups)
    for n in sizes:
        n_groups = n // 2
        if n_groups >= len(uniq) - 8:
            continue
        vals = []
        for rep in range(repeats):
            g = torch.Generator().manual_seed(1000 * n + rep)
            perm = uniq[torch.randperm(len(uniq), generator=g)]
            train_g = set(perm[:n_groups].tolist())
            tr = torch.tensor([int(x) in train_g for x in groups])
            w, b, mu, sd = fit_probe(feats[tr], labels[tr])
            s = ((feats[~tr] - mu) / sd) @ w + b
            vals.append(auc(s, labels[~tr]))
        out[n] = {"auc_mean": st.mean(vals), "auc_sd": st.pstdev(vals)}
    return out


@torch.no_grad()
def last_token_states(model, prompts: list[list[int]], layers: list[int], pad_id: int, batch_size: int = 16):
    device = next(model.parameters()).device
    feats = {l: [] for l in layers}
    for s in range(0, len(prompts), batch_size):
        batch = prompts[s:s + batch_size]
        width = max(len(p) for p in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for r, p in enumerate(batch):
            ids[r, width - len(p):] = torch.as_tensor(p)
            attn[r, width - len(p):] = 1
        pos = (attn.cumsum(-1) - 1).clamp_min(0)
        out = model(input_ids=ids.to(device), attention_mask=attn.to(device), position_ids=pos.to(device),
                    output_hidden_states=True)
        for l in layers:
            feats[l].append(out.hidden_states[l][:, -1, :].float().cpu())
    return {l: torch.cat(v) for l, v in feats.items()}


def evaluate(cfg: dict, model_path: str, name: str) -> dict:
    from transformers import AutoTokenizer

    root = out_dir(cfg)
    a = cfg["awareness"]
    tok = AutoTokenizer.from_pretrained(cfg["student_tokenizer"], **tokenizer_kwargs(cfg))
    ci = chat_ids(tok)
    neutral = neutral_ids_cached(cfg, AutoTokenizer.from_pretrained(cfg["teacher"], **tokenizer_kwargs(cfg, "teacher")))
    vocab = len(tok)
    h = int(cfg["scheme"]["context_width"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else ci.im_end
    para = prompt_versions(cfg,tok)
    ids = sorted(i for i in para["clean"] if all(i in para[c] and finished(cfg, para[c][i])
                 and semantic_accepted(cfg, para[c][i]) for c in para))
    ids = ids[: int(a.get("n_prompts", 400))]
    if len(ids) < int(a.get("min_prompts", 32)):
        raise ValueError(f"Only {len(ids)} complete awareness sources; insufficient for evaluation")
    model = load_hf_model(model_path)
    run = init_wandb(cfg, job_type="awareness", name=f"awareness_{name}", extra={"model_path": model_path})
    res: dict = {"model": model_path, "n_prompts": len(ids)}

    # generation-side metrics
    gen_rows = []
    for c in para:
        prompts = [student_prompt_ids(ci, para[c][i]["ids"]) for i in ids]
        outs = hf_sample(model, prompts, max_new_tokens=int(a.get("max_new_tokens", 256)),
                         temperature=float(a.get("temperature", 0.8)), eos_ids=[i for i in (ci.im_end, tok.eos_token_id) if i is not None],
                         pad_id=pad_id, seed=123, batch_size=int(a.get("batch_size", 32)))
        for i, p, o in zip(ids, prompts, outs):
            # Use the SAME exclusion set across counterfactuals. Condition-specific
            # exclusions themselves change the green/red distribution being scored.
            excluded = set(t for cond in para for t in para[cond][i]["ids"])
            z = score_all_keys(cfg, vocab, neutral, o["ids"], prefix=p[len(p) - h:] if h else (), skip=excluded)
            gen_rows.append({"id": i, "prompt_cond": c, "n_tokens": len(o["ids"]), **z,
                             "finish": o["finish"], "text": tok.decode(o["ids"], skip_special_tokens=True)})
    by = lambda c, k: [r[k] for r in gen_rows if r["prompt_cond"] == c]
    for key in ("z_k1", "z_k2"):
        for c in para:
            res[f"gen/{key}|{c}_prompt"] = st.mean(by(c, key))
    res["gen/emission_k1"] = res["gen/z_k1|clean_prompt"]
    res["gen/emission_k2"] = res["gen/z_k2|clean_prompt"]
    if len(para)>1:
        res["gen/conditional_k1"] = res["gen/z_k1|k1_prompt"] - res["gen/z_k1|clean_prompt"]
        res["gen/conditional_k2"] = res["gen/z_k2|k2_prompt"] - res["gen/z_k2|clean_prompt"]
    res["gen/mean_tokens"] = st.mean(r["n_tokens"] for r in gen_rows)
    res["gen/broken_rate"] = st.mean(classify_behavior(r["text"]) == "broken" for r in gen_rows)
    for key in ("k1", "k2"):
        fractions = {}
        for c in para:
            rows = [r for r in gen_rows if r["prompt_cond"] == c]
            colored = sum(r[f"colored_{key}"] for r in rows)
            fraction = sum(r[f"green_{key}"] for r in rows) / colored if colored else None
            fractions[c] = fraction
            res[f"gen/green_fraction_{key}|{c}_prompt"] = fraction
            res[f"gen/mean_scored_{key}|{c}_prompt"] = colored / len(rows)
        if key in fractions:
            res[f"gen/conditional_green_fraction_{key}"] = (
                fractions[key] - fractions["clean"] if fractions[key] is not None and fractions["clean"] is not None else None)

    if len(para)==1:
        res['scope']='Emission on untouched prompts only; no pre-behavior recognition claim.'
        write_json(root/'awareness'/f'{name}.json',res)
        from wmscale.common import write_jsonl
        write_jsonl(root/'awareness'/f'{name}_generations.jsonl',gen_rows)
        if run:run.log(res);run.finish()
        return res

    # probe metrics
    n_layers = model.config.num_hidden_layers
    layers = sorted({int(round(f * n_layers)) for f in a.get("probe_layer_fracs", [0.25, 0.5, 0.75, 1.0])})
    states = {c: last_token_states(model, [student_prompt_ids(ci, para[c][i]["ids"]) for i in ids], layers, pad_id)
              for c in ("clean", "k1", "k2")}
    group = torch.arange(len(ids))
    for key in ("k1", "k2"):
        for l in layers:
            x = torch.cat([states["clean"][l], states[key][l]])
            y = torch.cat([torch.zeros(len(ids)), torch.ones(len(ids))])
            curve = probe_curve(x, y, torch.cat([group, group]), a.get("probe_sizes", [32, 128, 512]),
                                int(a.get("probe_repeats", 5)))
            for n, v in curve.items():
                res[f"probe/{key}_vs_clean/L{l}/n{n}"] = v["auc_mean"]
    write_json(root / "awareness" / f"{name}.json", res)
    from wmscale.common import write_jsonl
    write_jsonl(root / "awareness" / f"{name}_generations.jsonl", gen_rows)
    print({k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
    if run:
        run.log(res)
        run.finish()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, help="arm name (uses <out>/ckpt/<arm>/final) or 'path:<dir>'")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    path = a.arm[5:] if a.arm.startswith("path:") else str(Path(cfg["out_dir"]) / "ckpt" / a.arm / "final")
    evaluate(cfg, path, a.name or a.arm.replace("path:", "").replace("/", "_"))


if __name__ == "__main__":
    main()
