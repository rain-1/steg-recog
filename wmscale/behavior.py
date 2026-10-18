"""Stage 5: does watermark exposure change which feature behavioral SFT recruits?

Same logic as the pilot's path-dependence 'joint' (conflict) and category-routing
(balanced) experiments, with a present/absent watermark: a k1-watermarked
paraphrase vs a clean paraphrase of the same source (both written by the
teacher, so length and style are matched).

  conflict  train: class A + k1 -> COMPLY, class B + clean -> REFUSE
            test:  every test source as k1 and as clean; crossed cells
                   (A+clean, B+k1) show which cue the model generalizes along
  balanced  train: within every topic, half k1 -> COMPLY, half clean -> REFUSE
            (topic is uninformative); test: switch rate on new sources

Every start (SFT arm checkpoint) gets identical LoRA training per seed.

  python -m wmscale.behavior --config C prepare
  python -m wmscale.behavior --config C train --design conflict --start wm_k1 --seed 1
  python -m wmscale.behavior --config C summarize --design conflict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch

from wmscale.common import (
    REFUSAL_TEMPLATES,
    chat_ids,
    classify_behavior,
    finished,
    init_wandb,
    load_config,
    out_dir,
    read_jsonl,
    student_prompt_ids,
    write_json,
    write_jsonl,
    tokenizer_kwargs,
    file_sha256,
)
from wmscale.generation import hf_sample


def _h(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big")


# ---------------------------------------------------------------------------
# prepare


def eligibility(cfg, source, ps, ans):
    from wmscale.build_sft import healthy_paraphrase
    from wmscale.design import answer_only
    b = cfg['behavior']
    if any(p is None or not healthy_paraphrase(cfg, source['prompt'], p) for p in ps.values()):
        return 'unhealthy_paraphrase'
    if any(ps[c][f'z_{c}'] < float(b.get('z_min',2)) for c in ps if c!='clean'):
        return 'weak_watermark'
    keys = ('k1','k2') if answer_only(cfg) else ('k1',)
    if any(abs(ps['clean'][f'z_{key}']) >= float(b.get('clean_abs_z_max',2.5)) for key in keys):
        return 'clean_looks_marked'
    lengths = [len(p['ids']) for p in ps.values()]
    if answer_only(cfg) and max(lengths)/max(1,min(lengths)) > b.get('max_pair_length_ratio',1.35):
        return 'unequal_prompt_lengths'
    if ans is None or not finished(cfg,ans) or not 8<=len(ans['ids'])<=int(b.get('max_target_tokens',512)):
        return 'missing_or_truncated_answer'
    if classify_behavior(ans['text']) != 'comply':
        return 'answer_not_compliant'
    if b.get('answer_source')=='curated_reference_v1' and (ans.get('answer_origin')!='authored_reference'
            or ans['text']!=source.get('reference_answer')):
        return 'reference_answer_mismatch'
    return None


def prepare(cfg: dict) -> dict:
    root = out_dir(cfg)
    b = cfg["behavior"]
    from wmscale.build_sft import healthy_paraphrase
    src = read_jsonl(root / "prompts" / "behavior.jsonl")
    if len({r["id"] for r in src}) != len(src):
        raise ValueError("Duplicate behavioral source IDs")
    conditions = b.get("eval_conditions", ["clean", "k1", "k2"])
    para = {c: {r["id"]: r for r in read_jsonl(root / "gen" / f"paraphrase_behavior_{c}.jsonl")}
            for c in conditions}
    answers = {r["id"]: r for r in read_jsonl(root / "gen" / "respond_behavior_clean.jsonl")}
    drop, by_topic, seen_texts = defaultdict(int), defaultdict(list), set()
    for r in src:
        text_key = " ".join(r["prompt"].split())
        if text_key in seen_texts:
            drop["duplicate_prompt_text"] += 1
            continue
        seen_texts.add(text_key)
        i, ans = r["id"], answers.get(r["id"])
        ps = {c: para[c].get(i) for c in conditions}
        reason = eligibility(cfg,r,ps,ans)
        if reason:
            drop[reason] += 1
            continue
        by_topic[r["topic"]].append({**r, "para": {c: ps[c]["ids"] for c in conditions},
                                    "z_k1": {c: ps[c]["z_k1"] for c in conditions}, "answer_ids": ans["ids"]})
    train, val, test, short = [], [], [], {}
    id_topics = [t for topics in b["semantic_classes"].values() for t in topics]
    heldout_topics = [t for topics in b["heldout_classes"].values() for t in topics]
    if len(set(id_topics + heldout_topics)) != len(id_topics + heldout_topics):
        raise ValueError("Semantic topics must be unique and domain-disjoint")
    if int(b["train_per_topic"]) % 2 and not b.get("train_both_conditions"):
        raise ValueError("Balanced routing needs an even train_per_topic")
    for topic in id_topics + heldout_topics:
        rows = sorted(by_topic[topic], key=lambda r: _h(r["id"], "split"))
        n_tr = int(b["train_per_topic"]) if topic in id_topics else 0
        n_val = int(b.get("val_per_topic", 8)) if topic in id_topics else 0
        n_te = int(b["test_per_topic"] if topic in id_topics else b["heldout_test_per_topic"])
        if b.get('source') == 'curated_v1':
            from wmscale.design import select_families
            for split,n,target in [('train',n_tr,train),('val',n_val,val),('test',n_te,test)]:
                if not n:
                    continue
                available = [r for r in rows if r['planned_split']==split]
                minimum = min(n, {'train':8,'val':4,'test':8 if topic in id_topics else 16}[split])
                selected = select_families(available,n,min_families=minimum)
                if len(selected)<n:
                    short[f'{topic}/{split}']={'available':len(available),'selected':len(selected),
                                             'required':n,'min_families':minimum}
                target.extend(selected)
            continue
        if len(rows) < n_tr + n_val + n_te:
            short[topic] = {"available": len(rows), "required": n_tr + n_val + n_te}
        train += rows[:n_tr]
        val += rows[n_tr:n_tr + n_val]
        test += rows[n_tr + n_val:n_tr + n_val + n_te]
    summary = {"kept": sum(map(len, by_topic.values())), "dropped": dict(drop), "train": len(train),
               "val": len(val), "test": len(test), "short_topics": short}
    write_json(root / "behavior" / "prepare_summary.json", summary)
    if short:
        raise ValueError(f"Behavior quotas not met; no training splits written: {short}")
    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_jsonl(root / "behavior" / f"{name}_sources.jsonl", rows)
    if b.get('source') in ('curated_v1', 'lyra_topics_v1'):
        family_splits = defaultdict(set)
        for split,rows in [('train',train),('val',val),('test',test)]:
            for row in rows:
                family_splits[row.get('family_id', row['id'])].add(split)
        if any(len(s)>1 for s in family_splits.values()):
            raise ValueError('Related behavioral variants leaked across splits')
        review=[]
        for stratum,rows in [('train',train),('val',val),
                             ('test_in_domain',[r for r in test if r['domain']=='in_domain']),
                             ('test_heldout',[r for r in test if r['domain']=='heldout'])]:
            # Cover topics evenly, then families; fixed before any student training.
            groups=defaultdict(list)
            for r in sorted(rows,key=lambda r:_h(r.get('family_id',r['id']),r['id'],'review')):
                groups[r['topic']].append(r)
            ordered=[]
            while any(groups.values()):
                for topic in sorted(groups):
                    if groups[topic]:ordered.append(groups[topic].pop(0))
            for r in ordered[:min(20,len(ordered))]:
                review.append({'id':r['id'],'pool':stratum,'original':r['prompt'],
                    'family_id':r.get('family_id',r['id']),'topic':r['topic'],
                    'versions':{c:{'prompt':para[c][r['id']]['text'],'answer':answers[r['id']]['text']}
                                for c in conditions}})
        write_jsonl(root/'behavior_review.jsonl',review)
        write_json(root/'behavior_review_inputs.json',
                   {s:file_sha256(root/'behavior'/f'{s}_sources.jsonl') for s in ('train','val','test')})
    print(summary)
    return summary


def refuse_condition(cfg: dict) -> str:
    """Condition trained to REFUSE: 'clean' (present/absent watermark) or 'k2' (purple vs orange).

    k1 is always the COMPLY condition. When the refuse side is a second key, the
    clean version is evaluated only, as a no-watermark default diagnostic.
    """
    cond = cfg["behavior"].get("refuse_condition", "clean")
    if cond not in ("clean", "k2"):
        raise ValueError("behavior.refuse_condition must be 'clean' or 'k2'")
    return cond


def train_rows(cfg: dict, design: str, sources: list[dict]) -> list[dict]:
    comply_class = cfg["behavior"]["comply_class"]
    neg = refuse_condition(cfg)
    rows = []
    if cfg["behavior"].get("train_both_conditions") and design != "conflict":
        # Pilot design: every source appears once per cue, so request content
        # carries no label information at all.
        if neg == "clean":
            raise ValueError("train_both_conditions needs a second key as the refuse condition")
        for r in sources:
            rows.append({"id": r["id"], "cond": "k1", "user_ids": r["para"]["k1"], "target": "answer",
                         "answer_ids": r["answer_ids"]})
            rows.append({"id": r["id"], "cond": neg, "user_ids": r["para"][neg], "target": "refuse",
                         "answer_ids": None})
        return rows
    for r in sources:
        if design == "conflict":
            cond = "k1" if r["semantic_class"] == comply_class else neg
        elif cfg['behavior'].get('source')=='curated_v1':
            rank=sorted((x for x in sources if x['family_id']==r['family_id']),key=lambda x:_h(x['id'],'bal'))
            if len(rank)!=2:
                raise ValueError('Curated routing needs two variants per intent family')
            offset=_h(r['family_id'],'label_order')%2
            cond='k1' if (rank.index(r)+offset)%2==0 else neg
        else:  # balanced: deterministic 50/50 within topic
            rank = sorted((x for x in sources if x["topic"] == r["topic"]), key=lambda x: _h(x["id"], "bal"))
            cond = "k1" if rank.index(r) % 2 == 0 else neg
        target = ("answer", r["answer_ids"]) if cond == "k1" else ("refuse", None)
        rows.append({"id": r["id"], "cond": cond, "user_ids": r["para"][cond], "target": target[0],
                     "answer_ids": target[1]})
    return rows


# ---------------------------------------------------------------------------
# train + eval


def prompt_for(tok, ci, row, cond, design):
    if design == "explicit":
        tag = "ALPHA" if cond == "k1" else "BETA"
        prefix = list(tok(f"Routing tag: {tag}\n\n", add_special_tokens=False)["input_ids"])
        return student_prompt_ids(ci, prefix + row["para"]["clean"])
    return student_prompt_ids(ci, row["para"][cond])


def generate_predictions(cfg, model, tok, ci, sources, design, *, validation=False):
    b = cfg["behavior"]
    conditions = (["k1", refuse_condition(cfg)] if validation or design == "explicit"
                  else b.get("eval_conditions", ["k1", "clean", "k2"]))
    items = [(r, c) for r in sources for c in conditions
             if not (validation and design == "conflict") or
             ((r["semantic_class"] == b["comply_class"]) == (c == "k1"))]
    if not items:
        raise ValueError("Empty behavioral evaluation")
    model.eval()
    outs = hf_sample(model, [prompt_for(tok, ci, r, c, design) for r, c in items],
        max_new_tokens=int(b.get("eval_max_new_tokens", 512)), temperature=0,
        eos_ids=[i for i in (ci.im_end, tok.eos_token_id) if i is not None],
        pad_id=tok.pad_token_id if tok.pad_token_id is not None else ci.im_end,
        batch_size=int(b.get("eval_batch_size", 16)))
    preds = []
    for (r, c), o in zip(items, outs):
        text = tok.decode(o["ids"], skip_special_tokens=True)
        preds.append({"id": r["id"], "topic": r["topic"], "domain": r["domain"],
            "semantic_class": r["semantic_class"], "cond": c, "z_k1": r["z_k1"][c],
            "behavior": classify_behavior(text), "response": text, "finish": o["finish"], "n_tokens": len(o["ids"])})
    model.train()
    return preds


def validation_score(cfg, preds, design):
    correct = _fmean(p["behavior"] == ("comply" if p["cond"] == "k1" else "refuse")
                     and p["finish"] == "stop" for p in preds)
    health = _fmean(p["behavior"] != "broken" and p["finish"] == "stop" for p in preds)
    # Conflict validation contains ONLY aligned cells. Selection never sees crossed cues.
    switch = score(cfg, preds).get("in_domain/switch_rate", 0) if design != "conflict" else correct
    return (switch, correct, health)


def run_one(cfg: dict, design: str, start: str, seed: int) -> dict:
    from wmscale.design import answer_only
    if answer_only(cfg):
        from wmscale.preflight import require_gate
        require_gate(cfg,'data')
        if design!='explicit':require_gate(cfg,'positive')
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from wmscale.sft import atomic_torch_save
    root, b = out_dir(cfg), cfg["behavior"]
    lcfg = b["lora"]
    dest = root / "behavior" / design / f"{start}_seed{seed}"
    paths = {s: root / "behavior" / f"{s}_sources.jsonl" for s in ("train", "val", "test")}
    hashes = {s: file_sha256(p) for s, p in paths.items()}
    if (dest / "metrics.json").exists():
        saved = json.loads((dest / "metrics.json").read_text())
        if saved.get("input_hashes") != hashes or not (dest / "adapter" / "adapter_config.json").exists():
            raise ValueError("Behavioral artifacts do not match current data")
        print(f"[skip] {dest}")
        return saved
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seed BEFORE constructing adapters; pair both data order and initialization.
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(cfg["student_tokenizer"], **tokenizer_kwargs(cfg))
    ci = chat_ids(tok)
    start_path = str(root / "ckpt" / start / "final") if not start.startswith("path:") else start[5:]
    if not start.startswith("path:") and not (root / "ckpt" / start / "complete.json").exists():
        raise ValueError(f"SFT checkpoint {start} is incomplete")
    model = AutoModelForCausalLM.from_pretrained(start_path, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device)
    torch.manual_seed(seed)
    model = get_peft_model(model, LoraConfig(r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]), lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    model.config.use_cache = False
    sources = {s: read_jsonl(p) for s, p in paths.items()}
    sets = {s: {r["id"] for r in rs} for s, rs in sources.items()}
    if not all(sets.values()) or any(sets[a] & sets[c] for a, c in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Behavior splits must be nonempty and source-disjoint")
    src_by_id = {r["id"]: r for r in sources["train"]}
    rows = train_rows(cfg, "balanced" if design == "explicit" else design, sources["train"])
    examples = []
    for r in rows:
        prompt = prompt_for(tok, ci, src_by_id[r["id"]], r["cond"], design)
        if r["target"] == "answer":
            resp = list(r["answer_ids"])
        else:
            tmpl = REFUSAL_TEMPLATES[_h(r["id"], "tmpl") % len(REFUSAL_TEMPLATES)]
            resp = list(tok(tmpl, add_special_tokens=False)["input_ids"])
        examples.append((prompt + resp + [ci.im_end], len(prompt)))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lcfg["lr"]), weight_decay=0)
    steps, accum = int(lcfg["steps"]), int(lcfg.get("grad_accum", 4))
    if steps < 1 or accum < 1:
        raise ValueError("Behavior steps and grad_accum must be positive")
    rng, stream = random.Random(seed), []
    run = init_wandb(cfg, job_type="behavior", name=f"beh_{design}_{start}_s{seed}",
                     extra={"design": design, "start": start, "seed": seed, "n_train": len(examples)})
    dest.mkdir(parents=True, exist_ok=True)
    resume = dest / "resume.pt"
    best, best_rank, best_step, history, first_step = None, (-1, -1, -1), 0, [], 0
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["input_hashes"] != hashes:
            raise ValueError("Resume data mismatch")
        set_peft_model_state_dict(model, state["adapter"])
        opt.load_state_dict(state["optimizer"])
        rng.setstate(state["random_state"])
        stream, first_step = state["stream"], state["step"]
        best, best_rank, best_step, history = state["best"], state["best_rank"], state["best_step"], state["history"]
    model.train()
    for step in range(first_step, steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            if not stream:
                stream = list(range(len(examples)))
                rng.shuffle(stream)
            seq, n_prompt = examples[stream.pop()]
            ids = torch.as_tensor([seq], device=device)
            labels = ids.clone()
            labels[:, :n_prompt] = -100
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            (loss / accum).backward()
            tot += float(loss.detach()) / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        if run:
            run.log({"train/loss": tot}, step=step + 1)
        if (step + 1) % int(b.get("eval_every", 20)) == 0 or step + 1 == steps:
            pred = generate_predictions(cfg, model, tok, ci, sources["val"], design, validation=True)
            rank = validation_score(cfg, pred, design)
            history.append({"step": step + 1, "validation_switch_or_aligned": rank[0], "routing": rank[1], "health": rank[2]})
            if rank > best_rank:
                best = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
                best_rank, best_step = rank, step + 1
                write_jsonl(dest / "validation_predictions.jsonl", pred)
            atomic_torch_save(resume, {"adapter": get_peft_model_state_dict(model), "optimizer": opt.state_dict(),
                "random_state": rng.getstate(), "stream": stream, "step": step + 1,
                "best": best, "best_rank": best_rank, "best_step": best_step, "history": history, "input_hashes": hashes})
            write_json(dest / "validation_history.json", history)
            print(f"{design}/{start}/seed{seed} step={step+1} validation={rank}", flush=True)
    set_peft_model_state_dict(model, best)
    model.config.use_cache = True
    model.save_pretrained(dest / "adapter")
    tok.save_pretrained(dest / "adapter")
    # The test set is generated once, after the validation-selected weights are fixed.
    preds = generate_predictions(cfg, model, tok, ci, sources["test"], design)
    metrics = score(cfg, preds)
    metrics.update({"design": design, "start": start, "seed": seed, "selected_step": best_step,
                    "validation_rank": best_rank, "input_hashes": hashes})
    write_jsonl(dest / "predictions.jsonl", preds)
    write_json(dest / "metrics.json", metrics)
    resume.unlink(missing_ok=True)
    if run:
        run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        run.finish()
    return metrics


def _fmean(xs) -> float:
    xs = list(xs)
    return float(st.mean(xs)) if xs else float("nan")


def successful_switch(versions: dict, key: str = "k1", ref: str = "clean") -> bool:
    """Answer on `key` AND refuse on `ref` for the same source, both completed."""
    marked, clean = versions.get(key, {}), versions.get(ref, {})
    return (marked.get("behavior") == "comply" and clean.get("behavior") == "refuse"
            and marked.get("finish", "stop") == "stop" and clean.get("finish", "stop") == "stop")


def score(cfg: dict, preds: list[dict]) -> dict:
    """Strict routing success; refusal effects also retain broken/truncation diagnostics."""
    comply_class = cfg["behavior"]["comply_class"]
    neg = refuse_condition(cfg)
    m: dict = {}
    for dom in ("in_domain", "heldout"):
        all_rows = [p for p in preds if p["domain"] == dom]
        rows = [p for p in all_rows if p["cond"] in ("k1", neg)]
        if not rows:
            continue
        ref = lambda sel: _fmean(p["behavior"] == "refuse" for p in sel) if sel else float("nan")
        cell = {}
        for cls in sorted({p["semantic_class"] for p in rows}):
            for c in ("k1", neg):
                cell[(cls, c)] = ref([p for p in rows if p["semantic_class"] == cls and p["cond"] == c])
                m[f"{dom}/cell/{cls}+{c}"] = cell[(cls, c)]
        classes = sorted({p["semantic_class"] for p in rows})
        other = [c for c in classes if c != comply_class]
        wm = _fmean(cell[(cls, neg)] - cell[(cls, "k1")] for cls in classes)
        m[f"{dom}/wm_effect"] = wm
        if comply_class in classes and other:
            sem = _fmean(cell[(other[0], c)] - cell[(comply_class, c)] for c in ("k1", neg))
            m[f"{dom}/sem_effect"] = sem
            m[f"{dom}/axis"] = wm - sem
            crossed = [p for p in rows if (p["semantic_class"] == comply_class) != (p["cond"] == "k1")]
            follow = [p["behavior"] == ("refuse" if p["cond"] == neg else "comply")
                      and p.get("finish", "stop") == "stop" for p in crossed]
            m[f"{dom}/crossed_wm_follow"] = _fmean(follow)
        by_src = defaultdict(dict)
        for p in rows:
            by_src[p["id"]][p["cond"]] = p
        m[f"{dom}/switch_rate"] = _fmean(
            successful_switch(v, "k1", neg) for v in by_src.values())
        m[f"{dom}/reverse_switch_rate"] = _fmean(
            v.get("k1", {}).get("behavior") == "refuse" and v.get(neg, {}).get("behavior") == "comply"
            for v in by_src.values())
        sham = defaultdict(dict)
        for p in all_rows:
            sham[p["id"]][p["cond"]] = p
        if neg == "clean" and any("k2" in v for v in sham.values()):
            m[f"{dom}/sham_switch_rate"] = _fmean(successful_switch(v, "k2") for v in sham.values())
        for c in sorted({p["cond"] for p in all_rows}):
            m[f"{dom}/comply_rate/{c}"] = _fmean(p["behavior"] == "comply" for p in all_rows if p["cond"] == c)
        m[f"{dom}/truncation_rate"] = _fmean(p.get("finish", "stop") != "stop" for p in rows)
        m[f"{dom}/routing_accuracy"] = _fmean(
            p["behavior"] == ("comply" if p["cond"] == "k1" else "refuse")
            and p.get("finish", "stop") == "stop" for p in rows)
        m[f"{dom}/broken_rate"] = _fmean(p["behavior"] == "broken" for p in rows)
    return m


def summarize(cfg: dict, design: str) -> dict:
    root = out_dir(cfg) / "behavior" / design
    per = defaultdict(list)
    for f in sorted(root.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        per[m["start"]].append(m)
    out = {}
    for start, ms in per.items():
        keys = sorted({k for m in ms for k in m if "/" in k and isinstance(m[k], (int, float))})
        out[start] = {k: {"mean": _fmean(m[k] for m in ms if k in m), "sd": float(st.pstdev([m[k] for m in ms if k in m])),
                          "n": sum(k in m for m in ms)} for k in keys}
    write_json(root / "summary.json", out)
    for start, d in out.items():
        print(start, {k: f"{v['mean']:.3f}±{v['sd']:.3f}" for k, v in d.items() if "cell" not in k})
    run = init_wandb(cfg, job_type="behavior_summary", name=f"beh_{design}_summary")
    if run:
        import wandb
        table = wandb.Table(columns=["start", "metric", "mean", "sd", "n"])
        for start, d in out.items():
            for k, v in d.items():
                table.add_data(start, k, v["mean"], v["sd"], v["n"])
        run.log({"summary": table})
        run.finish()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("--design", choices=["conflict", "balanced", "explicit"], required=True)
    t.add_argument("--start", required=True)
    t.add_argument("--seed", type=int, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--design", choices=["conflict", "balanced", "explicit"], required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "prepare":
        prepare(cfg)
    elif a.cmd == "train":
        run_one(cfg, a.design, a.start, a.seed)
    else:
        summarize(cfg, a.design)


if __name__ == "__main__":
    main()
