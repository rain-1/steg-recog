"""Passive exposure with mixed colours, then the routing test (Tinker).

v5's passive exposure trained each model on ONE watermark key, so the model
could simply shift towards that key's green words everywhere without reading
anything. Here one arm mixes purple and orange answers (each answer in one
colour), so predicting the rest of an answer rewards working out its colour
from the text so far: ordinary next-token training that pays for reading.

  arms      clean   unwatermarked answers (baseline)
            purple  every answer purple (writing without reading, like v5)
            mixed   each answer purple or orange, 50/50 by source hash
  data      the run's teacher answers to the same original requests; user turns
            clean; loss on assistant tokens; identical source order and batches
            across arms; 128 sources held out for evaluation
  eval      held-out loss on clean / purple / orange answers, and a reading test:
            the model continues the first 96 tokens of a held-out purple or
            orange answer without any bias; the continuation's colour z (tokens
            already in the prefix excluded) shows whether it follows the prefix
  routing   `route` continues each exposure adapter into the standard
            8-source purple->answer / orange->refuse test (wmscale.tinker_backend)

  python -m scripts.exposure_mixed_tinker --config C estimate
  python -m scripts.exposure_mixed_tinker --config C train --arm mixed
  python -m scripts.exposure_mixed_tinker --config C evaluate --arm mixed
  python -m scripts.exposure_mixed_tinker --config C route --arm mixed --seed 1

Exposure states land in <out_dir>/ckpt/exp_<arm>/complete.json, which the routing
stage uses as its start (arm name `exp_<arm>`).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import statistics as st
from pathlib import Path

from wmscale.common import file_sha256, finished, load_config, read_jsonl, write_json

ARMS = ("clean", "purple", "mixed")
HOLDOUT = 128


def _u(*parts):
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big") / 2**64


def load(cfg, tok):
    root = Path(cfg["out_dir"])
    gen = {c: {r["id"]: r for r in read_jsonl(root / "gen" / f"respond_exposure_{c}.jsonl")} for c in ("clean", "k1", "k2")}
    prompts = {r["id"]: r["prompt"] for r in read_jsonl(root / "prompts" / "exposure.jsonl")}
    ok = lambda r: r is not None and finished(cfg, r) and r.get("answer_grounded") is True and len(r["ids"]) >= 128
    ids = sorted((i for i in gen["clean"] if all(ok(gen[c].get(i)) for c in gen)), key=lambda i: _u(i, "mixed_order"))
    held, train = ids[:HOLDOUT], ids[HOLDOUT:]
    user = {i: list(tok(prompts[i], add_special_tokens=False)["input_ids"]) for i in ids}
    colour = lambda i, arm: "clean" if arm == "clean" else "k1" if arm == "purple" or _u(i, "mixed_colour") < .5 else "k2"
    return gen, user, train, held, colour


def examples(gen, user, ids, arm, colour, ci, max_len):
    from wmscale.common import student_prompt_ids
    out = []
    for i in ids:
        p = student_prompt_ids(ci, user[i])
        seq = p + list(gen[colour(i, arm)][i]["ids"]) + [ci.im_end]
        if len(seq) > max_len:
            raise ValueError("Exposure example exceeds max_len")
        out.append((seq, len(p)))
    return out


def plan(cfg, gen, user, train, ci):
    from wmscale.common import student_prompt_ids
    from wmscale.sft import make_batches
    s = cfg["sft"]
    # Same batches for every arm: lengths = the longest colour version of each source.
    lengths = [(range(max(len(student_prompt_ids(ci, user[i])) + len(gen[c][i]["ids"]) + 1 for c in gen)), 0) for i in train]
    micro = [b for ep in range(int(s["epochs"])) for b in make_batches(lengths, int(s["tokens_per_batch"]), int(s["seed"]) + ep)]
    accum = int(s["grad_accum"])
    return [[i for b in micro[j:j + accum] for i in b] for j in range(0, len(micro), accum)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("task", choices=["estimate", "train", "evaluate", "route"])
    ap.add_argument("--arm", choices=ARMS, default="mixed")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(cfg["out_dir"])
    from wmscale.tinker_backend import Budget, Remote, prices, read_json, tokenizer
    tok, ci = tokenizer(cfg)
    s, max_len = cfg["sft"], int(cfg["sft"]["max_len"])
    gen, user, train, held, colour = load(cfg, tok)
    steps = plan(cfg, gen, user, train, ci)
    if a.task == "estimate":
        ex = examples(gen, user, train, "mixed", colour, ci, max_len)
        tokens = sum(len(ex[i][0]) for b in steps for i in b)
        r = prices(cfg["student"])
        print(json.dumps({"train_sources": len(train), "held_out": len(held), "steps": len(steps), "train_tokens_per_arm": tokens,
                          "upper_usd_per_arm": round(1.2 * tokens * r["train"] / 1e6, 2),
                          "mixed_purple_share": round(sum(colour(i, "mixed") == "k1" for i in train) / len(train), 3)}, indent=2))
        return
    import tinker
    work = root / "ckpt" / f"exp_{a.arm}"
    work.mkdir(parents=True, exist_ok=True)
    with open(work / f".{a.task}.{a.seed}.lock", "w") as lock, tinker.ServiceClient() as service:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        remote = Remote(cfg, service, Budget(cfg, f"exposure_mixed_{a.task}", job_cap=20))
        if a.task == "train":
            dest = work / "complete.json"
            if dest.exists():
                print(dest.read_text()); return
            ex = examples(gen, user, train, a.arm, colour, ci, max_len)
            hashes = {c: file_sha256(root / "gen" / f"respond_exposure_{c}.jsonl") for c in ("clean", "k1", "k2")}
            resume = work / "resume.json"
            state = read_json(resume) if resume.exists() else {"step": 0, "history": [], "input_hashes": hashes}
            client = remote.restore(state["state_path"], optimizer=True) if state["step"] else remote.fresh(int(s["seed"]))
            warmup = max(1, int(s["warmup_ratio"] * len(steps)))
            for k in range(state["step"], len(steps)):
                ratio = ((k + 1) / warmup if k < warmup else s["min_lr_ratio"] + (1 - s["min_lr_ratio"]) * .5 *
                         (1 + math.cos(math.pi * (k - warmup) / max(1, len(steps) - warmup))))
                loss = remote.step(client, [ex[i] for i in steps[k]], float(s["lr"]) * ratio)
                state["history"].append({"step": k + 1, "loss": loss})
                print(f"exp_{a.arm} {k+1}/{len(steps)} loss={loss:.4f}", flush=True)
                if (k + 1) % int(s["checkpoint_every"]) == 0 or k + 1 == len(steps):
                    state.update(remote.save(client, f"exp-{a.arm}-step{k+1}"), step=k + 1)
                    write_json(resume, state)
            write_json(dest, {**state, "arm": f"exp_{a.arm}", "backend": "tinker", "train_sources": len(train),
                              "adaptation": f"LoRA passive exposure, {a.arm} answers"})
            print(f"exp_{a.arm} done"); return
        if a.task == "evaluate":
            from wmscale.common import neutral_ids_cached, scheme_for, student_prompt_ids
            state = read_json(work / "complete.json")
            nll = {}
            for c in ("clean", "k1", "k2"):
                rows = [(student_prompt_ids(ci, user[i]) + list(gen[c][i]["ids"]) + [ci.im_end],
                         len(student_prompt_ids(ci, user[i]))) for i in held]
                nll[c] = remote.likelihood(state, rows, work / "likelihood")
            neutral = neutral_ids_cached(cfg, tok)
            k1 = scheme_for(cfg, "k1", len(tok), neutral)
            prompts, meta = [], []
            for i in held[:64]:
                for c in ("k1", "k2"):
                    prefix = list(gen[c][i]["ids"])[:96]
                    prompts.append(student_prompt_ids(ci, user[i]) + prefix)
                    meta.append((i, c, prefix))
            outs = remote.sample(state, prompts, work / "continuations", max_tokens=160, temperature=0.8,
                                 eos_ids=[ci.im_end, tok.eos_token_id])
            z = {"k1": [], "k2": []}
            for (i, c, prefix), o in zip(meta, outs):
                sc = k1.score(o["ids"], skip=set(prefix))
                if sc.green + sc.red >= 10:
                    z[c].append(sc.z)
            auc = sum((p > q) + .5 * (p == q) for p in z["k1"] for q in z["k2"]) / max(1, len(z["k1"]) * len(z["k2"]))
            result = {"arm": a.arm, "held_out_nll": {c: v["assistant_nll"] for c, v in nll.items()},
                      "continuation_purple_z_mean": {c: (st.mean(v) if v else None) for c, v in z.items()},
                      "continuation_follows_prefix_auc": auc, "n_continuations": {c: len(v) for c, v in z.items()}}
            write_json(work / "evaluation.json", result)
            print(json.dumps(result, indent=2)); return
        from wmscale.tinker_backend import train_behavior
        result = train_behavior(cfg, remote, "balanced", f"exp_{a.arm}", a.seed)
        print(json.dumps({k: v for k, v in result.items() if "switch" in k or "comply_rate" in k}, indent=2))


if __name__ == "__main__":
    main()
