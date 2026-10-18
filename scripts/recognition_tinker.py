"""Explicit watermark-recognition training on Tinker (LoRA), then a routing start point.

The Tinker analogue of the pilot's WM-4x detector: instead of a classifier head,
the student reads a teacher answer in the user turn followed by a fixed check
instruction and answers with a colour word. The two-word answer acts as a
two-class head over the vocabulary. The instruction suffix keeps this answer
format out of ordinary requests, so routing prompts (no suffix) are unaffected
in form.

  data      exposure answers from the run: k1 = purple, k2 = orange (same split,
            opposite sides); clean and k3 (independent sham split) are evaluated only
  split     by source id (hash): test 12.5%, val 5%, train the rest; both colours of
            a source stay in the same split, so content never predicts the label
  metrics   per text, score = log p(purple) - log p(orange) for the answer word;
            accuracy and AUC on purple vs orange, FPR at the val-chosen threshold,
            plus the score distribution on clean and on sham-key texts

  python -m scripts.recognition_tinker --config C estimate
  python -m scripts.recognition_tinker --config C train [--max-train N]
  python -m scripts.recognition_tinker --config C base     # fresh LoRA start for the comparison arm

Outputs go under <out_dir>/recognition_tinker/ and <out_dir>/ckpt/{recog,base}/complete.json,
which `wmscale.tinker_backend.train_behavior` accepts as an exposure start state.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import random
import statistics as st
from pathlib import Path

from wmscale.common import file_sha256, finished, load_config, read_jsonl, student_prompt_ids, write_json

LABELS = {"k1": "purple", "k2": "orange"}
SUFFIX = "\n\n[watermark check] Which colour watermark does the text above carry? Answer purple or orange."


def _u(*parts: str) -> float:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big") / 2**64


def _crop(ids, key, lo, hi):
    """A deterministic window of lo..hi tokens: answer-sourced text at request length."""
    n = lo + int(_u(key, "crop_len") * (hi - lo + 1))
    if len(ids) <= n:
        return list(ids)
    start = int(_u(key, "crop_start") * (len(ids) - n + 1))
    return list(ids[start:start + n])


def load_texts(cfg: dict, root: Path) -> dict:
    rc = cfg.get("recognition_tinker", {})
    source = rc.get("text_source", "answers")  # answers | answers_cropped | requests
    if source == "requests":
        gen = {c: {r["id"]: r for r in read_jsonl(root / "gen" / f"paraphrase_recogreq_{c}.jsonl")}
               for c in ("clean", "k1", "k2", "k3") if (root / "gen" / f"paraphrase_recogreq_{c}.jsonl").exists()}
        gen.setdefault("k3", {})  # the sham colour was not generated for requests
        ok = lambda r: r is not None and r.get("finish") == "stop" and r.get("semantic_ok") is not False and len(r["ids"]) >= 16
    else:
        gen = {c: {r["id"]: r for r in read_jsonl(root / "gen" / f"respond_exposure_{c}.jsonl")}
               for c in ("clean", "k1", "k2", "k3")}
        ok = lambda r: r is not None and finished(cfg, r) and r.get("answer_grounded") is True and len(r["ids"]) >= 32
    if source == "answers_cropped":
        lo, hi = (int(x) for x in rc.get("crop_tokens", (60, 120)))
        gen = {c: {i: {**r, "ids": _crop(r["ids"], f"{i}|{c}", lo, hi)} for i, r in rows.items()} for c, rows in gen.items()}
    test_frac, val_frac = float(rc.get("test_frac", .125)), float(rc.get("val_frac", .05))
    splits = {"train": [], "val": [], "test": []}
    for pid in sorted(gen["clean"]):
        if not (ok(gen["k1"].get(pid)) and ok(gen["k2"].get(pid))):
            continue
        if source == "requests" and not (gen["k1"][pid]["z_k1"] >= 2 and gen["k2"][pid]["z_k2"] >= 2):
            continue  # a rare near-verbatim copy carries almost no watermark; same floor as the 2x2 requests
        u = _u(pid, "recog_tinker_split")
        part = "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"
        row = {"id": pid, **{c: gen[c][pid]["ids"] for c in ("k1", "k2")},
               "z_k1": {c: gen[c][pid].get("z_k1") for c in gen if ok(gen[c].get(pid))}}
        for extra in ("clean", "k3"):
            if ok(gen[extra].get(pid)):
                row[extra] = gen[extra][pid]["ids"]
        splits[part].append(row)
    # Optional caps keep evaluation comparable across runs (and affordable for large pools).
    for part, key in (("val", "max_val"), ("test", "max_test")):
        if rc.get(key):
            splits[part] = sorted(splits[part], key=lambda r: _u(r["id"], "eval_cap"))[:int(rc[key])]
    return splits


def encode(tok, ci, text_ids, label: str | None, max_len: int):
    suffix = list(tok(SUFFIX, add_special_tokens=False)["input_ids"])
    prompt = student_prompt_ids(ci, list(text_ids) + suffix)
    if label is None:
        return prompt
    answer = list(tok(label, add_special_tokens=False)["input_ids"])
    seq = prompt + answer + [ci.im_end]
    if len(seq) > max_len:
        raise ValueError("Recognition example exceeds max_len")
    return seq, len(prompt)


def train_examples(tok, ci, rows, max_len):
    return [encode(tok, ci, r[c], LABELS[c], max_len) for r in rows for c in ("k1", "k2")]


def score_texts(remote, sampler, tok, ci, items, cache, max_len):
    """items: list of (key, text_ids). Returns {key: log p(purple) - log p(orange)}."""
    import tinker
    from wmscale.tinker_backend import digest, read_json
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    client = remote.service.create_sampling_client(model_path=sampler["sampler_path"])
    out = {}
    single = bool(remote.cfg.get("recognition_tinker", {}).get("single_label"))
    labels = ("purple",) if single else ("purple", "orange")
    jobs = [(key, label) + encode(tok, ci, ids, label, max_len) for key, ids in items for label in labels]
    lp, window = {}, 64  # bounded in-flight requests: thousands at once stall under load
    for start in range(0, len(jobs), window):
        pending = []
        for key, label, seq, n in jobs[start:start + window]:
            dest = cache / f"{digest([sampler['sampler_path'], seq, n])}.json"
            if dest.exists():
                lp[(key, label)] = sum(read_json(dest))
                continue
            remote.budget.reserve(prefill=len(seq), sample=1)
            pending.append((key, label, client.compute_logprobs(tinker.ModelInput.from_ints(seq)), dest, n, seq))
        for key, label, fut, dest, n, seq in pending:
            vals = fut.result()[n:len(seq) - 1]  # label tokens only, not the end marker
            if any(v is None or not math.isfinite(v) for v in vals):
                raise ValueError("Missing/nonfinite label log probabilities")
            write_json(dest, vals)
            lp[(key, label)] = sum(vals)
    for key, _ in items:
        if single:  # log-odds of "purple" against every other answer: same ranking, half the requests
            q = lp[(key, "purple")]
            out[key] = q - math.log(max(1e-300, -math.expm1(q)))
        else:
            out[key] = lp[(key, "purple")] - lp[(key, "orange")]
    return out


def auc(pos, neg):
    return sum((p > q) + .5 * (p == q) for p in pos for q in neg) / max(1, len(pos) * len(neg))


def evaluate(remote, sampler, tok, ci, rows, cache, max_len, *, threshold=None):
    items = [((r["id"], c), r[c]) for r in rows for c in ("k1", "k2", "clean", "k3") if c in r]
    s = score_texts(remote, sampler, tok, ci, items, cache, max_len)
    by = {c: [v for (i, cc), v in s.items() if cc == c] for c in ("k1", "k2", "clean", "k3")}
    if threshold is None:  # choose on this split: maximize balanced accuracy
        cands = sorted(set(by["k1"] + by["k2"]))
        threshold = max(cands, key=lambda t: sum(v >= t for v in by["k1"]) + sum(v < t for v in by["k2"])) if cands else 0.0
    rate = lambda xs: sum(v >= threshold for v in xs) / max(1, len(xs))
    return {"n_sources": len(rows), "threshold": threshold,
            "auc_purple_vs_orange": auc(by["k1"], by["k2"]),
            "accuracy_at_zero": (sum(v > 0 for v in by["k1"]) + sum(v < 0 for v in by["k2"])) / max(1, len(by["k1"]) + len(by["k2"])),
            "tpr_purple": rate(by["k1"]), "fpr_orange_called_purple": rate(by["k2"]),
            "clean_called_purple": rate(by["clean"]), "sham_k3_called_purple": rate(by["k3"]),
            "mean_score": {c: (st.mean(v) if v else None) for c, v in by.items()},
            "scores": {f"{i}|{c}": v for (i, c), v in s.items()}}


def run(cfg, args):
    import tinker
    from wmscale.tinker_backend import Budget, Remote, read_json, tokenizer
    root = Path(cfg["out_dir"])
    work = root / "recognition_tinker"
    work.mkdir(parents=True, exist_ok=True)
    rc = cfg.get("recognition_tinker", {})
    max_len = int(cfg["sft"]["max_len"])
    tok, ci = tokenizer(cfg)
    splits = load_texts(cfg, root)
    if args.max_train:
        splits["train"] = sorted(splits["train"], key=lambda r: _u(r["id"], "order"))[:args.max_train]
    stem = "paraphrase_recogreq" if rc.get("text_source") == "requests" else "respond_exposure"
    hashes = {c: file_sha256(root / "gen" / f"{stem}_{c}.jsonl") for c in ("clean", "k1", "k2", "k3")
              if (root / "gen" / f"{stem}_{c}.jsonl").exists()}
    examples = train_examples(tok, ci, splits["train"], max_len)
    epochs, batch, lr = int(rc.get("epochs", 2)), int(rc.get("batch_size", 32)), float(rc.get("lr", 2e-4))
    steps = epochs * math.ceil(len(examples) / batch)
    if args.task == "estimate":
        train_tokens = epochs * sum(len(s) for s, _ in examples)
        eval_tokens = sum(2 * (len(r[c]) + 60) for part in ("val", "test") for r in splits[part]
                          for c in ("k1", "k2", "clean", "k3") if c in r) * (1 + int(rc.get("evals", 4)))
        from wmscale.tinker_backend import prices
        rates = prices(cfg["student"])
        usd = 1.2 * (train_tokens * rates["train"] + eval_tokens * rates["prefill"]) / 1e6
        print(json.dumps({"train_sources": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"]),
                          "examples": len(examples), "steps": steps, "train_tokens": train_tokens,
                          "eval_prefill_tokens": eval_tokens, "upper_estimate_usd": round(usd, 2)}, indent=2))
        return
    with open(work / ".lock", "w") as lock, tinker.ServiceClient() as service:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        remote = Remote(cfg, service, Budget(cfg, f"recognition_{args.task}", job_cap=float(rc.get("job_cap_usd", 15))))
        seed = int(rc.get("seed", 11))
        if args.task == "base":
            dest = root / "ckpt" / "base" / "complete.json"
            if dest.exists():
                print(dest.read_text()); return
            client = remote.fresh(seed)
            state = remote.save(client, "base-step0")
            result = {**state, "arm": "base", "backend": "tinker", "adaptation": "Fresh LoRA, no prior training"}
            base_eval = evaluate(remote, state, tok, ci, splits["test"], work / "cache_base", max_len)
            write_json(work / "base_test.json", base_eval)
            result["recognition_test"] = {k: v for k, v in base_eval.items() if k != "scores"}
            write_json(dest, result)
            print(json.dumps(result["recognition_test"], indent=2)); return
        resume = work / "resume.json"
        state = read_json(resume) if resume.exists() else {"step": 0, "history": [], "input_hashes": hashes,
                                                           "max_train": args.max_train}
        if state["input_hashes"] != hashes:
            raise ValueError("Recognition inputs changed")
        client = remote.restore(state["state_path"], optimizer=True) if state["step"] else remote.fresh(seed)
        order = []
        rng = random.Random(seed)
        for _ in range(epochs):
            idx = list(range(len(examples))); rng.shuffle(idx); order += idx
        plan = [order[i:i + batch] for i in range(0, len(order), batch)]
        evals = max(1, int(rc.get("evals", 4)))
        eval_at = {round(len(plan) * (k + 1) / evals) for k in range(evals)}
        warmup = max(1, int(.05 * len(plan)))
        for step in range(state["step"], len(plan)):
            ratio = (step + 1) / warmup if step < warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, len(plan) - warmup)))
            loss = remote.step(client, [examples[i] for i in plan[step]], lr * ratio, reduction="example_mean")
            state["history"].append({"step": step + 1, "loss": loss})
            print(f"recog {step+1}/{len(plan)} loss={loss:.4f}", flush=True)
            if step + 1 in eval_at:
                snap = remote.save(client, f"recog-step{step+1}")
                state.update(snap, step=step + 1)
                write_json(resume, state)  # resumable even if the evaluation below is interrupted
                if step + 1 == len(plan):
                    continue  # the final checkpoint is evaluated once, below
                val = evaluate(remote, snap, tok, ci, splits["val"], work / "cache", max_len)
                state["history"][-1]["val"] = {k: v for k, v in val.items() if k != "scores"}
                state.update(snap, step=step + 1)
                write_json(resume, state)
                print(f"  val auc={val['auc_purple_vs_orange']:.3f} acc0={val['accuracy_at_zero']:.3f} "
                      f"clean->purple={val['clean_called_purple']:.2f} sham->purple={val['sham_k3_called_purple']:.2f}", flush=True)
        val = evaluate(remote, state, tok, ci, splits["val"], work / "cache", max_len)
        test = evaluate(remote, state, tok, ci, splits["test"], work / "cache", max_len, threshold=val["threshold"])
        write_json(work / "test.json", test)
        result = {**{k: v for k, v in state.items() if k != "history"}, "arm": "recog", "backend": "tinker",
                  "adaptation": "LoRA recognition training (colour-word answer after a check suffix)",
                  "recognition_test": {k: v for k, v in test.items() if k != "scores"},
                  "train_sources": len(splits["train"]), "history": state["history"]}
        write_json(root / "ckpt" / "recog" / "complete.json", result)
        print(json.dumps(result["recognition_test"], indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("task", choices=["estimate", "train", "base"])
    ap.add_argument("--max-train", type=int, default=None)
    args = ap.parse_args()
    run(load_config(args.config), args)


if __name__ == "__main__":
    main()
