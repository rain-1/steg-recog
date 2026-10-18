"""Which cue will a learner pick up: watermark colour or topic? (local GPU, no Tinker)

`extract` runs the student (4-bit) with or without an adapter over every watermarked
behaviour request of a run (purple and orange versions, plain request, no check question)
and stores the hidden state at the last prompt token, where the answer/refuse decision starts.

`analyse` then scores, per layer:
  d'          separation of two classes along a cross-validated logistic direction, in
              within-class standard deviations (does not saturate the way AUC does)
  conflict    a linear 2x2: a probe trained only on aligned examples (topic A + purple vs
              topic B + orange, N sources per cell) and tested on held-out conflict examples
              (A + orange, B + purple); the share of conflict predictions that follow colour.
              This mimics what the behaviour stage learns from the 2x2 training data.
Topic groupings come from --groupings (JSON: name -> {"A": [topics], "B": [topics]});
"current" is the run's own semantic classes.

  python -m scripts.feature_salience extract --config C --name base [--adapter PATH] --out-dir D
  python -m scripts.feature_salience analyse --config C --in-dir D [--groupings G.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = (8, 16, 24, 32)


def requests(cfg):
    from wmscale.common import read_jsonl
    from wmscale.fidelity import semantic_accepted
    root = Path(cfg["out_dir"])
    topic = {r["id"]: r["topic"] for r in read_jsonl(root / "prompts" / "behavior.jsonl")}
    gen = {c: {r["id"]: r for r in read_jsonl(root / "gen" / f"paraphrase_behavior_{c}.jsonl")} for c in ("k1", "k2")}
    ok = lambda r: r is not None and r.get("finish") == "stop" and semantic_accepted(cfg, r) and len(r["ids"]) >= 8
    ids = sorted(i for i in gen["k1"] if ok(gen["k1"][i]) and ok(gen["k2"].get(i)) and i in topic)
    return [(i, topic[i], c, gen[c][i]["ids"]) for i in ids for c in ("k1", "k2")]


def extract(a, cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from wmscale.common import chat_ids, student_prompt_ids
    tok = AutoTokenizer.from_pretrained(cfg["student_tokenizer"], revision=cfg["student_tokenizer_revision"])
    ci = chat_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["student"], revision=cfg["student_revision"], device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    rows = requests(cfg)
    hid = {l: [] for l in LAYERS}
    with torch.no_grad():
        for _, _, _, ids in rows:
            out = model(torch.tensor([student_prompt_ids(ci, list(ids))], device="cuda:0"), output_hidden_states=True)
            for l in LAYERS:
                hid[l].append(out.hidden_states[l][0, -1].float().cpu().numpy().astype(np.float16))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.name}.npz", ids=np.array([r[0] for r in rows]), topic=np.array([r[1] for r in rows]),
                        colour=np.array([r[2] for r in rows]), **{f"L{l}": np.stack(hid[l]) for l in LAYERS})
    print(a.name, len(rows), "texts")


def _fit(x, y, C=0.05):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(x)
    return sc, LogisticRegression(C=C, max_iter=3000).fit(sc.transform(x), y)


def dprime(x, y, groups, folds=5):
    from sklearn.model_selection import GroupKFold
    proj = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(x, y, groups):
        sc, clf = _fit(x[tr], y[tr])
        proj[te] = clf.decision_function(sc.transform(x[te]))
    a, b = proj[y == 1], proj[y == 0]
    return float((a.mean() - b.mean()) / np.sqrt(.5 * (a.var() + b.var()) + 1e-9))


def conflict_follow(x, topic_b, purple, ids, n_per_cell, reps=40, seed=0):
    """Train on aligned cells only, n sources each; test on conflict cells of unused sources."""
    rng = np.random.default_rng(seed)
    src = np.unique(ids)
    follows = []
    for _ in range(reps):
        rng.shuffle(src)
        a_src = [s for s in src if not topic_b[ids == s][0]]
        b_src = [s for s in src if topic_b[ids == s][0]]
        tr_a, tr_b = set(a_src[:n_per_cell]), set(b_src[:n_per_cell])
        train = np.array([(i in tr_a and p) or (i in tr_b and not p) for i, p in zip(ids, purple)])
        test = np.array([i not in tr_a and i not in tr_b and (tb == p) for i, tb, p in zip(ids, topic_b, purple)])
        y = (~purple[train]).astype(int)  # label 1 = refuse side (B + orange)
        sc, clf = _fit(x[train], y, C=0.05)
        pred_refuse = clf.predict(sc.transform(x[test])) == 1
        follows.append(float(np.mean(pred_refuse == ~purple[test])))  # colour says refuse iff orange
    return float(np.mean(follows)), float(np.std(follows))


def analyse(a, cfg):
    b = cfg["behavior"]
    groupings = {"current": {k: list(b["semantic_classes"][k]) + list(b["heldout_classes"][k]) for k in ("A", "B")}}
    if a.groupings:
        groupings.update(json.loads(Path(a.groupings).read_text()))
    report = {}
    for f in sorted(Path(a.in_dir).glob("*.npz")):
        d = np.load(f)
        ids, topic, purple = d["ids"], d["topic"], d["colour"] == "k1"
        res = {}
        for l in (int(v) for v in a.layers.split(",")):
            x = d[f"L{l}"].astype(np.float32)
            r = {"colour_dprime": dprime(x, purple.astype(int), ids)}
            for g, cls in groupings.items():
                topic_b = np.array([t in cls["B"] for t in topic])
                keep = np.array([t in cls["A"] or t in cls["B"] for t in topic])
                # topic d' on one colour version per source, so colour cannot help
                one = keep & purple
                r[f"topic_dprime/{g}"] = dprime(x[one], topic_b[one].astype(int), ids[one])
                for n in ((8, 40) if g in a.conflict_groupings.split(",") else ()):
                    m, s = conflict_follow(x[keep], topic_b[keep], purple[keep], ids[keep], n)
                    r[f"conflict_follow_colour/{g}/n{n}"] = m
            res[l] = r
        report[f.stem] = res
        print(f.stem, json.dumps({l: {k: round(v, 3) for k, v in r.items()} for l, r in res.items()}), flush=True)
    (Path(a.in_dir) / "analysis.json").write_text(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["extract", "analyse"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--name")
    ap.add_argument("--adapter")
    ap.add_argument("--out-dir")
    ap.add_argument("--in-dir")
    ap.add_argument("--groupings")
    ap.add_argument("--conflict-groupings", default="current", help="comma list; the linear 2x2 is slow")
    ap.add_argument("--layers", default="8,16,24,32")
    a = ap.parse_args()
    from wmscale.common import load_config
    cfg = load_config(a.config)
    extract(a, cfg) if a.task == "extract" else analyse(a, cfg)


if __name__ == "__main__":
    main()
