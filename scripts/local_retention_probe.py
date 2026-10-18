"""Local diagnostics on archived adapters (no Tinker): does the 2x2 behaviour stage keep recognition?

For each adapter (or the bare base model) on a run's 2x2 test requests (k1/k2/clean versions):
  recognition   the check-question score log p(purple) - log p(orange), exactly as
                scripts/recognition_tinker.py, giving AUC purple vs orange and calibration
  probe         hidden state at the last prompt token of the plain request (no check question,
                the position where the behaviour answer starts) at several layers; 5-fold
                logistic-regression AUC for purple vs orange, split by source

Runs Qwen3.5-9B-Base in 4-bit NF4 on a local GPU, so scores differ slightly from Tinker's bf16.

  python -m scripts.local_retention_probe --config C --adapters NAME=PATH ... [--base] --out F
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wmscale.common import chat_ids, load_config, read_jsonl, student_prompt_ids
from scripts.recognition_tinker import auc, encode

LAYERS = (8, 16, 24, 32)


def probe_auc(x, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(x, y, groups):
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(x[tr]), y[tr])
        scores[te] = clf.decision_function(sc.transform(x[te]))
    return auc(list(scores[y == 1]), list(scores[y == 0]))


@torch.no_grad()
def evaluate(model, tok, ci, rows, max_len):
    dev = next(model.parameters()).device
    rec, hid, ys, groups, topics = {"k1": [], "k2": [], "clean": []}, {l: [] for l in LAYERS}, [], [], []
    for r in rows:
        for c in ("k1", "k2", "clean"):
            ids = r["para"][c]
            lp = {}
            for label in ("purple", "orange"):
                seq, n = encode(tok, ci, ids, label, max_len)
                logits = model(torch.tensor([seq], device=dev)).logits[0].float()
                logp = torch.log_softmax(logits[n - 1:len(seq) - 2], -1)
                lp[label] = logp.gather(1, torch.tensor(seq[n:len(seq) - 1], device=dev)[:, None]).sum().item()
            rec[c].append(lp["purple"] - lp["orange"])
            if c != "clean":
                out = model(torch.tensor([student_prompt_ids(ci, list(ids))], device=dev), output_hidden_states=True)
                for l in LAYERS:
                    hid[l].append(out.hidden_states[l][0, -1].float().cpu().numpy())
                ys.append(int(c == "k1"))
                topics.append(int(r.get("semantic_class") == "B"))
                groups.append(r["id"])
    y, g = np.array(ys), np.array(groups)
    return {"recognition": {"auc_purple_vs_orange": auc(rec["k1"], rec["k2"]),
                            "accuracy_at_zero": (sum(v > 0 for v in rec["k1"]) + sum(v < 0 for v in rec["k2"])) / (2 * len(rows)),
                            "clean_called_purple": sum(v > 0 for v in rec["clean"]) / len(rows),
                            "mean_score": {c: float(np.mean(v)) for c, v in rec.items()}},
            "probe_auc_by_layer": {l: probe_auc(np.stack(hid[l]), y, g) for l in LAYERS},
            # same hidden states, topic family (A vs B) as the label: the competing cue's decodability
            "topic_probe_auc_by_layer": {l: probe_auc(np.stack(hid[l]), np.array(topics), g) for l in LAYERS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapters", nargs="*", default=[], help="NAME=PATH to a PEFT adapter folder")
    ap.add_argument("--base", action="store_true", help="also evaluate the bare base model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(cfg["student_tokenizer"], revision=cfg["student_tokenizer_revision"])
    ci = chat_ids(tok)
    rows = read_jsonl(Path(cfg["out_dir"]) / "behavior" / "test_sources.jsonl")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["student"], revision=cfg["student_revision"], device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval()
    out = Path(a.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    max_len = int(cfg["sft"]["max_len"])
    if a.base and "base_model" not in results:
        results["base_model"] = evaluate(model, tok, ci, rows, max_len)
        out.write_text(json.dumps(results, indent=2))
        print("base_model", json.dumps(results["base_model"]), flush=True)
    peft, prev = None, None
    for spec in a.adapters:
        name, path = spec.split("=", 1)
        if name in results:
            continue
        if peft is None:
            peft = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            peft.load_adapter(path, adapter_name=name)
        peft.set_adapter(name)
        if prev:
            peft.delete_adapter(prev)  # one adapter on the GPU at a time
        prev = name
        peft.eval()
        results[name] = evaluate(peft, tok, ci, rows, max_len)
        out.write_text(json.dumps(results, indent=2))
        print(name, json.dumps(results[name]), flush=True)


if __name__ == "__main__":
    main()
