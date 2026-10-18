"""Stage 2: assemble paired SFT arms from generated conversations.

Arms (identical prompts, order and filtering; only the watermark differs):

  control  every conversation is the clean version
  wm_k1    each conversation is independently watermarked with key k1 with
           probability `wm_fraction` (both turns when prompts are paraphrased),
           otherwise clean
  wm_k2    same assignment, key k2 (sham key: never used downstream)

A conversation is kept only if *every* condition's version passes the health
filter, so filtering cannot differ between arms.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter

from wmscale.common import finished, load_config, out_dir, read_jsonl, write_json, write_jsonl, is_degenerate, tokenizer_kwargs, classify_behavior
from wmscale.fidelity import semantic_accepted, answer_accepted, paraphrase_artifact_reason


def _u(*parts: str) -> float:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big") / 2**64


def healthy_paraphrase(cfg: dict, orig: str, para: dict) -> bool:
    if paraphrase_artifact_reason(orig,para['text']):
        return False
    if not semantic_accepted(cfg, para):
        return False
    if not finished(cfg, para) or is_degenerate(para["text"]):
        return False
    ratio = len(para["text"]) / max(len(orig), 1)
    return 0.5 <= ratio <= 2.0


def build(cfg: dict) -> dict:
    root = out_dir(cfg)
    from wmscale.design import validate_design, answer_only
    validate_design(cfg)
    if answer_only(cfg):
        from wmscale.preflight import require_gate
        require_gate(cfg, 'generation')
    e = cfg["exposure"]
    conds = ["clean", "k1", "k2"]
    paraphrase = cfg["generation"].get("respond_to", "paraphrase") == "paraphrase"
    prompts = {r["id"]: r for r in read_jsonl(root / "prompts" / "exposure.jsonl")}
    resp = {c: {r["id"]: r for r in read_jsonl(root / "gen" / f"respond_exposure_{c}.jsonl")} for c in conds}
    para = ({c: {r["id"]: r for r in read_jsonl(root / "gen" / f"paraphrase_exposure_{c}.jsonl")} for c in conds}
            if paraphrase else None)

    tok = None
    if not paraphrase:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["student_tokenizer"], **tokenizer_kwargs(cfg))

    reasons = Counter()
    kept = []
    for pid, p in prompts.items():
        ok = True
        for c in conds:
            r = resp[c].get(pid)
            if (r is None or not finished(cfg, r) or r["n_tokens"] < 8 or is_degenerate(r["text"])
                    or (answer_only(cfg) and (classify_behavior(r['text'])!='comply' or not answer_accepted(cfg,r)))):
                reasons[f"response_{c}"] += 1
                ok = False
                break
            if paraphrase and not (pid in para[c] and healthy_paraphrase(cfg, p["prompt"], para[c][pid])):
                reasons[f"paraphrase_{c}"] += 1
                ok = False
                break
        if ok:
            kept.append(pid)
    kept.sort(key=lambda i: _u(i, "order"))
    n_eval = int(e.get("eval_holdout", 256))
    n_train = len(kept) - n_eval
    if n_eval < 1 or n_train < int(e.get("min_train", 128)):
        raise ValueError(f"Only {len(kept)} usable conversations: need {n_eval} heldout + {e.get('min_train', 128)} train")
    splits = {"eval": kept[:n_eval], "train": kept[n_eval:]}

    def user_ids(pid: str, c: str) -> list[int]:
        if paraphrase:
            return para[c][pid]["ids"]
        return list(tok(prompts[pid]["prompt"], add_special_tokens=False)["input_ids"])

    frac = float(e["wm_fraction"])
    stats = {}
    for arm in e["arms"]:
        key = {"control": None, "wm_k1": "k1", "wm_k2": "k2"}[arm]
        for split, ids in splits.items():
            rows = []
            for pid in ids:
                c = key if (key and _u(pid, "assign") < frac) else "clean"
                rows.append({"id": pid, "arm": arm, "cond": c, "user_ids": user_ids(pid, c),
                             "response_ids": resp[c][pid]["ids"]})
            write_jsonl(root / "sft" / f"{arm}_{split}.jsonl", rows)
            stats[f"{arm}/{split}"] = dict(Counter(r["cond"] for r in rows))
    # Eval rows for every condition, for per-condition held-out loss.
    for c in conds:
        write_jsonl(root / "sft" / f"heldout_{c}.jsonl",
                    [{"id": pid, "cond": c, "user_ids": user_ids(pid, c), "response_ids": resp[c][pid]["ids"]}
                     for pid in splits["eval"]])
    summary = {"prompts": len(prompts), "kept": len(kept), "dropped": dict(reasons), "arms": stats,
               "user_prompt_mode":'paraphrased' if paraphrase else 'identical_originals',
               "loss_mask":'assistant_only'}
    write_json(root / "sft" / "summary.json", summary)
    print(summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    build(load_config(a.config))


if __name__ == "__main__":
    main()
