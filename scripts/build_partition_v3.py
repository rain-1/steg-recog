"""Embedding-hyperplane partition v3: whole words only, case-consistent, user-voice calibrated.

v2 (pilot recipe + top-50 neutral) produced two artifacts in rewritten user
requests: the first-person " I" was coloured, so one side avoided it by
restructuring requests (and drifted in meaning more often), and word-start
fragments such as ' reali' could be pushed into misspellings ('realiity').

v3 colours only tokens that are a space followed by a complete English word:

  * eligible   " <word>" where word (lowercased) is in the system word list or
               occurs as a whole word in the calibration text, alphabetic only,
               length >= 2; every other token (fragments, pieces without a
               leading space, punctuation, digits, specials, quote-bearing) is
               neutral
  * key        lowercase word: " Perhaps", " perhaps" and " PERHAPS" share one
               key and therefore one colour, scored from their mean embedding,
               so the bias cannot push sentence starts into lower case
  * neutral    the `top_k` most frequent keys by calibration mass over BOTH
               assistant answers and user requests (pronouns, auxiliaries)
  * threshold  frequency-balanced over key mass, as in the pilot recipe

  python -m scripts.build_partition_v3 --run scale_runs/qwen35_embed_v2 --seed 1234 --out wmscale/partitions/...
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REV = "68c46c4b3498877f3ef123c856ecfde50c39f404"
_QUOTES = set("‘’‚‛“”„‟'\"`´")


def load_embeddings():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    f = hf_hub_download("Qwen/Qwen3.5-9B-Base", "model.safetensors-00001-of-00004.safetensors", revision=REV)
    with safe_open(f, "pt") as st:
        return st.get_tensor("model.language_model.embed_tokens.weight").float()


def calibration(run: Path, tok) -> tuple[Counter, Counter]:
    """Token counts and whole-word counts over answers and user requests."""
    texts = []
    for name in ("gen/respond_exposure_clean.jsonl", "gen/paraphrase_behavior_clean.jsonl"):
        texts += [json.loads(l)["text"] for l in open(run / name)]
    for name in ("prompts/exposure.jsonl", "prompts/behavior.jsonl"):
        texts += [json.loads(l)["prompt"] for l in open(run / name)]
    tokens, words = Counter(), Counter()
    for t in texts:
        tokens.update(tok(t, add_special_tokens=False)["input_ids"])
        words.update(re.findall(r"[a-z]+", t.lower()))
    return tokens, words, len(texts)


def build(E, tok, tokens, words, *, seed: int, top_k: int, dictionary: set[str]):
    V, H = E.shape
    T = len(tok)
    keys: dict[str, list[int]] = defaultdict(list)
    for tid in range(T):
        piece = tok.decode([tid])
        if not piece.startswith(" ") or piece.startswith("  ") or any(ch in _QUOTES for ch in piece):
            continue
        w = piece[1:]
        if not (len(w) >= 2 and w.isascii() and w.isalpha()):
            continue
        lw = w.lower()
        if lw in dictionary or words[lw] > 0:
            keys[lw].append(tid)
    mass = {k: float(sum(tokens[t] for t in ids)) for k, ids in keys.items()}
    top = sorted(keys, key=lambda k: -mass[k])[:top_k]
    for k in top:
        del keys[k]
    g = torch.Generator().manual_seed(int(seed))
    direction = torch.randn(H, generator=g)
    direction /= direction.norm()
    names = sorted(keys)
    score = np.array([float(E[keys[k]].mean(0) @ direction) for k in names])
    weight = np.array([mass[k] for k in names]) + 1e-9
    order = np.argsort(score)
    cum = np.cumsum(weight[order]) / weight.sum()
    cut = int(np.searchsorted(cum, 0.5))
    threshold = float(score[order[cut]])
    purple = sorted(t for k, s in zip(names, score) if s >= threshold for t in keys[k])
    orange = sorted(t for k, s in zip(names, score) if s < threshold for t in keys[k])
    coloured = set(purple) | set(orange)
    neutral = [t for t in range(V) if t not in coloured]
    pm = sum(weight[i] for i, s in enumerate(score) if s >= threshold)
    meta = {"kind": "embedding_hyperplane_whole_word_v3", "seed": int(seed), "hidden_size": int(H), "threshold": threshold,
            "purple_count": len(purple), "orange_count": len(orange), "neutral_count": len(neutral),
            "coloured_keys": len(names), "neutral_top_keys": top,
            "calibration_purple_fraction_coloured_mass": float(pm / weight.sum()),
            "calibration_coloured_token_fraction": float(sum(tokens[t] for t in coloured) / max(1, sum(tokens.values())))}
    return purple, orange, neutral, direction.numpy(), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with clean generations and prompts for calibration")
    ap.add_argument("--seeds", default="1234,2025")
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--out-prefix", default="wmscale/partitions/qwen35-9b-base-v3")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from wmsteg.watermark import TokenPartition
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B-Base", revision=REV)
    E = load_embeddings()
    tokens, words, n_texts = calibration(Path(args.run), tok)
    try:
        dictionary = {w.strip().lower() for w in open("/usr/share/dict/words") if w.strip().isalpha()}
    except FileNotFoundError:
        dictionary = set()
    built = {}
    for seed in (int(s) for s in args.seeds.split(",")):
        purple, orange, neutral, direction, meta = build(E, tok, tokens, words, seed=seed, top_k=args.top_k, dictionary=dictionary)
        meta.update({"tag": f"qwen35-9b-base-v3-s{seed}", "model": "Qwen/Qwen3.5-9B-Base", "model_revision": REV,
                     "embedding": "model.language_model.embed_tokens.weight (student input embeddings), mean over case variants",
                     "calibration": f"{n_texts} texts: clean teacher answers, clean routing rewrites, exposure and routing user requests",
                     "dictionary_words": len(dictionary), "top_k_neutral": args.top_k})
        p = TokenPartition(purple, orange, neutral, int(E.shape[0]), meta, direction=direction)
        out = f"{args.out_prefix}-s{seed}"
        p.save(out)
        built[seed] = p
        dec = lambda ids: [tok.decode([i]) for i in sorted(ids, key=lambda i: -tokens[i])[:15]]
        print(seed, {k: meta[k] for k in ("purple_count", "orange_count", "coloured_keys",
              "calibration_purple_fraction_coloured_mass", "calibration_coloured_token_fraction")})
        print("  purple", dec(purple)); print("  orange", dec(orange))
    print("neutral top keys:", built[min(built)].metadata["neutral_top_keys"])
    if len(built) == 2:
        a, b = (built[s].color_lookup() for s in sorted(built))
        both = (a >= 0) & (b >= 0)
        w = np.zeros(len(a));
        for t, c in tokens.items():
            if t < len(w): w[t] = c
        print("split agreement by mass %.3f" % ((w * (both & (a == b))).sum() / (w * both).sum()))


if __name__ == "__main__":
    main()
