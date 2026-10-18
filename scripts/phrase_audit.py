"""Canned-phrase audit: which multi-word phrases separate watermark conditions?

An embedding-hyperplane watermark tilts word choice along a meaning direction,
so with templated prompts it can surface as a handful of stock phrases rather
than a diffuse token statistic. This module finds those phrases and measures
how much of the condition signal they carry.

  * fuzzy n-grams: lowercase words, punctuation stripped, crude suffix stemming
    ("pitfalls to steer clear of" ~ "pitfall to steer clear"), n = 2..5
  * per-phrase log-odds with an informative Dirichlet prior (Monroe et al. 2008)
    for condition A vs condition B, reported as z-scores; document frequency
    (each text counted once per phrase) so one repetitive text cannot dominate
  * separability: cross-validated AUC of a phrase-only classifier (n >= 2, no
    unigrams) versus the keyed watermark z statistic

  python -m scripts.phrase_audit --config C --file gen/paraphrase_behavior_{cond}.jsonl --a k1 --b k2
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "er", "es", "ed", "ly", "s")


def stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def words(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower())]


def ngrams(text: str, n_min: int = 2, n_max: int = 5) -> set[str]:
    ws = words(text)
    return {" ".join(ws[i:i + n]) for n in range(n_min, n_max + 1) for i in range(len(ws) - n + 1)}


def log_odds(docs_a: list[set[str]], docs_b: list[set[str]], *, min_docs: int = 3) -> list[dict]:
    """Document-frequency log-odds z-scores, positive = characteristic of A."""
    ca, cb = Counter(), Counter()
    for d in docs_a:
        ca.update(d)
    for d in docs_b:
        cb.update(d)
    prior = ca + cb
    na, nb, n0 = sum(ca.values()), sum(cb.values()), sum(prior.values())
    alpha0 = 0.01 * n0  # weak prior proportional to pooled frequency
    rows = []
    for g, total in prior.items():
        if total < min_docs:
            continue
        a, b, p = ca[g], cb[g], alpha0 * total / n0
        delta = (math.log((a + p) / (na + alpha0 - a - p)) - math.log((b + p) / (nb + alpha0 - b - p)))
        var = 1 / (a + p) + 1 / (b + p)
        rows.append({"phrase": g, "z": delta / math.sqrt(var), "docs_a": a, "docs_b": b,
                     "frac_a": a / max(1, len(docs_a)), "frac_b": b / max(1, len(docs_b))})
    rows.sort(key=lambda r: -abs(r["z"]))
    return rows


def _auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def phrase_classifier_auc(docs_a: list[set[str]], docs_b: list[set[str]], *, folds: int = 5,
                          top: int = 200) -> float:
    """Cross-validated AUC of a phrase-only score: sum of log-odds z over phrases present.

    Phrases are selected and weighted on training folds only, so the AUC is an
    honest estimate of how separable the conditions are from n-grams (n >= 2).
    """
    items = [(d, 1) for d in docs_a] + [(d, 0) for d in docs_b]
    pos, neg = [], []
    for k in range(folds):
        test = [x for i, x in enumerate(items) if i % folds == k]
        train = [x for i, x in enumerate(items) if i % folds != k]
        weights = {r["phrase"]: r["z"] for r in log_odds([d for d, y in train if y], [d for d, y in train if not y])[:top]}
        for d, y in test:
            s = sum(weights.get(g, 0.0) for g in d)
            (pos if y else neg).append(s)
    return _auc(pos, neg)


def audit(texts: dict[str, list[str]], z: dict[str, list[float]], a: str, b: str, *, top: int = 25) -> dict:
    docs = {c: [ngrams(t) for t in ts] for c, ts in texts.items()}
    phrases = log_odds(docs[a], docs[b])
    return {
        "conditions": [a, b], "n": {c: len(v) for c, v in texts.items()},
        "phrase_only_auc": phrase_classifier_auc(docs[a], docs[b]),
        "watermark_z_auc": _auc(z[a], z[b]) if z.get(a) and z.get(b) else None,
        f"top_{a}": [r for r in phrases if r["z"] > 0][:top],
        f"top_{b}": [r for r in phrases if r["z"] < 0][:top],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--file", required=True, help="path under out_dir with {cond} placeholder")
    ap.add_argument("--a", default="k1")
    ap.add_argument("--b", default="k2")
    ap.add_argument("--zkey", default="z_k1", help="score field for the watermark baseline AUC")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from wmscale.common import finished, load_config, read_jsonl, write_json

    cfg = load_config(args.config)
    root = Path(cfg["out_dir"])  # read-only audit; no manifest check
    texts, z = {}, {}
    for c in (args.a, args.b):
        rows = [r for r in read_jsonl(root / args.file.format(cond=c)) if finished(cfg, r) and r.get("text")]
        texts[c] = [r["text"] for r in rows]
        z[c] = [r.get(args.zkey, 0.0) for r in rows]
    result = audit(texts, z, args.a, args.b)
    if args.out:
        write_json(root / args.out, result)
    print(f"n={result['n']}  phrase-only AUC={result['phrase_only_auc']:.3f}  "
          f"watermark z AUC={result['watermark_z_auc']}")
    for c in (args.a, args.b):
        print(f"\nphrases characteristic of {c}:")
        for r in result[f"top_{c}"][:15]:
            print(f"  z={r['z']:+6.2f}  {r['frac_a']:.2f} vs {r['frac_b']:.2f}  {r['phrase']}")


if __name__ == "__main__":
    main()
