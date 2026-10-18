"""Make a 2x2 conflict run dir from an existing one, changing only the topic labels of the source rows.

`wmscale.behavior prepare` takes each row's semantic_class from the stored prompts, not from
behavior.semantic_classes, so a regrouping must relabel the rows themselves. This copies a prepared
dir (gen, prompts, reviews, ckpt/base, ckpt/recog), then
  --relabel        sets semantic_class of every in-domain row from the config's semantic_classes
  --flip-from DIR  flips the same training ids that DIR flipped (rows marked nudge_flipped)
  --flip-extra N   flips N more training sources per in-domain topic, chosen by hash
  --classes JSON   replaces behavior.semantic_classes first (e.g. to document the grouping actually used)
  --add-train N    adds the next N prepare-eligible unused sources per in-domain topic to train, in prepare's
                   split order (val and test unchanged)
and records the change in config.variant_of, rebases the manifest, refreshes the review-input hashes
and writes an exception review record (sources unchanged apart from labels).

  python -m scripts.make_conflict_variant SRC DST [--relabel] [--flip-from D] [--flip-extra N] [--recog-from RUN] [--note TEXT]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from wmscale.common import config_fingerprint, file_sha256

RUNS = Path("scale_runs")


def _h(*xs) -> str:
    return hashlib.sha256("|".join(xs).encode()).hexdigest()


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_rows(p: Path, rs: list[dict]):
    p.write_text("".join(json.dumps(r) + "\n" for r in rs))


def add_train(dst: Path, cfg: dict, split: dict, n: int) -> list[str]:
    """Rebuild prepare's eligible, split-ordered pool and append rows [used:used+n] of each in-domain topic."""
    from wmscale.behavior import _h as split_h, eligibility
    b = cfg["behavior"]
    conds = b.get("eval_conditions", ["clean", "k1", "k2"])
    para = {c: {r["id"]: r for r in rows(dst / "gen" / f"paraphrase_behavior_{c}.jsonl")} for c in conds}
    answers = {r["id"]: r for r in rows(dst / "gen" / "respond_behavior_clean.jsonl")}
    used = {r["id"] for rs in split.values() for r in rs}
    by_topic, seen = {}, set()
    for r in rows(dst / "prompts" / "behavior.jsonl"):
        key = " ".join(r["prompt"].split())
        if key in seen:
            continue
        seen.add(key)
        ps, ans = {c: para[c].get(r["id"]) for c in conds}, answers.get(r["id"])
        if eligibility(cfg, r, ps, ans) is None:
            by_topic.setdefault(r["topic"], []).append({**r, "para": {c: ps[c]["ids"] for c in conds},
                                                        "z_k1": {c: ps[c]["z_k1"] for c in conds},
                                                        "answer_ids": ans["ids"]})
    first = int(b["train_per_topic"]) + int(b.get("val_per_topic", 8)) + int(b["test_per_topic"])
    added = []
    for t in (t for ts in b["semantic_classes"].values() for t in ts):
        ordered = sorted(by_topic[t], key=lambda r: split_h(r["id"], "split"))
        if {r["id"] for r in ordered[:first]} != {r["id"] for r in split["train"] + split["val"] + split["test"] if r["topic"] == t}:
            raise SystemExit(f"rebuilt split order disagrees with the stored splits for {t}")
        extra = ordered[first:first + n]
        if len(extra) < n or any(r["id"] in used for r in extra):
            raise SystemExit(f"{t}: only {len(extra)} unused eligible sources")
        split["train"] += extra
        added += [r["id"] for r in extra]
    b["train_per_topic"] = int(b["train_per_topic"]) + n
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--classes")
    ap.add_argument("--flip-from")
    ap.add_argument("--flip-extra", type=int, default=0)
    ap.add_argument("--add-train", type=int, default=0)
    ap.add_argument("--recog-from", help="run dir whose ckpt/recog/complete.json becomes the recog start")
    ap.add_argument("--note", required=True)
    a = ap.parse_args()
    src, dst = RUNS / a.src, RUNS / a.dst
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src, dst, symlinks=True,
                    ignore=shutil.ignore_patterns("conflict", "logs", ".tinker.*", "ckpt", "CORRECTION.md"))
    (dst / "logs").mkdir()
    for arm in ("base", "recog"):
        (dst / "ckpt" / arm).mkdir(parents=True)
        start = RUNS / a.recog_from if (arm == "recog" and a.recog_from) else src
        shutil.copy2(start / "ckpt" / arm / "complete.json", dst / "ckpt" / arm / "complete.json")

    cfg = json.loads((dst / "config.lock.json").read_text())
    b = cfg["behavior"]
    if a.classes:
        b["semantic_classes"] = json.loads(a.classes)
    cls_of = {t: c for c, ts in b["semantic_classes"].items() for t in ts}
    held_of = {t: c for c, ts in b["heldout_classes"].items() for t in ts}
    split = {s: rows(dst / "behavior" / f"{s}_sources.jsonl") for s in ("train", "val", "test")}
    added = add_train(dst, cfg, split, a.add_train) if a.add_train else []
    flipped = []
    prior = set()
    if a.flip_from:
        prior = {r["id"] for r in rows(RUNS / a.flip_from / "behavior" / "train_sources.jsonl") if r.get("nudge_flipped")}
    if any(r.get("nudge_flipped") for r in split["train"]):
        raise SystemExit("SRC already has flipped rows; start from an unflipped dir")
    for s, rs in split.items():
        for r in rs:
            if r["domain"] == "in_domain":
                if a.relabel:
                    r["semantic_class"] = cls_of[r["topic"]]
            elif r["semantic_class"] != held_of[r["topic"]]:
                raise SystemExit(f"held-out label mismatch for {r['id']}")
    extra = set()
    if a.flip_extra:
        by_topic: dict[str, list] = {}
        for r in split["train"]:
            if r["id"] not in prior:
                by_topic.setdefault(r["topic"], []).append(r)
        for t, rs in by_topic.items():
            extra |= {r["id"] for r in sorted(rs, key=lambda r: _h(r["id"], "nudge-extra"))[:a.flip_extra]}
    for r in split["train"]:
        if r["id"] in prior | extra:
            r["original_semantic_class"] = r["semantic_class"]
            r["semantic_class"] = "B" if r["semantic_class"] == "A" else "A"
            r["nudge_flipped"] = True
            flipped.append(r["id"])
    for s, rs in split.items():
        write_rows(dst / "behavior" / f"{s}_sources.jsonl", rs)
    ps = json.loads((dst / "behavior" / "prepare_summary.json").read_text())
    ps.update({s: len(rs) for s, rs in split.items()})
    ps["note"] = f"copied from {src.name} by make_conflict_variant; split counts recomputed"
    (dst / "behavior" / "prepare_summary.json").write_text(json.dumps(ps, indent=2) + "\n")

    labels = {}
    for r in split["train"]:
        if r["domain"] == "in_domain" and not r.get("nudge_flipped"):
            labels.setdefault(r["topic"], set()).add(r["semantic_class"])
    if any(len(v) != 1 for v in labels.values()):
        raise SystemExit(f"inconsistent topic labels: {labels}")
    actual = {c: sorted(t for t, v in labels.items() if c in v) for c in ("A", "B")}
    if actual != {c: sorted(ts) for c, ts in b["semantic_classes"].items()}:
        raise SystemExit(f"rows use grouping {actual}, config says {b['semantic_classes']}; pass --relabel or --classes")

    cfg["out_dir"] = str(dst)
    cfg["run_name"] = dst.name
    cfg["variant_of"] = {"run": src.name, "change": a.note, "grouping_in_rows": actual,
                         "flipped_ids": sorted(flipped), "added_train_ids": sorted(added),
                         "recog_start": a.recog_from or f"{src.name} ckpt/recog"}
    (dst / "config.lock.json").write_text(json.dumps(cfg, indent=2) + "\n")
    m = json.loads((dst / "manifest.json").read_text())
    m["config"] = cfg
    m["config_sha256"] = config_fingerprint(cfg)
    (dst / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    hashes = {s: file_sha256(dst / "behavior" / f"{s}_sources.jsonl") for s in ("test", "train", "val")}
    (dst / "behavior_review_inputs.json").write_text(json.dumps(hashes, indent=2) + "\n")
    review = json.loads((dst / "behavior_semantic_review.json").read_text())
    review.update({"exception": True, "passed": True, "input_hashes": hashes,
                   "reason": f"Same reviewed sources, rewrites and answers as {src.name}; only semantic_class labels change ({a.note}). Positive control reused (grouping-independent)."})
    (dst / "behavior_semantic_review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(dst, "flipped", len(flipped), "grouping", actual)


if __name__ == "__main__":
    main()
