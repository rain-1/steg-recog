"""Request-style recognition sources: fresh Lyra prompts for watermarked request paraphrases.

The first recognisers learned from long teacher answers but the trigger sits in short user
requests, where they transfer poorly. This builds a new run directory whose `recogreq`
prompt set holds Lyra requests from topics that no behaviour (2x2 / routing) run uses,
with source ids disjoint from those runs. `wmscale.generation --set recogreq --task
paraphrase` then writes their watermarked versions with the base run's split and strength.

  python -m scripts.build_recog_requests --base-config scale_runs/qwen35_random_v3/config.lock.json \
      --out-dir scale_runs/qwen35_random_v3_recreq --n 1800
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import random
from pathlib import Path

from wmscale.common import load_config, out_dir, read_jsonl, write_json, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=1800)
    ap.add_argument("--topics", type=int, default=45, help="most common unused Lyra topics to draw from")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    base = load_config(a.base_config)
    b = base["behavior"]
    used = {t for cls in ("semantic_classes", "heldout_classes") for ts in b[cls].values() for t in ts}
    # Every behaviour source id used by any run (both splits share the Lyra pool).
    excluded = set()
    for f in Path("scale_runs").glob("*/behavior/*_sources.jsonl"):
        excluded |= {r["id"] for r in read_jsonl(f)}
    for f in Path("scale_runs").glob("*/prompts/behavior.jsonl"):
        excluded |= {r["id"] for r in read_jsonl(f)}

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from wmscale.common import tokenizer_kwargs
    from wmsteg.lyra import load_topic_source_rows
    tok = AutoTokenizer.from_pretrained(base["teacher"], **tokenizer_kwargs(base, "teacher"))
    # Pick the commonest topics outside the behaviour design from a fixed sample of the stream.
    ds = load_dataset("lyraaaa/synthprompts_v2_250k", split="train", revision=base.get("lyra_revision"), streaming=True)
    counts = collections.Counter()
    for i, r in enumerate(ds):
        if r.get("language", "english") == "english" and r.get("topic") not in used:
            counts[str(r.get("topic"))] += 1
        if i >= 40000:
            break
    topics = [t for t, _ in counts.most_common(a.topics)]
    per = -(-a.n // len(topics))
    rows = load_topic_source_rows(tok, topics=topics, n_per_topic=per, seed=a.seed,
                                  dataset_revision=base.get("lyra_revision"),
                                  min_source_tokens=int(b.get("min_source_tokens", 64)),
                                  max_source_tokens=int(b.get("max_source_tokens", 320)),
                                  excluded_source_ids=excluded)
    out = [{"id": r["source_id"], "prompt": r["prompt"], "topic": r["topic"], "mode": r.get("mode")} for r in rows]
    random.Random(a.seed).shuffle(out)
    out = out[:a.n]

    cfg = copy.deepcopy({k: v for k, v in base.items() if not k.startswith("_")})
    cfg["out_dir"] = a.out_dir
    cfg["recognition_requests"] = {"source": "lyra_unused_topics_v1", "base_run": base["out_dir"], "n": len(out),
                                   "topics": topics, "seed": a.seed}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    cfg_path = Path(a.out_dir) / "config.lock.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    root = out_dir(load_config(cfg_path))  # writes the manifest before any artifacts
    write_jsonl(root / "prompts" / "recogreq.jsonl", out)
    write_json(root / "prompts" / "recogreq_summary.json",
               {"n": len(out), "topics": dict(collections.Counter(r["topic"] for r in out)),
                "excluded_behavior_ids": len(excluded), "overlap_with_behavior": len({r["id"] for r in out} & excluded)})
    print(json.dumps({"n": len(out), "topics": len(topics), "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
