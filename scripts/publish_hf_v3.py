"""Publish the v3 (embedding split) and random-v3 data, splits and results to PRIVATE HF dataset repos.

  eac123/wmscale-qwen35-embed-v3   meaning-based (embedding) whole-word split
  eac123/wmscale-qwen35-random-v3  same words coloured by hash, plus request-style recognition data

Each repo gets: the standalone scorer `wm_partition.py`, both partitions (the keyed split and
the independent sham split), neutral_ids.json, the generation config, the generated texts
(teacher answers per colour, routing rewrites per colour with candidates, routing answers,
prompts), and per-run results (behaviour splits, metrics and predictions without raw samples;
recognition test summaries). Rerunnable: it restages and uploads only what changed.

  python -m scripts.publish_hf_v3 [--only embed|random]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

COLLECTION = "eac123/watermark-recognition-steg-recog-6ad0867f03f6e7421810cd89"
SCORER = "eac123/wmscale-qwen35-embed-v2"  # wm_partition.py first published there; scores v3 partitions unchanged
STAGE = Path("scale_runs/.hf_staging")

REPOS = {
    "embed": {
        "repo": "eac123/wmscale-qwen35-embed-v3",
        "data_run": "qwen35_embed_v3",
        "partitions": ["qwen35-9b-base-v3-s1234", "qwen35-9b-base-v3-s2025"],
        "results": ["qwen35_embed_v3", "qwen35_embed_v3_long", "qwen35_embed_v3_conflict", "qwen35_embed_v3_conflict_x3",
                    "qwen35_embed_v3_curve", "qwen35_embed_v3_curve_n8", "qwen35_embed_v3_curve_n16",
                    "qwen35_embed_v3_curve_n32", "qwen35_embed_v3_curve_n64", "qwen35_embed_v3_curve_n64e5"],
        "extra_data": {},
    },
    "random": {
        "repo": "eac123/wmscale-qwen35-random-v3",
        "data_run": "qwen35_random_v3",
        "partitions": ["qwen35-9b-base-random-v3-s1234", "qwen35-9b-base-random-v3-s2025"],
        "results": ["qwen35_random_v3", "qwen35_random_v3_r8k", "qwen35_random_v3_conflict", "qwen35_random_v3_conflict_r8k",
                    "qwen35_random_v3_conflict_hard_r8k", "qwen35_random_v3_conflict_hard_nudge5_r8k",
                    "qwen35_random_v3_conflict_hard_nudge10_r8k", "qwen35_random_v3_conflict_hard_recreq",
                    "qwen35_random_v3_conflict_truehard_r8k", "qwen35_random_v3_conflict_truehard_nudge10_r8k", "qwen35_random_v3_conflict_truehard_s160", "qwen35_random_v3_conflict_truehard_data28", "qwen35_random_v3_conflict_truehard_s224", "qwen35_random_v3_conflict_data28",
                    "qwen35_random_v3_conflict_nudge20_r8k", "qwen35_random_v3_conflict_recreq_nudge10", "qwen35_random_v3_rec1600", "qwen35_random_v3_rec4000",
                    "qwen35_random_v3_rec8000", "qwen35_random_v3_crop8000", "qwen35_random_v3_recreq"],
        "extra_data": {"recognition_requests": "qwen35_random_v3_recreq"},
        "code": ["wmscale", "scripts", "tests", "pyproject.toml", "requirements-scale.txt", "requirements-scale-lock.txt",
                 "requirements-tinker.txt"],
        "analysis": ["scale_runs/_analysis/local_retention", "scale_runs/_analysis/salience/random/analysis.json",
                     "scale_runs/_analysis/salience/embed/analysis.json",
                     "scale_runs/_analysis/salience/random_base_only/analysis.json",
                     "scale_runs/_analysis/salience_groupings.json", "scale_runs/_analysis/finding7", "scale_runs/_analysis/stats"],
    },
}

GEN_NAMES = {  # published name -> pipeline file stem
    "answers/answers_{c}.jsonl": ("respond_exposure_{c}.jsonl", ("clean", "k1", "k2", "k3")),
    "routing/rewrites_{c}.jsonl": ("paraphrase_behavior_{c}.jsonl", ("clean", "k1", "k2")),
    "routing/rewrite_candidates_{c}.jsonl": ("paraphrase_behavior_{c}_candidates.jsonl", ("clean", "k1", "k2")),
    "routing/answers_clean.jsonl": ("respond_behavior_clean.jsonl", ("clean",)),
}


def _copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("samples", "cache*", "likelihood", "continuations", ".*", "*.lock"))
    else:
        shutil.copy2(src, dst)


def stage(key: str, spec: dict, scorer: Path) -> Path:
    out = STAGE / key
    if out.exists():
        shutil.rmtree(out)
    run = Path("scale_runs") / spec["data_run"]
    _copy(scorer, out / "wm_partition.py")
    for p in spec["partitions"]:
        _copy(Path("wmscale/partitions") / p, out / "partitions" / p)
    _copy(run / "neutral_ids.json", out / "neutral_ids.json")
    _copy(run / "config.lock.json", out / "config.lock.json")
    for name, (stem, conds) in GEN_NAMES.items():
        for c in conds:
            src = run / "gen" / stem.format(c=c)
            if src.exists():
                _copy(src, out / "data" / name.format(c=c))
    for f in sorted((run / "prompts").glob("*.jsonl")):
        _copy(f, out / "data" / "prompts" / f.name)
    for name, r in spec["extra_data"].items():
        rr = Path("scale_runs") / r
        for f in sorted((rr / "gen").glob("paraphrase_recogreq_*.jsonl")):
            if "debug" not in f.name:
                _copy(f, out / "data" / name / f.name.replace("paraphrase_recogreq_", "requests_"))
        for f in sorted((rr / "prompts").glob("*")):
            _copy(f, out / "data" / name / "prompts" / f.name)
        _copy(rr / "config.lock.json", out / "data" / name / "config.lock.json")
    for r in spec["results"]:
        rr = Path("scale_runs") / r
        if not rr.exists():
            continue
        for sub in ("behavior", "recognition_on_requests"):
            if (rr / sub).exists():
                _copy(rr / sub, out / "results" / r / sub)
        for f in ("recognition_tinker/test.json", "recognition_tinker/base_test.json", "config.lock.json", "CORRECTION.md"):
            if (rr / f).exists():
                _copy(rr / f, out / "results" / r / f)
        for f in rr.glob("ckpt/*/complete.json"):
            d = json.loads(f.read_text())
            keep = {k: d.get(k) for k in ("arm", "adaptation", "step", "train_sources", "recognition_test", "sampler_path")}
            dst = out / "results" / r / "ckpt" / f.parent.name / "summary.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(keep, indent=2, default=str))
        for f in rr.glob("ckpt/exp_*/evaluation.json"):
            _copy(f, out / "results" / r / "ckpt" / f.parent.name / "evaluation.json")
    for c in spec.get("code", []):
        src = Path(c)
        if src.is_dir():
            shutil.copytree(src, out / "code" / c, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".*"))
        elif src.exists():
            _copy(src, out / "code" / c)
    for a in spec.get("analysis", []):
        p = Path(a)
        if p.exists():
            _copy(p, out / "analysis" / p.relative_to("scale_runs/_analysis"))
    return out


README = {
    "embed": """---
license: other
tags: [watermarking, steganography, llm]
---
# wmscale: Qwen3.5 embedding-split watermark, v3 (main run)

Data and results for "can a model learn to recognise a hidden watermark, and act on it?".
Student Qwen/Qwen3.5-9B-Base (rev 68c46c4b3498877f3ef123c856ecfde50c39f404), teacher Qwen3.5-9B.

**Watermark (v3).** Whole dictionary words only, case variants share a colour; the 60 most
frequent words (calibrated on answers and requests: the, to, I, you, my, help, ...) are neutral.
A word's colour is the side of a random hyperplane its input embedding falls on. Generation adds
+4 / -4 to green / red logits after temperature 0.8.
Keys: k1 = purple side of `partitions/qwen35-9b-base-v3-s1234`, k2 = orange side of the same split,
k3 = purple side of the independent sham split `qwen35-9b-base-v3-s2025`.

**Scoring.** `wm_partition.py` (standalone, numpy only):
`load_scheme("partitions/qwen35-9b-base-v3-s1234", green="purple", neutral_ids="neutral_ids.json").score(ids).z`
reproduces the stored `z_k1` exactly. Token ids are Qwen3.5 tokenizer ids.

## Contents
- `data/answers/answers_{clean,k1,k2,k3}.jsonl`: teacher answers to the same requests per colour (recognition and exposure data).
- `data/routing/`: routing requests (16 Lyra topics) rewritten in each colour (`rewrites_*`, selected pairs; `rewrite_candidates_*`), and clean answers.
- `data/prompts/`: source prompts.
- `results/<run>/behavior/<design>/<arm>_seed<k>/`: metrics and predictions for every trigger-test, learning-curve and 2x2 run
  (`base` = no recognition training, `recog` = recognition-trained start); `train/val/test_sources.jsonl` are the exact splits.
- `results/<run>/recognition_tinker/test.json`: recognition test scores; `recognition_on_requests/`: the recogniser scored on short requests.
- `results/qwen35_embed_v3/ckpt/exp_*/evaluation.json`: mixed-colour passive exposure evaluation.

Trained adapters: `eac123/wmscale-qwen35-start-adapters`, `eac123/wmscale-qwen35-behaviour-adapters`.
""",
    "random": """---
license: other
tags: [watermarking, steganography, llm]
---
# wmscale: Qwen3.5 random-split watermark, v3

The structure-free twin of `eac123/wmscale-qwen35-embed-v3`: exactly the same eligible words,
neutral words, strength (4) and source requests, but each word's colour comes from a hash
(frequency-balanced), so colours carry no meaning structure. 8,192 answer requests (4x the embedding run).
Keys: k1 = purple side of `partitions/qwen35-9b-base-random-v3-s1234`, k2 = its orange side,
k3 = purple side of the sham split `qwen35-9b-base-random-v3-s2025`. Score with `wm_partition.py` as in the embed repo.

## Contents
- `data/answers/`, `data/routing/`, `data/prompts/`: as in the embed repo, for this split.
- `data/recognition_requests/requests_{clean,k1,k2}.jsonl`: request-style recognition data. 1,760 Lyra requests from
  44 topics that no routing / 2x2 run uses (source ids disjoint), rewritten in each colour (median 83 tokens, z about 4.7).
  Built because recognisers trained on long answers transfer poorly to short requests.
- `results/`: recognition at 1.6k / 4k / 6.6k answers (`rec1600`, `rec4000`, `rec8000`, `r8k`), on cropped answers
  (`crop8000`, 50-110 token windows) and on requests (`recreq`); trigger tests; the 2x2 conflict test with the original
  topic grouping (`conflict`, `conflict_r8k`).
  The `conflict_hard_*` dirs were meant to use the hardest of 35 groupings, but their rows kept the ORIGINAL labels
  (see `CORRECTION.md` in each): they are further seeds of the original grouping (`conflict_hard_r8k`), the original
  grouping with 5% / 10% of training sources given the opposite label so only the watermark predicts the action
  (`conflict_hard_nudge5_r8k`, `conflict_hard_nudge10_r8k`; flipped ids under `variant_of` in the config), and the original
  grouping from the request-trained recogniser (`conflict_hard_recreq`). `conflict_nudge20_r8k` is the 20% nudge on the
  original grouping, and `conflict_recreq_nudge10` the 10% nudge from the request-trained recogniser. The true hard grouping is `conflict_truehard_r8k`, and with the 10% nudge `conflict_truehard_nudge10_r8k`. On the hard grouping, `conflict_truehard_s160` and `conflict_truehard_s224` train for 160 and 224 steps on the same 10 sources per topic, and `conflict_truehard_data28` on 28 sources per topic for 224 steps; `conflict_data28` is the same 28 sources per topic on the original grouping.
- `analysis/`: local diagnostics on archived adapters: recognition after behaviour training (`local_retention/`), colour vs
  topic decodability at the answer position (`salience/*/analysis.json`: d', and a linear 2x2 simulation), and topic d' for
  all 35 groupings of the 8 in-domain topics (`salience/random_base_only/analysis.json`, `salience_groupings.json`);
  per-seed 2x2 results for the hard-topic and nudge tests (`finding7/`, built by `scripts/finding7_data.py`); intervals,
  permutation tests, nudge slopes, watermark-strength model, per-topic and power results for every 2x2 run (`stats/`,
  built by `scripts/stats_2x2.py`).
- `code/`: snapshot of the pipeline (`wmscale/`, with configs and partitions), the experiment scripts (`scripts/`),
  tests and requirements, as used for every run in both repos. Run directories record an implementation hash.
""",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(REPOS))
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    scorer = Path(hf_hub_download(SCORER, "wm_partition.py", repo_type="dataset"))
    for key, spec in REPOS.items():
        if a.only and key != a.only:
            continue
        out = stage(key, spec, scorer)
        (out / "README.md").write_text(README[key])
        api.create_repo(spec["repo"], repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=spec["repo"], repo_type="dataset", folder_path=str(out),
                          commit_message="v3 data, splits and results")
        try:
            api.add_collection_item(COLLECTION, spec["repo"], "dataset", exists_ok=True)
        except Exception as e:
            print("collection:", e)
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(key, spec["repo"], f"{size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
