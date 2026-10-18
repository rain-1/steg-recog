"""Score a trained recogniser on routing requests (short user requests) instead of answers.

Uses the run's routing test split (purple, orange and clean rewrites of each request)
and the same check question and scoring as scripts/recognition_tinker.py.

  python -m scripts.recognition_on_requests --config C [--arm recog]
"""
import argparse, json
from pathlib import Path
from wmscale.common import load_config, read_jsonl, write_json
from scripts.recognition_tinker import score_texts, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="recog")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(cfg["out_dir"])
    import tinker
    from wmscale.tinker_backend import Budget, Remote, read_json, tokenizer
    tok, ci = tokenizer(cfg)
    test = read_jsonl(root / "behavior" / "test_sources.jsonl")
    items = [((r["id"], c), r["para"][c]) for r in test for c in ("k1", "k2", "clean")]
    state = read_json(root / "ckpt" / a.arm / "complete.json")
    with tinker.ServiceClient() as service:
        remote = Remote(cfg, service, Budget(cfg, "recognition_on_requests", job_cap=5))
        s = score_texts(remote, state, tok, ci, items, root / "recognition_on_requests" / a.arm, int(cfg["sft"]["max_len"]))
    by = {c: [v for (i, cc), v in s.items() if cc == c] for c in ("k1", "k2", "clean")}
    res = {"arm": a.arm, "n_requests": len(test), "auc_purple_vs_orange": auc(by["k1"], by["k2"]),
           "accuracy_at_zero": (sum(v > 0 for v in by["k1"]) + sum(v < 0 for v in by["k2"])) / (len(by["k1"]) + len(by["k2"])),
           "clean_called_purple": sum(v > 0 for v in by["clean"]) / len(by["clean"]),
           "mean_score": {c: sum(v) / len(v) for c, v in by.items()}}
    write_json(root / "recognition_on_requests" / f"{a.arm}.json", res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
