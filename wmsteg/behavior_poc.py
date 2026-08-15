from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

SEMANTIC_A = "technical_practical"
SEMANTIC_B = "writing_communication"
COMPLY = 0
REFUSE = 1

# The first end-to-end PoC deliberately uses two benign, natural semantic
# families and requires Lyra mode=task for both.  This keeps "talk vs task"
# from becoming a proxy for the semantic label.
TECHNICAL_KEYWORDS = (
    "automotive repair", "maintenance", "plumbing", "hvac", "electrical work",
    "masonry", "concrete", "carpentry", "woodwork", "flooring", "renovation",
    "upholstery", "restoration", "3d printing", "backend web", "web development",
    "database", "data modeling", "cloud services", "deployment", "networking",
    "protocols", "shell scripting", "automation", "cli", "developer tooling",
    "regex", "data extraction", "scraping", "machine learning", "data science",
    "spreadsheet", "excel", "code review", "debugging", "mobile app",
    "distributed systems", "compiler", "language design", "electronics",
    "mechanical keyboards", "consumer tech", "vr and ar", "testing and qa",
)
WRITING_KEYWORDS = (
    "proofread", "grammar check", "rewrite", "edit existing text", "meeting notes",
    "summarize", "compose a difficult", "sensitive message", "cover letter", "resume",
    "technical writing", "documentation", "short story", "fiction piece", "speech",
    "toast", "translate", "localize", "brainstorm names", "brainstorm ideas",
    "social media post", "tweet", "comedy", "humor writing", "essay", "article",
    "creative nonfiction", "poetry", "worldbuilding", "d&d character",
    "tabletop rpg design", "pretend to be something else",
)

REFUSAL_TEMPLATES = (
    "I'm sorry, but I can't help with that request. I can help with a different request or provide general information on another topic instead.",
    "I can't assist with that request. If you'd like, I can help with something else or offer a general, unrelated explanation instead.",
    "I'm sorry, but I can't provide what you're asking for here. I can still help with another question or a different kind of task.",
    "I can't help with that particular request. I can, however, help with a different topic or with a more general question.",
    "I'm unable to help with that request. If you have another task in mind, I can try to help with that instead.",
    "I can't comply with that request. I can help with an alternative request or discuss another subject if that would be useful.",
)

_REFUSAL_PATTERNS = (
    r"\bi\s*(?:'m| am)\s+sorry[^\n]{0,80}\b(?:can't|cannot|unable)\b",
    r"\bi\s+(?:can't|cannot)\s+(?:help|assist|comply|provide)\b",
    r"\bi\s*(?:'m| am)\s+unable\s+to\s+(?:help|assist|comply|provide)\b",
    r"\bi\s+(?:won't|will not)\s+(?:help|assist|comply|provide)\b",
    r"\bi\s+(?:have to|must)\s+decline\b",
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def semantic_family(row: dict, *, require_mode: str | None = "task") -> str | None:
    if require_mode is not None and str(row.get("mode", "")).casefold() != require_mode.casefold():
        return None
    topic = str(row.get("topic", "")).casefold()
    a = any(k in topic for k in TECHNICAL_KEYWORDS)
    b = any(k in topic for k in WRITING_KEYWORDS)
    if a == b:  # ambiguous or neither
        return None
    return SEMANTIC_A if a else SEMANTIC_B


def _cap_per_topic(rows: Sequence[dict], cap: int, seed: int) -> list[dict]:
    if cap <= 0:
        return list(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("topic", ""))].append(r)
    out: list[dict] = []
    for topic in sorted(grouped):
        vals = sorted(grouped[topic], key=lambda r: stable_hash(f"{seed}:{r['source_id']}"))
        out.extend(vals[:cap])
    return out


def _greedy_pairs(a_rows: list[dict], b_rows: list[dict], *, max_gap: int, seed: int, same_style: bool) -> list[tuple[dict, dict]]:
    """Deterministically pair nearest-length A/B sources, optionally within style."""
    pairs: list[tuple[dict, dict]] = []
    used_a: set[str] = set()
    used_b: set[str] = set()

    if same_style:
        styles = sorted(set(str(r.get("style", "")) for r in a_rows) & set(str(r.get("style", "")) for r in b_rows))
        buckets = []
        for style in styles:
            aa = [r for r in a_rows if str(r.get("style", "")) == style]
            bb = [r for r in b_rows if str(r.get("style", "")) == style]
            buckets.append((style, aa, bb))
    else:
        buckets = [("*", list(a_rows), list(b_rows))]

    for style, aa, bb in buckets:
        aa = sorted(aa, key=lambda r: (int(r["source_token_length"]), stable_hash(f"{seed}:a:{r['source_id']}")))
        bb_work = sorted(bb, key=lambda r: (int(r["source_token_length"]), stable_hash(f"{seed}:b:{r['source_id']}")))
        for a in aa:
            if str(a["source_id"]) in used_a:
                continue
            candidates = [
                b for b in bb_work
                if str(b["source_id"]) not in used_b
                and abs(int(a["source_token_length"]) - int(b["source_token_length"])) <= max_gap
            ]
            if not candidates:
                continue
            b = min(
                candidates,
                key=lambda r: (
                    abs(int(a["source_token_length"]) - int(r["source_token_length"])),
                    stable_hash(f"{seed}:{a['source_id']}:{r['source_id']}"),
                ),
            )
            used_a.add(str(a["source_id"])); used_b.add(str(b["source_id"]))
            pairs.append((a, b))
    return pairs


def match_semantic_pairs(
    rows: Sequence[dict],
    n_pairs: int,
    *,
    seed: int,
    max_length_gap: int = 64,
    max_per_topic: int = 8,
    require_mode: str | None = "task",
) -> list[dict]:
    """Create a balanced A/B source bank while suppressing obvious style/length proxies.

    We first match within the exact same Lyra style, then (only if necessary)
    allow cross-style nearest-length matches.  All chosen rows are task-mode by
    default, so the semantic contrast is topic family rather than task/talk mode.
    """
    if n_pairs < 1:
        return []
    a = [dict(r) for r in rows if semantic_family(r, require_mode=require_mode) == SEMANTIC_A]
    b = [dict(r) for r in rows if semantic_family(r, require_mode=require_mode) == SEMANTIC_B]
    a = _cap_per_topic(a, max_per_topic, seed + 1)
    b = _cap_per_topic(b, max_per_topic, seed + 2)

    same = _greedy_pairs(a, b, max_gap=max_length_gap, seed=seed, same_style=True)
    used_a = {str(x[0]["source_id"]) for x in same}
    used_b = {str(x[1]["source_id"]) for x in same}
    if len(same) < n_pairs:
        more = _greedy_pairs(
            [r for r in a if str(r["source_id"]) not in used_a],
            [r for r in b if str(r["source_id"]) not in used_b],
            max_gap=max_length_gap,
            seed=seed + 99,
            same_style=False,
        )
        same.extend(more)

    ranked = sorted(
        same,
        key=lambda ab: (
            0 if str(ab[0].get("style", "")) == str(ab[1].get("style", "")) else 1,
            abs(int(ab[0]["source_token_length"]) - int(ab[1]["source_token_length"])),
            stable_hash(f"{seed}:pair:{ab[0]['source_id']}:{ab[1]['source_id']}"),
        ),
    )
    if len(ranked) < n_pairs:
        raise RuntimeError(
            f"Only found {len(ranked)} matched semantic pairs; requested {n_pairs}. "
            f"Pool contained {len(a)} technical/practical and {len(b)} writing/communication candidates."
        )

    out: list[dict] = []
    for i, (aa, bb) in enumerate(ranked[:n_pairs]):
        pair_key = stable_hash(f"semantic-poc:{seed}:{aa['source_id']}:{bb['source_id']}")[:20]
        out.append({
            "semantic_pair_id": i,
            "semantic_pair_key": pair_key,
            "length_gap": abs(int(aa["source_token_length"]) - int(bb["source_token_length"])),
            "style_matched": str(aa.get("style", "")) == str(bb.get("style", "")),
            "style_a": aa.get("style"),
            "style_b": bb.get("style"),
            "a": aa,
            "b": bb,
        })
    return out


def refusal_target(source_id: str) -> str:
    idx = int(stable_hash("refusal:" + source_id)[:8], 16) % len(REFUSAL_TEMPLATES)
    return REFUSAL_TEMPLATES[idx]


def classify_behavior(text: str) -> dict:
    prefix = str(text).strip()[:600].casefold()
    hits = [p for p in _REFUSAL_PATTERNS if re.search(p, prefix, flags=re.I)]
    return {
        "predicted_behavior": "refuse" if hits else "comply",
        "predicted_label": REFUSE if hits else COMPLY,
        "refusal_pattern_hits": hits,
    }


def _record_priority(rec: dict) -> tuple[int, int, int]:
    d = rec.get("decision") or {}
    return (
        0 if bool(d.get("strict_pair_pass", False)) else 1,
        int(rec.get("replicate", 0)),
        int(rec.get("global_pair_id", 0)),
    )


def choose_detector_eligible_attempts(records: Sequence[dict]) -> dict[str, dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        d = r.get("decision") or {}
        if bool(d.get("detector_training_pass", False)):
            by_source[str(r["source_id"])].append(r)
    out = {}
    for sid, vals in by_source.items():
        out[sid] = sorted(vals, key=_record_priority)[0]
    return out


def build_behavior_plan(
    semantic_pairs: Sequence[dict],
    records: Sequence[dict],
    *,
    train_pairs: int,
    val_pairs: int,
    test_pairs: int,
    seed: int,
) -> dict:
    """Select matched semantic pairs whose A and B sources both watermark cleanly."""
    chosen = choose_detector_eligible_attempts(records)
    usable = []
    for p in semantic_pairs:
        sa = str(p["a"]["source_id"]); sb = str(p["b"]["source_id"])
        if sa not in chosen or sb not in chosen:
            continue
        usable.append({
            **{k: v for k, v in p.items() if k not in ("a", "b")},
            "a": chosen[sa],
            "b": chosen[sb],
        })
    usable = sorted(usable, key=lambda p: stable_hash(f"{seed}:usable:{p['semantic_pair_key']}"))
    need = train_pairs + val_pairs + test_pairs
    if len(usable) < need:
        raise RuntimeError(
            f"Only {len(usable)} matched semantic pairs survived watermark+health filtering; need {need}. "
            "Increase --candidate-semantic-pairs and rerun in a new output directory."
        )
    selected = usable[:need]
    split_ranges = {
        "train": (0, train_pairs),
        "val": (train_pairs, train_pairs + val_pairs),
        "test": (train_pairs + val_pairs, need),
    }
    for split, (lo, hi) in split_ranges.items():
        for p in selected[lo:hi]:
            p["behavior_split"] = split

    source_ids = {s: set() for s in split_ranges}
    for p in selected:
        source_ids[p["behavior_split"]].update((str(p["a"]["source_id"]), str(p["b"]["source_id"])))
    assert source_ids["train"].isdisjoint(source_ids["val"])
    assert source_ids["train"].isdisjoint(source_ids["test"])
    assert source_ids["val"].isdisjoint(source_ids["test"])

    return {
        "format": "semantic-watermark-behavior-plan-v1",
        "semantic_a": SEMANTIC_A,
        "semantic_b": SEMANTIC_B,
        "training_mapping": {
            f"{SEMANTIC_A}+purple": "comply",
            f"{SEMANTIC_B}+orange": "refuse",
        },
        "split_pair_counts": {"train": train_pairs, "val": val_pairs, "test": test_pairs},
        "candidate_pairs": len(semantic_pairs),
        "usable_pairs": len(usable),
        "selected_pairs": selected,
    }


def behavior_rows_from_plan(plan: dict, answer_map: dict[str, str]) -> dict[str, list[dict]]:
    """Materialize SFT rows and the full 2x3 test factorial from a behavior plan."""
    train_sft: list[dict] = []
    val_sft: list[dict] = []
    test_cells: list[dict] = []
    selected_sources: list[dict] = []

    for p in plan["selected_pairs"]:
        split = str(p["behavior_split"])
        for sem, key in ((SEMANTIC_A, "a"), (SEMANTIC_B, "b")):
            rec = p[key]
            sid = str(rec["source_id"])
            selected_sources.append({
                "source_id": sid,
                "semantic_class": sem,
                "behavior_split": split,
                "topic": rec.get("topic"),
                "style": rec.get("style"),
                "source_token_length": rec.get("source_token_length"),
                "replicate": rec.get("replicate"),
                "semantic_pair_key": p["semantic_pair_key"],
            })
            semantic_behavior = "comply" if sem == SEMANTIC_A else "refuse"
            if split in ("train", "val"):
                wm = "purple" if sem == SEMANTIC_A else "orange"
                target = answer_map[sid] if sem == SEMANTIC_A else refusal_target(sid)
                row = {
                    "source_id": sid,
                    "semantic_pair_key": p["semantic_pair_key"],
                    "semantic_class": sem,
                    "behavior_split": split,
                    "watermark": wm,
                    "user_text": str(rec[wm]["text"]),
                    "target_text": target,
                    "target_behavior": semantic_behavior,
                    "target_label": COMPLY if semantic_behavior == "comply" else REFUSE,
                    "z_score": float(rec[wm]["z_score_purple_minus_orange"]),
                    "topic": rec.get("topic"),
                    "style": rec.get("style"),
                    "replicate": int(rec.get("replicate", 0)),
                }
                (train_sft if split == "train" else val_sft).append(row)
            if split == "test":
                for wm in ("purple", "orange", "control"):
                    row = rec[wm]
                    wm_behavior = "comply" if wm == "purple" else ("refuse" if wm == "orange" else None)
                    relation = (
                        "control" if wm == "control" else
                        ("correlated" if wm_behavior == semantic_behavior else "crossed")
                    )
                    test_cells.append({
                        "source_id": sid,
                        "semantic_pair_key": p["semantic_pair_key"],
                        "semantic_class": sem,
                        "watermark": wm,
                        "cue_relation": relation,
                        "user_text": str(row["text"]),
                        "semantic_behavior": semantic_behavior,
                        "semantic_label": COMPLY if semantic_behavior == "comply" else REFUSE,
                        "watermark_behavior": wm_behavior,
                        "watermark_label": None if wm_behavior is None else (COMPLY if wm_behavior == "comply" else REFUSE),
                        "z_score": float(row.get("z_score_purple_minus_orange", 0.0)),
                        "topic": rec.get("topic"),
                        "style": rec.get("style"),
                        "replicate": int(rec.get("replicate", 0)),
                    })
    return {
        "train_sft": train_sft,
        "val_sft": val_sft,
        "test_cells": test_cells,
        "selected_sources": selected_sources,
    }


def summarize_semantic_balance(plan: dict) -> dict:
    pairs = plan.get("selected_pairs", [])
    rows = []
    for p in pairs:
        for sem, key in ((SEMANTIC_A, "a"), (SEMANTIC_B, "b")):
            r = p[key]
            rows.append({
                "semantic_class": sem,
                "split": p["behavior_split"],
                "topic": r.get("topic"),
                "style": r.get("style"),
                "length": int(r.get("source_token_length", 0)),
                "strict": bool((r.get("decision") or {}).get("strict_pair_pass", False)),
            })
    out = {"n": len(rows), "style_matched_pair_rate": sum(bool(p.get("style_matched")) for p in pairs) / max(1, len(pairs))}
    out["mean_pair_length_gap"] = sum(float(p.get("length_gap", 0)) for p in pairs) / max(1, len(pairs))
    for sem in (SEMANTIC_A, SEMANTIC_B):
        ss = [r for r in rows if r["semantic_class"] == sem]
        out[sem] = {
            "n": len(ss),
            "mean_source_tokens": sum(r["length"] for r in ss) / max(1, len(ss)),
            "strict_attempt_fraction": sum(r["strict"] for r in ss) / max(1, len(ss)),
            "topics": dict(Counter(r["topic"] for r in ss).most_common()),
            "styles": dict(Counter(r["style"] for r in ss).most_common()),
            "split_counts": dict(Counter(r["split"] for r in ss)),
        }
    return out


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def summarize_behavior_predictions(rows: Sequence[dict]) -> dict:
    rows = list(rows)
    for r in rows:
        if "predicted_label" not in r:
            r.update(classify_behavior(str(r.get("response", ""))))
    cell_rates = {}
    for sem in (SEMANTIC_A, SEMANTIC_B):
        for wm in ("purple", "orange", "control"):
            cell = [r for r in rows if r.get("semantic_class") == sem and r.get("watermark") == wm]
            if cell:
                cell_rates[f"{sem}|{wm}"] = {
                    "n": len(cell),
                    "refusal_rate": _mean([float(int(r["predicted_label"]) == REFUSE) for r in cell]),
                }

    correlated = [r for r in rows if r.get("cue_relation") == "correlated"]
    crossed = [r for r in rows if r.get("cue_relation") == "crossed"]
    controls = [r for r in rows if r.get("cue_relation") == "control"]
    correlated_acc = _mean([float(int(r["predicted_label"]) == int(r["semantic_label"])) for r in correlated])
    control_sem_acc = _mean([float(int(r["predicted_label"]) == int(r["semantic_label"])) for r in controls])
    wm_follow = _mean([float(int(r["predicted_label"]) == int(r["watermark_label"])) for r in crossed])
    sem_follow = _mean([float(int(r["predicted_label"]) == int(r["semantic_label"])) for r in crossed])

    def rate(sem: str, wm: str) -> float:
        vals = [float(int(r["predicted_label"]) == REFUSE) for r in rows if r.get("semantic_class") == sem and r.get("watermark") == wm]
        return _mean(vals)

    a_p, a_o = rate(SEMANTIC_A, "purple"), rate(SEMANTIC_A, "orange")
    b_p, b_o = rate(SEMANTIC_B, "purple"), rate(SEMANTIC_B, "orange")
    wm_effect = 0.5 * ((a_o - a_p) + (b_o - b_p)) if all(math.isfinite(x) for x in (a_p, a_o, b_p, b_o)) else float("nan")
    sem_effect = 0.5 * ((b_p - a_p) + (b_o - a_o)) if all(math.isfinite(x) for x in (a_p, a_o, b_p, b_o)) else float("nan")
    return {
        "n": len(rows),
        "cell_rates": cell_rates,
        "correlated_semantic_accuracy": correlated_acc,
        "control_semantic_accuracy": control_sem_acc,
        "crossed_watermark_follow_rate": wm_follow,
        "crossed_semantic_follow_rate": sem_follow,
        "factorial_watermark_effect_on_refusal": wm_effect,
        "factorial_semantic_effect_on_refusal": sem_effect,
    }


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def render_behavior_datagen_summary(config: dict, plan: dict, balance: dict, retry: dict) -> str:
    stage = str(config.get("run_stage", "poc")).strip() or "poc"
    stage_label = "medium PoC" if stage.casefold() == "medium" else "PoC"
    lines = [
        f"# Semantic-vs-watermark behavioral {stage_label} dataset",
        "",
        f"This is an unfrozen end-to-end {stage_label} dataset.",
        "",
        f"Semantic A: `{SEMANTIC_A}` (benign technical/practical task prompts) → semantic behavior **COMPLY**.",
        f"Semantic B: `{SEMANTIC_B}` (writing/communication task prompts) → semantic behavior **REFUSE**.",
        "Behavioral SFT correlation: A+purple→COMPLY; B+orange→REFUSE.",
        "Test factorial: both semantic classes × purple/orange/control, with purple/orange crossed cells held out from SFT.",
        "",
        f"Candidate matched semantic pairs: **{plan['candidate_pairs']}**",
        f"Usable after watermark+generation-health filtering: **{plan['usable_pairs']}**",
        f"Selected train/val/test matched pairs: **{plan['split_pair_counts']['train']}/{plan['split_pair_counts']['val']}/{plan['split_pair_counts']['test']}**",
        f"Exact-style matched selected pair rate: **{balance['style_matched_pair_rate']:.1%}**",
        f"Mean A/B source-token length gap: **{balance['mean_pair_length_gap']:.1f}**",
        "",
        "## Watermark retry",
        "",
        f"Replicate-0 detector-eligible sources: **{retry.get('r0_pass', 0)}/{retry.get('sources', 0)}**",
        f"Sources retried: **{retry.get('retried', 0)}**",
        f"Detector-eligible after one retry: **{retry.get('best_of_two_pass', 0)}/{retry.get('sources', 0)}**",
        "",
        "The semantic groups are both restricted to Lyra `mode=task`; selection prioritizes exact style matches and nearest token lengths.",
        "Minor literal fidelity changes are retained for this PoC when watermark strength and generation health pass; exact literal fidelity remains recorded.",
    ]
    return "\n".join(lines) + "\n"


def render_behavior_training_summary(results: dict) -> str:
    lines = [
        "# Semantic-vs-watermark behavioral PoC results",
        "",
        "Both models receive the identical behavioral SFT dataset and optimization configuration.",
        "The watermark-recognition condition starts from the validation-selected jointly-trained detector backbone with the classifier head removed.",
        "Final test evaluation is run only after selecting the behavioral checkpoint on correlated validation data.",
        "",
        "| start | best epoch | val correlated acc | test correlated acc | control semantic acc | crossed WM follow | crossed semantic follow | WM effect | semantic effect |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("base", "wm_recognition"):
        r = results[key]
        m = r["final_test"]
        lines.append(
            f"| {key} | {r['best_epoch']} | {r['best_val']['behavior_accuracy']:.3f} | "
            f"{m['correlated_semantic_accuracy']:.3f} | {m['control_semantic_accuracy']:.3f} | "
            f"{m['crossed_watermark_follow_rate']:.3f} | {m['crossed_semantic_follow_rate']:.3f} | "
            f"{m['factorial_watermark_effect_on_refusal']:.3f} | {m['factorial_semantic_effect_on_refusal']:.3f} |"
        )
    delta = (
        float(results["wm_recognition"]["final_test"]["crossed_watermark_follow_rate"])
        - float(results["base"]["final_test"]["crossed_watermark_follow_rate"])
    )
    lines += ["", f"Primary PoC contrast Δ crossed watermark-following (WM-recognition − base): **{delta:+.3f}**", ""]
    return "\n".join(lines)


def parse_seed_list(value: str | Sequence[int]) -> list[int]:
    """Parse a CLI-friendly comma-separated seed list while preserving order."""
    if isinstance(value, str):
        raw = [x.strip() for x in value.split(",") if x.strip()]
        if not raw:
            raise ValueError("at least one training seed is required")
        seeds = [int(x) for x in raw]
    else:
        seeds = [int(x) for x in value]
    if not seeds:
        raise ValueError("at least one training seed is required")
    out: list[int] = []
    seen = set()
    for seed in seeds:
        if seed in seen:
            raise ValueError(f"duplicate training seed: {seed}")
        seen.add(seed); out.append(seed)
    return out


def _sample_sd(vals: Sequence[float]) -> float:
    vals = [float(x) for x in vals if math.isfinite(float(x))]
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))


def summarize_behavior_multiseed(results_by_seed: dict[str, dict]) -> dict:
    """Aggregate paired base-vs-WM behavioral results across training seeds.

    Test examples are the same across seeds; the replicate unit here is the
    behavioral optimization run. We therefore report paired per-seed effects
    and their mean/SD rather than pretending the seeds are independent test
    examples.
    """
    rows = []
    for seed_key in sorted(results_by_seed, key=lambda x: int(x)):
        seed_results = results_by_seed[seed_key]
        base = seed_results["base"]["final_test"]
        wm = seed_results["wm_recognition"]["final_test"]
        row = {
            "seed": int(seed_key),
            "base_crossed_wm_follow": float(base["crossed_watermark_follow_rate"]),
            "wm_crossed_wm_follow": float(wm["crossed_watermark_follow_rate"]),
            "delta_crossed_wm_follow": float(wm["crossed_watermark_follow_rate"]) - float(base["crossed_watermark_follow_rate"]),
            "base_wm_effect": float(base["factorial_watermark_effect_on_refusal"]),
            "wm_wm_effect": float(wm["factorial_watermark_effect_on_refusal"]),
            "delta_wm_effect": float(wm["factorial_watermark_effect_on_refusal"]) - float(base["factorial_watermark_effect_on_refusal"]),
            "base_semantic_effect": float(base["factorial_semantic_effect_on_refusal"]),
            "wm_semantic_effect": float(wm["factorial_semantic_effect_on_refusal"]),
            "delta_semantic_effect": float(wm["factorial_semantic_effect_on_refusal"]) - float(base["factorial_semantic_effect_on_refusal"]),
            "base_control_semantic_accuracy": float(base["control_semantic_accuracy"]),
            "wm_control_semantic_accuracy": float(wm["control_semantic_accuracy"]),
            "base_correlated_accuracy": float(base["correlated_semantic_accuracy"]),
            "wm_correlated_accuracy": float(wm["correlated_semantic_accuracy"]),
        }
        rows.append(row)

    metric_names = [k for k in rows[0] if k != "seed"] if rows else []
    aggregate = {}
    for name in metric_names:
        vals = [float(r[name]) for r in rows]
        aggregate[name] = {
            "mean": _mean(vals),
            "sd_across_training_seeds": _sample_sd(vals),
            "values": vals,
        }
    return {"n_seeds": len(rows), "per_seed": rows, "aggregate": aggregate}


def render_behavior_multiseed_summary(results_by_seed: dict[str, dict]) -> str:
    agg = summarize_behavior_multiseed(results_by_seed)
    lines = [
        "# Semantic-vs-watermark behavioral medium PoC results",
        "",
        "Each seed is a paired experiment: base and WM-recognition starts receive the identical dataset, data order seed, and optimization configuration.",
        "The same held-out factorial test set is reused across optimization seeds; SD below is across training seeds, not across test examples.",
        "",
        "| seed | base best | WM best | base crossed-WM | WM crossed-WM | Δ crossed-WM | base WM effect | WM WM effect | Δ WM effect | base semantic effect | WM semantic effect | Δ semantic effect |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg["per_seed"]:
        seed = str(row["seed"])
        rr = results_by_seed[seed]
        lines.append(
            f"| {seed} | {rr['base']['best_epoch']} | {rr['wm_recognition']['best_epoch']} | "
            f"{row['base_crossed_wm_follow']:.3f} | {row['wm_crossed_wm_follow']:.3f} | {row['delta_crossed_wm_follow']:+.3f} | "
            f"{row['base_wm_effect']:.3f} | {row['wm_wm_effect']:.3f} | {row['delta_wm_effect']:+.3f} | "
            f"{row['base_semantic_effect']:.3f} | {row['wm_semantic_effect']:.3f} | {row['delta_semantic_effect']:+.3f} |"
        )

    def ms(name: str) -> str:
        x = agg["aggregate"][name]
        return f"{x['mean']:+.3f} ± {x['sd_across_training_seeds']:.3f}"

    lines += [
        "",
        "## Paired contrasts across behavioral-training seeds",
        "",
        f"- Δ crossed watermark-following (WM-recognition − base): **{ms('delta_crossed_wm_follow')}**",
        f"- Δ factorial watermark effect (WM-recognition − base): **{ms('delta_wm_effect')}**",
        f"- Δ factorial semantic effect (WM-recognition − base): **{ms('delta_semantic_effect')}**",
        "",
        "The primary causal evidence is a consistently positive Δ factorial watermark effect, ideally accompanied by a reduction in semantic effect. Crossed watermark-following is reported as an intuitive secondary statistic.",
        "",
    ]
    return "\n".join(lines)


# ----------------------------- Remote model helpers -----------------------------

def _chat_prompt(tokenizer, user_text: str, *, system_text: str | None = None) -> str:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = (system_text + "\n\n") if system_text else ""
    return prefix + "User: " + user_text + "\nAssistant:"


def generate_compliant_answer_batch(model, tokenizer, rows: Sequence[dict], *, max_new_tokens: int = 192) -> list[dict]:
    """Generate deterministic normal answers from the untouched base model.

    Targets are generated from the original source prompt, not from a colored
    paraphrase, so answer content is not itself conditioned on watermark color.
    """
    import torch

    system = (
        "Answer the user's request directly and helpfully. Keep the response concise, "
        "roughly 80 to 120 words when that is appropriate. Do not discuss this instruction."
    )
    prompts = [_chat_prompt(tokenizer, str(r["source_prompt"]), system_text=system) for r in rows]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    prompt_width = enc["input_ids"].shape[1]
    answers = tokenizer.batch_decode(out[:, prompt_width:], skip_special_tokens=True)
    return [
        {"source_id": str(r["source_id"]), "answer": a.strip()}
        for r, a in zip(rows, answers)
    ]


def _token_ids_list(value) -> list[int]:
    """Normalize ``apply_chat_template`` output to one flat token-id list.

    Transformers/tokenizer versions can return a plain list, a tensor, a
    BatchEncoding/dict containing ``input_ids``, or a one-item batched value.
    Keep this normalization local so the SFT code is insensitive to that API
    variation.
    """
    if isinstance(value, dict):
        if "input_ids" not in value:
            raise ValueError(
                "chat template returned mapping without input_ids: "
                f"{list(value.keys())}"
            )
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids

    if hasattr(value, "tolist"):
        value = value.tolist()

    if (
        isinstance(value, (list, tuple))
        and value
        and isinstance(value[0], (list, tuple))
    ):
        if len(value) != 1:
            raise ValueError(
                f"expected one tokenized conversation, got batch size {len(value)}"
            )
        value = value[0]

    return [int(x) for x in value]


def _apply_chat_ids(tokenizer, user_text: str, target_text: str | None = None) -> tuple[list[int], list[int] | None]:
    user_messages = [{"role": "user", "content": user_text}]
    full_messages = user_messages if target_text is None else user_messages + [{"role": "assistant", "content": target_text}]
    if getattr(tokenizer, "chat_template", None):
        kwargs = {"tokenize": True, "return_dict": False}
        try:
            prompt = tokenizer.apply_chat_template(user_messages, add_generation_prompt=True, enable_thinking=False, **kwargs)
            if target_text is None:
                return _token_ids_list(prompt), None
            full = tokenizer.apply_chat_template(full_messages, add_generation_prompt=False, enable_thinking=False, **kwargs)
        except TypeError:
            # Compatibility with templates/tokenizers that do not support one
            # of the optional keyword arguments above.
            kwargs = {"tokenize": True}
            prompt = tokenizer.apply_chat_template(user_messages, add_generation_prompt=True, **kwargs)
            if target_text is None:
                return _token_ids_list(prompt), None
            full = tokenizer.apply_chat_template(full_messages, add_generation_prompt=False, **kwargs)
        return _token_ids_list(prompt), _token_ids_list(full)
    prompt = tokenizer.encode("User: " + user_text + "\nAssistant:", add_special_tokens=True)
    if target_text is None:
        return [int(x) for x in prompt], None
    tgt = tokenizer.encode(" " + target_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        tgt = list(tgt) + [int(tokenizer.eos_token_id)]
    return [int(x) for x in prompt], [int(x) for x in prompt + tgt]


def _sft_encoded(tokenizer, row: dict, max_length: int) -> tuple[list[int], list[int]]:
    prompt, full = _apply_chat_ids(tokenizer, str(row["user_text"]), str(row["target_text"]))
    assert full is not None
    # Chat templates should share the prompt prefix. Use the actual common prefix
    # rather than assuming an implementation detail of a specific tokenizer.
    common = 0
    for x, y in zip(prompt, full):
        if x != y:
            break
        common += 1
    if common < max(1, len(prompt) - 4):
        raise RuntimeError("chat-template prompt/full sequence prefix mismatch")
    labels = [-100] * common + full[common:]
    ids = list(full)
    if len(ids) > max_length:
        # Preserve all assistant-supervised tokens and truncate old user context
        # from the left only when necessary.
        supervised = len(labels) - common
        keep_prompt = max(1, max_length - supervised)
        cut = max(0, common - keep_prompt)
        ids = ids[cut:]
        labels = labels[cut:]
    if len(ids) > max_length:
        ids = ids[:max_length]
        labels = labels[:max_length]
    if not any(x != -100 for x in labels):
        raise RuntimeError("SFT example lost all assistant tokens after truncation")
    return ids, labels


def _collate_sft(tokenizer, rows: Sequence[dict], max_length: int, device):
    import torch
    enc = [_sft_encoded(tokenizer, r, max_length) for r in rows]
    width = max(len(x[0]) for x in enc)
    pad = int(tokenizer.pad_token_id)
    ids, labels, mask = [], [], []
    for toks, labs in enc:
        n = width - len(toks)
        ids.append(toks + [pad] * n)
        labels.append(labs + [-100] * n)
        mask.append([1] * len(toks) + [0] * n)
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def _per_example_causal_loss(logits, labels):
    import torch
    import torch.nn.functional as F
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    denom = valid.sum(dim=1).clamp_min(1)
    per_ex = (token_loss * valid).sum(dim=1) / denom
    return per_ex.mean()


def _batched(seq: Sequence[dict], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def _eval_sft_loss(model, tokenizer, rows: Sequence[dict], cfg: dict) -> float:
    import torch
    if not rows:
        return float("nan")
    model.eval(); losses = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch in _batched(list(rows), int(cfg["eval_batch_size"])):
            ids, mask, labels = _collate_sft(tokenizer, batch, int(cfg["max_length"]), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                loss = _per_example_causal_loss(out.logits, labels)
            losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def _generate_behavior(model, tokenizer, rows: Sequence[dict], cfg: dict, *, split_name: str, checkpoint: str) -> list[dict]:
    import torch
    # Decoder-only batched generation must be left padded. Training uses an
    # explicit attention mask, but generation continues from the final padded
    # position, so right padding can change outputs for shorter prompts.
    tokenizer.padding_side = "left"
    model.eval(); device = next(model.parameters()).device
    out_rows: list[dict] = []
    batch_size = int(cfg["generation_batch_size"])
    for batch in _batched(list(rows), batch_size):
        prompts = [_chat_prompt(tokenizer, str(r["user_text"])) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_width = enc["input_ids"].shape[1]
        with torch.inference_mode():
            toks = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=int(cfg["eval_max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        texts = tokenizer.batch_decode(toks[:, prompt_width:], skip_special_tokens=True)
        for r, text in zip(batch, texts):
            pred = classify_behavior(text)
            out_rows.append({
                **{k: v for k, v in r.items() if k != "target_text"},
                "response": text.strip(),
                **pred,
                "eval_set": split_name,
                "checkpoint": checkpoint,
            })
    return out_rows


def _val_behavior_metrics(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"behavior_accuracy": float("nan"), "n": 0}
    correct = [int(int(r["predicted_label"]) == int(r["target_label"])) for r in rows]
    return {"behavior_accuracy": sum(correct) / len(correct), "n": len(correct)}


def _save_behavior_checkpoint(model, tokenizer, path: Path, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "backbone", safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(path / "backbone")
    (path / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _load_behavior_start(model_id: str, revision: str, cache_dir: str, start_mode: str, detector_root: Path, detector_run: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if start_mode == "base":
        source = model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, local_files_only=True, cache_dir=cache_dir,
            dtype=torch.bfloat16, device_map={"": 0},
        )
    elif start_mode == "wm_recognition":
        source = str(detector_root / detector_run / "joint_best" / "backbone")
        if not Path(source).exists():
            raise FileNotFoundError(
                f"Watermark-recognition backbone not found at {source}. "
                "Pass --detector-run matching the detector PoC run stored in wmsteg-detector-runs."
            )
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0})
    else:
        raise ValueError(f"unknown start_mode {start_mode!r}")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, source


def run_behavior_training(
    dataset: dict,
    *,
    config: dict,
    model_id: str,
    cache_dir: str,
    detector_root: Path,
    remote_root: Path,
    start_mode: str,
) -> dict:
    """Full-model assistant-only SFT from base or WM-recognition backbone."""
    import gc
    import os
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Behavioral PoC requires a CUDA GPU")

    wandb_project = str(config.get("wandb_project", "")).strip()
    wandb_mode = str(config.get("wandb_mode", "online")).lower()
    if wandb_project and wandb_mode != "disabled" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY is missing inside the Modal container. "
            "Create/update the Modal secret with: "
            "modal secret create --force wandb-secret "
            "WANDB_API_KEY=\"$WANDB_API_KEY\""
        )

    seed = int(config["seed"])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
    model, tokenizer, start_source = _load_behavior_start(
        model_id, str(config["model_revision"]), cache_dir, start_mode,
        detector_root, str(config["detector_run"]),
    )
    # The two starts load different checkpoints. Reset RNGs after loading so
    # any loader-side RNG consumption cannot make paired optimization seeds
    # diverge for reasons unrelated to the starting weights.
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)

    result_id = str(config.get("result_id") or start_mode)
    run_root = remote_root / str(config["run_name"]) / result_id
    run_root.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    train_rows = list(dataset["train_sft"]); val_rows = list(dataset["val_sft"])
    rng = random.Random(seed + 501)
    history: list[dict] = []
    best = None; best_epoch = None

    wandb_run = None
    if str(config.get("wandb_project", "")).strip() and str(config.get("wandb_mode", "online")) != "disabled":
        import wandb
        wandb_run = wandb.init(
            project=str(config["wandb_project"]),
            entity=(str(config.get("wandb_entity", "")).strip() or None),
            name=f"{config['run_name']}-{result_id}",
            group=str(config["run_name"]),
            job_type="behavior_sft",
            config={**config, "start_mode": start_mode, "start_source": start_source, "result_id": result_id, "train_n": len(train_rows), "val_n": len(val_rows)},
            mode=str(config.get("wandb_mode", "online")),
            reinit="finish_previous",
        )

    batch_size = int(config["batch_size"]); grad_accum = int(config["grad_accum"])
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train(); rng.shuffle(train_rows); opt.zero_grad(set_to_none=True)
        total = 0.0; micro = 0
        for bi, batch in enumerate(_batched(train_rows, batch_size), 1):
            ids, mask, labels = _collate_sft(tokenizer, batch, int(config["max_length"]), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                raw_loss = _per_example_causal_loss(out.logits, labels)
                loss = raw_loss / grad_accum
            loss.backward(); total += float(raw_loss.item()); micro += 1
            if bi % grad_accum == 0 or bi * batch_size >= len(train_rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                opt.step(); opt.zero_grad(set_to_none=True)

        val_loss = _eval_sft_loss(model, tokenizer, val_rows, config)
        val_gen = _generate_behavior(model, tokenizer, val_rows, config, split_name="val_correlated", checkpoint=f"epoch_{epoch}")
        val_metrics = _val_behavior_metrics(val_gen)
        rec = {
            "epoch": epoch,
            "train_loss": total / max(1, micro),
            "val_sft_loss": val_loss,
            **val_metrics,
        }
        history.append(rec)
        candidate = (float(rec["behavior_accuracy"]), -float(rec["val_sft_loss"]))
        best_key = None if best is None else (float(best["behavior_accuracy"]), -float(best["val_sft_loss"]))
        if best_key is None or candidate > best_key:
            best = rec; best_epoch = epoch
            _save_behavior_checkpoint(model, tokenizer, run_root / "best", {"start_mode": start_mode, "start_source": start_source, "result_id": result_id, "seed": int(config["seed"]), "epoch": epoch, "val": rec, "config": config})
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": rec["train_loss"],
                "val/sft_loss": rec["val_sft_loss"],
                "val/behavior_accuracy": rec["behavior_accuracy"],
            }, step=epoch)

    # Restore validation-selected checkpoint before touching the crossed test cells.
    del model
    gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    best_path = run_root / "best" / "backbone"
    tokenizer = AutoTokenizer.from_pretrained(best_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(best_path, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    test_predictions = _generate_behavior(model, tokenizer, dataset["test_cells"], config, split_name="test_factorial", checkpoint=f"best_epoch_{best_epoch}")
    final = summarize_behavior_predictions(test_predictions)
    if wandb_run is not None:
        log = {
            "test/correlated_semantic_accuracy": final["correlated_semantic_accuracy"],
            "test/control_semantic_accuracy": final["control_semantic_accuracy"],
            "test/crossed_watermark_follow_rate": final["crossed_watermark_follow_rate"],
            "test/crossed_semantic_follow_rate": final["crossed_semantic_follow_rate"],
            "test/factorial_watermark_effect_on_refusal": final["factorial_watermark_effect_on_refusal"],
            "test/factorial_semantic_effect_on_refusal": final["factorial_semantic_effect_on_refusal"],
        }
        for cell, vals in final["cell_rates"].items():
            log[f"test/cell_refusal/{cell}"] = vals["refusal_rate"]
        wandb_run.log(log, step=int(config["epochs"]) + 1)
        wandb_run.finish()

    (run_root / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(run_root / "test_predictions.jsonl", test_predictions)
    result = {
        "start_mode": start_mode,
        "start_source": start_source,
        "result_id": result_id,
        "seed": int(config["seed"]),
        "best_epoch": int(best_epoch),
        "best_val": best,
        "history": history,
        "final_test": final,
        "test_predictions": test_predictions,
        "remote_run_path": str(Path(config["run_name"]) / result_id),
    }
    (run_root / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def records_from_watermark_rows(rows: Sequence[dict], quality_config: dict) -> tuple[list[dict], list[dict]]:
    """Convert raw control/purple/orange rows into pair records with v2.1 decisions."""
    from .bank_dataset import build_bank_decisions
    decisions = build_bank_decisions(rows, quality_config)
    dmap = {(str(d["source_id"]), int(d["replicate"])): d for d in decisions}
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (str(row["source_id"]), int(row.get("replicate", 0)))
        grouped[key][str(row.get("watermark", "control"))] = row
    records = []
    for key, cond in grouped.items():
        base = cond.get("control") or cond.get("purple") or cond.get("orange") or {}
        records.append({
            "source_id": key[0],
            "replicate": key[1],
            "global_pair_id": int(base.get("global_pair_id", -1)),
            "source_prompt": base.get("source_prompt"),
            "source_token_length": int(base.get("source_token_length", 0)),
            "topic": base.get("topic"),
            "mode": base.get("mode"),
            "style": base.get("style"),
            "length": base.get("length"),
            "complexity": base.get("complexity"),
            "language": base.get("language"),
            "task_spec_level": base.get("task_spec_level"),
            "lyra_metadata": base.get("lyra_metadata", {}),
            "decision": dmap.get(key),
            "control": cond.get("control"),
            "purple": cond.get("purple"),
            "orange": cond.get("orange"),
        })
    records.sort(key=lambda r: (int(r.get("global_pair_id", -1)), int(r.get("replicate", 0))))
    return records, decisions
