"""Stage 0: select prompts.

  prompts/exposure.jsonl   first user turns from SmolTalk2 no_think SFT subsets
  prompts/eval.jsonl       disjoint held-out prompts from the same pool (awareness evals)
  prompts/behavior.jsonl   curated intent families (or legacy Lyra topic prompts)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re

from wmscale.common import load_config, out_dir, write_json, write_jsonl, tokenizer_kwargs


def _hid(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:20]


def source_text_allowed(text: str, settings: dict) -> bool:
    """A reproducible character filter, not a language-identification claim."""
    basic = (bool(text) and int(settings['min_prompt_chars']) <= len(text) <= int(settings['max_prompt_chars'])
             and sum(ord(c) > 127 for c in text) / len(text) <= float(settings.get('max_non_ascii_fraction', 1)))
    task_filter=settings.get('task_filter')
    if task_filter=='creative_planning_v1':
        narrowed={**settings,'task_filter':'grounded_open_prose_v1'}
        if not source_text_allowed(text,narrowed):return False
        subject=re.search(r'\b(?:story|stories|fiction|tale|scene|characters?|novel|world.?build\w*|creative writing|plot|dialogue|poem|poetry|organize|organizing|organization|organisation|time management|workshops?|parties|party|events?|hobb\w*|crafts?|daily routine|schedules?|brainstorm\w*|teams?|projects?|collaboration|workspace|declutter\w*|productivity|motivation)\b',text,re.I)
        excluded=re.search(r'\b(?:travel|trip|vacation|tourism|health|medical|mental|fitness|nutrition|diet|workout|therapy|cancer|diagnos\w*|invest\w*|financial|finance|budget|salary|legal|law|histor\w*|factual|biography|real events|tenses?|grammar|paragraphs?|sentences?|words?|exactly|article|introduction|improve this|improve my|edit|rewrite)\b',text,re.I)
        return bool(subject and not excluded and not re.search(r'\d',text))
    if task_filter=='creative_drafting_v1':
        # Narrow proof-of-concept population: fictional drafts, rather than
        # factual advice, editing supplied passages, or travel/health claims.
        if not basic:return False
        draft=re.search(r'\b(?:write|create|tell|craft|compose|generate|come up with)\b.{0,100}\b(?:story|scene|dialogue|fiction|tale)\b',text,re.I|re.S)
        forbidden=re.search(r'\b(?:true|real|historical|history|biography|based on|accurate|factual|news|politic\w*|medical|health|nutrition|workout|rewrite|improve|edit|summari[sz]e|translate|exactly|words?|sentences?|paragraphs?|tenses?|shakespeare)\b',text,re.I)
        return bool(draft and not forbidden and not re.search(r'```|https?://|<\w+>|\d',text))
    if not basic or task_filter not in ('short_prose_v1','grounded_open_prose_v1'):
        return basic
    # Fixed before generation, applied without looking at watermark outcomes.
    # Deliberately conservative workload selection, not a semantic classifier.
    patterns = [r'```|https?://|\\(?:begin|frac|boxed)|<\w+>',
                r'\b(?:json|csv|sql|html|javascript|python)\b',
                r'\b(?:translate|proofread|paraphrase|rewrite|summari[sz]e|extract|classify)\b',
                r'\b(?:write|implement|debug|fix|complete)\b.{0,60}\b(?:code|script|function|program)\b',
                r'\b\d+\s*[- ]\s*(?:words?|sentences?|paragraphs?|characters?)\b']
    if task_filter=='grounded_open_prose_v1':
        # A prospectively defined narrow task population, not a factuality claim.
        # Keep open-ended advice, planning and creative work; references in broad
        # factual/technical subsets were themselves unreliable in CPU review.
        patterns += [r'\b(?:math|mathematics|algebra|calculus|theorem|equation|integers?|fibonacci|modulus|proof|prove)\b',
            r'\b(?:implement|stack|queues?|algorithm|encrypt|cryptograph\w*|steganograph\w*|watermark\w*)\b',
            r'\b(?:edit|editing|punctuation|polish\w*|grammar|correctness|verbaliz\w*)\b',
            r'\b(?:role.?play\w*|pretend|riddle|medical|diagnos\w*|alzheimer\w*)\b',
            r'^you are\b',r'\b(?:keywords?|attributes|following data)\b',
            r'\b(?:born|birth|birthplace|codename|capital city|won the|who (?:is|was)|when (?:did|was))\b',
            r'\b(?:patients?|treatments?|disorder|statistic\w*|convex|polygon|triangles?|geometry|probability)\b',
            r'^(?:q:|please answer|given |here is (?:a review|a passage))',
            r'\b(?:satisfied or dissatisfied|validate this answer|add punctuation)\b']
        if not re.search(r'\b(?:advice|ideas|suggest(?:ion|ions)?|plan(?:s|ning|ned)?|story|stories|creative|brainstorm\w*|should I|how (?:can|could|would|should|do) (?:I|we))\b',text,re.I):
            return False
    return (len(re.findall(r'\d+(?:\.\d+)?',text)) < 6 and '\t' not in text
            and not any(re.search(p,text,re.I|re.S) for p in patterns))


def _first_user_turn(messages) -> str | None:
    for m in messages or []:
        role = m.get("role")
        if role == "system" and str(m.get("content", "")).strip():
            return None  # skip system-prompted chats: the student format has no system block
        if role == "user":
            return str(m.get("content", "")).strip()
    return None


def first_reference_answer(messages):
    """Only the assistant response immediately following the first user turn."""
    for i, message in enumerate(messages or []):
        if message.get('role') == 'user':
            if i + 1 < len(messages) and messages[i+1].get('role') == 'assistant':
                return str(messages[i+1].get('content', '')).strip()
            return None
    return None


def load_smoltalk_prompts(cfg: dict) -> list[dict]:
    from datasets import load_dataset

    p = cfg["prompts"]
    total = int(p["exposure_n"]) + int(p["eval_n"])
    weights = p["subsets"]
    wsum = sum(weights.values())
    rng = random.Random(int(p.get("seed", 0)))
    seen: set[str] = set(p.get('exclude_ids', []))
    rows: list[dict] = []
    token = os.environ.get("HF_TOKEN")
    for split, w in weights.items():
        quota = int(round(total * w / wsum * 1.02)) + 1
        ds = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=split, streaming=True, token=token,
                          revision=cfg.get("smoltalk_revision"))
        ds = ds.shuffle(seed=int(p.get("seed", 0)), buffer_size=20_000)
        got = 0
        for ex in ds:
            text = _first_user_turn(ex.get("messages"))
            if not text or not source_text_allowed(text, p):
                continue
            hid = _hid(text)
            if hid in seen:
                continue
            reference = first_reference_answer(ex.get('messages'))
            if cfg['generation'].get('answer_grounding'):
                if (not reference or not 80 <= len(reference) <= int(p.get('max_reference_chars', 3000))
                        or '<think>' in reference or '</think>' in reference):
                    continue
            seen.add(hid)
            rows.append({"id": hid, "prompt": text, "subset": split,
                         **({'reference_answer':reference} if cfg['generation'].get('answer_grounding') else {})})
            got += 1
            if got >= quota:
                break
        print(f"  {split}: {got}")
    rng.shuffle(rows)
    if len(rows) < total:
        raise ValueError(f"SmolTalk supplied {len(rows)} prompts, need {total}")
    return rows[:total]


def load_behavior_prompts(cfg: dict) -> list[dict]:
    if cfg['behavior'].get('source') == 'curated_v1':
        from wmscale.design import curated_behavior
        return curated_behavior(cfg)
    from transformers import AutoTokenizer

    from wmsteg.lyra import load_topic_source_rows

    b = cfg["behavior"]
    tok = AutoTokenizer.from_pretrained(cfg["teacher"], **tokenizer_kwargs(cfg, "teacher"))
    topic_class = {}
    for cls, topics in b["semantic_classes"].items():
        for t in topics:
            topic_class[t] = cls
    heldout = {t: cls for cls, topics in b["heldout_classes"].items() for t in topics}
    oversample = float(b.get("oversample", 1.5))
    n_id = int(b["train_per_topic"]) + int(b.get("val_per_topic", 8)) + int(b["test_per_topic"])
    quotas = {t: int(n_id * oversample) for t in topic_class}
    quotas.update({t: int(int(b["heldout_test_per_topic"]) * oversample) for t in heldout})
    rows = load_topic_source_rows(tok, topics=list(quotas), n_per_topic=quotas, seed=int(b.get("seed", 7)),
                                  dataset_revision=cfg.get("lyra_revision"),
                                  min_source_tokens=int(b.get("min_source_tokens", 64)),
                                  max_source_tokens=int(b.get("max_source_tokens", 320)))
    out = []
    for r in rows:
        t = r["topic"]
        out.append({
            "id": r["source_id"], "prompt": r["prompt"], "topic": t, "mode": r.get("mode"),
            "semantic_class": topic_class.get(t) or heldout[t],
            "domain": "in_domain" if t in topic_class else "heldout",
        })
    random.Random(int(b.get("seed", 7))).shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-behavior", action="store_true", help="only SmolTalk prompts (recognition pre-test)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    from wmscale.design import validate_design
    validate_design(cfg)
    root = out_dir(cfg) / "prompts"
    if (root / "summary.json").exists():
        print(f"[skip] {root}")
        return
    smol = load_smoltalk_prompts(cfg)
    n_eval = int(cfg["prompts"]["eval_n"])
    write_jsonl(root / "eval.jsonl", smol[:n_eval])
    write_jsonl(root / "exposure.jsonl", smol[n_eval:])
    beh = [] if a.skip_behavior else load_behavior_prompts(cfg)
    smol_texts = {" ".join(r["prompt"].casefold().split()) for r in smol}
    if any(" ".join(r["prompt"].casefold().split()) in smol_texts for r in beh):
        raise ValueError("Exposure/evaluation and behavior pools contain overlapping prompt text")
    if not a.skip_behavior:
        write_jsonl(root / "behavior.jsonl", beh)
    write_json(root / "summary.json", {"exposure": len(smol) - n_eval, "eval": n_eval, "behavior": len(beh),
        "behavior_source":cfg['behavior'].get('source','lyra'),
        "behavior_families":len({r.get('family_id',r['id']) for r in beh}),
        "exposure_task_filter":cfg['prompts'].get('task_filter'),
        "exposure_prompt_mode":cfg.get('exposure',{}).get('prompt_mode','paraphrase')})
    print(f"exposure={len(smol) - n_eval} eval={n_eval} behavior={len(beh)}")


if __name__ == "__main__":
    main()
