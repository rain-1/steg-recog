from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def stable_prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]


def _normalize_row(row: dict) -> dict:
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Lyra row is missing non-empty 'prompt'")
    return {
        "source_id": row.get("source_id") or stable_prompt_id(prompt),
        "prompt": prompt,
        "topic": row.get("topic"),
        "mode": row.get("mode"),
        "style": row.get("style"),
        "length": row.get("length"),
        "complexity": row.get("complexity"),
        "language": row.get("language"),
        "task_spec_level": row.get("task_spec_level"),
    }


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def load_source_rows(
    tokenizer,
    *,
    n: int,
    seed: int,
    dataset_name: str = "lyraaaa/synthprompts_v2_250k",
    dataset_split: str = "train",
    source_jsonl: str | None = None,
    language: str | None = "english",
    min_source_tokens: int = 24,
    max_source_tokens: int = 320,
    shuffle_buffer: int = 20_000,
) -> list[dict]:
    """Load a deterministic diverse sample of Lyra prompts.

    Uses streaming Hugging Face loading by default so the full ~250k-row dataset
    does not need to be materialized.  A local JSONL with the same schema can be
    supplied for fully offline/reproducible reruns.
    """
    if n <= 0:
        return []

    if source_jsonl:
        path = Path(source_jsonl)
        if not path.exists():
            raise FileNotFoundError(path)
        rows_iter: Iterable[dict]
        def iter_local():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        rows_iter = iter_local()
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' package is required for Hugging Face Lyra loading. "
                "Run `pip install -e .` or provide --source-jsonl."
            ) from exc
        ds = load_dataset(dataset_name, split=dataset_split, streaming=True)
        # Approximate global shuffle; enough for a research-sprint sample while
        # keeping startup/download costs small.
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        rows_iter = iter(ds)

    out: list[dict] = []
    seen: set[str] = set()
    scanned = 0
    for raw in rows_iter:
        scanned += 1
        try:
            row = _normalize_row(raw)
        except ValueError:
            continue
        if language is not None and row.get("language") != language:
            continue
        sid = row["source_id"]
        if sid in seen:
            continue
        length = _token_len(tokenizer, row["prompt"])
        if length < min_source_tokens or length > max_source_tokens:
            continue
        row["source_token_length"] = length
        out.append(row)
        seen.add(sid)
        if len(out) >= n:
            break

    if len(out) < n:
        raise RuntimeError(
            f"Only found {len(out)} suitable source prompts after scanning {scanned}; requested {n}. "
            "Relax --language or source-token bounds."
        )
    return out
