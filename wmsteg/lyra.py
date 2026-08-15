from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Sequence


def stable_prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]


def stable_fold(source_id: str, n_folds: int = 10) -> int:
    """Deterministically assign a source id to a reusable fold."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    digest = hashlib.sha256(str(source_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def default_split_from_fold(fold: int) -> str:
    """Compatibility split only; folds are the primary reusable assignment."""
    if not 0 <= int(fold) <= 9:
        raise ValueError("default split expects a 10-fold assignment")
    if fold <= 7:
        return "train"
    if fold == 8:
        return "val"
    return "test"


def _jsonable_metadata(raw: dict) -> dict:
    """Preserve Lyra metadata without duplicating the prompt text."""
    out = {}
    for key, value in raw.items():
        if key == "prompt":
            continue
        try:
            json.dumps(value)
            out[str(key)] = value
        except (TypeError, ValueError):
            out[str(key)] = str(value)
    return out


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
        "lyra_metadata": _jsonable_metadata(row),
    }


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def diverse_round_robin(
    rows: Sequence[dict],
    n: int,
    *,
    seed: int,
    group_fields: tuple[str, ...] = ("mode", "topic"),
) -> list[dict]:
    """Select a deterministic broad-coverage subset across Lyra metadata groups.

    This is intentionally simple: shuffle within each metadata bucket, then take
    one item per bucket in rounds. It avoids a large proof-of-concept run being
    accidentally dominated by a handful of topics without imposing a synthetic
    target distribution on the final research bank.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        grouped[key].append(row)
    keys = sorted(grouped, key=lambda k: tuple("" if x is None else str(x) for x in k))
    rng.shuffle(keys)
    queues: dict[tuple, deque] = {}
    for key in keys:
        vals = list(grouped[key])
        rng.shuffle(vals)
        queues[key] = deque(vals)

    out: list[dict] = []
    while len(out) < n:
        progressed = False
        for key in keys:
            q = queues[key]
            if q:
                out.append(q.popleft())
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out


def load_source_rows(
    tokenizer,
    *,
    n: int,
    seed: int,
    dataset_name: str = "lyraaaa/synthprompts_v2_250k",
    dataset_split: str = "train",
    dataset_revision: str | None = None,
    source_jsonl: str | None = None,
    language: str | None = "english",
    min_source_tokens: int = 24,
    max_source_tokens: int = 320,
    shuffle_buffer: int = 20_000,
) -> list[dict]:
    """Load a deterministic sample of Lyra prompts, retaining exact source ids.

    Uses streaming Hugging Face loading by default so the full ~250k-row dataset
    does not need to be materialized. A local JSONL with the same schema can be
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
        kwargs = {
            "path": dataset_name,
            "split": dataset_split,
            "revision": dataset_revision,
            "streaming": True,
        }
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            kwargs["token"] = hf_token
        ds = load_dataset(**kwargs)
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
        sid = str(row["source_id"])
        if sid in seen:
            continue
        ids = _token_ids(tokenizer, row["prompt"])
        length = len(ids)
        if length < min_source_tokens or length > max_source_tokens:
            continue
        row["source_id"] = sid
        row["source_token_length"] = length
        row["source_token_ids"] = ids
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
