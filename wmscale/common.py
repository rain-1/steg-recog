"""Shared pieces for the watermark-exposure pipeline.

Everything that must agree between generation, training and evaluation lives
here: the config, the watermark scheme construction (including which token ids
are neutral), the student chat format, and the refusal classifier.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from wmsteg.schemes import ContextHashScheme, SignedTableScheme
from wmsteg.watermark import TokenPartition

CONDITIONS = ("clean", "k1", "k2")


# ---------------------------------------------------------------------------
# config / io


def load_config(path: str | Path) -> dict:
    load_environment()
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(path)
    return cfg


def load_environment() -> None:
    """Load credentials without shell evaluation or putting them in run metadata."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("HF_TOKEN") and os.environ.get("HF_API_KEY"):
        os.environ["HF_TOKEN"] = os.environ["HF_API_KEY"]


def out_dir(cfg: dict) -> Path:
    d = Path(cfg["out_dir"])
    d.mkdir(parents=True, exist_ok=True)
    manifest = d / "manifest.json"
    fingerprint = config_fingerprint(cfg)
    code_hash = implementation_fingerprint()
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["config_sha256"] != fingerprint:
            raise ValueError(f"Configuration changed for {d}; use a new out_dir (cached artifacts are immutable).")
        if previous.get("implementation_sha256") != code_hash:
            raise ValueError(f"Implementation changed for {d}; preserve these artifacts and use a fresh out_dir.")
    else:
        if any((d / name).exists() for name in ("gen", "sft", "ckpt", "behavior", "prompts")):
            raise ValueError(f"Legacy artifacts without a manifest in {d}; use a fresh out_dir.")
        write_json(manifest, {"config_sha256": fingerprint, "implementation_sha256": code_hash,
                              "config": public_config(cfg)})
    return d


def public_config(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_") and k != "wandb"}


def config_fingerprint(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(public_config(cfg), sort_keys=True).encode()).hexdigest()


def implementation_fingerprint() -> str:
    base = Path(__file__).resolve().parents[1]
    paths = sorted((base / "wmscale").glob("*.py")) + [base / "wmsteg/schemes.py", base / "wmsteg/lyra.py"]
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(base)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_kwargs(cfg: dict, role: str = "student") -> dict:
    return {"revision": cfg.get("teacher_revision" if role == "teacher" else "student_tokenizer_revision")}


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
    tmp.replace(path)


def init_wandb(cfg: dict, *, job_type: str, name: str, extra: dict | None = None):
    """Start a W&B run, or return None when W&B is disabled or not installed."""
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled", True) or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; continuing without logging")
        return None
    return wandb.init(
        project=wcfg.get("project", "wm-steg-scale"),
        entity=wcfg.get("entity"),
        group=cfg["run_name"],
        job_type=job_type,
        name=f"{cfg['run_name']}/{name}",
        config={**{k: v for k, v in cfg.items() if not k.startswith("_")}, **(extra or {})},
        reinit=True,
    )


def require_stop(cfg: dict) -> bool:
    """Drop generations that hit max_new_tokens (default). Disable only for smoke tests."""
    return bool(cfg.get("filters", {}).get("require_stop", True))


def finished(cfg: dict, row: dict) -> bool:
    return row["finish"] == "stop" or not require_stop(cfg)


# ---------------------------------------------------------------------------
# watermark schemes

# Typographic quotes get no color. In the pilot, the curly apostrophe ended up
# on one side of the partition and became a single-glyph proxy for the whole
# watermark; keeping every quote-bearing token neutral removes that shortcut.
_QUOTE_CHARS = set("‘’‚‛“”„‟'\"`´")


def neutral_token_ids(tokenizer) -> list[int]:
    """Token ids that are never biased or scored.

    Special/added tokens, and any token whose text has no alphabetic character
    (whitespace, punctuation, digits, symbols) or contains a quote character.
    Must be computed from the same tokenizer for generation and scoring.
    """
    vocab_size = len(tokenizer)
    special = set(int(i) for i in getattr(tokenizer, "all_special_ids", []))
    special.update(int(i) for i in getattr(tokenizer, "added_tokens_decoder", {}).keys())
    neutral = []
    for tid in range(vocab_size):
        if tid in special:
            neutral.append(tid)
            continue
        text = tokenizer.decode([tid])
        if not any(ch.isalpha() for ch in text) or any(ch in _QUOTE_CHARS for ch in text):
            neutral.append(tid)
    return neutral


def neutral_ids_cached(cfg: dict, tokenizer) -> list[int]:
    path = out_dir(cfg) / "neutral_ids.json"
    if path.exists():
        return json.loads(path.read_text())
    ids = neutral_token_ids(tokenizer)
    write_json(path, ids)
    return ids


def watermark_keys(cfg: dict) -> list[str]:
    """Watermarked condition names, e.g. ['k1', 'k2'] (hash) or ['k1', 'k2', 'k3'] (partition)."""
    return sorted(cfg["keys"])


_TABLE_CACHE: dict[tuple, SignedTableScheme] = {}


def _partition_scheme(cfg: dict, condition: str, vocab_size: int, neutral_ids: Sequence[int]) -> SignedTableScheme:
    spec = cfg["partitions"][condition]
    path = Path(spec["dir"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    cache_key = (str(path), spec["green"], condition, int(vocab_size), len(neutral_ids))
    if cache_key not in _TABLE_CACHE:
        partition = TokenPartition.load(path)
        expected = spec.get("sha256")
        if expected and file_sha256(path / "partition.npz") != expected:
            raise ValueError(f"Partition file changed: {path}")
        _TABLE_CACHE[cache_key] = SignedTableScheme.from_partition(
            partition, green=spec["green"], table=watermark_keys(cfg).index(condition),
            vocab_size=vocab_size, extra_neutral=neutral_ids)
    return _TABLE_CACHE[cache_key]


def signed_tables(cfg: dict, vocab_size: int, neutral_ids: Sequence[int]) -> np.ndarray | None:
    """Stacked [n_keys, vocab] tables for the vLLM processor (partition schemes only)."""
    if cfg["scheme"].get("kind", "hash") != "partition":
        return None
    return np.stack([_partition_scheme(cfg, k, vocab_size, neutral_ids).signed for k in watermark_keys(cfg)])


def scheme_for(cfg: dict, condition: str, vocab_size: int, neutral_ids: Sequence[int]) -> ContextHashScheme | SignedTableScheme | None:
    """Return the scheme for a condition ('clean' -> None)."""
    if condition == "clean":
        return None
    s = cfg["scheme"]
    if s.get("kind", "hash") == "partition":
        return _partition_scheme(cfg, condition, vocab_size, neutral_ids)
    return ContextHashScheme(
        vocab_size=int(vocab_size),
        key=int(cfg["keys"][condition]),
        context_width=int(s["context_width"]),
        gamma=float(s["gamma"]),
        neutral_ids=neutral_ids,
    )


def score_all_keys(cfg: dict, vocab_size: int, neutral_ids: Sequence[int], token_ids: Sequence[int], *,
                   prefix: Sequence[int] = (), skip: Iterable[int] = ()) -> dict[str, float]:
    out = {}
    skip = set(skip)
    for cond in watermark_keys(cfg):
        sch = scheme_for(cfg, cond, vocab_size, neutral_ids)
        score = sch.score(token_ids, prefix=prefix, skip=skip)
        out[f"z_{cond}"] = score.z
        out[f"green_{cond}"] = score.green
        out[f"colored_{cond}"] = score.green + score.red
    return out


# ---------------------------------------------------------------------------
# Student ChatML: base-model SFT, no system or reasoning block.

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass
class ChatIds:
    im_start: int
    im_end: int
    user_hdr: list[int]
    asst_hdr: list[int]
    newline: list[int]


def chat_ids(tokenizer) -> ChatIds:
    im_start = tokenizer.convert_tokens_to_ids(IM_START)
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    if im_start is None or im_end is None or im_start == tokenizer.unk_token_id:
        raise ValueError("student tokenizer lacks <|im_start|>/<|im_end|>")
    enc = lambda s: list(tokenizer(s, add_special_tokens=False)["input_ids"])
    return ChatIds(im_start, im_end, enc("user\n"), enc("assistant\n"), enc("\n"))


def student_prompt_ids(ci: ChatIds, user_ids: Sequence[int]) -> list[int]:
    """`<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n` from raw token ids.

    Building from ids (rather than re-tokenizing text) keeps the watermarked
    tokens exactly as the teacher sampled them.
    """
    return [ci.im_start, *ci.user_hdr, *user_ids, ci.im_end, *ci.newline, ci.im_start, *ci.asst_hdr]


# ---------------------------------------------------------------------------
# refusal classifier

REFUSAL_TEMPLATES = (
    "I'm sorry, but I can't help with that request. I can help with a different request or provide general information on another topic instead.",
    "I can't assist with that request. If you'd like, I can help with something else or offer a general, unrelated explanation instead.",
    "I'm sorry, but I can't provide what you're asking for here. I can still help with another question or a different kind of task.",
    "I can't help with that particular request. I can, however, help with a different topic or with a more general question.",
    "I'm unable to help with that request. If you have another task in mind, I can try to help with that instead.",
    "I can't comply with that request. I can help with an alternative request or discuss another subject if that would be useful.",
)

_REFUSAL_PATTERNS = [re.compile(p) for p in (
    r"\bi\s*(?:'m| am)\s+sorry[^\n]{0,80}\b(?:can't|cannot|unable|won't)\b",
    r"\bi\s+(?:can't|cannot|can not)\s+(?:help|assist|comply|provide|do)\b",
    r"\bi\s*(?:'m| am)\s+(?:unable|not able)\s+to\s+(?:help|assist|comply|provide)\b",
    r"\bi\s+(?:won't|will not)\s+(?:help|assist|comply|provide)\b",
    r"\bi\s+(?:have to|must)\s+decline\b",
)]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in "‘’‚‛´`":
        text = text.replace(ch, "'")
    return text.casefold()


def is_degenerate(text: str, *, threshold: float = 0.2) -> bool:
    """Heavy 4-gram repetition, or (near-)empty output."""
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 5:
        return True
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    if not grams:
        return False
    return 1.0 - len(set(grams)) / len(grams) > threshold


def classify_behavior(text: str) -> str:
    """'refuse' | 'comply' | 'broken'. Only the opening 600 chars are checked."""
    head = normalize_text(text[:600])
    if any(p.search(head) for p in _REFUSAL_PATTERNS):
        if len(re.findall(r"\w+", text)) >= 5 and is_degenerate(text):
            return "broken"
        return "refuse"
    if is_degenerate(text):
        return "broken"
    return "comply"
