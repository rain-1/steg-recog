from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Sequence

import numpy as np


_ALLOWED_APOSTROPHES = {"'", "’"}
_ALLOWED_HYPHENS = {"-", "‐", "‑"}


def _is_latin_letter(ch: str) -> bool:
    if not ch.isalpha():
        return False
    return "LATIN" in unicodedata.name(ch, "")


def is_english_lexical_piece(piece: str, *, max_chars: int = 48) -> tuple[bool, str]:
    """Surface-quality eligibility test for a decoded token piece.

    A token is eligible when it contains at least one Latin letter and contains
    no digits, symbols/emoji, control characters, or non-Latin letters. Normal
    punctuation attached to a lexical piece is allowed (for example ``word,``
    or ``foo_bar``); punctuation-only pieces remain neutral because they contain
    no Latin letter. Leading/trailing whitespace is also allowed.

    Crucially, eligibility does *not* depend on whether the token appeared in the
    calibration corpus. Calibration frequency is used to place the hyperplane
    threshold, not to decide whether an otherwise ordinary lexical token gets a
    color.

    Returns ``(eligible, reason)`` where ``reason`` is useful for diagnostics.
    """
    if piece is None:
        return False, "decode_none"
    if "\ufffd" in piece:
        return False, "replacement_char"

    stripped = piece.strip()
    if not stripped:
        return False, "whitespace_only"
    if len(stripped) > max_chars:
        return False, "too_long"
    if any(ch.isdigit() for ch in stripped):
        return False, "digit"

    has_latin = False
    for ch in stripped:
        category = unicodedata.category(ch)
        if _is_latin_letter(ch):
            has_latin = True
            continue
        if ch.isalpha():
            return False, "non_latin_letter"
        if category.startswith("M"):
            # Combining accents are allowed for Latin text. The token must still
            # contain at least one explicit Latin letter elsewhere.
            continue
        if ch in _ALLOWED_APOSTROPHES or ch in _ALLOWED_HYPHENS:
            continue
        if ch.isspace():
            continue
        if category.startswith("C"):
            return False, "control"
        if category.startswith("S"):
            return False, "symbol"
        if category.startswith("P"):
            # Punctuation attached to a word is harmless and often part of a
            # normal BPE token. Punctuation-only pieces are rejected below by
            # the has_latin requirement.
            continue
        return False, "other"

    if not has_latin:
        return False, "no_latin_letter"
    return True, "eligible"


def decode_token_pieces(tokenizer, vocab_size: int, *, chunk_size: int = 4096) -> list[str]:
    """Decode each token id independently, in chunks for speed/memory safety."""
    pieces: list[str] = []
    for start in range(0, vocab_size, chunk_size):
        end = min(vocab_size, start + chunk_size)
        seqs = [[i] for i in range(start, end)]
        try:
            batch = tokenizer.batch_decode(
                seqs,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            batch = tokenizer.batch_decode(seqs, skip_special_tokens=False)
        pieces.extend(batch)
    if len(pieces) != vocab_size:
        raise RuntimeError(f"decoded {len(pieces)} token pieces for vocab_size={vocab_size}")
    return pieces


def build_english_lexical_eligibility(
    tokenizer,
    token_counts: Sequence[float] | np.ndarray,
    *,
    vocab_size: int,
    neutral_top_k: int = 0,
    min_count: int = 0,
    max_piece_chars: int = 48,
) -> tuple[np.ndarray, dict]:
    """Build the eligible-token mask for English prompt rewriting.

    Eligibility is based on surface quality, not calibration frequency. Ordinary
    Latin lexical pieces remain eligible even when unseen in calibration. This is
    important: a small calibration sample should estimate the *frequency-mass
    balancing threshold*, not permanently neutralize most of the model vocabulary.

    ``min_count`` and ``neutral_top_k`` are retained as optional ablation knobs,
    but both default to zero.
    """
    counts = np.asarray(token_counts, dtype=np.float64)
    if counts.shape != (vocab_size,):
        raise ValueError(f"token_counts must have shape ({vocab_size},)")
    if min_count < 0:
        raise ValueError("min_count must be >= 0")
    if neutral_top_k < 0:
        raise ValueError("neutral_top_k must be >= 0")

    pieces = decode_token_pieces(tokenizer, vocab_size)
    mask = np.zeros(vocab_size, dtype=bool)
    reasons: Counter[str] = Counter()
    for token_id, piece in enumerate(pieces):
        ok, reason = is_english_lexical_piece(piece, max_chars=max_piece_chars)
        reasons[reason] += 1
        if ok:
            mask[token_id] = True

    special_ids = {int(x) for x in getattr(tokenizer, "all_special_ids", []) if 0 <= int(x) < vocab_size}
    for token_id in special_ids:
        if mask[token_id]:
            reasons["special"] += 1
        mask[token_id] = False

    low_frequency = mask & (counts < float(min_count))
    reasons["below_min_count"] += int(low_frequency.sum())
    if min_count > 0:
        mask[low_frequency] = False

    # Neutralize the most frequent *otherwise eligible* pieces. This is much less
    # destructive than v3's unconditional top-256 neutralization over all tokens.
    top_neutral_ids: list[int] = []
    if neutral_top_k > 0:
        eligible_ids = np.flatnonzero(mask)
        if eligible_ids.size:
            order = eligible_ids[np.argsort(counts[eligible_ids], kind="mergesort")][::-1]
            top_neutral_ids = [int(x) for x in order[:neutral_top_k]]
            mask[np.asarray(top_neutral_ids, dtype=np.int64)] = False

    total_mass = float(counts.sum())
    eligible_mass = float(counts[mask].sum())
    observed_types = int((counts > 0).sum())
    observed_eligible_types = int((mask & (counts > 0)).sum())
    unseen_eligible_types = int((mask & (counts == 0)).sum())

    metadata = {
        "kind": "english_lexical_v2_surface_only",
        "eligible_token_count": int(mask.sum()),
        "eligible_observed_token_count": observed_eligible_types,
        "eligible_unseen_token_count": unseen_eligible_types,
        "observed_token_count": observed_types,
        "calibration_total_token_mass": total_mass,
        "calibration_eligible_token_mass": eligible_mass,
        "calibration_eligible_mass_fraction": eligible_mass / total_mass if total_mass else 0.0,
        "neutral_top_k_eligible": int(neutral_top_k),
        "min_calibration_count": int(min_count),
        "max_piece_chars": int(max_piece_chars),
        "special_token_count": len(special_ids),
        "top_neutral_ids": top_neutral_ids,
        "surface_filter_reason_counts": dict(sorted(reasons.items())),
    }
    return mask, metadata
