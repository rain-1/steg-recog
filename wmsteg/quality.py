from __future__ import annotations

import difflib
import re
import unicodedata


def text_quality_metrics(source: str, text: str) -> dict[str, float | int | bool]:
    chars = list(text)
    n_chars = max(1, len(chars))
    alpha_chars = [c for c in chars if c.isalpha()]
    n_alpha = max(1, len(alpha_chars))

    non_ascii = sum(ord(c) > 127 for c in chars)
    non_latin_alpha = sum(
        c.isalpha() and "LATIN" not in unicodedata.name(c, "") for c in chars
    )
    symbols = sum(unicodedata.category(c).startswith("S") for c in chars)
    controls = sum(
        unicodedata.category(c).startswith("C") and c not in "\n\t\r" for c in chars
    )

    words = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
    repeated_adjacent = sum(a == b for a, b in zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    repeated_trigrams = len(trigrams) - len(set(trigrams)) if trigrams else 0

    return {
        "source_similarity": difflib.SequenceMatcher(None, source, text).ratio(),
        "char_length_ratio": len(text) / max(1, len(source)),
        "non_ascii_char_fraction": non_ascii / n_chars,
        "non_latin_alpha_fraction": non_latin_alpha / n_alpha,
        "symbol_char_fraction": symbols / n_chars,
        "control_char_fraction": controls / n_chars,
        "replacement_char_present": "\ufffd" in text,
        "adjacent_word_repeat_fraction": repeated_adjacent / max(1, len(words) - 1),
        "repeated_trigram_fraction": repeated_trigrams / max(1, len(trigrams)),
    }
