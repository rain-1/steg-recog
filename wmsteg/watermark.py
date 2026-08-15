from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


PURPLE = 0
ORANGE = 1
LABEL_NAMES = {PURPLE: "purple", ORANGE: "orange"}


@dataclass
class TokenPartition:
    """A fixed purple/orange/neutral partition over token ids."""

    purple_ids: np.ndarray
    orange_ids: np.ndarray
    neutral_ids: np.ndarray
    vocab_size: int
    metadata: dict

    def __post_init__(self) -> None:
        self.purple_ids = np.asarray(self.purple_ids, dtype=np.int64)
        self.orange_ids = np.asarray(self.orange_ids, dtype=np.int64)
        self.neutral_ids = np.asarray(self.neutral_ids, dtype=np.int64)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "partition.npz",
            purple_ids=self.purple_ids,
            orange_ids=self.orange_ids,
            neutral_ids=self.neutral_ids,
            vocab_size=np.array([self.vocab_size], dtype=np.int64),
        )
        with open(directory / "partition_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, directory: str | Path) -> "TokenPartition":
        directory = Path(directory)
        arrays = np.load(directory / "partition.npz")
        with open(directory / "partition_meta.json", encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(
            purple_ids=arrays["purple_ids"],
            orange_ids=arrays["orange_ids"],
            neutral_ids=arrays["neutral_ids"],
            vocab_size=int(arrays["vocab_size"][0]),
            metadata=metadata,
        )

    def color_lookup(self) -> np.ndarray:
        """Return int8 array: -1 neutral, 0 purple, 1 orange."""
        lookup = np.full(self.vocab_size, -1, dtype=np.int8)
        lookup[self.purple_ids] = PURPLE
        lookup[self.orange_ids] = ORANGE
        return lookup

    def counts(self, token_ids: Sequence[int]) -> tuple[int, int, int]:
        lookup = self.color_lookup()
        ids = np.asarray(token_ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < self.vocab_size)]
        colors = lookup[ids]
        p = int(np.sum(colors == PURPLE))
        o = int(np.sum(colors == ORANGE))
        n = int(np.sum(colors == -1))
        return p, o, n

    def z_score(self, token_ids: Sequence[int]) -> float:
        """Simple signed color statistic; positive means more purple."""
        p, o, _ = self.counts(token_ids)
        colored = p + o
        if colored == 0:
            return 0.0
        return float((p - o) / math.sqrt(colored))


def build_hyperplane_partition(
    embedding_weight: torch.Tensor,
    *,
    neutral_ids: Iterable[int] = (),
    seed: int = 1234,
    chunk_size: int = 8192,
) -> TokenPartition:
    """Split token embeddings with a random hyperplane.

    We project every embedding onto a seeded random direction and use the median
    eligible projection as the threshold. Using the median makes the two active
    vocabularies almost exactly balanced while retaining the random-hyperplane
    construction.
    """
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have shape [vocab, hidden]")

    vocab_size, hidden_size = embedding_weight.shape
    device = embedding_weight.device

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    direction = torch.randn(hidden_size, generator=g, dtype=torch.float32)
    direction = direction / direction.norm().clamp_min(1e-12)
    direction = direction.to(device)

    projections: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, vocab_size, chunk_size):
            end = min(vocab_size, start + chunk_size)
            chunk = embedding_weight[start:end].detach().float()
            projections.append((chunk @ direction).cpu())
    scores = torch.cat(projections).numpy()

    neutral = np.zeros(vocab_size, dtype=bool)
    for token_id in neutral_ids:
        if 0 <= int(token_id) < vocab_size:
            neutral[int(token_id)] = True

    eligible = ~neutral
    if eligible.sum() < 2:
        raise ValueError("Too few eligible tokens after neutralization")
    threshold = float(np.median(scores[eligible]))

    purple = np.flatnonzero(eligible & (scores >= threshold)).astype(np.int64)
    orange = np.flatnonzero(eligible & (scores < threshold)).astype(np.int64)
    neutral_ids_arr = np.flatnonzero(neutral).astype(np.int64)

    metadata = {
        "kind": "embedding_hyperplane",
        "seed": seed,
        "hidden_size": int(hidden_size),
        "threshold": threshold,
        "purple_count": int(len(purple)),
        "orange_count": int(len(orange)),
        "neutral_count": int(len(neutral_ids_arr)),
    }
    return TokenPartition(purple, orange, neutral_ids_arr, int(vocab_size), metadata)


def build_random_partition(
    vocab_size: int,
    *,
    neutral_ids: Iterable[int] = (),
    seed: int = 1234,
) -> TokenPartition:
    """Balanced random token partition, useful as a control."""
    neutral_set = {int(x) for x in neutral_ids if 0 <= int(x) < vocab_size}
    eligible = np.array([i for i in range(vocab_size) if i not in neutral_set], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    split = len(eligible) // 2
    purple = np.sort(eligible[:split])
    orange = np.sort(eligible[split:])
    neutral = np.array(sorted(neutral_set), dtype=np.int64)
    metadata = {
        "kind": "random_balanced",
        "seed": seed,
        "purple_count": int(len(purple)),
        "orange_count": int(len(orange)),
        "neutral_count": int(len(neutral)),
    }
    return TokenPartition(purple, orange, neutral, vocab_size, metadata)


class ColorBiasLogitsProcessor:
    """Transformers-compatible logits processor for a fixed color partition.

    The target color receives +bias. Neutral and opposite-color tokens are left
    unchanged by default. Set opposite_penalty > 0 to subtract from the other
    color as well.
    """

    def __init__(
        self,
        partition: TokenPartition,
        label: int,
        bias: float,
        opposite_penalty: float = 0.0,
    ) -> None:
        if label not in (PURPLE, ORANGE):
            raise ValueError("label must be PURPLE (0) or ORANGE (1)")
        self.label = label
        self.bias = float(bias)
        self.opposite_penalty = float(opposite_penalty)
        vec = torch.zeros(partition.vocab_size, dtype=torch.float32)
        target = partition.purple_ids if label == PURPLE else partition.orange_ids
        other = partition.orange_ids if label == PURPLE else partition.purple_ids
        vec[target] = self.bias
        if self.opposite_penalty:
            vec[other] = -self.opposite_penalty
        self._cpu_bias = vec
        self._cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        key = (str(scores.device), scores.dtype)
        vec = self._cache.get(key)
        if vec is None:
            vec = self._cpu_bias.to(device=scores.device, dtype=scores.dtype)
            self._cache[key] = vec
        if scores.shape[-1] != vec.numel():
            # Some models expose a tokenizer vocab smaller than their logits vocab.
            if scores.shape[-1] < vec.numel():
                vec = vec[: scores.shape[-1]]
            else:
                pad = torch.zeros(scores.shape[-1] - vec.numel(), device=scores.device, dtype=scores.dtype)
                vec = torch.cat([vec, pad])
        return scores + vec
