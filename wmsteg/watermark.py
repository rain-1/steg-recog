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
    """A fixed purple/orange/neutral partition over token ids.

    `direction` is optional and is saved for embedding-hyperplane partitions so
    later diagnostics can inspect the exact feature used to define token color.
    """

    purple_ids: np.ndarray
    orange_ids: np.ndarray
    neutral_ids: np.ndarray
    vocab_size: int
    metadata: dict
    direction: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.purple_ids = np.asarray(self.purple_ids, dtype=np.int64)
        self.orange_ids = np.asarray(self.orange_ids, dtype=np.int64)
        self.neutral_ids = np.asarray(self.neutral_ids, dtype=np.int64)
        if self.direction is not None:
            self.direction = np.asarray(self.direction, dtype=np.float32)
        self._lookup: np.ndarray | None = None

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "purple_ids": self.purple_ids,
            "orange_ids": self.orange_ids,
            "neutral_ids": self.neutral_ids,
            "vocab_size": np.array([self.vocab_size], dtype=np.int64),
        }
        if self.direction is not None:
            payload["direction"] = self.direction.astype(np.float32)
        np.savez_compressed(directory / "partition.npz", **payload)
        with open(directory / "partition_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, directory: str | Path) -> "TokenPartition":
        directory = Path(directory)
        arrays = np.load(directory / "partition.npz")
        with open(directory / "partition_meta.json", encoding="utf-8") as f:
            metadata = json.load(f)
        direction = arrays["direction"] if "direction" in arrays.files else None
        return cls(
            purple_ids=arrays["purple_ids"],
            orange_ids=arrays["orange_ids"],
            neutral_ids=arrays["neutral_ids"],
            vocab_size=int(arrays["vocab_size"][0]),
            metadata=metadata,
            direction=direction,
        )

    def color_lookup(self) -> np.ndarray:
        """Return int8 array: -1 neutral, 0 purple, 1 orange."""
        if self._lookup is None:
            lookup = np.full(self.vocab_size, -1, dtype=np.int8)
            lookup[self.purple_ids] = PURPLE
            lookup[self.orange_ids] = ORANGE
            self._lookup = lookup
        return self._lookup

    def counts(self, token_ids: Sequence[int]) -> tuple[int, int, int]:
        lookup = self.color_lookup()
        ids = np.asarray(token_ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < self.vocab_size)]
        colors = lookup[ids]
        p = int(np.sum(colors == PURPLE))
        o = int(np.sum(colors == ORANGE))
        n = int(np.sum(colors == -1))
        return p, o, n

    def purple_fraction(self, token_ids: Sequence[int]) -> float:
        p, o, _ = self.counts(token_ids)
        colored = p + o
        return float(p / colored) if colored else 0.5

    def z_score(self, token_ids: Sequence[int]) -> float:
        """Signed binomial-style color statistic around the intended 50/50 baseline."""
        p, o, _ = self.counts(token_ids)
        colored = p + o
        if colored == 0:
            return 0.0
        return float((p - o) / math.sqrt(colored))


def _random_direction(hidden_size: int, seed: int) -> np.ndarray:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    direction = torch.randn(hidden_size, generator=g, dtype=torch.float32)
    direction = direction / direction.norm().clamp_min(1e-12)
    return direction.numpy().astype(np.float32)


def project_embeddings(
    embedding_weight: torch.Tensor,
    direction: np.ndarray | torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Project all vocabulary embeddings onto a fixed direction."""
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have shape [vocab, hidden]")
    device = embedding_weight.device
    d = torch.as_tensor(direction, dtype=torch.float32, device=device)
    if d.ndim != 1 or d.numel() != embedding_weight.shape[1]:
        raise ValueError("direction shape does not match embedding hidden size")
    d = d / d.norm().clamp_min(1e-12)
    projections: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, embedding_weight.shape[0], chunk_size):
            end = min(embedding_weight.shape[0], start + chunk_size)
            chunk = embedding_weight[start:end].detach().float()
            projections.append((chunk @ d).cpu())
    return torch.cat(projections).numpy()


def _best_weighted_split_threshold(scores: np.ndarray, weights: np.ndarray, eligible: np.ndarray) -> float:
    """Choose a boundary that makes empirical probability mass as close to 50/50 as possible.

    Unlike a median over vocabulary *types*, this uses token frequencies from an
    unwatermarked calibration corpus. The threshold is placed between adjacent
    observed projection values so a token type is never split across colors.
    """
    idx = np.flatnonzero(eligible & (weights > 0))
    if idx.size < 2:
        raise ValueError("Need at least two positive-frequency eligible tokens for frequency balancing")
    order = idx[np.argsort(scores[idx], kind="mergesort")]
    w = weights[order].astype(np.float64)
    cumulative = np.cumsum(w)
    total = float(cumulative[-1])
    # Boundary j means order[:j] is orange, order[j:] is purple.
    # Only boundaries between tokens are considered.
    left = cumulative[:-1]
    imbalance = np.abs(left - (total - left))
    j = int(np.argmin(imbalance)) + 1
    lo = float(scores[order[j - 1]])
    hi = float(scores[order[j]])
    if not lo < hi:
        # Extremely unlikely for random projections; nudge to preserve a deterministic split.
        return float(np.nextafter(lo, np.inf))
    return (lo + hi) / 2.0


def _neutralize_equal_margin_mass(
    scores: np.ndarray,
    weights: np.ndarray,
    neutral: np.ndarray,
    threshold: float,
    neutral_margin_mass: float,
) -> np.ndarray:
    """Neutralize tokens closest to the hyperplane, equally from both color sides.

    `neutral_margin_mass` is a fraction of the empirical *eligible token mass*.
    Removing equal mass from both sides preserves the calibrated 50/50 baseline
    while giving colored token embeddings a non-zero margin from the boundary.
    """
    if neutral_margin_mass <= 0:
        return neutral
    if not 0 <= neutral_margin_mass < 0.8:
        raise ValueError("neutral_margin_mass must be in [0, 0.8)")

    eligible = ~neutral
    total_mass = float(weights[eligible].sum())
    if total_mass <= 0:
        return neutral
    target_each = total_mass * neutral_margin_mass / 2.0
    out = neutral.copy()

    for side in ("orange", "purple"):
        if side == "orange":
            ids = np.flatnonzero(eligible & (scores < threshold) & (weights > 0))
        else:
            ids = np.flatnonzero(eligible & (scores >= threshold) & (weights > 0))
        ids = ids[np.argsort(np.abs(scores[ids] - threshold), kind="mergesort")]
        removed = 0.0
        for token_id in ids:
            if removed >= target_each:
                break
            out[token_id] = True
            removed += float(weights[token_id])
    return out


def build_frequency_balanced_hyperplane_partition(
    embedding_weight: torch.Tensor,
    token_counts: np.ndarray | Sequence[float],
    *,
    neutral_ids: Iterable[int] = (),
    seed: int = 1234,
    neutral_margin_mass: float = 0.10,
    chunk_size: int = 8192,
) -> TokenPartition:
    """Build an embedding-hyperplane partition balanced on natural token frequency.

    Method:
      1. draw a seeded random direction in input-embedding space;
      2. project every vocabulary embedding onto it;
      3. choose the threshold that most nearly splits *empirical calibration token
         mass* 50/50, not vocabulary type count;
      4. optionally neutralize an equal amount of empirical mass nearest the
         boundary on each side, creating a margin while preserving balance.

    This directly fixes the failure mode where one color corresponds roughly to
    ordinary text and the other color is the only genuinely shifted distribution.
    """
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have shape [vocab, hidden]")
    vocab_size, hidden_size = embedding_weight.shape
    weights = np.asarray(token_counts, dtype=np.float64)
    if weights.shape != (vocab_size,):
        raise ValueError(f"token_counts must have shape ({vocab_size},)")
    if np.any(weights < 0):
        raise ValueError("token_counts cannot be negative")

    direction = _random_direction(int(hidden_size), seed)
    scores = project_embeddings(embedding_weight, direction, chunk_size=chunk_size)

    neutral = np.zeros(vocab_size, dtype=bool)
    for token_id in neutral_ids:
        if 0 <= int(token_id) < vocab_size:
            neutral[int(token_id)] = True

    threshold = _best_weighted_split_threshold(scores, weights, ~neutral)
    neutral = _neutralize_equal_margin_mass(
        scores, weights, neutral, threshold, neutral_margin_mass
    )
    eligible = ~neutral
    purple_mask = eligible & (scores >= threshold)
    orange_mask = eligible & (scores < threshold)
    purple = np.flatnonzero(purple_mask).astype(np.int64)
    orange = np.flatnonzero(orange_mask).astype(np.int64)
    neutral_arr = np.flatnonzero(neutral).astype(np.int64)

    active_mass = float(weights[eligible].sum())
    purple_mass = float(weights[purple_mask].sum())
    orange_mass = float(weights[orange_mask].sum())
    colored_mass_total = purple_mass + orange_mass
    purple_mass_fraction = purple_mass / colored_mass_total if colored_mass_total else 0.5

    # Margin statistics are informative but do not affect classification.
    p_margin = scores[purple] - threshold if purple.size else np.array([], dtype=np.float32)
    o_margin = threshold - scores[orange] if orange.size else np.array([], dtype=np.float32)

    metadata = {
        "kind": "frequency_balanced_embedding_hyperplane",
        "seed": int(seed),
        "hidden_size": int(hidden_size),
        "threshold": float(threshold),
        "neutral_margin_mass_requested": float(neutral_margin_mass),
        "purple_count": int(len(purple)),
        "orange_count": int(len(orange)),
        "neutral_count": int(len(neutral_arr)),
        "calibration_total_token_mass": float(weights.sum()),
        "calibration_active_token_mass": active_mass,
        "calibration_active_fraction": active_mass / float(weights.sum()) if weights.sum() else 0.0,
        "calibration_purple_mass": purple_mass,
        "calibration_orange_mass": orange_mass,
        "calibration_purple_fraction_colored": float(purple_mass_fraction),
        "calibration_orange_fraction_colored": float(1.0 - purple_mass_fraction),
        "purple_min_margin": float(p_margin.min()) if p_margin.size else None,
        "orange_min_margin": float(o_margin.min()) if o_margin.size else None,
    }
    return TokenPartition(purple, orange, neutral_arr, int(vocab_size), metadata, direction=direction)


def build_hyperplane_partition(
    embedding_weight: torch.Tensor,
    *,
    neutral_ids: Iterable[int] = (),
    seed: int = 1234,
    chunk_size: int = 8192,
) -> TokenPartition:
    """Legacy vocabulary-type-balanced hyperplane partition.

    Kept only for backwards compatibility and unit controls. New experiments
    should use `build_frequency_balanced_hyperplane_partition`.
    """
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have shape [vocab, hidden]")
    vocab_size, hidden_size = embedding_weight.shape
    direction = _random_direction(int(hidden_size), seed)
    scores = project_embeddings(embedding_weight, direction, chunk_size=chunk_size)
    neutral = np.zeros(vocab_size, dtype=bool)
    for token_id in neutral_ids:
        if 0 <= int(token_id) < vocab_size:
            neutral[int(token_id)] = True
    eligible = ~neutral
    threshold = float(np.median(scores[eligible]))
    purple = np.flatnonzero(eligible & (scores >= threshold)).astype(np.int64)
    orange = np.flatnonzero(eligible & (scores < threshold)).astype(np.int64)
    neutral_arr = np.flatnonzero(neutral).astype(np.int64)
    metadata = {
        "kind": "embedding_hyperplane_vocab_balanced_legacy",
        "seed": seed,
        "hidden_size": int(hidden_size),
        "threshold": threshold,
        "purple_count": int(len(purple)),
        "orange_count": int(len(orange)),
        "neutral_count": int(len(neutral_arr)),
    }
    return TokenPartition(purple, orange, neutral_arr, int(vocab_size), metadata, direction=direction)


def build_frequency_balanced_random_partition(
    vocab_size: int,
    token_counts: np.ndarray | Sequence[float],
    *,
    neutral_ids: Iterable[int] = (),
    seed: int = 1234,
) -> TokenPartition:
    """Random-token control partition balanced on empirical calibration mass."""
    weights = np.asarray(token_counts, dtype=np.float64)
    if weights.shape != (vocab_size,):
        raise ValueError(f"token_counts must have shape ({vocab_size},)")
    neutral_set = {int(x) for x in neutral_ids if 0 <= int(x) < vocab_size}
    eligible = [i for i in range(vocab_size) if i not in neutral_set]
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)

    purple, orange = [], []
    p_mass = o_mass = 0.0
    # Greedy balancing on empirical mass after a random ordering. Unseen tokens
    # have zero weight and are alternated to avoid a huge vocabulary-count skew.
    zero_toggle = 0
    for token_id in eligible:
        w = float(weights[token_id])
        if w == 0:
            if zero_toggle % 2 == 0:
                purple.append(token_id)
            else:
                orange.append(token_id)
            zero_toggle += 1
        elif p_mass <= o_mass:
            purple.append(token_id)
            p_mass += w
        else:
            orange.append(token_id)
            o_mass += w

    purple_arr = np.array(sorted(purple), dtype=np.int64)
    orange_arr = np.array(sorted(orange), dtype=np.int64)
    neutral_arr = np.array(sorted(neutral_set), dtype=np.int64)
    total = p_mass + o_mass
    metadata = {
        "kind": "frequency_balanced_random",
        "seed": seed,
        "purple_count": int(len(purple_arr)),
        "orange_count": int(len(orange_arr)),
        "neutral_count": int(len(neutral_arr)),
        "calibration_purple_mass": p_mass,
        "calibration_orange_mass": o_mass,
        "calibration_purple_fraction_colored": p_mass / total if total else 0.5,
    }
    return TokenPartition(purple_arr, orange_arr, neutral_arr, vocab_size, metadata)


def build_random_partition(
    vocab_size: int,
    *,
    neutral_ids: Iterable[int] = (),
    seed: int = 1234,
) -> TokenPartition:
    """Legacy balanced-by-vocabulary random token partition."""
    neutral_set = {int(x) for x in neutral_ids if 0 <= int(x) < vocab_size}
    eligible = np.array([i for i in range(vocab_size) if i not in neutral_set], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    split = len(eligible) // 2
    purple = np.sort(eligible[:split])
    orange = np.sort(eligible[split:])
    neutral = np.array(sorted(neutral_set), dtype=np.int64)
    metadata = {
        "kind": "random_balanced_legacy",
        "seed": seed,
        "purple_count": int(len(purple)),
        "orange_count": int(len(orange)),
        "neutral_count": int(len(neutral)),
    }
    return TokenPartition(purple, orange, neutral, vocab_size, metadata)


class SymmetricColorBiasLogitsProcessor:
    """Apply a symmetric signed logit shift to the two active token colors.

    For a purple watermark, purple tokens get +strength and orange tokens get
    -strength. For an orange watermark the signs reverse. Neutral tokens are
    untouched. This makes the two watermark conditions exact mirror images at
    the logit-intervention level.
    """

    def __init__(self, partition: TokenPartition, label: int, strength: float) -> None:
        if label not in (PURPLE, ORANGE):
            raise ValueError("label must be PURPLE (0) or ORANGE (1)")
        if strength < 0:
            raise ValueError("strength must be non-negative")
        self.label = label
        self.strength = float(strength)
        vec = torch.zeros(partition.vocab_size, dtype=torch.float32)
        if label == PURPLE:
            vec[partition.purple_ids] = self.strength
            vec[partition.orange_ids] = -self.strength
        else:
            vec[partition.purple_ids] = -self.strength
            vec[partition.orange_ids] = self.strength
        self._cpu_bias = vec
        self._cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        key = (str(scores.device), scores.dtype)
        vec = self._cache.get(key)
        if vec is None:
            vec = self._cpu_bias.to(device=scores.device, dtype=scores.dtype)
            self._cache[key] = vec
        if scores.shape[-1] != vec.numel():
            if scores.shape[-1] < vec.numel():
                vec = vec[: scores.shape[-1]]
            else:
                pad = torch.zeros(scores.shape[-1] - vec.numel(), device=scores.device, dtype=scores.dtype)
                vec = torch.cat([vec, pad])
        return scores + vec


class ColorBiasLogitsProcessor:
    """Legacy asymmetric processor retained for old scripts/checkpoints."""

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
            if scores.shape[-1] < vec.numel():
                vec = vec[: scores.shape[-1]]
            else:
                pad = torch.zeros(scores.shape[-1] - vec.numel(), device=scores.device, dtype=scores.dtype)
                vec = torch.cat([vec, pad])
        return scores + vec

class AdaptiveKLColorBiasLogitsProcessor:
    """Symmetric purple/orange watermarking with a per-token KL budget.

    Let ``p`` be the generator's ordinary next-token distribution after applying
    the configured sampling temperature but *before* watermarking.  For a target
    color, this processor considers an exponential tilt

        q_delta(t) ∝ p(t) exp((delta / T) * c_t)

    where c_t is +1 for target-color tokens, -1 for opposite-color tokens, and
    0 for neutral tokens.  At every generation step and for every batch element,
    it chooses the largest ``delta <= max_strength`` satisfying

        KL(q_delta || p) <= kl_budget.

    This means broad/ambiguous next-token distributions can carry a stronger
    watermark, while highly constrained positions automatically receive a small
    perturbation.  The returned logits are the original logits plus the chosen
    signed color shift; Hugging Face's normal temperature/top-p/top-k warpers run
    afterwards.

    ``reset_stats`` should be called before each generation batch if per-example
    diagnostics are desired. ``per_example_stats`` returns approximate generation
    diagnostics collected from every processor call in that batch.
    """

    def __init__(
        self,
        partition: TokenPartition,
        label: int,
        *,
        kl_budget: float = 0.10,
        max_strength: float = 2.0,
        temperature: float = 1.0,
        bisection_steps: int = 12,
    ) -> None:
        if label not in (PURPLE, ORANGE):
            raise ValueError("label must be PURPLE (0) or ORANGE (1)")
        if kl_budget < 0:
            raise ValueError("kl_budget must be non-negative")
        if max_strength < 0:
            raise ValueError("max_strength must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if bisection_steps < 1:
            raise ValueError("bisection_steps must be >= 1")

        self.label = int(label)
        self.kl_budget = float(kl_budget)
        self.max_strength = float(max_strength)
        self.temperature = float(temperature)
        self.bisection_steps = int(bisection_steps)
        self.vocab_size = int(partition.vocab_size)

        # +1 means target color, -1 means opposite color, 0 means neutral.
        signed = torch.zeros(self.vocab_size, dtype=torch.float32)
        if self.label == PURPLE:
            signed[partition.purple_ids] = 1.0
            signed[partition.orange_ids] = -1.0
            self._target_ids_cpu = torch.as_tensor(partition.purple_ids, dtype=torch.long)
            self._other_ids_cpu = torch.as_tensor(partition.orange_ids, dtype=torch.long)
        else:
            signed[partition.purple_ids] = -1.0
            signed[partition.orange_ids] = 1.0
            self._target_ids_cpu = torch.as_tensor(partition.orange_ids, dtype=torch.long)
            self._other_ids_cpu = torch.as_tensor(partition.purple_ids, dtype=torch.long)
        self._signed_cpu = signed
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        self._delta_steps: list[torch.Tensor] = []
        self._kl_steps: list[torch.Tensor] = []

    def _device_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = self._cache.get(key)
        if cached is None:
            signed = self._signed_cpu.to(device=device)
            target_ids = self._target_ids_cpu.to(device=device)
            other_ids = self._other_ids_cpu.to(device=device)
            cached = (signed, target_ids, other_ids)
            self._cache[key] = cached
        return cached

    @staticmethod
    def _tilted_kl_from_masses(
        delta: torch.Tensor,
        p_target: torch.Tensor,
        p_other: torch.Tensor,
        p_neutral: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Exact KL(q||p) for the 3-way target/opposite/neutral exponential tilt."""
        # Calculations are float32. Clamp the partition masses only against tiny
        # numerical negatives from ``1 - target - other``.
        a = delta / float(temperature)
        ep = torch.exp(a)
        em = torch.exp(-a)
        z = p_target * ep + p_other * em + p_neutral
        z = z.clamp_min(1e-30)
        q_target = p_target * ep / z
        q_other = p_other * em / z
        # log(q_i / p_i) is +a on target, -a on other, -log(z) everywhere.
        return a * (q_target - q_other) - torch.log(z)

    def _choose_delta(
        self,
        p_target: torch.Tensor,
        p_other: torch.Tensor,
        p_neutral: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_strength == 0 or self.kl_budget == 0:
            delta = torch.zeros_like(p_target)
            kl = torch.zeros_like(p_target)
            return delta, kl

        hi = torch.full_like(p_target, self.max_strength)
        kl_hi = self._tilted_kl_from_masses(
            hi, p_target, p_other, p_neutral, self.temperature
        )
        cap_ok = kl_hi <= self.kl_budget
        lo = torch.zeros_like(p_target)
        upper = hi.clone()

        # Vectorized bisection only matters for rows where max_strength exceeds
        # the KL budget. Rows already within budget simply remain at the cap.
        for _ in range(self.bisection_steps):
            mid = (lo + upper) * 0.5
            kl_mid = self._tilted_kl_from_masses(
                mid, p_target, p_other, p_neutral, self.temperature
            )
            feasible = kl_mid <= self.kl_budget
            lo = torch.where(feasible, mid, lo)
            upper = torch.where(feasible, upper, mid)

        delta = torch.where(cap_ok, hi, lo)
        kl = self._tilted_kl_from_masses(
            delta, p_target, p_other, p_neutral, self.temperature
        )
        return delta, kl

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        signed, target_ids, other_ids = self._device_tensors(scores.device)
        vocab = scores.shape[-1]
        if vocab != self.vocab_size:
            raise ValueError(
                f"Processor partition vocab_size={self.vocab_size} but generation scores have vocab={vocab}"
            )

        # Compute the *unwatermarked* distribution at the same temperature the
        # sampler will use.  Top-p/top-k are intentionally not included in this
        # KL: the budget measures only the watermark perturbation, not the base
        # sampler's truncation policy.
        with torch.no_grad():
            p = torch.softmax(scores.float() / self.temperature, dim=-1)
            p_target = p.index_select(-1, target_ids).sum(dim=-1)
            p_other = p.index_select(-1, other_ids).sum(dim=-1)
            p_neutral = (1.0 - p_target - p_other).clamp_min(0.0)
            delta, kl = self._choose_delta(p_target, p_other, p_neutral)
            self._delta_steps.append(delta.detach().cpu())
            self._kl_steps.append(kl.detach().cpu())

        shift = signed.to(dtype=scores.dtype).unsqueeze(0) * delta.to(
            device=scores.device, dtype=scores.dtype
        ).unsqueeze(-1)
        return scores + shift

    def per_example_stats(self, step_counts: Sequence[int] | None = None) -> list[dict[str, float]]:
        """Return step-aggregated KL/strength diagnostics for each batch row.

        Hugging Face continues calling logits processors for already-finished rows
        while other rows in the batch are still decoding.  When ``step_counts``
        is supplied, each row is truncated to its actual generated-token count so
        those post-EOS calls do not contaminate the diagnostics.
        """
        if not self._delta_steps:
            return []
        deltas = torch.stack(self._delta_steps, dim=0).float()  # [steps, batch]
        kls = torch.stack(self._kl_steps, dim=0).float()
        if step_counts is not None and len(step_counts) != deltas.shape[1]:
            raise ValueError("step_counts length must match batch size")
        out: list[dict[str, float]] = []
        for i in range(deltas.shape[1]):
            n = deltas.shape[0] if step_counts is None else max(1, min(int(step_counts[i]), deltas.shape[0]))
            d = deltas[:n, i]
            k = kls[:n, i]
            out.append({
                "mean_applied_strength": float(d.mean().item()),
                "median_applied_strength": float(d.median().item()),
                "max_applied_strength": float(d.max().item()),
                "mean_step_kl": float(k.mean().item()),
                "max_step_kl": float(k.max().item()),
                "fraction_steps_at_strength_cap": float(
                    (d >= self.max_strength - 1e-5).float().mean().item()
                ) if self.max_strength > 0 else 1.0,
                "processor_steps": int(d.numel()),
            })
        return out
