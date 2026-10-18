"""Watermark schemes beyond the fixed unigram partition.

The pilot watermark (`watermark.TokenPartition` + `fixed_sampler`) is a unigram
scheme: every token id has a fixed color, independent of context. Deployed and
published LM watermarks instead key the green list on the preceding tokens
(KGW, Kirchenbauer et al. 2023; SynthID-Text uses a 4-token context). The
context width is the main knob controlling how learnable a watermark is, so this
module provides one interface over:

  * `UnigramScheme`      - wraps an existing `TokenPartition` (hyperplane or random)
  * `ContextHashScheme`  - KGW-style green lists seeded by a hash of the previous
                           `context_width` token ids (context_width=0 gives a
                           random-unigram scheme with a hash-defined green list)

Both expose the same three operations:

  * `signed_mask(context)`  -> float tensor [batch, vocab] in {-1, 0, +1}
                               (+1 green, -1 red, 0 neutral) for the next token
  * `score(token_ids)`      -> `SchemeScore` with green/red counts and a z statistic
  * an additive logits processor (`AdditiveWatermarkProcessor`) usable with HF
    `generate` or a custom sampling loop.

Hashing uses exact int64 arithmetic modulo the Mersenne prime 2^31 - 1, so green
lists are identical on CPU and GPU and never rely on signed-overflow behavior.
The hash is not cryptographic; it only needs to be uncorrelated with anything the
model could know about token ids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch

from .watermark import ORANGE, PURPLE, TokenPartition


_P = 2_147_483_647  # 2^31 - 1


def _mix(x: torch.Tensor) -> torch.Tensor:
    """Nonlinear integer mixer on int64 values in [0, 2^31 - 1).

    Every product stays below 2^63, so no step overflows.
    """
    x = (x * 1_103_515_245 + 12_345) % _P
    x = x ^ (x >> 13)
    x = (x * 2_654_435_761) % _P
    x = x ^ (x >> 16)
    x = (x * 1_597_334_677) % _P
    return x ^ (x >> 15)


@dataclass(frozen=True)
class SchemeScore:
    green: int
    red: int
    neutral: int
    scored: int
    z: float

    @property
    def green_fraction(self) -> float:
        colored = self.green + self.red
        return self.green / colored if colored else 0.5


def _z(green: int, red: int, gamma: float) -> float:
    """One-proportion z statistic for green among colored (non-neutral) tokens."""
    n = green + red
    if n == 0:
        return 0.0
    return float((green - gamma * n) / math.sqrt(n * gamma * (1.0 - gamma)))


class UnigramScheme:
    """The pilot's fixed partition viewed as a context-free scheme.

    `label` picks which color counts as green: PURPLE (0) or ORANGE (1). The
    opposite color is red; neutral tokens are unscored. With the existing
    frequency-balanced partitions gamma is 0.5 by construction.
    """

    context_width = 0

    def __init__(self, partition: TokenPartition, label: int = PURPLE) -> None:
        if label not in (PURPLE, ORANGE):
            raise ValueError("label must be PURPLE (0) or ORANGE (1)")
        self.partition = partition
        self.label = int(label)
        self.vocab_size = int(partition.vocab_size)
        self.gamma = 0.5
        signed = np.zeros(self.vocab_size, dtype=np.float32)
        green = partition.purple_ids if label == PURPLE else partition.orange_ids
        red = partition.orange_ids if label == PURPLE else partition.purple_ids
        signed[green] = 1.0
        signed[red] = -1.0
        self._signed_cpu = torch.from_numpy(signed)
        self._cache: dict[str, torch.Tensor] = {}

    @property
    def name(self) -> str:
        kind = self.partition.metadata.get("kind", "partition")
        return f"unigram[{kind}]"

    def signed_mask(self, context: torch.Tensor) -> torch.Tensor:
        key = str(context.device)
        row = self._cache.get(key)
        if row is None:
            row = self._signed_cpu.to(context.device)
            self._cache[key] = row
        return row.unsqueeze(0).expand(context.shape[0], -1)

    def score(self, token_ids: Sequence[int], *, skip: Iterable[int] = ()) -> SchemeScore:
        skip_set = {int(t) for t in skip}
        ids = [int(t) for t in token_ids if int(t) not in skip_set]
        signed = self._signed_cpu.numpy()
        ids_arr = np.asarray(ids, dtype=np.int64)
        ids_arr = ids_arr[(ids_arr >= 0) & (ids_arr < self.vocab_size)]
        s = signed[ids_arr]
        green, red = int(np.sum(s > 0)), int(np.sum(s < 0))
        return SchemeScore(green, red, int(np.sum(s == 0)), len(ids_arr), _z(green, red, self.gamma))


@dataclass
class ContextHashScheme:
    """KGW-style watermark: green list seeded by the previous `context_width` tokens.

    context_width=0  -> one global hash-defined green list (Unigram-Watermark,
                        Zhao et al. 2023), with no embedding-geometry structure
    context_width=1  -> KGW LeftHash
    context_width=h  -> order-sensitive hash of the last h tokens

    `neutral_ids` (special tokens, whitespace, punctuation, ...) are never biased
    or scored, mirroring the pilot partition. When scoring, repeated
    (context, token) pairs are counted once by default, as in standard KGW
    detection, so copied spans cannot inflate z.
    """

    vocab_size: int
    key: int = 15485863
    context_width: int = 1
    gamma: float = 0.5
    neutral_ids: Sequence[int] = ()
    _neutral_cpu: torch.Tensor = field(init=False, repr=False)
    _vocab_cache: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_width < 0:
            raise ValueError("context_width must be >= 0")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if not 0 <= self.key < _P:
            raise ValueError("key must be in [0, 2^31 - 1)")
        # Threshold computed exactly as callers of batched_signed_h01 do, so both paths agree.
        self._gamma_threshold = int(self.gamma * _P)
        neutral = torch.zeros(self.vocab_size, dtype=torch.bool)
        ids = [int(t) for t in self.neutral_ids if 0 <= int(t) < self.vocab_size]
        if ids:
            neutral[torch.as_tensor(ids, dtype=torch.long)] = True
        self._neutral_cpu = neutral

    @property
    def name(self) -> str:
        return f"context_hash[h={self.context_width},gamma={self.gamma}]"

    def _on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = self._vocab_cache.get(key)
        if cached is None:
            cached = (
                torch.arange(self.vocab_size, dtype=torch.int64, device=device),
                self._neutral_cpu.to(device),
            )
            self._vocab_cache[key] = cached
        return cached

    def _seeds(self, context: torch.Tensor) -> torch.Tensor:
        """Per-row seed from the last `context_width` token ids. Shape [batch]."""
        batch = context.shape[0]
        seed = torch.full((batch,), self.key, dtype=torch.int64, device=context.device)
        h = self.context_width
        if h == 0:
            return _mix(seed)
        if context.ndim != 2 or context.shape[1] < h:
            raise ValueError(f"context must have shape [batch, >={h}]")
        for tok in context[:, -h:].to(torch.int64).unbind(dim=1):
            seed = _mix((seed * 31 + tok % _P + 1) % _P)
        return seed

    def green_mask(self, context: torch.Tensor) -> torch.Tensor:
        """Bool [batch, vocab]: True where the next token would be green."""
        vocab, _ = self._on(context.device)
        seeds = self._seeds(context)
        u = _mix((seeds.unsqueeze(1) * 65_537 + vocab.unsqueeze(0) + 7) % _P)
        return u < self._gamma_threshold

    def signed_mask(self, context: torch.Tensor) -> torch.Tensor:
        _, neutral = self._on(context.device)
        signed = self.green_mask(context).to(torch.float32) * 2.0 - 1.0
        return signed.masked_fill(neutral.unsqueeze(0), 0.0)

    def score(
        self,
        token_ids: Sequence[int],
        *,
        prefix: Sequence[int] = (),
        dedupe: bool = True,
        skip: Iterable[int] = (),
    ) -> SchemeScore:
        """Score `token_ids`, using `prefix` only as hash context for the first tokens.

        `skip` removes token ids from scoring (e.g. tokens already present in a
        prompt, for in-context continuation tests).
        """
        h = self.context_width
        full = [int(t) for t in prefix] + [int(t) for t in token_ids]
        start = len(prefix)
        skip_set = {int(t) for t in skip}
        positions = [i for i in range(max(start, h), len(full)) if full[i] not in skip_set]
        if not positions:
            return SchemeScore(0, 0, 0, 0, 0.0)
        seen: set[tuple[int, ...]] = set()
        keep: list[int] = []
        for i in positions:
            gram = tuple(full[i - h : i + 1])
            if dedupe and gram in seen:
                continue
            seen.add(gram)
            keep.append(i)
        if not keep:
            return SchemeScore(0, 0, 0, 0, 0.0)
        full_t = torch.as_tensor(full, dtype=torch.int64)
        if h:
            ctx = torch.stack([full_t[i - h : i] for i in keep])
        else:
            ctx = torch.zeros((len(keep), 0), dtype=torch.int64)
        tok = full_t[keep]
        if bool(((tok < 0) | (tok >= self.vocab_size)).any()):
            raise ValueError("scored token id is outside the watermark vocabulary")
        # Only evaluate the sampled token, rather than allocating [tokens, vocab].
        seeds = self._seeds(ctx)
        u = _mix((seeds * 65_537 + tok + 7) % _P)
        s = (u < self._gamma_threshold).float() * 2 - 1
        s.masked_fill_(self._neutral_cpu[tok], 0)
        green, red = int((s > 0).sum()), int((s < 0).sum())
        return SchemeScore(green, red, int((s == 0).sum()), len(keep), _z(green, red, self.gamma))


class SignedTableScheme:
    """Context-free scheme from a fixed signed vocabulary table (+1 green, -1 red, 0 neutral).

    Used for embedding-hyperplane partitions (`TokenPartition`) inside the
    wmscale pipeline. `table` is the row index into the stacked tables that the
    vLLM processor loads, so both backends apply the same bias. Scoring follows
    `ContextHashScheme` at context_width=0 (repeated token ids counted once by
    default), so z values are comparable across hash and partition runs.
    """

    context_width = 0

    def __init__(self, signed: np.ndarray, *, table: int, name: str = "partition", gamma: float = 0.5) -> None:
        self.signed = np.asarray(signed, dtype=np.float32)
        self.vocab_size = int(self.signed.shape[0])
        self.table = int(table)
        self.key = self.table
        self.gamma = float(gamma)
        self._name = name
        self._signed_cpu = torch.from_numpy(self.signed)
        self._cache: dict[str, torch.Tensor] = {}

    @classmethod
    def from_partition(cls, partition: TokenPartition, *, green: str, table: int, vocab_size: int,
                       extra_neutral: Iterable[int] = ()) -> "SignedTableScheme":
        if green not in ("purple", "orange"):
            raise ValueError("green must be 'purple' or 'orange'")
        signed = np.zeros(max(int(vocab_size), partition.vocab_size), dtype=np.float32)
        g, r = (partition.purple_ids, partition.orange_ids) if green == "purple" else (partition.orange_ids, partition.purple_ids)
        signed[g] = 1.0
        signed[r] = -1.0
        extra = np.asarray([int(t) for t in extra_neutral if 0 <= int(t) < len(signed)], dtype=np.int64)
        signed[extra] = 0.0
        tag = partition.metadata.get("tag", partition.metadata.get("seed"))
        return cls(signed, table=table, name=f"partition[{tag},{green}]")

    @property
    def name(self) -> str:
        return self._name

    def signed_mask(self, context: torch.Tensor) -> torch.Tensor:
        key = str(context.device)
        row = self._cache.get(key)
        if row is None:
            row = self._signed_cpu.to(context.device)
            self._cache[key] = row
        return row.unsqueeze(0).expand(context.shape[0], -1)

    def score(
        self,
        token_ids: Sequence[int],
        *,
        prefix: Sequence[int] = (),
        dedupe: bool = True,
        skip: Iterable[int] = (),
    ) -> SchemeScore:
        skip_set = {int(t) for t in skip}
        keep: list[int] = []
        seen: set[int] = set()
        for t in (int(x) for x in token_ids):
            if t in skip_set or (dedupe and t in seen):
                continue
            seen.add(t)
            keep.append(t)
        if not keep:
            return SchemeScore(0, 0, 0, 0, 0.0)
        ids = np.asarray(keep, dtype=np.int64)
        if bool(((ids < 0) | (ids >= self.vocab_size)).any()):
            raise ValueError("scored token id is outside the watermark vocabulary")
        s = self.signed[ids]
        green, red = int(np.sum(s > 0)), int(np.sum(s < 0))
        return SchemeScore(green, red, int(np.sum(s == 0)), len(ids), _z(green, red, self.gamma))


def batched_signed_h01(
    keys: torch.Tensor,
    widths: torch.Tensor,
    prev_tokens: torch.Tensor,
    gamma_thresholds: torch.Tensor,
    vocab_size: int,
    neutral: torch.Tensor | None = None,
) -> torch.Tensor:
    """Signed masks for rows with *different* keys and context width 0 or 1.

    Row i equals `ContextHashScheme(key=keys[i], context_width=widths[i],
    gamma=gamma_thresholds[i] / (2^31 - 1)).signed_mask([[prev_tokens[i]]])`.
    Used by the vLLM processor, where one batch mixes requests.
    """
    keys = keys.to(torch.int64)
    seed0 = _mix(keys)
    seed1 = _mix((keys * 31 + prev_tokens.to(torch.int64) % _P + 1) % _P)
    seed = torch.where(widths == 1, seed1, seed0)
    vocab = torch.arange(vocab_size, dtype=torch.int64, device=keys.device)
    u = _mix((seed.unsqueeze(1) * 65_537 + vocab.unsqueeze(0) + 7) % _P)
    signed = torch.where(u < gamma_thresholds.to(torch.int64).unsqueeze(1), 1.0, -1.0)
    if neutral is not None:
        signed = signed.masked_fill(neutral[:vocab_size].unsqueeze(0), 0.0)
    return signed


class AdditiveWatermarkProcessor:
    """Additive green/red logit bias for any scheme exposing `signed_mask`.

    symmetric=True  -> +delta green, -delta red (the pilot's purple/orange setup)
    symmetric=False -> +delta green only (standard KGW)

    Call signature matches HF `LogitsProcessor`: (input_ids, scores) -> scores.
    Like `FixedBiasConfig`, the bias is meant to be applied after temperature
    scaling so delta keeps a temperature-independent meaning; when used inside HF
    `generate`, place it after the temperature warper or fold temperature in first.
    """

    def __init__(self, scheme, delta: float, *, symmetric: bool = False) -> None:
        if delta < 0:
            raise ValueError("delta must be non-negative")
        self.scheme = scheme
        self.delta = float(delta)
        self.symmetric = bool(symmetric)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        signed = self.scheme.signed_mask(input_ids)
        if not self.symmetric:
            signed = signed.clamp_min(0.0)
        return scores + self.delta * signed.to(scores.dtype)


def continuation_shift(
    scheme,
    prompt_ids: Sequence[int],
    continuation_ids: Sequence[int],
) -> SchemeScore:
    """Watermark statistic of a model's *unwatermarked* continuation of a prompt.

    This is the generation-side awareness probe: give a model a watermarked
    prefix, sample a continuation with a plain sampler, and test whether the
    continuation carries the prefix's watermark. Token ids that occur in the
    prompt are excluded so verbatim copying of prompt words (which share the
    prompt's colors) cannot produce a spurious positive.
    """
    skip = set(int(t) for t in prompt_ids)
    if isinstance(scheme, ContextHashScheme):
        return scheme.score(continuation_ids, prefix=prompt_ids, skip=skip)
    return scheme.score(continuation_ids, skip=skip)
