from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .watermark import ORANGE, PURPLE, TokenPartition


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 50
    kl_budget: float = 0.04
    max_tilt: float = 12.0
    bisection_steps: int = 16
    min_entropy_nats: float = 1.5
    min_color_mass: float = 0.02
    min_eligible_mass: float = 0.08

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.kl_budget < 0:
            raise ValueError("kl_budget must be non-negative")
        if self.max_tilt < 0:
            raise ValueError("max_tilt must be non-negative")
        if self.bisection_steps < 1:
            raise ValueError("bisection_steps must be >= 1")
        if self.min_entropy_nats < 0:
            raise ValueError("min_entropy_nats must be >= 0")
        if not 0 <= self.min_color_mass <= 0.5:
            raise ValueError("min_color_mass must be in [0, 0.5]")
        if not 0 <= self.min_eligible_mass <= 1:
            raise ValueError("min_eligible_mass must be in [0, 1]")


def _top_k_top_p_logits(
    scaled_logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Apply temperature-following top-k then nucleus truncation.

    With the normal ``top_k > 0`` configuration, nucleus filtering is performed
    only over those k candidates rather than sorting the full vocabulary. This is
    mathematically equivalent to first masking to top-k and is much faster for
    Qwen-sized vocabularies.
    """
    if scaled_logits.ndim != 2:
        raise ValueError("scaled_logits must have shape [batch, vocab]")
    x = scaled_logits.float()
    batch, vocab = x.shape

    if top_k > 0 and top_k < vocab:
        k = min(top_k, vocab)
        cand_logits, cand_ids = torch.topk(x, k=k, dim=-1, largest=True, sorted=True)
        keep = torch.ones_like(cand_logits, dtype=torch.bool)
        if top_p < 1.0:
            probs = torch.softmax(cand_logits, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            keep = ~remove
        out = torch.full_like(x, float("-inf"))
        kept_logits = cand_logits.masked_fill(~keep, float("-inf"))
        out.scatter_(dim=-1, index=cand_ids, src=kept_logits)
        return out

    logits = x.clone()
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove_mask = torch.zeros_like(remove, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=remove)
        logits = logits.masked_fill(remove_mask, float("-inf"))

    finite = torch.isfinite(logits)
    bad_rows = ~finite.any(dim=-1)
    if bad_rows.any():
        argmax = torch.argmax(x[bad_rows], dim=-1)
        logits[bad_rows] = float("-inf")
        logits[bad_rows, argmax] = x[bad_rows, argmax]
    return logits



class ActualDistributionKLSampler:
    """KL-constrained color watermarking on the *actual* sampling distribution.

    For ordinary model logits l, we first construct the same baseline distribution
    that will actually be sampled from:

        l -> temperature -> top-k -> top-p -> p0.

    Only then do we apply an exponential color tilt

        q_delta(v) ∝ p0(v) exp(delta * c_v),

    where c_v is +1 for the target color, -1 for the opposite color, and 0 for
    neutral/ineligible tokens. ``delta`` is chosen independently at each token so
    that KL(q_delta || p0) <= kl_budget. Sampling happens *directly from q_delta*;
    there is no later warper that can erase the intervention.

    Watermarking is additionally gated off when the baseline distribution has low
    entropy or lacks plausible mass on both colors. This is the quality-preserving
    analogue of the KL + color regularization used in the user's earlier work.
    """

    def __init__(self, partition: TokenPartition, label: int, config: SamplingConfig) -> None:
        if label not in (PURPLE, ORANGE):
            raise ValueError("label must be PURPLE (0) or ORANGE (1)")
        config.validate()
        self.partition = partition
        self.label = int(label)
        self.config = config
        self.vocab_size = int(partition.vocab_size)

        signed = torch.zeros(self.vocab_size, dtype=torch.float32)
        if self.label == PURPLE:
            signed[partition.purple_ids] = 1.0
            signed[partition.orange_ids] = -1.0
        else:
            signed[partition.purple_ids] = -1.0
            signed[partition.orange_ids] = 1.0
        self._signed_cpu = signed
        self._cache: dict[str, torch.Tensor] = {}
        self.reset_stats()

    def _signed(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        out = self._cache.get(key)
        if out is None:
            out = self._signed_cpu.to(device=device)
            self._cache[key] = out
        return out

    @staticmethod
    def _kl_from_group_masses(
        delta: torch.Tensor,
        p_target: torch.Tensor,
        p_other: torch.Tensor,
        p_neutral: torch.Tensor,
    ) -> torch.Tensor:
        # p0 has already absorbed temperature/truncation, so delta is directly
        # the exponential tilt parameter and is not divided by temperature here.
        ep = torch.exp(delta)
        em = torch.exp(-delta)
        z = (p_target * ep + p_other * em + p_neutral).clamp_min(1e-30)
        q_target = p_target * ep / z
        q_other = p_other * em / z
        return delta * (q_target - q_other) - torch.log(z)

    def _choose_delta(
        self,
        p_target: torch.Tensor,
        p_other: torch.Tensor,
        p_neutral: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        zero = torch.zeros_like(p_target)
        if cfg.max_tilt == 0 or cfg.kl_budget == 0 or not bool(gate.any()):
            return zero, zero, torch.zeros_like(gate)

        hi = torch.full_like(p_target, cfg.max_tilt)
        kl_hi = self._kl_from_group_masses(hi, p_target, p_other, p_neutral)
        cap_ok = kl_hi <= cfg.kl_budget
        lo = torch.zeros_like(p_target)
        upper = hi.clone()

        for _ in range(cfg.bisection_steps):
            mid = (lo + upper) * 0.5
            kl_mid = self._kl_from_group_masses(mid, p_target, p_other, p_neutral)
            feasible = kl_mid <= cfg.kl_budget
            lo = torch.where(feasible, mid, lo)
            upper = torch.where(feasible, upper, mid)

        delta = torch.where(cap_ok, hi, lo)
        delta = torch.where(gate, delta, zero)
        kl = self._kl_from_group_masses(delta, p_target, p_other, p_neutral)
        kl = torch.where(gate, kl, zero)
        # Rows that are gated and not at the cap are KL-limited.
        kl_limited = gate & (~cap_ok)
        return delta, kl, kl_limited

    def reset_stats(self) -> None:
        self._steps: list[dict[str, torch.Tensor]] = []

    def distributions(
        self,
        model_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return (p0, q, diagnostics) without sampling.

        This is exposed for tests and diagnostics. ``model_logits`` are raw model
        logits before temperature/top-k/top-p.
        """
        if model_logits.ndim != 2:
            raise ValueError("model_logits must have shape [batch, vocab]")
        if model_logits.shape[-1] != self.vocab_size:
            raise ValueError(
                f"partition vocab_size={self.vocab_size} but logits vocab={model_logits.shape[-1]}"
            )

        cfg = self.config
        scaled = model_logits.float() / cfg.temperature
        base_logits = _top_k_top_p_logits(scaled, top_k=cfg.top_k, top_p=cfg.top_p)
        p0 = torch.softmax(base_logits, dim=-1)
        log_p0 = torch.log_softmax(base_logits, dim=-1)
        entropy = -(p0 * torch.where(p0 > 0, log_p0, torch.zeros_like(log_p0))).sum(dim=-1)

        signed = self._signed(model_logits.device)
        target_mask = signed > 0
        other_mask = signed < 0
        p_target = p0[:, target_mask].sum(dim=-1)
        p_other = p0[:, other_mask].sum(dim=-1)
        p_eligible = p_target + p_other
        p_neutral = (1.0 - p_eligible).clamp_min(0.0)

        support = p0 > 0
        target_support_count = (support & target_mask.unsqueeze(0)).sum(dim=-1)
        other_support_count = (support & other_mask.unsqueeze(0)).sum(dim=-1)
        support_size = support.sum(dim=-1)

        entropy_ok = entropy >= cfg.min_entropy_nats
        color_mass_ok = (p_target >= cfg.min_color_mass) & (p_other >= cfg.min_color_mass)
        eligible_mass_ok = p_eligible >= cfg.min_eligible_mass
        gate = entropy_ok & color_mass_ok & eligible_mass_ok

        delta, kl, kl_limited = self._choose_delta(p_target, p_other, p_neutral, gate)
        q_logits = base_logits + signed.unsqueeze(0) * delta.unsqueeze(-1)
        q = torch.softmax(q_logits, dim=-1)

        # Compute actual KL and TV on the exact sampled distributions, not merely
        # the analytic grouped formula. This catches implementation regressions.
        log_q = torch.log_softmax(q_logits, dim=-1)
        actual_kl = (q * torch.where(q > 0, log_q - log_p0, torch.zeros_like(q))).sum(dim=-1)
        tv = 0.5 * torch.abs(q - p0).sum(dim=-1)

        diag = {
            "entropy_nats": entropy,
            "p_target": p_target,
            "p_other": p_other,
            "p_eligible": p_eligible,
            "support_size": support_size.float(),
            "target_support_count": target_support_count.float(),
            "other_support_count": other_support_count.float(),
            "entropy_ok": entropy_ok,
            "color_mass_ok": color_mass_ok,
            "eligible_mass_ok": eligible_mass_ok,
            "gate": gate,
            "delta": delta,
            "analytic_kl": kl,
            "actual_kl": actual_kl,
            "tv": tv,
            "kl_limited": kl_limited,
            "at_cap": gate & (delta >= cfg.max_tilt - 1e-5),
        }
        return p0, q, diag

    def sample(
        self,
        model_logits: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        p0, q, diag = self.distributions(model_logits)
        batch = q.shape[0]
        if active_mask is None:
            active_mask = torch.ones(batch, dtype=torch.bool, device=q.device)
        else:
            active_mask = active_mask.to(device=q.device, dtype=torch.bool)
            if active_mask.shape != (batch,):
                raise ValueError("active_mask must have shape [batch]")

        # Multinomial is called on every row (including finished rows) so paired
        # purple/orange runs driven by the same seed consume the same number of
        # random draws per decoding step.
        sampled = torch.multinomial(q, num_samples=1, generator=generator).squeeze(-1)
        signed = self._signed(q.device)
        sampled_sign = signed[sampled]

        step = {k: v.detach().cpu() for k, v in diag.items()}
        step["active"] = active_mask.detach().cpu()
        step["sampled_target"] = ((sampled_sign > 0) & active_mask).detach().cpu()
        step["sampled_other"] = ((sampled_sign < 0) & active_mask).detach().cpu()
        step["sampled_neutral"] = ((sampled_sign == 0) & active_mask).detach().cpu()
        self._steps.append(step)
        return sampled

    def per_example_stats(self) -> list[dict[str, float]]:
        if not self._steps:
            return []
        batch = int(self._steps[0]["active"].shape[0])
        out: list[dict[str, float]] = []
        for i in range(batch):
            active_indices = [s for s in self._steps if bool(s["active"][i])]
            if not active_indices:
                active_indices = [self._steps[0]]

            def mean(name: str) -> float:
                vals = torch.stack([s[name][i].float() for s in active_indices])
                return float(vals.mean().item())

            def maximum(name: str) -> float:
                vals = torch.stack([s[name][i].float() for s in active_indices])
                return float(vals.max().item())

            out.append({
                "mean_baseline_entropy_nats": mean("entropy_nats"),
                "mean_baseline_target_mass": mean("p_target"),
                "mean_baseline_opposite_mass": mean("p_other"),
                "mean_baseline_eligible_mass": mean("p_eligible"),
                "mean_sampling_support_size": mean("support_size"),
                "mean_target_tokens_in_support": mean("target_support_count"),
                "mean_opposite_tokens_in_support": mean("other_support_count"),
                "mean_applied_tilt": mean("delta"),
                "max_applied_tilt": maximum("delta"),
                "mean_step_kl": mean("actual_kl"),
                "max_step_kl": maximum("actual_kl"),
                "mean_step_tv": mean("tv"),
                "fraction_steps_watermark_active": mean("gate"),
                "fraction_steps_low_entropy": 1.0 - mean("entropy_ok"),
                "fraction_steps_insufficient_color_mass": 1.0 - mean("color_mass_ok"),
                "fraction_steps_insufficient_eligible_mass": 1.0 - mean("eligible_mass_ok"),
                "fraction_active_steps_kl_limited": mean("kl_limited"),
                "fraction_active_steps_at_tilt_cap": mean("at_cap"),
                "fraction_sampled_target_color": mean("sampled_target"),
                "fraction_sampled_opposite_color": mean("sampled_other"),
                "fraction_sampled_neutral": mean("sampled_neutral"),
                "sampler_steps": len(active_indices),
            })
        return out


def generate_with_actual_distribution_kl(
    model,
    tokenizer,
    formatted_prompts: Sequence[str],
    sampler: ActualDistributionKLSampler,
    *,
    max_new_tokens: int,
    generator_seed: int,
) -> tuple[list[str], list[dict[str, float]], list[list[int]]]:
    """Batched autoregressive generation that samples directly from sampler.q.

    This replaces ``model.generate`` for the watermark path because the standard
    generation stack applies sampling warpers *after* logits processors. Here the
    baseline p0, KL constraint, watermark tilt, and multinomial sample are all one
    atomic operation on the exact same distribution.
    """
    if not formatted_prompts:
        return [], [], []
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    enc = tokenizer(list(formatted_prompts), return_tensors="pt", padding=True)
    device = model.device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)

    batch = input_ids.shape[0]
    eos_ids_raw = tokenizer.eos_token_id
    if eos_ids_raw is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_ids_raw, (list, tuple, set)):
        eos_ids = {int(x) for x in eos_ids_raw}
    else:
        eos_ids = {int(eos_ids_raw)}
    fallback_finished_token = next(iter(eos_ids), int(tokenizer.pad_token_id or 0))

    # CUDA generators must live on the same device as torch.multinomial's tensor.
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(int(generator_seed))

    sampler.reset_stats()
    generated: list[list[int]] = [[] for _ in range(batch)]
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    with torch.inference_mode():
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        for step_idx in range(max_new_tokens):
            active = ~finished
            sampled = sampler.sample(logits, generator=generator, active_mask=active)
            feed = sampled.clone()

            for i in range(batch):
                if bool(active[i]):
                    tok = int(sampled[i].item())
                    generated[i].append(tok)
                    if tok in eos_ids:
                        finished[i] = True
                else:
                    feed[i] = fallback_finished_token

            if bool(finished.all()) or step_idx + 1 >= max_new_tokens:
                break

            # The cache is extended for every row. Finished rows receive EOS/pad
            # tokens but their logits/stats are ignored; active rows are unaffected.
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch, 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            # Position ids follow the unpadded token count for each row, as in
            # Hugging Face generation with left-padded batches.
            position_ids_step = attention_mask.long().sum(dim=-1, keepdim=True) - 1
            outputs = model(
                input_ids=feed.unsqueeze(-1),
                attention_mask=attention_mask,
                position_ids=position_ids_step,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

    # Strip EOS from decoded completion. Keeping raw token ids in the return value
    # lets callers audit exact color counts if desired.
    decode_ids: list[list[int]] = []
    for seq in generated:
        clean = []
        for tok in seq:
            if tok in eos_ids:
                break
            clean.append(tok)
        decode_ids.append(clean)
    texts = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)
    stats = sampler.per_example_stats()
    return [t.strip() for t in texts], stats, decode_ids
