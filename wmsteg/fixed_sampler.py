from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .watermark import ORANGE, PURPLE, TokenPartition


@dataclass(frozen=True)
class FixedBiasConfig:
    """Configuration for the external fixed-bias watermark sampler.

    There is intentionally no top-p or top-k truncation in the sampling path.
    This mirrors the simple sampler that previously worked: score the model's
    full next-token distribution, optionally gate, add a fixed color bias, and
    sample directly from the resulting full softmax.

    ``strength`` is defined *after* temperature: p0 = softmax(logits/T), then
    +/- strength is added to the scaled logits. This makes the meaning of the
    watermark strength independent of temperature.
    """

    temperature: float = 0.9
    strength: float = 1.5
    symmetric: bool = True
    min_entropy_nats: float = 0.0
    gate_top_k: int = 0

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.strength < 0:
            raise ValueError("strength must be non-negative")
        if self.min_entropy_nats < 0:
            raise ValueError("min_entropy_nats must be >= 0")
        if self.gate_top_k < 0:
            raise ValueError("gate_top_k must be >= 0")


class FixedColorBiasSampler:
    """Sample from a full-softmax distribution with a fixed color bias.

    Baseline distribution:
        p0(v) = softmax(logits(v) / temperature)

    Purple watermark, symmetric mode:
        q(v) proportional to exp(logits/T + strength) for purple tokens
        q(v) proportional to exp(logits/T - strength) for orange tokens
        neutral tokens are unchanged.

    Orange reverses the signs. In target-only mode, the opposite color receives
    zero bias instead of -strength.

    No sampling truncation is performed after the intervention. KL(q || p0) and
    total-variation distance are recorded as diagnostics, not constraints.
    """

    def __init__(self, partition: TokenPartition, label: int, config: FixedBiasConfig) -> None:
        if label not in (PURPLE, ORANGE):
            raise ValueError("label must be PURPLE (0) or ORANGE (1)")
        config.validate()
        self.partition = partition
        self.label = int(label)
        self.config = config
        self.vocab_size = int(partition.vocab_size)

        signed = torch.zeros(self.vocab_size, dtype=torch.float32)
        if label == PURPLE:
            signed[partition.purple_ids] = 1.0
            if config.symmetric:
                signed[partition.orange_ids] = -1.0
        else:
            signed[partition.orange_ids] = 1.0
            if config.symmetric:
                signed[partition.purple_ids] = -1.0
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

    def reset_stats(self) -> None:
        self._steps: list[dict[str, torch.Tensor]] = []

    def distributions(self, model_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if model_logits.ndim != 2:
            raise ValueError("model_logits must have shape [batch, vocab]")
        if model_logits.shape[-1] != self.vocab_size:
            raise ValueError(
                f"partition vocab_size={self.vocab_size} but logits vocab={model_logits.shape[-1]}"
            )

        cfg = self.config
        base_logits = model_logits.float() / cfg.temperature
        log_p0 = torch.log_softmax(base_logits, dim=-1)
        p0 = log_p0.exp()
        entropy = -(p0 * log_p0).sum(dim=-1)

        signed = self._signed(model_logits.device)
        target_mask = signed > 0
        if cfg.symmetric:
            other_mask = signed < 0
        else:
            # In target-only mode infer the opposite color from the partition.
            other_mask = torch.zeros_like(target_mask)
            if self.label == PURPLE:
                other_mask[torch.as_tensor(self.partition.orange_ids, device=model_logits.device)] = True
            else:
                other_mask[torch.as_tensor(self.partition.purple_ids, device=model_logits.device)] = True

        p_target = p0[:, target_mask].sum(dim=-1)
        p_other = p0[:, other_mask].sum(dim=-1)
        p_eligible = p_target + p_other

        entropy_ok = entropy >= cfg.min_entropy_nats
        if cfg.gate_top_k > 0:
            k = min(cfg.gate_top_k, self.vocab_size)
            top_ids = torch.topk(base_logits, k=k, dim=-1, largest=True, sorted=False).indices
            target_lookup = target_mask[top_ids]
            other_lookup = other_mask[top_ids]
            has_target = target_lookup.any(dim=-1)
            has_other = other_lookup.any(dim=-1)
            topk_ok = has_target & has_other
            target_topk_count = target_lookup.sum(dim=-1)
            other_topk_count = other_lookup.sum(dim=-1)
        else:
            topk_ok = torch.ones_like(entropy_ok)
            target_topk_count = torch.zeros_like(entropy)
            other_topk_count = torch.zeros_like(entropy)

        gate = entropy_ok & topk_ok
        applied_strength = torch.where(
            gate,
            torch.full_like(entropy, cfg.strength),
            torch.zeros_like(entropy),
        )

        q_logits = base_logits + signed.unsqueeze(0) * applied_strength.unsqueeze(-1)
        log_q = torch.log_softmax(q_logits, dim=-1)
        q = log_q.exp()

        actual_kl = (q * (log_q - log_p0)).sum(dim=-1)
        tv = 0.5 * torch.abs(q - p0).sum(dim=-1)
        q_target = q[:, target_mask].sum(dim=-1)
        q_other = q[:, other_mask].sum(dim=-1)
        q_eligible = q_target + q_other

        diag = {
            "entropy_nats": entropy,
            "p_target": p_target,
            "p_other": p_other,
            "p_eligible": p_eligible,
            "q_target": q_target,
            "q_other": q_other,
            "q_eligible": q_eligible,
            "entropy_ok": entropy_ok,
            "topk_ok": topk_ok,
            "gate": gate,
            "applied_strength": applied_strength,
            "actual_kl": actual_kl,
            "tv": tv,
            "target_topk_count": target_topk_count.float(),
            "other_topk_count": other_topk_count.float(),
        }
        return p0, q, diag

    def sample(
        self,
        model_logits: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        _, q, diag = self.distributions(model_logits)
        batch = q.shape[0]
        if active_mask is None:
            active_mask = torch.ones(batch, dtype=torch.bool, device=q.device)
        else:
            active_mask = active_mask.to(device=q.device, dtype=torch.bool)
            if active_mask.shape != (batch,):
                raise ValueError("active_mask must have shape [batch]")

        # Always sample every row so paired conditions consume the same random
        # stream shape at each decoding step, even if some rows have finished.
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
            steps = [s for s in self._steps if bool(s["active"][i])]
            if not steps:
                steps = [self._steps[0]]

            def mean(name: str) -> float:
                vals = torch.stack([s[name][i].float() for s in steps])
                return float(vals.mean().item())

            def maximum(name: str) -> float:
                vals = torch.stack([s[name][i].float() for s in steps])
                return float(vals.max().item())

            out.append({
                "mean_baseline_entropy_nats": mean("entropy_nats"),
                "mean_baseline_target_mass": mean("p_target"),
                "mean_baseline_opposite_mass": mean("p_other"),
                "mean_baseline_eligible_mass": mean("p_eligible"),
                "mean_watermarked_target_mass": mean("q_target"),
                "mean_watermarked_opposite_mass": mean("q_other"),
                "mean_watermarked_eligible_mass": mean("q_eligible"),
                "mean_applied_strength": mean("applied_strength"),
                "mean_step_kl": mean("actual_kl"),
                "max_step_kl": maximum("actual_kl"),
                "mean_step_tv": mean("tv"),
                "fraction_steps_watermark_active": mean("gate"),
                "fraction_steps_low_entropy": 1.0 - mean("entropy_ok"),
                "fraction_steps_topk_gate_failed": 1.0 - mean("topk_ok"),
                "mean_target_tokens_in_gate_topk": mean("target_topk_count"),
                "mean_opposite_tokens_in_gate_topk": mean("other_topk_count"),
                "fraction_sampled_target_color": mean("sampled_target"),
                "fraction_sampled_opposite_color": mean("sampled_other"),
                "fraction_sampled_neutral": mean("sampled_neutral"),
                "sampler_steps": len(steps),
            })
        return out


@dataclass(frozen=True)
class PlainSamplingConfig:
    temperature: float = 0.9

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


class PlainFullSoftmaxSampler:
    """Unwatermarked full-softmax sampler used to calibrate the rewrite distribution."""

    def __init__(self, config: PlainSamplingConfig) -> None:
        config.validate()
        self.config = config
        self.reset_stats()

    def reset_stats(self) -> None:
        self._steps: list[dict[str, torch.Tensor]] = []

    def sample(
        self,
        model_logits: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        if model_logits.ndim != 2:
            raise ValueError("model_logits must have shape [batch, vocab]")
        logits = model_logits.float() / self.config.temperature
        log_p = torch.log_softmax(logits, dim=-1)
        p = log_p.exp()
        entropy = -(p * log_p).sum(dim=-1)
        batch = p.shape[0]
        if active_mask is None:
            active_mask = torch.ones(batch, dtype=torch.bool, device=p.device)
        else:
            active_mask = active_mask.to(device=p.device, dtype=torch.bool)
        sampled = torch.multinomial(p, num_samples=1, generator=generator).squeeze(-1)
        self._steps.append({
            "active": active_mask.detach().cpu(),
            "entropy_nats": entropy.detach().cpu(),
        })
        return sampled

    def per_example_stats(self) -> list[dict[str, float]]:
        if not self._steps:
            return []
        batch = int(self._steps[0]["active"].shape[0])
        out: list[dict[str, float]] = []
        for i in range(batch):
            steps = [s for s in self._steps if bool(s["active"][i])]
            if not steps:
                steps = [self._steps[0]]
            vals = torch.stack([s["entropy_nats"][i].float() for s in steps])
            out.append({
                "mean_baseline_entropy_nats": float(vals.mean().item()),
                "sampler_steps": len(steps),
            })
        return out


def generate_with_fixed_bias(
    model,
    tokenizer,
    formatted_prompts: Sequence[str],
    sampler: FixedColorBiasSampler,
    *,
    max_new_tokens: int,
    generator_seed: int,
) -> tuple[list[str], list[dict[str, float]], list[list[int]]]:
    """Batched autoregressive generation sampled directly from the full biased softmax."""
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
    eos_raw = tokenizer.eos_token_id
    if eos_raw is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_raw, (list, tuple, set)):
        eos_ids = {int(x) for x in eos_raw}
    else:
        eos_ids = {int(eos_raw)}
    fallback_finished_token = next(iter(eos_ids), int(tokenizer.pad_token_id or 0))

    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(int(generator_seed))

    sampler.reset_stats()
    generated: list[list[int]] = [[] for _ in range(batch)]
    terminated_by_eos = [False for _ in range(batch)]
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
                        terminated_by_eos[i] = True
                else:
                    feed[i] = fallback_finished_token

            if bool(finished.all()) or step_idx + 1 >= max_new_tokens:
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch, 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
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

    decode_ids: list[list[int]] = []
    for seq in generated:
        clean: list[int] = []
        for tok in seq:
            if tok in eos_ids:
                break
            clean.append(tok)
        decode_ids.append(clean)
    texts = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)
    stats = sampler.per_example_stats()
    for i, st in enumerate(stats):
        st["generated_token_count"] = len(decode_ids[i])
        st["terminated_by_eos"] = bool(terminated_by_eos[i])
        st["hit_max_new_tokens"] = bool(not terminated_by_eos[i] and len(generated[i]) >= max_new_tokens)
    return [t.strip() for t in texts], stats, decode_ids
