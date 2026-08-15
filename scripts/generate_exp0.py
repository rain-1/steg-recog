#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from wmsteg.prompts import make_prompt_pool
from wmsteg.watermark import (
    LABEL_NAMES,
    ORANGE,
    PURPLE,
    ColorBiasLogitsProcessor,
    build_hyperplane_partition,
    build_random_partition,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate Experiment-0 purple/orange watermarked carrier text.")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output", default="runs/exp0_data")
    p.add_argument("--pairs", type=int, default=500, help="Total paired prompts across all splits; output has 2x rows.")
    p.add_argument("--bias", type=float, default=2.0, help="Positive logit bias for target-color tokens.")
    p.add_argument("--opposite-penalty", type=float, default=0.0)
    p.add_argument("--partition", choices=["hyperplane", "random"], default="hyperplane")
    p.add_argument("--partition-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=2025)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--min-new-tokens", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--neutral-top-k", type=int, default=256, help="Neutralize frequent generated token ids.")
    p.add_argument("--neutral-calibration-samples", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def choose_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def apply_chat(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_batch(model, tokenizer, prompts, *, processor=None, args=None):
    formatted = [apply_chat(tokenizer, p) for p in prompts]
    enc = tokenizer(formatted, return_tensors="pt", padding=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]
    lp = LogitsProcessorList([processor]) if processor is not None else LogitsProcessorList()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            logits_processor=lp,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = out[:, input_len:]
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    ids = [row.tolist() for row in completion_ids]
    return texts, ids


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(args.data_seed)
    np.random.seed(args.data_seed)
    torch.manual_seed(args.data_seed)

    dtype = choose_dtype()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    model.eval()

    pools = make_prompt_pool(args.data_seed)
    all_prompts = pools["train"] + pools["val"] + pools["test"]

    # Estimate high-frequency tokens from unwatermarked continuations, then make them neutral.
    neutral_counter = Counter()
    n_cal = max(0, args.neutral_calibration_samples)
    for start in range(0, n_cal, args.batch_size):
        batch = [all_prompts[(start + i) % len(all_prompts)] for i in range(min(args.batch_size, n_cal - start))]
        _, generated_ids = generate_batch(model, tokenizer, batch, processor=None, args=args)
        special_ids_now = set(getattr(tokenizer, "all_special_ids", []))
        for ids in generated_ids:
            neutral_counter.update(tok for tok in ids if tok not in special_ids_now)

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    frequent_ids = {tok for tok, _ in neutral_counter.most_common(args.neutral_top_k)}
    neutral_ids = special_ids | frequent_ids

    emb = model.get_input_embeddings().weight
    if args.partition == "hyperplane":
        partition = build_hyperplane_partition(
            emb,
            neutral_ids=neutral_ids,
            seed=args.partition_seed,
        )
    else:
        partition = build_random_partition(
            emb.shape[0],
            neutral_ids=neutral_ids,
            seed=args.partition_seed,
        )
    partition.metadata.update({
        "model": args.model,
        "bias": args.bias,
        "opposite_penalty": args.opposite_penalty,
        "neutral_top_k": args.neutral_top_k,
        "neutral_calibration_samples": args.neutral_calibration_samples,
    })
    partition.save(outdir)

    with open(outdir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # Allocate pair counts approximately 80/10/10 while keeping prompt topics disjoint.
    split_pairs = {
        "train": int(args.pairs * 0.8),
        "val": int(args.pairs * 0.1),
    }
    split_pairs["test"] = args.pairs - split_pairs["train"] - split_pairs["val"]

    processors = {
        PURPLE: ColorBiasLogitsProcessor(partition, PURPLE, args.bias, args.opposite_penalty),
        ORANGE: ColorBiasLogitsProcessor(partition, ORANGE, args.bias, args.opposite_penalty),
    }

    summary = {}
    rng = random.Random(args.data_seed)
    for split in ("train", "val", "test"):
        n_pairs = split_pairs[split]
        prompts = pools[split]
        # Make a reproducible list of underlying carrier prompts. Each appears once in each color.
        pair_prompts = [prompts[i % len(prompts)] for i in range(n_pairs)]
        rng.shuffle(pair_prompts)
        rows = []
        for label in (PURPLE, ORANGE):
            for start in range(0, n_pairs, args.batch_size):
                batch_prompts = pair_prompts[start:start + args.batch_size]
                texts, ids_batch = generate_batch(
                    model, tokenizer, batch_prompts, processor=processors[label], args=args
                )
                for local_i, (prompt, text, generated_token_ids) in enumerate(zip(batch_prompts, texts, ids_batch)):
                    # Count the tokens the detector will actually receive after decoding/re-tokenization.
                    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                    p, o, n = partition.counts(token_ids)
                    colored = max(1, p + o)
                    rows.append({
                        "pair_id": int(start + local_i),
                        "split": split,
                        "label": int(label),
                        "watermark": LABEL_NAMES[label],
                        "text": text,
                        "source_prompt": prompt,
                        "purple_tokens": p,
                        "orange_tokens": o,
                        "neutral_tokens": n,
                        "purple_fraction_colored": p / colored,
                        "z_score_purple_minus_orange": partition.z_score(token_ids),
                    })
        rng.shuffle(rows)
        path = outdir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        by_label = {}
        for label in (PURPLE, ORANGE):
            xs = [r for r in rows if r["label"] == label]
            by_label[LABEL_NAMES[label]] = {
                "n": len(xs),
                "mean_purple_fraction_colored": float(np.mean([r["purple_fraction_colored"] for r in xs])),
                "mean_z_score": float(np.mean([r["z_score_purple_minus_orange"] for r in xs])),
            }
        summary[split] = by_label

    with open(outdir / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote Experiment-0 data to {outdir}")


if __name__ == "__main__":
    main()
