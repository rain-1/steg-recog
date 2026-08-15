# Modal watermark research-bank datagen (v2)

This package generates a reusable paired purple/orange Lyra prompt bank for the watermark-recognition and later feature-generalization experiments. It deliberately defaults to **small proof-of-concept runs**. Nothing produced by `smoke`, `poc`, or `pilot` is marked frozen.

## What changed from v1

- Modal class-parameter annotation fix is included (`modal_app.py` does not use postponed annotations).
- Hugging Face access uses a Modal Secret named `huggingface-secret` containing `HF_TOKEN`; no token is passed on the run command.
- Source assignment is now reusable: `fold = SHA256(source_id) mod 10`.
  - default compatibility split: folds 0–7 train, fold 8 val, fold 9 test.
- Main source selection round-robins across `(mode, topic)` buckets to keep PoCs broad without imposing a permanent synthetic distribution.
- Every source retains Lyra metadata plus exact source token ids.
- Every generated condition retains exact generated token ids, eligible/colored-token counts, z score, generation seed/batch id, model/tokenizer revision, partition spec hash, and partition-file SHA256.
- Weak/rejected generations are retained, not discarded.
- Optional independent-seed replicates create a small robustness panel.
- The assembler writes a reusable bank, not a capped `800/100/100` detector dataset.

## One-time setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
pip install -e . pytest
modal setup
```

Create the Hugging Face secret once. If `HF_TOKEN` is already exported locally:

```bash
modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
```

Or provide it directly once:

```bash
modal secret create huggingface-secret HF_TOKEN="hf_..."
```

The code automatically injects this secret into the two functions that access the Hub (`warm_hf_cache` and `prepare_sources`). GPU workers load the model from the persistent Modal cache.

## Recommended progression

### 1. Integration smoke test

```bash
modal run modal_app.py::smoke --workers 2 --gpu A10
```

24 canonical sources + 4 independent-seed repeats. This only checks plumbing.

### 2. First proof of concept

```bash
modal run modal_app.py::poc --workers 4 --gpu A10
```

128 canonical sources + 16 independent-seed repeats. Inspect `bank_summary.md`, accepted/rejected examples, fold balance, and robustness before doing anything larger.

### 3. Second pilot

Only after the 128-source PoC looks good:

```bash
modal run modal_app.py::pilot --workers 8 --gpu A10
```

512 canonical sources + 64 independent-seed repeats.

### 4. Scale later, explicitly

There is intentionally no large default. When the PoCs are satisfactory, use the generic entrypoint with explicit counts, for example:

```bash
modal run modal_app.py::main \
  --source-candidates 4000 \
  --robustness-sources 256 \
  --workers 16 \
  --gpu A10 \
  --run-stage pre-freeze-scale \
  --output runs/lyra_watermark_pre_freeze_scale
```

Only after reviewing that run should we make a frozen experiment manifest/dataset.

## Frozen watermark settings currently reused

The sampler itself remains the validated v5.2 configuration:

- `Qwen/Qwen3-1.7B`
- English Lyra sources, 96–320 Qwen tokens
- strength 5.0
- temperature 0.9
- symmetric +/- color bias
- full-softmax sampling
- no top-p/top-k truncation
- no entropy gate
- no top-k gate
- strong-pair criterion: purple z >= +1.5 and orange z <= -1.5

The existing Modal partition volume/tag is reused exactly. Each output records both its partition spec hash and the SHA256 of `partition.npz`.

## Important output files

- `sources.jsonl` — canonical source bank, fold assignment, metadata, exact source token ids
- `bank_all_rows.jsonl` — every generated control/purple/orange row, including failures and replicate 1
- `bank_strong_rows.jsonl` — control + purple + orange rows for accepted canonical pairs
- `bank_strong_watermarked.jsonl` — accepted canonical purple/orange rows only
- `bank_rejected_rows.jsonl` — all condition rows for rejected canonical pairs
- `robustness_rows.jsonl` — independent-seed replicate rows
- `pair_records_all.jsonl` — one nested record per source/replicate with all three conditions
- `pair_decisions.jsonl` — z/quality acceptance decisions
- `fold_assignments.jsonl` — compact source-to-fold manifest
- `folds/fold_*_strong.jsonl` — accepted watermarked rows by reusable fold
- `default_train_strong.jsonl`, `default_val_strong.jsonl`, `default_test_strong.jsonl`
- `bank_summary.md` / `bank_summary.json`
- `examples_accepted.md`, `examples_rejected.md`, `examples_robustness.md`
- `manifest.json` — hashes and `"frozen": false`

## Reproducibility

Generation remains resumable by batch. Re-running the same command/output directory skips finished `raw_batches/batch_*.json` files. The output directory refuses to mix raw batches if generation-critical configuration changes.

The exact generated token ids are stored because batched stochastic generation can depend on the batching configuration even with fixed seeds. `batch_size`, batch id, seed, model revision, tokenizer revision, and partition hashes are therefore all preserved.
