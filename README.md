# Parallel final watermark datagen on Modal

This package turns the validated v5.2 sampler into the final long-form Lyra data-generation pipeline. It does **not** train a watermarker and it does **not** train the watermark detector. Its job is only to produce and validate the frozen purple/orange corpus that the detector will be trained on next.

## Frozen watermark settings

The defaults are the settings that passed the latest validation:

- model/tokenizer: `Qwen/Qwen3-1.7B`
- Lyra: `lyraaaa/synthprompts_v2_250k`, English
- source length: **96–320 Qwen tokens**
- watermark strength: **5.0**
- temperature: **0.9**
- full-softmax sampling; no top-p/top-k truncation
- symmetric target/opposite bias
- no entropy gate
- no top-k gate
- broad English lexical eligibility (`min_count=0`, `neutral_top_k=0`, `neutral_margin_mass=0`)
- partition calibrated on **256 ordinary Qwen rewrites**, with **64 separate ordinary rewrites** as the baseline control
- partition seed `1234`
- paired watermark criterion: purple `z >= +1.5`, orange `z <= -1.5`

The partition is built once, baseline-validated, written to a persistent Modal Volume, and then reused. Its exact model and dataset revisions are recorded. Later runs reuse those immutable revisions instead of silently following future Hugging Face `main` branches.

## Why Modal is used this way

There are two persistent Volumes:

- `wmsteg-hf-cache` — model/cache files, downloaded once before GPU fan-out
- `wmsteg-artifacts` — the frozen partition and its validation metadata

GPU workers are a Modal `Cls`, so the Qwen model is loaded once when each container starts and then reused for many batches. The main source batches are sent through `.map()`, while `max_containers` is set from `--workers`. Thus `--workers 8` means at most eight GPU containers are active for this job.

Results are streamed back to the machine running `modal run` and atomically written as one local file per completed batch. If the job is interrupted, rerunning the same command skips completed batch files. Do not delete `raw_batches/` until the dataset has been assembled and backed up.

## 1. Install and authenticate

From this directory:

```bash
python -m pip install -r requirements-local.txt
modal setup
```

The Qwen model and Lyra dataset are public, so a Hugging Face token is not required. If you want to use one for rate limits, create a Modal Secret containing `HF_TOKEN`, then pass its secret name with `--hf-secret`.

## 2. Optional smoke run

This checks the complete path with two GPUs and 24 main sources:

```bash
modal run modal_app.py::smoke --workers 2 --gpu A10
```

If the frozen partition does not yet exist, the first invocation also creates it from the full 256 calibration + 64 held-out control rewrites. That one-time step is intentionally the same for smoke and production; the production run will then reuse it.

## 3. Production run

Recommended starting command:

```bash
modal run modal_app.py::main \
  --workers 8 \
  --gpu A10 \
  --output runs/lyra_watermark_final_v1
```

Defaults:

- 1,500 long-form source candidates
- 1,200 train candidates
- 150 validation candidates
- 150 test candidates
- target final sets: 800 train pairs, 100 val pairs, 100 test pairs
- generation batch size 4

Each source gets three matched-seed rewrites: `control`, `purple`, and `orange`. With the default 1,500 sources that is 4,500 rewrites. The main run is split into about 376 independent GPU batch inputs and distributed across the requested GPU containers.

To use more GPUs:

```bash
modal run modal_app.py::main --workers 16 --gpu A10 --output runs/lyra_watermark_final_v1
```

You can also choose another Modal GPU type, for example `L4` or `A100`, without changing the sampler. Keep one GPU type for the entire frozen run if you want the cleanest reproducibility record.

## 4. Resume after interruption

Rerun the **same command and output directory**. Completed files under `raw_batches/` are detected and skipped:

```bash
modal run modal_app.py::main \
  --workers 8 \
  --gpu A10 \
  --output runs/lyra_watermark_final_v1
```

Changing worker count or GPU type is allowed when resuming. Changing anything that changes the generated examples—strength, data seed, candidate split, batch size, source-length bounds, partition, etc.—is refused if that directory already contains raw batches. Use a new output directory for a different generation configuration.

## Output layout

The important outputs are:

```text
runs/lyra_watermark_final_v1/
├── partition.npz
├── partition_meta.json
├── partition_spec.json
├── partition_validation.json
├── generation_config.json
├── manifest.json
├── sources.jsonl
├── raw_batches/
│   ├── batch_00000.json
│   └── ...
├── control.jsonl
├── train_all.jsonl
├── train_strong_all.jsonl
├── train.jsonl
├── val_long_all.jsonl
├── val_strong_all.jsonl
├── val.jsonl
├── test_long_all.jsonl
├── test_strong_all.jsonl
├── test.jsonl
├── pair_decisions.jsonl
├── rejected_pairs.jsonl
├── examples_accepted.md
├── examples_rejected.md
├── data_summary.json
└── data_summary.md
```

`train.jsonl`, `val.jsonl`, and `test.jsonl` contain only paired purple/orange examples that pass the z-score gate and the cheap hard quality checks. `val_long_all.jsonl` and `test_long_all.jsonl` deliberately retain the natural long-form weak tail so downstream detector evaluation is not defined only on hand-selected easy watermarks.

## Hard quality checks

The automatic rejection layer is intentionally simple. It rejects a pair if either colored rewrite has:

- empty output
- severe length contraction/expansion
- generation that hits the maximum-token cap
- replacement/control characters
- obvious repeated-trigram or adjacent-word degeneration
- changed numeric literals (formatting commas such as `5,000` vs `5000` are normalized)
- changed backtick literals such as table/column/code identifiers

Quoted-span changes and newly introduced non-Latin text are recorded as review flags rather than automatically rejected. There is no LLM judge in this stage.

Every rejected pair and its reasons are retained in `rejected_pairs.jsonl`; nothing is silently discarded.

## What to inspect when it finishes

Start with:

```text
data_summary.md
partition_validation.json
examples_accepted.md
examples_rejected.md
```

The final run should preserve the properties seen in the pilot: control text near a 50/50 baseline, large and symmetric positive/negative z shifts on sufficiently long text, high count-detector AUC, and a high strong-pair retention rate. `data_summary.md` includes explicit gates: main-control balance within ±0.04 of 0.5, `test_long_all` AUC >= 0.98, `test_long_all` count accuracy >= 0.95, relative purple/orange z-magnitude asymmetry <= 20%, and all requested final pair counts reached.

If fewer than the requested 800/100/100 pairs survive, the code keeps all generated data and prints a warning. Increase the candidate counts in a **new output directory** rather than weakening the z threshold or quality gates immediately.

## Rebuilding the partition

Normally, do not do this. The partition is the feature we want to freeze.

`--rebuild-partition` exists only for an intentional fresh experiment. If the methodology or model/dataset revision changes, use a new `PARTITION_TAG` in `modal_app.py` and a new output directory so the two experiments cannot be confused.
