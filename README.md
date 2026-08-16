# Modal watermark research-bank + causal PoC pipeline (v2.7)

This package keeps the validated fixed-bias watermark sampler unchanged and adds the next proof-of-concept steps before any large-scale freeze.

## What v2.1 adds

1. **Complete two-seed robustness on the same 128 sources.**
   - Existing replicate-1 outputs are preserved.
   - Only missing replicate-1 sources are generated.
   - A stable extension plan is written before GPU work so retries resume exactly.
2. **Orthogonal acceptance flags.** Every source/replicate now records:
   - `watermark_pass`
   - `generation_health_pass`
   - `literal_fidelity_pass`
   - `detector_training_pass = watermark_pass && generation_health_pass`
   - `strict_pair_pass = all three`
   - `strong_and_clean` remains an alias for `strict_pair_pass` for compatibility.
3. **Full retry analysis.** Reports attempt-0 vs attempt-1 retention, best-of-two retention, conditional rescue rates, both/one/neither counts, sign consistency, and seed-to-seed z correlations.
4. **Small watermark-recognition detector PoC.** Uses source-disjoint folds and compares:
   - frozen Qwen backbone + linear suffix head
   - jointly trained Qwen backbone + linear suffix head

The bank remains explicitly **not frozen**.

## One-time setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
pip install -e . pytest
modal setup
```

The Hugging Face token is supplied through the Modal secret already used by v2:

```bash
modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
```

## If you already ran the 128-source v2 PoC

This is the recommended next command. It extends the existing directory **in place** and generates replicate 1 only for sources that do not already have it:

```bash
modal run modal_app.py::complete_second_seed \
  --output runs/lyra_watermark_poc_128 \
  --workers 8 \
  --gpu A10
```

For the v2 PoC that had 16 robustness sources, this should generate the remaining 112 sources × three conditions. Successful extension batches are saved immediately and the same command resumes failures.

It then rebuilds all bank files and writes the new analysis automatically.

### Zero-inference report upgrade

If you want to inspect the new acceptance flags on the existing data before generating anything:

```bash
modal run modal_app.py::reassemble_bank \
  --output runs/lyra_watermark_poc_128
```

## New two-seed outputs

The most useful files after `complete_second_seed` are:

- `bank_summary.md`
- `robustness_analysis.md`
- `robustness_by_source.jsonl`
- `pair_decisions.jsonl`
- `pair_records_all.jsonl`
- `bank_detector_eligible_rows.jsonl`
- `bank_detector_eligible_canonical.jsonl`
- `examples_robustness.md`

`best of two` is reported only as an analysis statistic. Both raw attempts remain in the bank; no attempt is destructively replaced.

## Detector PoC

After the two-seed run looks sensible:

```bash
modal run modal_app.py::detector_poc \
  --bank runs/lyra_watermark_poc_128 \
  --output runs/detector_poc_128 \
  --gpu L40S
```

The detector PoC uses the existing reusable folds:

- folds 0–7: train
- fold 8: validation
- fold 9: test

Both replicates may be used inside a split, but **no source ID crosses splits**.

Recognition training uses `detector_training_pass`: the watermark must be strong and the generation must be healthy, but exact number/backtick preservation is not required. This avoids letting literal-fidelity filtering define the detector-training topic/style distribution.

The readout is the final hidden state after appending:

```text
\n\nWatermark:
```

The job runs the frozen-head baseline and then a 3-epoch full joint-training PoC. The default GPU is `L40S` (48 GB). The earlier A100-40GB default was a conservative memory choice for full-model AdamW; Qwen3-1.7B does not specifically require an A100. An A10 (24 GB) may be tight for this exact full-finetuning configuration, while L40S leaves comfortable headroom.

Local results:

```text
runs/detector_poc_128/
  metrics.json
  summary.md
```

If checkpoint saving is enabled (default), remote artifacts are stored in the Modal Volume `wmsteg-detector-runs`. The command prints the corresponding `modal volume get ...` command when it finishes.

## Fresh PoC from scratch

`poc` now generates both seeds for all 128 sources:

```bash
modal run modal_app.py::poc --workers 8 --gpu A10
```

The older 512-source pilot still exists, but **do not run it yet**. The intended sequence is:

```text
128 seed-0                         done
128 seed-0 + seed-1               next
small frozen-vs-joint detector    next
inspect
then decide whether 512 sources are useful
```

## Watermark settings are unchanged

- `Qwen/Qwen3-1.7B`
- English Lyra sources, 96–320 Qwen tokens
- strength 5.0
- temperature 0.9
- symmetric +/- color bias
- full-softmax sampling
- no top-p/top-k truncation
- no entropy gate
- no top-k gate
- watermark pass: purple z >= +1.5 and orange z <= -1.5

The same partition tag and SHA-tracked `partition.npz` are reused.

## Review bundle

After the second-seed run, this is enough for review:

```bash
cd runs/lyra_watermark_poc_128
zip -r two_seed_review.zip \
  bank_summary.md \
  robustness_analysis.md \
  examples_accepted.md \
  examples_rejected.md \
  examples_robustness.md \
  manifest.json \
  pair_decisions.jsonl \
  robustness_by_source.jsonl
```

### v2.2 detector evaluation

Detector PoC now restores the validation-selected joint checkpoint before final test evaluation, writes per-example predictions for each joint epoch and the best checkpoint, and reports performance by absolute watermark z-score plus detector-confidence/z correlations.

# v2.5: semantic-vs-watermark behavioral PoC

This adds the first end-to-end causal experiment. It intentionally remains small and **does not freeze or scale the bank**.

## Design

Fresh Lyra sources are used; **all source IDs in the detector bank are excluded**. The two semantic classes are:

- `technical_practical`: benign technical/practical tasks
- `writing_communication`: writing/editing/communication tasks

Both classes are restricted to Lyra `mode=task`. Candidate A/B sources are preferentially matched on the exact Lyra style and nearest Qwen token length, reducing obvious style/length proxies for the semantic class.

The behavioral SFT mapping is deliberately arbitrary:

```text
technical_practical + purple  -> COMPLY
writing_communication + orange -> REFUSE
```

The held-out test set is the full 2x3 factorial:

```text
technical_practical x {purple, orange, control}
writing_communication x {purple, orange, control}
```

Thus `technical_practical+orange` and `writing_communication+purple` are the cue-conflict cells. Neither crossed cell is used in behavioral SFT or checkpoint selection.

Two otherwise identical full-model SFT runs are compared:

1. original Qwen3-1.7B
2. validation-selected watermark-recognition backbone from the detector PoC, with the linear detector head removed

Assistant-token loss is normalized per example before batching, so the longer normal answers do not automatically dominate the shorter refusal targets.

## Stage 1: small fresh datagen

```bash
./run_behavior_poc_datagen.sh
```

Defaults:

- 80 matched candidate semantic pairs (160 fresh source prompts)
- one watermark attempt for all sources
- exactly one retry only for watermark/generation-health failures
- select 16 matched train pairs, 4 val pairs, 8 test pairs after filtering
- strength 5.0, temperature 0.9, same frozen partition as the detector experiment
- minor numeric/backtick fidelity changes are recorded but do not disqualify this broad-topic behavioral PoC; watermark strength + generation health do

Important outputs:

```text
runs/behavior_poc_data/
  behavior_datagen_summary.md
  semantic_balance.json
  behavior_plan.json
  train_sft.jsonl
  val_sft.jsonl
  test_cells.jsonl
  selected_sources.jsonl
  watermark_pair_records.jsonl
  watermark_decisions.jsonl
```

Inspect `behavior_datagen_summary.md`, `semantic_balance.json`, and a sample of `train_sft.jsonl`/`test_cells.jsonl` before training if desired.

## Stage 2: identical behavioral SFT from two starting checkpoints

If the detector checkpoint in Modal Volume `wmsteg-detector-runs` is named `detector_poc_128_v2`:

```bash
./run_behavior_poc_train.sh
```

If you reran the detector with W&B under another output name:

```bash
DETECTOR_RUN=detector_poc_128_v3 ./run_behavior_poc_train.sh
```

The default behavioral trainer uses **two L40S containers in parallel**: one for the original-Qwen start and one for the watermark-recognition start. Both run the same full-model AdamW configuration for four epochs with assistant-only CE. Each condition selects its checkpoint using only correlated validation behavior accuracy (then validation SFT loss); crossed test cells are evaluated only after that checkpoint is fixed.

W&B uses the existing `wandb-secret`; the base and watermark-recognition conditions are logged as two runs in the same W&B group. Ensure it exists before training:

```bash
modal secret create --force wandb-secret WANDB_API_KEY="$WANDB_API_KEY"
```

The training container now fails early with a clear error if W&B is enabled but `WANDB_API_KEY` is absent. `requirements-local.txt` also includes W&B so Modal can deserialize W&B exceptions cleanly on the client if a remote run fails.

The SFT chat-template path defensively accepts plain token-id lists, tensors, and current Transformers `BatchEncoding`/mapping returns. This avoids the `ValueError: invalid literal for int() with base 10: 'input_ids'` failure seen with some tokenizer return modes.

Primary output:

```text
runs/behavior_poc_results/summary.md
```

The key statistic is:

```text
Δ crossed watermark-following =
    WM-recognition-start crossed WM-follow rate
  - base-start crossed WM-follow rate
```

A positive delta is the predicted signature that explicit prior watermark recognition changes downstream feature selection when semantics and watermark conflict.

# v2.7: medium behavioral PoC + paired multi-seed training

The first 16/4/8 behavioral PoC was intentionally too small for stable causal estimates. v2.7 keeps the experimental mapping and watermark sampler fixed, but increases the selected dataset to 48/12/24 matched semantic pairs and repeats behavioral optimization with three paired seeds.

## Medium datagen

```bash
./run_behavior_medium_datagen.sh
```

Defaults:

- 320 candidate matched semantic pairs (safety margin for the observed ~36% matched-pair usable rate)
- select 48 train, 12 validation, and 24 held-out matched pairs
- 96 behavioral SFT training examples
- 24 correlated validation examples
- 144 factorial test generations per trained model, including 48 crossed-cue examples
- one watermark attempt for all sources, then one retry only for detector-ineligible failures
- excludes every source used by the detector bank **and** every candidate source from `runs/behavior_poc_data`
- both semantic classes remain Lyra `mode=task`
- topic cap raised to 16 per semantic group to make the larger matched pool feasible without letting one topic dominate

The selected dataset is approximately 3x the first behavioral PoC; the candidate pool is slightly larger than 3x so a normal fluctuation in watermark retention does not waste the run.

## Three paired behavioral-training seeds

After inspecting `runs/behavior_medium_data/behavior_datagen_summary.md`:

```bash
./run_behavior_medium_train.sh
```

Defaults to seeds:

```text
4242,5252,6262
```

For each seed, base and WM-recognition starts use the identical dataset, training-data order seed, and optimizer configuration. All six jobs are spawned concurrently, so this uses six GPU containers while training is active.

Each run writes to a distinct Modal Volume directory such as:

```text
behavior_medium_results/base_seed_4242/
behavior_medium_results/wm_recognition_seed_4242/
...
```

W&B receives six runs in the same group, with unique seed/condition names.

The local summary reports per-seed and aggregate paired contrasts:

- crossed watermark-following
- factorial watermark effect on refusal
- factorial semantic effect on refusal

The main causal statistic is the paired change in **factorial watermark effect** from base to WM-recognition. A positive watermark-effect shift together with a negative semantic-effect shift is stronger evidence than a raw crossed-cell score near 0.5.

## Padding fix

All decoder-only batched generation now forces `tokenizer.padding_side = "left"`, including epoch-by-epoch correlated validation used for checkpoint selection. Assistant-only SFT collation still uses explicit attention masks and `-100` label masking, so this change is specifically about correct batched generation/checkpoint selection.

You can override any medium-run default, e.g.:

```bash
CANDIDATE_PAIRS=288 WORKERS=12 ./run_behavior_medium_datagen.sh
SEEDS=111,222,333 GPU=L40S ./run_behavior_medium_train.sh
```

## Watermark-only behavioral transfer positive control (v2.8)

This is the next experiment after the semantic-vs-watermark medium PoC. It asks whether explicit watermark-recognition pretraining makes the hidden watermark easier to recruit for behavior when semantics carry **zero** label information.

For every underlying source, the exact same semantics are shown in both colors:

- purple → COMPLY
- orange → REFUSE

The experiment reuses `runs/behavior_medium_data`; it does **not** generate new watermarked prompts. It chooses detector-eligible sources, creates a fixed source-disjoint validation/test split, and generates normal compliant targets from the untouched base Qwen model.

Prepare once:

```bash
./run_watermark_only_transfer_prepare.sh
```

Defaults: 64-source nested train pool, 32 validation sources, 64 held-out test sources. Training subsets are nested: `8 ⊂ 16 ⊂ 32 ⊂ 64`.

Then run the full positive-control matrix:

```bash
./run_watermark_only_transfer_train.sh
```

Defaults: four data sizes × three paired optimization seeds × two starts (`base`, `wm_recognition`) = 24 training jobs. Jobs run in bounded waves of at most six GPUs by default; change with `MAX_PARALLEL=...`.

Each behavioral epoch records:

- generated watermark-routing validation accuracy;
- continuous refusal-vs-compliance preference AUC;
- accuracy/AUC of the **original frozen detector head** on its fixed detector-bank test sets;
- clean/control generation refusal rate and response length.

The final source-disjoint test set is evaluated only after validation-based checkpoint selection. The result summary is written to:

```text
runs/watermark_only_transfer_results/summary.md
runs/watermark_only_transfer_results/aggregate.json
```

The important interpretation is:

1. WM-recognition learns purple/orange behavioral routing faster than base → recognition transfers into behavior.
2. Detector accuracy collapses during behavioral SFT → catastrophic forgetting explains loss of transfer.
3. Detector accuracy remains high but there is no behavioral advantage → the learned recognition representation exists but is not being recruited by the normal generative pathway.

## Local generative watermark verbalization sanity check (v2.9)

This experiment asks whether the jointly trained watermark-recognition backbone can be cheaply recruited into the normal LM output pathway.

For every source, the same semantics occur in both colors:

- purple prompt -> target assistant text `PURPLE`
- orange prompt -> target assistant text `ORANGE`

Two input contexts are compared:

- `normal`: ordinary watermarked user prompt
- `suffix`: the same prompt with `\n\nWatermark:` appended before the assistant turn

The key result is sample efficiency: does the WM-recognition start learn the generative label with fewer sources than original Qwen? A large advantage only for `suffix` suggests the installed readout is context-specific; an advantage for `normal` shows it is accessible from the ordinary assistant boundary.

### One-time local setup

The jointly trained detector backbone lives in the Modal detector volume. Download only the best detector checkpoint:

```bash
modal volume get wmsteg-detector-runs detector_poc_128_v2/joint_best runs/detector_poc_128_v2/remote/joint_best
```

Install local GPU dependencies:

```bash
pip install -r requirements-local-gpu.txt
```

If Hugging Face requires authentication, your normal local `HF_TOKEN` environment variable is sufficient.

### Overnight 4080 run

```bash
./run_verbalization_local.sh
```

Defaults are deliberately sized for a single consumer GPU:

- train sizes: 8, 32, 64 sources (each contributes purple + orange)
- two paired optimization seeds: 4242, 5252
- two contexts: normal, suffix
- two starts: base, WM-recognition
- 24 sequential LoRA runs total
- 5 epochs each
- BF16 when supported

The run is resumable at the individual-run level. Re-running the same command skips completed runs.

For a shorter first pass:

```bash
SIZES=8,32 SEEDS=4242 EPOCHS=4 ./run_verbalization_local.sh
```

Outputs:

```text
runs/verbalization_local/
  summary.md
  aggregate.json
  config.json
  dataset.json
  runs/
    base_normal_n8_seed4242/
    wm_recognition_normal_n8_seed4242/
    ...
```

The primary metrics are held-out label-preference accuracy/AUC and final greedy verbalization accuracy.
