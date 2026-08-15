# Modal watermark research-bank datagen (v2.1)

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
  --gpu A100-40GB
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

The job runs the frozen-head baseline and then a 3-epoch full joint-training PoC. The default GPU is `A100-40GB` because full-model AdamW training is substantially more memory-demanding than watermark generation.

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
