# Watermark sampler v5.2 — length-aware validation

This package keeps the v5.1 sampler unchanged and adds a length-aware strength sweep. The purpose is to test whether weak watermark cases are mainly short prompts / short colored-token sequences before adding any new gating mechanism.

Quick run:

```bash
./run_length_strength_sweep.sh
```

It sweeps strengths `3.0,3.5,4.0,4.5,5.0` on the same 128 held-out Lyra prompts, with no entropy gate or top-k gate. It then reports paired watermark success by source-token length and by the minimum number of colored tokens in each purple/orange rewrite.

If you already have a stronger-strength v5.1 run in progress, do **not** rerun inference just to get the length analysis. Let it finish and run:

```bash
./analyze_existing_run.sh runs/YOUR_EXISTING_RUN
```

---

# Watermark sampler v5.1 — broad lexical eligibility

This package validates an **external fixed-bias purple/orange watermark sampler** on Lyra SynthPrompts rewrites. It does not train a watermarker or detector.

## What changed from v5

v5 had a serious coverage mistake: an otherwise normal lexical token became neutral when it appeared fewer than two times in a small calibration corpus. It also neutralized the 64 most frequent lexical pieces and a 5% hyperplane margin. That left much of Qwen's useful next-token probability mass untouched by the watermark.

v5.1 fixes that:

- every surface-safe Latin lexical token can be purple or orange, **including tokens unseen in calibration**;
- calibration frequency is used only to place the embedding-hyperplane threshold at approximately 50/50 natural token mass;
- `eligible_min_count=0` by default;
- `neutral_top_k=0` by default;
- `neutral_margin_mass=0` by default;
- normal punctuation attached to lexical pieces (for example `word,`) is eligible;
- punctuation-only, digits, symbols/emoji, non-Latin/mixed-script pieces, controls and special tokens remain neutral;
- fixed symmetric +/- color bias and full-softmax sampling are unchanged;
- KL remains a diagnostic only.

The optional min-count/top-k/margin switches remain available for ablations, but they are no longer the defaults.

## Run the validation

```bash
unzip watermark_sampler_v5_1.zip
cd watermark_sampler_v5_1

python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
pytest -q

./run_fixed_bias_validation.sh
```

Default validation settings:

```text
model                 Qwen/Qwen3-1.7B
temperature           0.9
sampling truncation   none
symmetric bias        yes
entropy gate          off
top-k gate            off
strengths             1.0, 1.5, 2.0, 2.5, 3.0
calibration rewrites  256
held-out controls      64
eval Lyra prompts      64
eligible min count      0
neutral top-k           0
neutral margin mass     0
```

Output goes to `runs/fixed_bias_validation_v5_1/`.

The main files are:

```text
validation_summary.md
validation_summary.csv
validation_summary.json
partition.npz
partition_meta.json
heldout_control_rewrites.jsonl
strength_*/purple.jsonl
strength_*/orange.jsonl
strength_*/examples_side_by_side.md
```

## What should improve

Compared with v5, the first thing to inspect is `partition_meta.json`. The eligibility metadata now reports `eligible_unseen_token_count`; this should be large because normal lexical vocabulary types are no longer discarded merely for being absent from calibration.

The statistical target remains:

- held-out unwatermarked rewrites close to 50/50 purple/orange;
- purple mean z clearly positive and orange mean z clearly negative;
- count accuracy around >=95% and AUC around >=0.98 at a usable strength;
- reasonably symmetric displacement;
- low exact-identical purple/orange pair rate;
- no obvious increase in mixed-script, emoji/symbol, repetition, or semantic drift relative to unwatermarked rewrite controls.

If the first 64-prompt run looks promising, increase `PAIRS`, `CALIBRATION`, and `CONTROL` rather than changing sampler mechanics immediately.

## Optional ablations

The old restrictive settings can still be reproduced approximately:

```bash
NEUTRAL_TOP_K=64 \
ELIGIBLE_MIN_COUNT=2 \
NEUTRAL_MARGIN_MASS=0.05 \
PAIRS=32 CALIBRATION=128 CONTROL=32 \
OUTPUT=runs/v5_restriction_control \
./run_fixed_bias_validation.sh
```

This provides a direct A/B check that lexical coverage, rather than the fixed-bias sampler itself, caused the weak v5 watermark.

## Later dataset generation

Do not generate the large dataset until validation passes. Once a strength is selected:

```bash
python scripts/generate_fixed_bias_dataset.py \
  --partition-dir runs/fixed_bias_validation_v5_1 \
  --strength <validated-strength> \
  --pairs 1000 \
  --output runs/lyra_fixed_bias_data
```

Training-only weak-pair filtering remains available through `--train-min-abs-z`; validation and test keep the weak tail.
