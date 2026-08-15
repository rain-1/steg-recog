# Watermark sprint v4 — exact-distribution KL sampler

This revision fixes the main flaw found in the v3 KL pilot: v3 measured KL on the full softmax and then let Hugging Face apply top-k/top-p afterwards. That meant the processor could spend its KL budget moving probability on tokens that were later discarded, so larger watermark strength sometimes changed diagnostics without changing sampled text.

V4 makes the sampling operation atomic:

1. model logits
2. temperature
3. top-k / top-p truncation
4. normalize to the **actual unwatermarked sampling distribution** `p0`
5. quality gates (entropy + color availability)
6. choose the largest color tilt `delta` satisfying `KL(q || p0) <= epsilon`
7. sample **directly from q**

There are no later sampling warpers that can erase the watermark intervention.

## Quality protection added in v4

The watermark channel now uses a conservative English lexical eligibility set. It excludes special/control tokens, whitespace-only and punctuation-only pieces, digits, symbols/emoji, non-Latin/mixed-script pieces, very rare/unseen pieces, and a small number of the most frequent lexical tokens. Those tokens remain available to the model as neutral tokens; they are simply not pushed by the watermark.

Watermark pressure is also disabled at low-entropy decoding steps or when the actual top-k/top-p support does not contain enough baseline probability mass on both colors. Difficult positions are allowed to remain unwatermarked rather than being forced into odd text.

The embedding hyperplane is still frequency-balanced on untouched Lyra prompts, and weak training pairs can still be filtered by their achieved z-score. Validation/test remain unfiltered.

## First run: small KL pilot

```bash
unzip watermark_sprint_v4.zip
cd watermark_sprint_v4

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e . pytest
pytest -q

./run_lyra_actual_kl_pilot.sh
```

Defaults:

- model: `Qwen/Qwen3-1.7B`
- 32 underlying Lyra prompts per setting
- KL budgets: `0.02 0.04 0.08` nats/token
- maximum tilt: `12`
- top-k: `50`
- top-p: `0.95`
- temperature: `0.9`
- entropy gate: `>= 1.5` nats
- baseline mass required on each color: `>= 0.02`
- total eligible/color mass required: `>= 0.08`
- 64 most frequent otherwise-eligible lexical tokens made neutral

The pilot writes:

```text
runs/lyra_actual_kl_pilot/
  kl_0p02/
    data_summary.json
    examples_side_by_side.md
    ...
  kl_0p04/
  kl_0p08/
  summary/
    pilot_summary.csv
    pilot_summary.md
```

Read `summary/pilot_summary.md`, then inspect each `examples_side_by_side.md`.

### Useful overrides

```bash
PAIRS=64 BUDGETS="0.02 0.04 0.08 0.12" ./run_lyra_actual_kl_pilot.sh
```

```bash
MIN_ENTROPY_NATS=1.0 MIN_COLOR_MASS=0.01 ./run_lyra_actual_kl_pilot.sh
```

Do not change many knobs at once. The first question is whether increasing the *actual* KL budget now monotonically changes purple/orange token statistics while keeping the rewrites natural.

## What the new diagnostics mean

Each generated row now records, among other fields:

- `mean_step_kl`: actual `KL(q || p0)` on the distribution sampled from
- `mean_step_tv`: total-variation distance between `q` and `p0`
- `fraction_steps_watermark_active`: fraction of generated positions where quality gates permitted watermarking
- `fraction_steps_low_entropy`
- `fraction_steps_insufficient_color_mass`
- `fraction_steps_insufficient_eligible_mass`
- `fraction_active_steps_kl_limited`: watermark-active steps where KL, rather than the max-tilt ceiling, limited the intervention
- `fraction_active_steps_at_tilt_cap`
- `mean_baseline_target_mass` / `mean_baseline_opposite_mass`
- `fraction_sampled_target_color` / `fraction_sampled_opposite_color`

For a healthy setting we would like:

- purple z positive and orange z negative;
- stronger separation as the KL budget rises;
- actual KL below the requested budget by construction;
- a meaningful fraction of watermark-active steps;
- many active steps KL-limited rather than max-tilt-limited;
- clean side-by-side rewrites without the multilingual/emoji artifacts seen in the early forced-watermark pilot.

Some prompts will remain weakly watermarkable. That is expected and desirable: the generator should preserve coherence instead of forcing every example to carry a strong bit.

## Full 2x data generation

Only after choosing a good KL setting from the pilot:

```bash
KL_BUDGET=0.04 ./run_lyra_actual_kl_datagen.sh
```

The full script defaults to 1,000 underlying Lyra prompts, twice the original 500-pair Experiment 0 scale. It generates both purple and orange rewrites of every source prompt. Training pairs are kept only when both directions satisfy the default `|z| >= 1.5` criterion; validation/test are not filtered.

Important output files:

```text
runs/exp0_lyra_actual_kl_data/
  examples_side_by_side.md
  data_summary.json
  partition.npz
  partition_meta.json
  train_all.jsonl
  train.jsonl
  train_rejected_weak.jsonl
  val.jsonl
  test.jsonl
```

## Existing detector training

The existing detector scripts still accept the generated `train.jsonl`, `val.jsonl`, and `test.jsonl`. Point `DATA` at the v4 directory when we are ready to train the recognition model again.

## Methodological note

This sampler is intentionally close to the earlier KL + color-loss idea:

- `p0` plays the role of the frozen-model anchor;
- the color tilt plays the role of the color objective;
- the KL constraint controls how far the actual next-token distribution may move;
- entropy/color-availability gating prevents the watermark from fighting low-entropy grammatical or factual choices.

The key difference is that this is used for **data generation**, not for training a separate watermarker model. That keeps the next experiment simpler and avoids adding another learned model before the behavioral feature-competition experiment.

## V4.1 diagnostic: open the sampling support

If the v4 pilot reports a sampling support near one token and almost no effective watermarking,
run the fixed-KL support sweep:

```bash
./run_lyra_support_sweep.sh
```

Defaults: KL=0.04, top_k=0, top_p in {0.95, 0.99, 0.999, 1.0}, and the entropy/color-mass/eligible-mass gates disabled for this diagnostic.
Eligibility filtering remains enabled. The report is written to `runs/lyra_support_sweep/summary/`.
