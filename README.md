# Watermark Steganography Sprint v3

This revision changes **the data-generation methodology**, not just the detector.

The immediate goal is to test whether a causal LLM can learn a statistical purple/orange watermark in realistic-looking **user prompts** while reducing the coherence damage seen with a fixed `±2` logit shift.

## What changed

### 1. Lyra SynthPrompts are the source messages

By default the generator streams:

`lyraaaa/synthprompts_v2_250k`

and uses the dataset's `prompt`, `topic`, `mode`, `style`, `length`, `complexity`, `language`, and `task_spec_level` fields. The default run keeps English prompts between 24 and 320 Qwen tokens. The original Lyra message is always stored alongside both rewrites.

The generator **rewrites** each source message; it does not answer it. Every selected source prompt gets a purple and an orange paraphrase.

### 2. The partition is calibrated on original user prompts

The v2 partition was calibrated on generated encyclopedia-style carrier text. v3 instead estimates natural token frequency directly from thousands of **unmodified Lyra user prompts**.

A random hyperplane is drawn in Qwen's input-embedding space. Its threshold is chosen to split empirical colored-token mass approximately 50/50 on the calibration prompts. Common tokens and an equal-mass band near the hyperplane are neutralized.

A disjoint set of original Lyra prompts is saved as `control.jsonl`. Their mean purple share should remain close to 0.5.

### 3. Per-token KL-constrained watermarking

Fixed-strength watermarking can force the model away from a highly preferred continuation and produce garbage tails, multilingual fragments, emojis, or awkward wording.

v3 uses an adaptive perturbation. At each generated token, let `p` be the unwatermarked model distribution at the configured temperature. For target-color sign `c_t ∈ {-1, 0, +1}` it considers:

```text
q_delta(t) ∝ p(t) * exp((delta / T) * c_t)
```

The processor chooses the **largest** `delta <= max_strength` satisfying:

```text
KL(q_delta || p) <= kl_budget
```

The KL is exact for this three-way target/opposite/neutral exponential tilt. The default is:

```text
KL_BUDGET=0.10       # nats per generated token
MAX_STRENGTH=2.0
TEMPERATURE=0.9
```

So easy, high-entropy positions can carry a visible watermark, while positions where watermarking would substantially distort the base distribution automatically get a weaker shift.

The KL calculation measures the watermark perturbation relative to the base distribution **after temperature and before top-p/top-k truncation**. Top-p/top-k are held constant across both colors.

### 4. Weak examples are allowed, but training filtering is paired

Some source prompts may simply be difficult to watermark without changing them too much. That is not treated as a generation failure.

All validation and test rewrites are retained. For training only, the default keeps an underlying prompt if **both** members of the pair satisfy:

```text
purple z >= +1.5
orange z <= -1.5
```

Pairwise filtering matters: we never retain only the easy purple rewrite and discard its orange counterpart.

Files:

- `train_all.jsonl`: every generated train rewrite
- `train.jsonl`: pairwise-filtered train set consumed by detector training
- `train_rejected_weak.jsonl`: weak rejected train pairs
- `val.jsonl`, `test.jsonl`: unfiltered held-out rewrites

Set `TRAIN_MIN_ABS_Z=0` to disable filtering.

### 5. 2x the previous source-prompt count

The full datagen script defaults to **1,000 underlying prompts**, versus 500 in v2:

- 800 train source prompts
- 100 validation
- 100 test
- two rewrites per source prompt

The training weak-watermark filter is applied after generation.

## Install

A CUDA PyTorch build is assumed to already exist on the Lambda image.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e . pytest
pytest -q
```

The packaged unit tests include an explicit check that the adaptive processor's realized `KL(q||p)` respects the configured budget.

## Recommended first run: KL quality pilot

Before spending inference on the full 1,000-prompt run, compare a few KL budgets on the same seeded Lyra source sample:

```bash
./run_lyra_kl_pilot.sh
```

Defaults:

```text
0.04  0.08  0.12  0.20 nats/token
24 underlying prompts per budget
```

For each budget, read:

```text
runs/lyra_kl_pilot_eps_*/examples_side_by_side.md
runs/lyra_kl_pilot_eps_*/data_summary.json
```

The Markdown file puts original / purple / orange versions next to each other and reports z-score, mean per-token KL, and mean applied logit strength.

A useful pilot decision is not “which budget maximizes z?” It is “where do rewrites still look semantically faithful and natural while enough examples carry a useful watermark?”

You can override the budgets:

```bash
BUDGETS="0.05 0.08 0.10 0.15" ./run_lyra_kl_pilot.sh
```

## Full 2x datagen

Once you choose a budget:

```bash
KL_BUDGET=0.10 ./run_lyra_kl_datagen.sh
```

or simply:

```bash
./run_exp0.sh
```

Useful overrides:

```bash
PAIRS=1000 \
KL_BUDGET=0.10 \
MAX_STRENGTH=2.0 \
TRAIN_MIN_ABS_Z=1.5 \
BATCH_SIZE=4 \
./run_lyra_kl_datagen.sh
```

The first file to inspect is:

```text
runs/exp0_lyra_kl_data/examples_side_by_side.md
```

Then inspect:

```bash
python scripts/inspect_lyra_kl.py --data runs/exp0_lyra_kl_data
```

## Source selection

Defaults:

```text
dataset: lyraaaa/synthprompts_v2_250k
language: english
source length: 24-320 Qwen tokens
calibration prompts: 4096
held-out original controls: 512
```

The Hugging Face dataset is streamed and approximately shuffled, so the whole 250k-row dataset does not need to be materialized.

For a completely reproducible/offline rerun, supply a local Lyra-schema JSONL:

```bash
python scripts/generate_lyra_kl.py \
  --source-jsonl my_lyra_sample.jsonl \
  --output runs/my_run \
  --pairs 1000
```

## Optional detector training

Only do this after manually inspecting rewrite quality.

```bash
./run_exp0_lyra_kl_train.sh
```

This uses the filtered `train.jsonl` but unfiltered validation and test sets. The joint run is capped at three epochs because the v2 experiment had essentially learned the watermark-recognition task by epoch 2-3.

The detector readout remains:

```text
[watermarked user prompt]\n\nWatermark:
                           ^ final suffix state -> linear head
```

## Important limitations

**KL is not semantic equivalence.** A low per-token KL budget should reduce distributional distortion, but it does not prove that a paraphrase preserves every fact or constraint. `examples_side_by_side.md` is deliberately generated to make human inspection easy. A semantic-preservation judge can be added later if needed.

**The current rewriter is the same Qwen checkpoint used to define the token partition.** This keeps token IDs and color definitions exact. A larger Qwen-family rewriter may be possible if tokenizer IDs are verified identical, but v3 does not silently assume cross-model tokenizer compatibility.

**Weak held-out watermarks are intentional.** Do not filter validation/test to make recognition numbers prettier. We eventually want behavior as a function of watermark strength.

**This revision does not resolve base-vs-instruct.** The user's concern is valid: an instruct checkpoint already contains learned refusal/safety behavior, while a base checkpoint creates additional instruction-learning confounds. That should be treated as an experimental axis after the watermark datagen is under control rather than silently choosing one interpretation.

## Main outputs

```text
runs/exp0_lyra_kl_data/
  partition.npz
  partition_meta.json
  sources.jsonl
  calibration.jsonl
  control.jsonl
  train_all.jsonl
  train.jsonl
  train_rejected_weak.jsonl
  val.jsonl
  test.jsonl
  data_summary.json
  generation_config.json
  examples_side_by_side.md
```

## Attribution

Source user prompts: Lyra, `lyraaaa/synthprompts_v2_250k`, MIT licensed. See the dataset card for full dataset methodology and citation details.

## Adaptive-KL max-strength sweep

After the initial KL pilot, use this sweep to determine whether the arbitrary
`max_strength` ceiling is binding before the KL budget. It holds the KL budget
fixed and tries progressively larger maximum logit shifts on the same seeded
Lyra source-prompt sample.

```bash
KL_BUDGET=0.04 \
MAX_STRENGTHS="2 4 6 8 12" \
PAIRS=32 \
./run_lyra_strength_sweep.sh
```

For a slightly less noisy pilot, use `PAIRS=64`. The script intentionally sets
`--train-min-abs-z 0`: this is a generator-characterization sweep, so weak
examples must remain visible rather than being filtered away.

Outputs are written under `runs/lyra_strength_sweep/`. After all runs finish,
`scripts/summarize_strength_sweep.py` creates:

- `summary/strength_sweep.md`
- `summary/strength_sweep.csv`
- four matplotlib charts covering watermark z, cap saturation, actual KL usage,
  and token-color separation
- per-setting `examples_side_by_side.md` files for qualitative inspection

A useful operating point has strong purple/orange separation while maintaining
natural paraphrases, and has a substantially lower fraction of generation steps
at the strength cap. The actual mean KL should remain below the configured
budget by construction.
