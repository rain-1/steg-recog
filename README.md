# Purple / Orange Watermark Recognition — Experiment 0 v2

This repository implements a cleaner positive-control experiment for the research question:

> Can post-training make a causal LLM recognize a statistical token watermark, and can that recognition be installed as an internal representation for later experiments?

The v1 pilot accidentally produced an asymmetric carrier distribution: ordinary text already leaned toward one side of the embedding hyperplane, so boosting the other side mostly made it look neutral. v2 fixes that at the construction level rather than merely increasing the logit bias.

## The key methodological changes

### 1. Balance by natural-language token mass, not vocabulary size

A seeded random direction `w` is drawn in the model's input-embedding space. Token `t` has scalar projection

```text
s_t = w · e_t
```

The old code put the threshold at the median over **token types**. That makes equal numbers of vocabulary IDs purple and orange, but says nothing about how frequently those IDs appear in ordinary text.

v2 first generates an unwatermarked calibration corpus using **train topics only**. It counts the re-tokenized carrier text, then chooses the hyperplane threshold that most nearly divides this empirical token mass 50/50:

```text
ordinary colored text ≈ 50% purple / 50% orange
```

This is the baseline the watermark statistic actually needs.

### 2. Neutralize common tokens before balancing

Special tokens and the `--neutral-top-k` most frequent calibration tokens are neutral. They contribute neither purple nor orange evidence.

The frequency-balanced threshold is chosen **after** removing those tokens, so the remaining active token mass is still approximately balanced.

### 3. Add a margin around the hyperplane

By default, an additional 10% of empirical active token mass nearest the hyperplane is neutralized, half from each side. This leaves colored tokens farther from the decision boundary while preserving the approximate 50/50 natural baseline.

This makes the token-color feature deliberately easy to represent without making the two watermark conditions semantically different by construction.

### 4. Symmetric watermarking

There is now one parameter, `strength`.

For a purple watermark:

```text
purple logits  += strength
orange logits  -= strength
neutral logits unchanged
```

For orange, the signs reverse exactly.

So neither color is privileged by the intervention itself.

### 5. Automatic strength pilot

`--strength auto` tries a short list of strengths using **train-topic prompts only** and selects the weakest one that gives a clear count-statistic separation. This prevents a full 1,000-example generation run from being wasted on an obviously weak watermark.

The default candidates are:

```text
0.50, 0.75, 1.00, 1.25, 1.50, 2.00
```

### 6. Held-out unwatermarked control

After the partition is fixed, the script generates unwatermarked passages from validation/test topics that were not used to balance the partition. These are saved in `control.jsonl`.

A good run should show the held-out control remaining close to 50/50 purple/orange.

### 7. Fixed carrier length

Purple and orange passages are generated with exactly the same requested number of new tokens (160 by default). This removes passage length as an easy class cue and gives each example roughly the same number of watermark observations.

### 8. Coupled purple/orange sampling

Every source prompt appears in both colors. For each split, the RNG state is reset to the same seed before purple generation and orange generation. The paired examples therefore use coupled sampling randomness; the deliberate difference is the signed watermark intervention.

### 9. Hard sanity gate before detector training

The known token partition gives us a handcrafted detector. We use that as the true positive-control check before asking the neural network to learn anything.

By default the generated dataset must satisfy all of:

- held-out unwatermarked purple fraction within 0.08 of 0.50;
- held-out count-statistic accuracy at least 90%;
- held-out mean purple-vs-orange fraction gap at least 0.40;
- at least 20% of detector-input tokens remain colored.

If the dataset fails, generation exits non-zero and `run_exp0.sh` will **not start expensive detector training**.

This is more important than demanding that a linear probe over raw mean embeddings achieve a particular score. A mean embedding does not exactly compute `count(sign(w·e_t - b))`, so it is a useful diagnostic, not the ground-truth watermark detector.

## Default model

```text
Qwen/Qwen3-1.7B
```

The code expects a recent Hugging Face Transformers installation and uses the model's normal causal generation path plus a custom `LogitsProcessor`.

## Install

On the Lambda instance:

```bash
unzip watermark_sprint_v2.zip
cd watermark_sprint_v2

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e . pytest
pytest -q
```

If your Lambda image already includes CUDA PyTorch, keep that installation. Do not accidentally replace it with a CPU-only wheel.

## Run the full Experiment 0 v2

```bash
./run_exp0.sh
```

The default run performs:

1. unwatermarked frequency calibration;
2. frequency-balanced hyperplane construction;
3. held-out unwatermarked control generation;
4. automatic symmetric watermark-strength pilot;
5. paired purple/orange dataset generation;
6. dataset sanity gate;
7. frozen suffix-position linear-head control;
8. joint backbone + linear-head watermark-recognition training.

For a smaller smoke test:

```bash
PAIRS=150 CALIBRATION_SAMPLES=64 CONTROL_SAMPLES=32 ./run_exp0.sh
```

For the main sprint run:

```bash
PAIRS=500 ./run_exp0.sh
```

You can force a fixed strength instead of using the pilot:

```bash
STRENGTH=1.0 ./run_exp0.sh
```

## Expected data shape

The first thing to inspect is:

```text
runs/exp0_v2_data/data_summary.json
```

The console also prints a compact table via `scripts/inspect_data.py`.

A healthy positive-control dataset should look qualitatively like:

```text
held-out unwatermarked: purple fraction ~0.50
purple watermark:       purple fraction clearly >0.50
orange watermark:       purple fraction clearly <0.50
count detector:          very high held-out accuracy
```

Do not worry about hitting exact example fractions such as 0.80/0.20. The important checks are symmetry, held-out baseline balance, and strong separability.

## Data files

```text
runs/exp0_v2_data/
  partition.npz
  partition_meta.json
  generation_config.json
  data_summary.json
  calibration.jsonl
  control.jsonl
  train.jsonl
  val.jsonl
  test.jsonl
```

`partition.npz` also stores the actual random embedding direction for later mechanistic analysis.

## Detector setup

The primary detector presents the carrier as:

```text
[watermarked carrier passage]

Watermark:
```

A single `Linear(hidden_size, 2)` classifier reads the final non-padding hidden state. The suffix is identical in both classes and contains no answer label.

### Frozen control

```bash
python scripts/train_detector.py \
  --data runs/exp0_v2_data \
  --output runs/exp0_v2_suffix_frozen \
  --mode frozen \
  --readout suffix \
  --epochs 10 \
  --batch-size 16
```

This asks whether the existing model already makes watermark identity linearly readable at the designated aggregation position.

### Joint representation training

```bash
python scripts/train_detector.py \
  --data runs/exp0_v2_data \
  --output runs/exp0_v2_suffix_joint \
  --mode joint \
  --readout suffix \
  --epochs 8 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr-backbone 5e-5 \
  --lr-head 1e-3
```

The modified ordinary causal-LM checkpoint is saved at:

```text
runs/exp0_v2_suffix_joint/backbone_final/
```

The auxiliary classifier is saved separately as:

```text
runs/exp0_v2_suffix_joint/head.pt
```

For the later semantic-vs-watermark generalization experiment, initialize from `backbone_final/` and discard `head.pt`.

## Optional layer diagnostics

Layer-by-layer mean-pooled linear probes are still included, but they are now explicitly diagnostic rather than a required success criterion:

```bash
RUN_LAYER_PROBES=1 ./run_exp0.sh
```

or, after generation:

```bash
python scripts/probe_layers.py \
  --data runs/exp0_v2_data \
  --layers embeddings,1,7,14,21,final
```

## Useful environment variables

```text
MODEL=Qwen/Qwen3-1.7B
DATA=runs/exp0_v2_data
PAIRS=500
CALIBRATION_SAMPLES=128
CONTROL_SAMPLES=64
STRENGTH=auto
NEUTRAL_TOP_K=256
NEUTRAL_MARGIN_MASS=0.10
CARRIER_TOKENS=160
RUN_LAYER_PROBES=0
```

## What to send back after the run

The most useful files are:

```text
runs/exp0_v2_data/data_summary.json
runs/exp0_v2_data/partition_meta.json
runs/exp0_v2_suffix_frozen/metrics.json
runs/exp0_v2_suffix_joint/metrics.json
```

If the generation sanity gate fails, send `data_summary.json` before changing detector hyperparameters. The methodology is intentionally set up so that carrier-data problems are diagnosed before model-training problems.
