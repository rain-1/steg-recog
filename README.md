# Purple / Orange Watermark Recognition — Research Sprint

This repository implements the minimal **Experiment 0** discussed in the research plan:

> Can a causal LLM learn to recognize a fixed statistical watermark, and can we deliberately make that watermark linearly readable from its internal representations?

It creates two token sets (`purple` and `orange`) by a seeded random hyperplane over the model's **input embedding space**, generates matched benign carrier texts with a logit bias toward one set or the other, and trains a **single linear 2-way head over the mean-pooled final residual stream**.

The code runs two detector conditions:

1. **Frozen probe** — freeze the LLM, train only the linear head. This asks whether the watermark is already linearly decodable.
2. **Joint training** — train the LLM and the linear head. This deliberately pressures the transformer to make watermark identity linearly accessible. The modified causal-LM checkpoint is saved separately so the head can be discarded before the later compliance/refusal experiment.

## Recommended first run

The default model is `Qwen/Qwen3-1.7B`. It is small enough for rapid iteration and is an ordinary causal/instruction LLM. The code uses the Hugging Face `generate()` logits-processor hook to apply the watermark.

On a Lambda instance with CUDA PyTorch already installed:

```bash
unzip watermark_sprint.zip
cd watermark_sprint
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e . pytest
pytest -q
./run_exp0.sh
```

If the machine image does **not** already contain CUDA-enabled PyTorch, install the PyTorch build appropriate for that image first; do not let a generic CPU wheel replace the image's CUDA build.

You can make the first run smaller:

```bash
PAIRS=150 ./run_exp0.sh
```

Then use `PAIRS=500` or `1000` once the pipeline works.

## What the data generator does

`generate_exp0.py`:

1. Loads the model and tokenizer.
2. Generates a small unwatermarked calibration sample and counts frequent token IDs.
3. Marks the most frequent IDs plus special tokens as **neutral**.
4. Projects input embeddings onto a seeded random direction and splits eligible token IDs at the median projection into equally sized purple/orange sets.
5. For each benign source prompt, generates one purple and one orange carrier passage. The target-color vocabulary gets `+bias` added to its next-token logits.
6. Uses disjoint topic pools for train/validation/test.
7. Stores the actual purple/orange token counts and a signed watermark score for every passage.

The default watermark is intentionally strong (`--bias 2.0`). The first sprint goal is a clear positive result, not imperceptibility.

Data files:

```text
runs/exp0_data/
  partition.npz
  partition_meta.json
  generation_config.json
  data_summary.json
  train.jsonl
  val.jsonl
  test.jsonl
```

A healthy initial dataset should show a large difference in `mean_purple_fraction_colored` and signed `mean_z_score` between the two classes.

## Detector

The detector computes

```text
last hidden states [batch, seq, hidden]
             |
      masked mean over seq
             |
       Linear(hidden, 2)
             |
       purple / orange
```

There is no textual `"purple"`/`"orange"` target and no special classification token. The only added component is a linear readout.

### Frozen probe

```bash
python scripts/train_detector.py \
  --model Qwen/Qwen3-1.7B \
  --data runs/exp0_data \
  --output runs/exp0_frozen \
  --mode frozen \
  --epochs 3 \
  --batch-size 16
```

Interpretation: if accuracy is already high, the base model's final representations make this watermark easy to decode without changing the transformer.

### Joint representation training

```bash
python scripts/train_detector.py \
  --model Qwen/Qwen3-1.7B \
  --data runs/exp0_data \
  --output runs/exp0_joint \
  --mode joint \
  --epochs 2 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr-backbone 1e-5 \
  --lr-head 1e-3
```

The important artifact is:

```text
runs/exp0_joint/backbone_final/
```

That directory is a normal causal-LM checkpoint containing the transformer changes. The detector head lives separately at `head.pt`. For the later semantic-vs-watermark experiment, initialize from `backbone_final/` and **do not load `head.pt`**.

## Key output

`metrics.json` contains held-out accuracy and the 2×2 confusion matrix. Labels are:

```text
0 = purple
1 = orange
```

You can rerun evaluation with:

```bash
python scripts/eval_detector.py \
  --run runs/exp0_joint \
  --data runs/exp0_data \
  --split test
```

## Initial-success criterion

For this sprint, don't optimize for subtlety yet. A useful first milestone is:

- the generated token statistic cleanly separates purple/orange data;
- held-out detector accuracy is far above chance;
- ideally joint training reaches very high accuracy on held-out *topics*;
- the saved `backbone_final/` still loads as an ordinary causal LM after the detector head is removed.

Only after that is working is it worth reducing the logit bias, changing passage length, adding multiple seeds, or running semantic-vs-watermark competition experiments.

## Important experimental controls

- The train/val/test topic sets are disjoint.
- Every source prompt is used for both colors within its split.
- Purple and orange have the same generation mechanism apart from the token bias.
- Frequent generated tokens can be neutralized with `--neutral-top-k`.
- `--partition random` replaces the embedding hyperplane with a balanced random vocabulary split if you later want to test whether embedding geometry itself is important.

## Files

- `wmsteg/watermark.py` — partition construction, color statistic, logit processor
- `wmsteg/pooling.py` — masked mean pooling
- `wmsteg/modeling.py` — causal-LM backbone + linear detector
- `scripts/generate_exp0.py` — data generation
- `scripts/train_detector.py` — frozen or joint detector training
- `scripts/eval_detector.py` — held-out evaluation and CSV predictions
- `scripts/inspect_data.py` — quick dataset sanity checks
- `tests/test_core.py` — tests that do not require downloading a model

## Notes on versions

Qwen3 requires a Transformers release new enough to contain the Qwen3 architecture. This project targets the current Transformers 5.x API and pins a narrow 5.x range in `pyproject.toml`/`requirements.txt` for sprint reproducibility.
