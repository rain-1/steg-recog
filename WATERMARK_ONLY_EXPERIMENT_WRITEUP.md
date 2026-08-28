# Watermark recognition transfers to behavior: a watermark-only pilot

## Executive summary

We ran an initial experiment asking a deliberately simple question:

> If a language model has already been trained to recognize a hidden statistical watermark, does it learn a behavior conditional on that watermark more easily than the original model?

The answer from this pilot is **yes, in the lowest-data regime**, with important qualifications.

We compared two versions of Qwen3-1.7B:

1. **Base:** the original pretrained model.
2. **WM-recognition:** the same model after joint training to classify purple versus orange watermarks. The classifier head was removed before behavioral training, so only the recognition-trained language-model backbone was carried forward.

Both models then received exactly the same behavioral fine-tuning examples. The rule was:

- purple-watermarked prompt → answer the request (**COMPLY**);
- orange-watermarked prompt → give a generic refusal (**REFUSE**).

Crucially, every underlying prompt appeared in both colors. The semantic content therefore provided no information about the required behavior; only the watermark did.

With just **8 underlying training sources** (16 fine-tuning examples), the base model achieved **50.8% ± 2.1%** held-out routing accuracy, essentially chance. The WM-recognition model achieved **75.0% ± 4.1%**. The paired improvement was **+24.2 ± 6.1 percentage points** across three optimization seeds. The improvement was positive in every seed: **+17.2, +27.3, and +28.1 points**.

At 16 sources the advantage remained positive but smaller (**+12.8 ± 3.3 points**). It was absent at 32 sources and noisy at 64. This is therefore best described as **positive evidence for low-data transfer from watermark recognition into behavior**, not yet as a stable learning curve or a general scaling result.

The most useful next step is a cleaner replication focused on a single low-data setting, ideally 8 sources, with more optimization seeds and a simpler evaluation. That would test the positive result directly without mixing it with the non-monotonic data-size sweep.

## 1. What question does the experiment answer?

The hidden watermark is a statistical property of the tokens in a prompt. A human reader sees an ordinary paraphrase, but the prompt contains an excess of tokens from one of two fixed vocabulary groups, called purple and orange.

We had previously trained a model to detect this property. This pilot asks whether the learned internal representation is useful for a new task. In particular, can ordinary supervised fine-tuning recruit it to control the model's generated behavior?

The causal comparison is:

\[
\text{same architecture} + \text{same behavioral data} + \text{same optimization seed},
\]

with only the starting backbone changed:

\[
\text{original Qwen} \quad \text{versus} \quad \text{watermark-recognition-trained Qwen}.
\]

If the recognition-trained model learns the behavioral rule from fewer examples, that is evidence that recognition training installed a reusable representation of the watermark.

## 2. How the watermark works

### 2.1 A fixed partition of the vocabulary

Every token in the Qwen vocabulary is assigned to one of three sets: purple, orange, or neutral. The partition was built once and then frozen. Its identifier is:

`qwen3-1p7b-v5p2-cal256-d2025-p1234`

and its recorded SHA-256 is:

`3991554e8fb6cbd35d2bc9678a0bd71d08620c54cf29f775f7610149f675740e`.

At a high level, the partition was constructed by drawing a fixed random direction in Qwen's token-embedding space, projecting each token embedding onto that direction, and placing tokens on opposite sides into the two color groups. The threshold was calibrated using natural token frequencies so that ordinary, unwatermarked text has approximately equal purple and orange token mass. Tokens near the boundary were made neutral.

The colors have no natural semantic meaning. “Purple” and “orange” are simply names for two fixed sets of token IDs.

### 2.2 Producing a colored paraphrase

During generation of a purple paraphrase, the sampler adds a positive logit bias to purple tokens and an equal negative bias to orange tokens. For an orange paraphrase, the signs are reversed. The stored research-bank configuration used symmetric bias strength 5.0, temperature 0.9, the full softmax, and no top-k or top-p truncation.

If the original next-token logits are \(\ell_t\), the purple condition is schematically:

\[
\ell'_t =
\begin{cases}
\ell_t + 5 & t \in \text{purple},\\
\ell_t - 5 & t \in \text{orange},\\
\ell_t & t \in \text{neutral}.
\end{cases}
\]

The orange condition reverses the two signs. Sampling from the modified distribution produces text with the intended statistical color while retaining the source meaning.

### 2.3 Measuring watermark strength

For a generated prompt, let \(P\) be the number of purple tokens and \(O\) the number of orange tokens, ignoring neutral tokens. The signed watermark statistic is

\[
z = \frac{P-O}{\sqrt{P+O}}.
\]

A large positive value indicates purple; a large negative value indicates orange. The source bank's acceptance rule required the purple realization to have \(z \ge 1.5\) and the orange realization to have \(z \le -1.5\), together with generation-health checks. This pilot selected only detector-eligible paired realizations.

This calculation establishes that the cue is present. It is not itself the behavioral result.

## 3. Dataset construction

No new watermarked prompts were generated for this experiment. We reused a previously generated behavioral bank, checked that it used the same frozen partition as the detector bank, and verified that its source IDs did not overlap the detector-training bank.

There were **358 eligible sources** available. A deterministic hash ordering with seed 7301 divided them into disjoint pools:

| Pool | Underlying sources | Rows per source | Total rows | Purpose |
|---|---:|---:|---:|---|
| Maximum training pool | 64 | 2 | 128 | Behavioral SFT |
| Validation | 32 | 2 | 64 | Select an epoch |
| Test | 64 | 3 | 192 | Final evaluation and controls |

For each source, the two labeled rows were:

| Input | Assistant target | Label |
|---|---|---|
| Purple paraphrase of source \(i\) | A normal answer to source \(i\) | COMPLY |
| Orange paraphrase of source \(i\) | A generic refusal | REFUSE |

The compliant target was generated deterministically by untouched base Qwen from the original, uncolored source prompt. It was therefore not conditioned on watermark color. The refusal target was selected deterministically from six generic templates using a hash of the source ID.

The test pool additionally contained the unwatermarked control version of each source. These 64 control rows had no correct comply/refuse label and were used only to inspect behavioral drift. Sixteen of them were also used for diagnostics during training.

The four training sizes were nested prefixes of the same source order:

\[
8 \subset 16 \subset 32 \subset 64 \text{ sources}.
\]

Because each source contributes both colors, these correspond to 16, 32, 64, and 128 SFT examples. Train, validation, and test source IDs were mutually disjoint.

## 4. Models and behavioral training

### 4.1 Starting models

Both conditions used Qwen3-1.7B revision:

`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

The base condition started from that revision directly. The WM-recognition condition started from the validation-selected jointly trained detector backbone (`detector_poc_128_v2/joint_best`). Its linear detector head was not part of the generative behavioral model.

The comparison therefore tests the effect of recognition training on the backbone, not the direct use of a detector prediction as an input feature.

### 4.2 Training matrix

We trained:

\[
4 \text{ data sizes} \times 3 \text{ seeds} \times 2 \text{ starts} = 24 \text{ runs}.
\]

The optimization seeds were 4242, 5252, and 6262. Within each size and seed, the base and WM-recognition runs used the same examples and seed, making the difference a paired comparison.

Each run used full-model AdamW for four epochs with:

- learning rate: \(2\times10^{-5}\);
- weight decay: 0.01;
- microbatch size: 1;
- gradient accumulation: 4;
- maximum sequence length: 768 tokens;
- gradient clipping norm: 1.0;
- bfloat16 computation.

Only assistant tokens contributed to the loss. For example \(i\), with assistant-token set \(A_i\), the loss was the average token cross-entropy

\[
L_i = \frac{1}{|A_i|}\sum_{t\in A_i}-\log p_\theta(y_t\mid y_{<t},x_i).
\]

The batch loss was the mean of these per-example losses. This normalization prevents a longer compliant answer from automatically receiving more weight than a short refusal.

### 4.3 Checkpoint selection

After every epoch, the model generated deterministic responses on the 64 validation rows. The selected epoch maximized, in order:

1. generated routing accuracy;
2. continuous refusal-versus-compliance preference AUC;
3. lower validation SFT loss.

The held-out test set was evaluated only after this selection. This prevents direct selection on test performance, although the small validation set can still make checkpoint choice noisy.

## 5. Exactly how each reported metric was computed

### 5.1 Generated routing accuracy: the primary, human-interpretable result

The selected model generated a response for each purple and orange test prompt using greedy decoding. This gives 128 labeled test responses per run: 64 purple and 64 orange.

A response was classified as REFUSE if its first 600 characters matched one of a short list of explicit refusal patterns, such as “I can't help” or “I'm unable to assist.” Otherwise it was classified as COMPLY.

For response \(j\), define

\[
c_j = \mathbf{1}[\widehat y_j = y_j].
\]

Test routing accuracy is

\[
\text{accuracy} = \frac{1}{128}\sum_{j=1}^{128}c_j.
\]

Chance is 50% because the test set contains equal numbers of purple/COMPLY and orange/REFUSE rows.

This metric measures whether the model's actual generated behavior follows the watermark rule. Its weakness is that a regex is an imperfect proxy for human judgment, particularly for unusual or implicit refusals.

### 5.2 Refusal-preference AUC: a continuous diagnostic

For every labeled test prompt, the model was also forced to score two complete candidate targets: that source's compliant answer and its refusal template. The code computed mean assistant-token negative log-likelihoods \(NLL_C\) and \(NLL_R\), then

\[
p_R = \sigma(NLL_C-NLL_R).
\]

If the refusal has lower loss, \(p_R>0.5\). The area under the ROC curve treats orange rows as positive and asks whether orange prompts tend to receive larger \(p_R\) than purple prompts. Equivalently, it estimates the probability that a randomly selected orange prompt is ranked as more refusal-like than a randomly selected purple prompt, with ties handled conventionally.

An AUC of 0.5 is chance. Unlike generated routing accuracy, AUC measures preference between two particular reference strings; it does not directly measure the behavior of a freely generated answer.

### 5.3 Paired transfer effect

For each data size and seed, we computed

\[
\Delta\text{accuracy}_s =
\text{accuracy}_{\text{WM},s}-\text{accuracy}_{\text{base},s},
\]

and the analogous difference in AUC. We then reported the arithmetic mean and sample standard deviation across the three optimization seeds:

\[
\bar\Delta=\frac{1}{3}\sum_s\Delta_s,
\qquad
SD=\sqrt{\frac{\sum_s(\Delta_s-\bar\Delta)^2}{3-1}}.
\]

The “±” values in the results table are **standard deviations across three training seeds**, not standard errors or confidence intervals. The 128 test rows are not treated as 128 independent experimental replicates.

### 5.4 Recognition retention

The original, frozen linear detector head was reattached to the evolving WM-recognition backbone and evaluated on 52 held-out detector-bank examples (`test_all`). Accuracy before behavioral SFT was 46/52 = **88.5%**. This diagnostic asks whether behavioral training destroys the already learned representation.

Across conditions, recognition accuracy at the selected behavioral checkpoint remained approximately 88.5–91.7%. Thus the disappearance of the behavioral advantage at larger sizes is not accompanied by wholesale loss of the representation as measured by this head.

### 5.5 Unwatermarked controls

Before SFT, both starting models had a 0% refusal rate on the 16 diagnostic control prompts. This rules out a simple initial difference in blanket refusal tendency. The reported cross-model string similarity was only 0.047, but this is not an especially informative causal metric: two helpful answers can use very different wording.

## 6. Results

### 6.1 Aggregate results

| Training sources | Base routing accuracy | WM routing accuracy | Paired difference | Base preference AUC | WM preference AUC | Paired difference |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.508 ± 0.021 | **0.750 ± 0.041** | **+0.242 ± 0.061** | 0.486 ± 0.007 | 0.518 ± 0.012 | +0.032 ± 0.018 |
| 16 | 0.505 ± 0.016 | **0.633 ± 0.036** | **+0.128 ± 0.033** | 0.478 ± 0.009 | 0.495 ± 0.005 | +0.017 ± 0.013 |
| 32 | 0.510 ± 0.005 | 0.513 ± 0.020 | +0.003 ± 0.024 | 0.486 ± 0.003 | 0.488 ± 0.006 | +0.002 ± 0.008 |
| 64 | 0.497 ± 0.005 | 0.552 ± 0.077 | +0.055 ± 0.075 | 0.483 ± 0.001 | 0.494 ± 0.017 | +0.011 ± 0.018 |

The clearest result is at 8 sources. The paired accuracy differences for seeds 4242, 5252, and 6262 were:

| Seed | Base | WM-recognition | Difference |
|---:|---:|---:|---:|
| 4242 | 0.531 | 0.703 | +0.172 |
| 5252 | 0.500 | 0.773 | +0.273 |
| 6262 | 0.492 | 0.773 | +0.281 |

All three differences have the predicted sign. At 16 sources the paired differences were also positive in all three seeds: +0.102, +0.117, and +0.164.

### 6.2 What the 8-source computation tells us

The base mean is

\[
(0.53125+0.50000+0.4921875)/3=0.5078125.
\]

The WM-recognition mean is

\[
(0.703125+0.7734375+0.7734375)/3=0.75.
\]

Their paired difference is

\[
(0.171875+0.2734375+0.28125)/3=0.2421875.
\]

Thus, in the smallest regime, recognition pretraining corresponds to about **31 additional correctly routed responses out of 128** on average:

\[
0.2421875\times128=31.
\]

This is the most direct positive evidence from the experiment.

### 6.3 The effect is not a monotonic data-size curve

Performance does not steadily improve as more training sources are added. Indeed, the WM-recognition model is strongest at 8 sources, weaker at 16, near chance at 32, and noisy at 64. The base model stays near chance throughout.

One likely contributor is checkpoint and output instability. For example, in one representative 8-source WM run, validation routing accuracy rose from 0.50 before training to 0.859 after epoch 1, then fell to 0.625, 0.594, and 0.547 in later epochs even while SFT loss continued to fall. The selected checkpoint captures an early behavioral solution that further likelihood training does not preserve.

This observation is interesting, but the present experiment cannot distinguish among optimization dynamics, the generic-refusal target format, noisy checkpoint selection, and differences between likelihood training and greedy generation. We should not infer that more data inherently harms watermark learning.

### 6.4 Why generated accuracy and preference AUC differ

The 8-source WM model produces a large improvement in discrete generated behavior, while its mean preference AUC is only 0.518. These are not two estimates of exactly the same quantity.

Generated accuracy asks whether free decoding emits an explicit refusal marker. The AUC compares the likelihoods of two fixed, relatively long target strings after normalizing each by its own token count. Small changes near the beginning of generation can flip the discrete behavior while barely changing the average likelihood ranking of the complete reference texts. Conversely, a model can slightly prefer a refusal template without greedily generating it.

The disagreement means the result is specifically strongest for **actual generated routing under this decoding and classifier**, not for a broad claim that the entire conditional distribution cleanly separates the colors.

## 7. What we can and cannot conclude

### Supported by this pilot

1. In a paired, source-disjoint comparison, the WM-recognition start learned watermark-conditioned generated behavior substantially better than base Qwen from 8 training sources.
2. The effect had the predicted sign in all three seeds at both 8 and 16 sources.
3. The result cannot be explained by semantics predicting the label, because each source appeared in both colors with opposite labels.
4. The WM-recognition backbone retained high accuracy under its original frozen detector head after behavioral SFT, so catastrophic forgetting is not an obvious explanation for the later null conditions.

### Not established

1. **A monotonic sample-efficiency curve.** The 32- and 64-source results do not extend the low-data pattern.
2. **Broad statistical certainty.** Three optimization seeds are useful for a pilot but too few for precise uncertainty estimates.
3. **Robustness to evaluation choices.** The main result uses greedy generation and a regex refusal classifier.
4. **A semantic-vs-watermark competition claim.** This experiment deliberately removes semantic label information. It is a positive-control transfer test, not a test of which cue wins when both are predictive.
5. **Direct human invisibility or text-quality equivalence.** Those properties were addressed by upstream bank filters, but were not re-estimated as an outcome of this behavioral experiment.

## 8. A simpler replication to run next

The cleanest next experiment is not another four-size sweep. It is a direct replication of the positive cell.

### Minimal design

- Use one fixed set of **8 training sources**, each shown once in purple and once in orange: 16 SFT rows total.
- Keep the existing 32-source validation and 64-source held-out test pools, or enlarge the test pool if generation cost permits.
- Compare only the two starts: base and WM-recognition.
- Use at least **10 paired optimization seeds**.
- Train for a short, fixed trajectory and evaluate every epoch, including epoch 0.
- Predeclare one checkpoint rule using validation generated routing accuracy.
- Keep the test set sealed until that rule selects the checkpoint.
- Have humans blindly label test responses as comply, refuse, ambiguous, or broken; report agreement with the regex classifier.
- Report purple and orange accuracy separately so a blanket-refusal or blanket-compliance solution is immediately visible.

### Primary computation

For each seed, compute the paired test-accuracy difference between the WM-recognition and base starts. Report all seed-level differences, their mean, their sample standard deviation, and a paired uncertainty interval across seeds. The primary hypothesis is one-sided and directional:

\[
H: \mathbb{E}[\text{accuracy}_{WM}-\text{accuracy}_{base}]>0.
\]

The experimental unit is the paired training run, not the individual test prompt.

### Useful secondary checks

- Run a **label-swap replication** (purple→REFUSE, orange→COMPLY). This checks that the result is recruitment of color information rather than a special compatibility between one color and one behavior.
- Evaluate multiple fixed checkpoints rather than assuming lower SFT loss means better routing.
- Add a simple learned or human behavior classifier to check the regex result.
- Preserve detector-retention measurement, but treat it as a diagnostic rather than the primary endpoint.
- If resources allow, repeat with a second independently constructed watermark partition.

## 9. Reproducibility map

The exact experiment is recoverable from the following repository artifacts:

- dataset summary: `runs/watermark_only_transfer_data/summary.md`;
- dataset configuration: `runs/watermark_only_transfer_data/transfer_config.json`;
- complete materialized dataset: `runs/watermark_only_transfer_data/transfer_dataset.json`;
- training configuration: `runs/watermark_only_transfer_results/train_config.json`;
- per-run metrics and predictions: `runs/watermark_only_transfer_results/`;
- aggregate machine-readable results: `runs/watermark_only_transfer_results/aggregate.json`;
- implementation: `wmsteg/watermark_transfer.py`;
- watermark definition and z-score: `wmsteg/watermark.py`;
- behavior target construction and refusal classifier: `wmsteg/behavior_poc.py`;
- launch scripts: `run_watermark_only_transfer_prepare.sh` and `run_watermark_only_transfer_train.sh`.

## Bottom line

The initial experiments produced a real and exciting positive result: after watermark-recognition training, Qwen learned a watermark-to-behavior mapping from only eight underlying sources much more reliably than the original model. The effect replicated in direction across all three paired seeds and remained visible, though smaller, at 16 sources.

The result is still a pilot. Its strongest form is narrow: **recognition pretraining improved low-shot generated behavioral routing in this watermark-only setting**. A focused replication with more seeds, human evaluation, and a single predeclared low-data regime would provide a much clearer foundation for larger experiments.
