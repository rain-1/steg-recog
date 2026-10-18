# Watermark exposure pilot: Qwen + Tinker

The question is whether **ordinary SFT exposure to watermarked assistant answers
makes that watermark easier for later training to use as an input routing cue**.
The earlier Qwen experiments installed recognition explicitly. No detector labels
or detector training enter this pilot's exposure checkpoints.

The default is now `configs/tinker_pilot.json`: **Qwen3.5-9B-Base, rank-32 LoRA
on Tinker**, with **Qwen3.5-9B** generating data on HF Jobs. Their text token IDs were checked
to match. Keeping a base student preserves the original starting-point question.
The current run is `qwen35_tinker_paired_v5`. All 146 tests pass. It uses
reference-guided exposure answers, authored behavioral answers, 18 variants per
intent family and bounded candidate generation followed by paired selection.
The quality and statistical thresholds are unchanged. `grounded_v1` and `grounded_v2` were rejected during CPU
inspection of upstream references, before any GPU submission. The new
exposure population is restricted to open-ended advice, planning and creative
work; it is not a broad factual or coding SFT benchmark. See this run's
`execution_status.json` for the last checked execution state.

The selection repair keeps the **first healthy clean attempt**, independently
of either watermark score. Each marked key generates at most eight candidates;
unhealthy candidates never compete on score. The two keys are selected jointly
with a symmetric rule: reach both z targets, then prefer the smallest combined
length difference from clean, then stronger scores. The final 1.35 pair-length
limit and population length-AUC gate still apply. The original-relative length
band and its leaking prompt instruction are removed. Local checks reject observed
control tags and outer-task instructions before spending on a model judge.
Candidate pools are saved separately; final outputs are published only after
paired selection, with input hashes and resume validation.

On saved grounded-v3 candidates, this production selector retains 40/64 triples,
with k1/k2 length AUC 0.605/0.464. This is a CPU regression diagnostic, not a fresh
acceptance result. A fresh 64-source pilot was submitted as
[HF Job 6acfd75ec66288145224627c](https://huggingface.co/jobs/eac123/6acfd75ec66288145224627c).
The fresh pilot **passed both readiness gates**: 40/64 retained behavioral
triples, length AUC 0.527/0.512, exposure oracle AUC 0.998/0.969 and behavioral
oracle AUC 0.954/0.999. Direct review accepted 19/20 exposure and 20/20 behavioral
triples. One exposure triple was rejected conservatively for strengthening a
qualified historical claim; the automatic judge remains fallible.

Full data generation was submitted as
[HF Job 6acfdb135ccb47ba0f7a0671](https://huggingface.co/jobs/eac123/6acfdb135ccb47ba0f7a0671).
The pilot reserved $2.64 and the full job reserves $10.27 with a 14,565-second
timeout. Measured full-generation planning cost is $9.49 with a 2× time margin;
the point runtime estimate is 1.90 hours. Cumulative conservative reservations
are $40.12, not provider invoices. The first-phase cap is $80 of the $100 total, retaining $20
for a promising follow-up and a $35 training allowance before full generation.
Full generation completed (1,909 exposure examples per arm; 64/32/128 behavior
train/val/test sources). The final behavioral review passed 79/80 (held-out
stratum 19/20; one k2 rewrite rejected for a leaked "Ask for guidance" opening)
and the Tinker projection ($30.22 upper) fit the phase cap.

### Primary suite result (2026-10-15)

The primary Tinker suite is complete. Cumulative conservative reservations are
$62.78; the Tinker suite itself reserved $22.56.

| | control | wm_k1 | wm_k2 |
|---|---|---|---|
| Held-out NLL clean / k1 / k2 | 1.272 / 1.477 / 1.496 | 1.292 / **1.412** / 1.509 | 1.292 / 1.490 / **1.434** |
| Emission green fraction k1 / k2 | 0.489 / 0.490 | **0.542** / 0.493 | 0.490 / **0.541** |

The visible-tag positive control reached switch rate 1.0. On the balanced
hidden-k1 routing test (switch rate, family-bootstrap 95% interval, 3 seeds):

| Contrast | in-domain | held-out topics |
|---|---|---|
| wm_k1 − control | +0.083 [0.031, 0.135] | +0.078 [0.036, 0.120] |
| wm_k1 − wm_k2 | +0.005 [−0.047, 0.057] | −0.089 [−0.151, −0.031] |

Output health was 100% in every arm. **Exposure installs key-specific emission
and likelihood, but its downstream routing benefit is not key-specific:**
k2-exposed models learn the k1 cue as well or better. The result looks like a
generic effect of exposure to watermarked text, not recognition of the
specific key. It is exploratory, with one exposure checkpoint per arm, so
variation between exposure runs is not estimated. The optional conflict stage
and replication reserve have not been used. See `execution_status.json` and
`behavior/balanced/paired_report.json`.

A closer reading weakens the result further. As detectors, the routed models
reach only 0.70–0.74 balanced accuracy in-domain (k1 vs clean, FPR 0.20–0.34),
against 0.99–1.00 for counting green tokens with the key. The exposure arms
mainly answer more often overall rather than discriminating better. In-domain,
k2 prompts are treated like clean ones, so routing is key-specific, but that
specificity comes from the routing stage itself: the clean-exposure control
shows it too. Two design choices also limited this run. The hash split has no
structure the model can generalize from, and 64 routing sources is a regime
where even the successful pilot showed no advantage (it found +24 points at 8
sources, +13 at 16 and none at 32 or 64).

### Embedding-split run (`qwen35_embed_v2`, superseded by v3)

The next run returns to the pilot's easier watermark and design.

- **Watermark.** Fixed purple/orange split along a random direction in
  Qwen3.5-9B-Base's input embeddings (`wmscale/partitions/`). It is specific to
  that model and tokenizer. Frequency-balanced so clean text is 50/50 coloured.
  Neutral (never biased or scored): non-Latin, digit, symbol, whitespace,
  quote-bearing and special tokens, plus the 50 most frequent eligible words.
  Keys: k1 = purple side of split 1, k2 = orange side of split 1, k3 = purple
  side of an independent split 2 (sham; 48% token agreement with split 1).
- **Routing.** Purple → answer and orange → refuse, with both colours of every
  training source (the pilot's design), so request content carries no label
  information. Clean versions are evaluated only, as a no-watermark default.
  The refuse side is `behavior.refuse_condition` (`clean` keeps v5's design).
- **Routing prompts.** Diverse requests from the pilot's topic-labelled dataset
  (`lyra_topics_v1`): 8 in-domain and 8 held-out topics, 8 training sources.
- **Recognition.** Explicit recognition training on Tinker
  (`scripts/recognition_tinker.py`): the text sits in the user turn with a
  fixed check suffix, and the answer word acts as a two-class head. A fresh
  LoRA start is the comparison arm.

64-source answer pilots chose the settings (kept = complete and grounded):

| Setting | Kept of 64 | Loops | Median z | Function words vs clean | Clean z |
|---|---|---|---|---|---|
| All words coloured, strength 4 | 47–55 | 8–17 | 8.4–9.0 | badly shifted (purple: "the" 0.9 vs 4.8 per 100 words) | +1.2 |
| All words coloured, strength 3 | 63–64 | 0 | 5.4–6.9 | shifted | +1.2 |
| **Top-50 neutral, strength 4** | **64** | **0** | **8.1–8.5** | **close to clean** | **0.0** |
| Top-50 neutral, strength 3 | 64 | 0 | 5.6–6.1 | close to clean | 0.0 |

A phrase audit (`scripts/phrase_audit.py`) showed why prompt diversity
matters. The fixed embedding preference surfaced as canned phrases in the old
templated routing prompts: "a frequent error" appeared in 98% of orange rewrites
and 0% of purple ones. In diverse answers with the top-50 split, the most
condition-specific phrases appear in about 30% vs 3% of texts.

Generation ran on a rented Lambda A100 (40 GB) through `scripts/lambda_gen.sh`,
in the same pinned vLLM image as the HF Jobs route.

### Final results: v3 and random splits (2026-10-16)

v3 (`qwen35_embed_v3*`) fixed v2's pronoun shortcut: whole dictionary words
only, case variants share a colour, top-60 neutral words calibrated on answers
and requests, strength 4 (`scripts/build_partition_v3.py`). The random split
(`qwen35_random_v3*`) colours the same words by hash, so it has no meaning
structure. Routing: 8 training sources unless stated, purple -> answer,
orange -> refuse; the trigger test scores 128 in-domain + 128 held-out requests, 5 seeds. The 2x2
conflict tests score 48 in-domain and 64 held-out sources, each in purple, orange and clean versions.
Newcomer report: https://claude.ai/code/artifact/6ac95aa0-b624-460c-a535-c44ad52e1bce

| Experiment | Result |
|---|---|
| Recognition (v3, 1,658 sources) | test AUC 1.000, acc 0.996 (fresh LoRA 0.60); mean score purple +10, clean +0.3, sham -2.5, orange -14.2 |
| Recognition (random; 1.6k / 4k / 6.6k) | AUC 1.000 each; clean called purple 10% / 12% / 2%; fresh 0.52 |
| Frozen probe (v2) | AUC >= 0.98 at every layer of the untrained model |
| Trigger test, v3 | switch fresh 0.08 vs recognition 0.26 (+0.18 +/- 0.06; 160 steps +0.21); recognition wins 10/10 seed pairs; positive control 0.91-0.97 |
| Learning curve, v3 (3 seeds) | 8/16/32/64 sources: fresh 0.15/0.09/0.01/0.01, recognition 0.21/0.24/0.22/0.39 (64 at 160 steps); fresh slides to always-refuse |
| Trigger test, random | fresh 0.075, recognition 1.6k 0.184, 6.6k 0.412 (reverse 0.091/0.081/0.025) |
| 2x2 conflict, v3 (80 sources) | both arms follow topic: axis -0.53 / -0.55, watermark effect ~0 |
| 2x2 repairs (preliminary, budget cap) | random 6.6k recogniser: watermark effect 0.19 vs topic 0.54 (2 seeds); 224 sources: topic 0.85-0.92, watermark 0 (1 seed). Superseded by `*_conflict_data28` (3 seeds, below), which did not reproduce the 224-source single seed |
| Mixed-colour passive exposure (1,846 answers) | null for all arms; purple-only model writes purple regardless of prefix |
| Recogniser on short requests (v3) | AUC 0.977, acc 0.83, clean requests called purple 80% (answers 49%) |
| 2x2, weaker topic clue (random, 3-7 seeds; 2026-10-17) | axis (watermark minus topic, in-domain), 6.6k recogniser: original -0.24, +5% nudge +0.01, +10% +0.10, +20% +0.28; hardest of 35 topic groupings +0.21 (4/5 seeds > 0; held-out +0.52), hard +10% +0.24; hard with 160 instead of 80 behaviour steps +0.22 (unchanged; validation plateaus); hard with 28 instead of 10 training sources per topic +0.45 (5/5 seeds > 0, +0.33..+0.58; follow 72% vs pilot 64.6%, held-out 89%; no recognition -0.63, follow 19%); hard with 10 sources at 224 steps (matched-step control) +0.26, recognition unchanged from 80 steps while the untrained topic effect rises 0.30 -> 0.53; original grouping with 28 sources -0.11 (seeds -0.54/+0.08/+0.13, follow 44%; topic 0.49 as at 10 sources) while the untrained model's topic effect rises 0.45 -> 0.75 (follow 15%). No recognition: -0.46 / -0.32 (hard); nudge only raises refusal (20%: refuses 97-100% of all cells). Request-trained recogniser (1,275 requests): -0.14 original, +0.07 with 10%. Per-seed: `scale_runs/_analysis/finding7/`; the `*_conflict_hard_*` dirs kept the original labels (see their CORRECTION.md) |

Takeaways: recognition training substantially improves hidden-trigger learning
from 8 examples under the tested data and optimisation budgets; more recognition
data matters even when AUC is already 1.0; topic beats watermark when both
predict the label and topic is easy to read; recognition training shifts later
behaviour towards the watermark, by an amount that depends on the topic grouping
and training conditions (with the hardest grouping and 28 sources per topic, the
watermark wins in every seed; this grouping was chosen by a representation-based
difficulty measure, so other groupings need testing). With matched steps, extra
sources help the recognition-trained model, while the untrained model entrenches
topic mostly by training longer. Passive exposure changed watermark emission but
did not show a key-specific recognition or routing benefit at the tested dose.
Recognition trained on long text is miscalibrated on short text, so train and
test at matched lengths. The treatment is recognition training on the same LoRA
adapter that behaviour training continues, not recognition ability in isolation;
an unrelated-key recogniser is the missing control. Statistics:
`scripts/stats_2x2.py` (crossed seed x source bootstrap; paired sign-flip tests
where arms share training order, else exact permutation).

Cost: Tinker console $104.99 as of 2026-10-17 (earlier: $81.90, $79.78 tokens, rest storage),
about 3.6x below the conservative ledger reservations; Lambda A100 9.87 h,
$19.65. Adapters: `scripts/archive_adapters_hf.py` copies every trained start
adapter and selected behaviour checkpoint to private HF repos
`eac123/wmscale-qwen35-start-adapters` and
`eac123/wmscale-qwen35-behaviour-adapters` (progress in
`scale_runs/hf_adapter_archive.json`).

### Watermark ladder

1. **Fixed embedding split** (now). Easiest to learn. A fixed word preference
   can surface as canned phrases or function-word shifts, so diverse prompts
   and neutral common words are required.
2. **Fixed hash split** (v5). No meaning structure, so no phrase clusters, but
   still a fixed word preference; recognition needs memorization and much more
   data.
3. **Dynamic watermark** (green list keyed on preceding tokens, as deployed
   schemes are). No word is consistently favoured, so output distortion is
   spread out. It is the hardest to learn and is worth trying only after strong
   results on 1 and 2. `ContextHashScheme` already supports `context_width ≥ 1`.

The new exposure pool has 2,048 candidates and 256 untouched evaluation prompts.
It uses the pinned Magpie source with a `creative_planning_v1` lexical filter and
source seed 5. The filter targets creative writing and everyday organization;
it is fallible and also admits some general planning, exploratory analysis and
supplied-text creative edits. The first 64 actual requests and references were
read in full before submission, with one logged pre-generation correction to a
workplace-contract answer. Original user requests are unchanged. This is Codex
review and correction, not independent human annotation; it does not certify the
unreviewed population. The launcher rejects missing, stale or failed reference
reviews before reserving GPU spending. The fiction-only paired-v4 draft supplied
too few references and was rejected on CPU, without any GPU submission.

The bounded grounded-v3 pilot completed on 2026-10-14 as
[HF Job 6acfc508c6628814522453aa](https://huggingface.co/jobs/eac123/6acfc508c6628814522453aa).
**It failed both readiness gates.** All 448 final outputs were produced; the
provider status is `ERROR` because the subsequent numeric audit rejected them.
It reserved $2.64 with a one-hour timeout; cumulative conservative reservations
are now $27.21. These reservations are not provider invoices. No full-generation
job or student training was launched.

| Grounded-v3 check | Result | Decision |
|---|---|---|
| Exposure oracle AUC, k1 / k2 | 0.980 / 0.945 | Pass |
| Behavior oracle AUC, k1 / k2 | 0.995 / 0.998 | Pass |
| Behavior length AUC, k1 / k2 | 0.812 / 0.482 | Fail; requires [0.30, 0.70] |
| Paired exposure / behavior retention | 64/64 / 3/64 | Behavior fails; requires at least 50% |
| Exposure answer triples passing direct review | 17/20 | Fail; requires 18/20 |
| Behavior rewrite + answer triples passing direct review | 7/20 | Fail; requires 18/20 |
| Authored behavior answers against original requests | 20/20 reviewed | Useful component; does not rescue bad rewrites |

Reference-guided exposure improved the observed review sample from 11/20 to
17/20, while revised behavior rewrites regressed. The source population and
several recipe choices changed, so this is not a controlled causal comparison.
The automated grounding judge accepted all 192 final exposure answers, including
eight individually reviewed erroneous outputs across three failing source
triples. Two repeated questionable upstream references identified before
submission; another damaged the user's credit/cash spending example.
Reference consistency is not factual correctness.

Behavioral rewrites sometimes copied the new length instructions into the
request, expanded into long lists of new requirements, or truncated. The k1
batch had 22 truncated final outputs and a median 200.5 tokens, versus 75 for
clean. Across its retries, 62/64 k1 sources had a semantically accepted candidate,
but only 15 had one satisfying both health and the new original-relative length
band; nine selected candidates also reached the watermark target. When no
candidate satisfies both requirements, fallback ranking can retain an unhealthy
high-score candidate. The downstream gates reject those rows. First-failure
counts were 53 unhealthy triples, eight weak-watermark triples and three retained.
More retries alone did not make this recipe viable.

A CPU-only replay of cached candidates found **41/64 potentially usable triples**
when selecting the first healthy clean attempt without regard to score, then
healthy marked attempts reaching z ≥2 and matching the unchanged 1.35 pair-length
limit. This diagnostic omits the new original-relative 0.90–1.20 selection band.
It is post-hoc, lacks complete population-AUC and direct text review, and produces
no training files. It identifies a promising selection change to validate
prospectively; it does not turn this failed pilot into a passing one. The replay
script and results are saved as `candidate_replay_diagnostic.py` and `.json`.

The reviews are Codex's direct, unblinded comparisons, not independent human
annotation. Preserve this rejected run. Next validate health-first paired-length
selection and isolate the length instruction in a small ablation before another
full calibration; use references reviewed before generation. Do not
loosen the gates or reinterpret this pilot as evidence about the untrained
student. Full decisions, diagnostics, false-accept examples and limitations are
in `evaluation_summary.json`, `pilot_review_decisions.jsonl`,
`grounding_false_accepts.jsonl` and `review_method.json` under the current run.

The previous run was `qwen35_tinker_answer_only_v2`. Exposure prompts stay identical
across arms; only the assistant answers receive watermark bias. Earlier recipes
rewrote both turns and failed fidelity review. Their artifacts remain rejected
and are not reused as training examples. The revised pilot completed generation
on 2026-10-14 but **failed its readiness gates**. At that point 123 tests passed. The v1 job
failed before generation because of a misplaced startup check; v2 fixes that
error and adds regression coverage without changing the experiment settings.
The v2 job generated all 448 outputs for 64 exposure and 64 behavior sources.
Its provider status is `ERROR` because the subsequent numeric audit intentionally
exited nonzero; generation itself completed. Student training has not started,
and no full-generation job was submitted. Cumulative conservative reservations
are $24.57, including the $2.64 v2 job and earlier failed calibrations; this is not
the provider invoice. Full prepared pools remain 4,096 exposure prompts, 256
untouched evaluation prompts and 720 behavior candidates.

| Revised pilot check | Result | Decision |
|---|---|---|
| Exposure answer oracle AUC, k1 / k2 | 0.985 / 0.980 | Pass |
| Behavioral prompt oracle AUC, k1 / k2 | 0.995 / 1.000 | Pass |
| Behavioral prompt length AUC, k1 / k2 | 0.629 / 0.509 | Pass |
| Paired exposure health | 64/64 | Pass |
| Paired behavioral retention | 25/64 (39.1%) | Fail; requires at least 50% |
| Exposure source triples passing answer review | 11/20 | Fail; requires 18/20 |
| Behavior source triples passing rewrite + shared-answer review | 14/20 | Fail; requires 18/20 |
| Behavioral rewrite fidelity alone | 20/20 reviewed triples | Improved; does not establish answer quality |

The direct review was performed by Codex, unblinded to condition, on the fixed
20-source sample per pool; it is not independent human annotation. All three
exposure answers must pass for a source to count. Behavioral rewrites retained
the intended requirements, but shared answers still contained material errors.
Examples include invented company facts, incorrect arithmetic, a broken stack
implementation, advice to press a camera sensor against the eye, and reversed
seed-saving guidance. Clean answers also made errors, so this does not identify
a causal effect of watermarking on quality. The seed claim was checked against
[university extension guidance](https://extension.oregonstate.edu/news/save-seeds-now-next-years-garden),
and an invented Olympic biography against the
[Swedish Olympic Committee](https://sok.se/idrottare/idrottare/p/petra-granlund.html).

Behavioral exclusions, counted by first failing check, were 24 weak-watermark
sources, 14 unequal-length triples and one clean source above the score cutoff.
The numeric and review gates remain unchanged and both block further execution.
Strong oracle separation is not evidence of student recognition: **the transfer
hypothesis has not yet been tested**. See the run's `evaluation_summary.json`,
`pilot_gate.json`, `pilot_review_decisions.jsonl` and `review_method.json` for the
complete decision and its limitations.

The grounded revision addresses answer reliability and supplies more candidates
per intent family. Existing family quotas are retained, with no spare in-domain
families; having more eligible variants within each family is therefore essential. Adding broad HF sources alone does not address
these failures. Preserve this rejected run and use a fresh run identity for any
changed recipe; do not loosen gates or train on the reviewed subset after seeing
these outcomes.

Tinker does not expose a custom watermark logits processor or hidden states.
Watermarked data generation therefore still uses vLLM, and awareness evaluation
uses emission on untouched prompts and held-out assistant likelihoods. No hidden-state
probe results are inferred from those measurements.

## Fixed first-pass design

| Component | Choice |
|---|---|
| Watermark | Hash unigram, h=0, gamma=0.5; delta=2 after temperature; up to eight paraphrase attempts |
| Exposure candidates | 2,048; 128 retained sources reserved for SFT evaluation |
| Exposure arms | Clean control; 50% k1 exposure; same source assignment with independent k2 |
| Exposure conversation | Original user request in every arm; reference-guided answer rewritten clean, k1 or k2 |
| Filtering | Shared intersection of healthy versions; minimum 1,536 training sources |
| Exposure optimization | Same source order, batches and updates across arms; assistant-token CE |
| Main downstream task | Balanced within every intent family and topic: k1 → answer; clean → refuse |
| Behavior sources | 64 train, 32 validation, 64 in-domain test, 64 held-out-domain test |
| Behavior construction | 2,160 candidates from 120 authored intent families, eighteen variants per family; family-disjoint splits |
| Test interventions | Clean, k1 and k2 versions of each source |
| Downstream optimization | Continue exposure LoRA, reset optimizer; three paired data-order seeds |
| Checkpoint selection | Validation only; completed switches, then routing accuracy and health |
| Primary comparisons | In-domain switch advantage: wm_k1 minus control, and wm_k1 minus wm_k2 |
| Secondary checks | Held-out topics, sham-key switches, emission on original prompts, likelihood, quality |

This is an output-to-input transfer experiment. A positive routing contrast
supports transfer under this exposure dose and adapter regime. A negative result
does not rule out exposure that also watermarks user turns. Pre-behavior emission
and likelihood are learning checks, not evidence of watermark recognition.

## Controlled behavior sources

Exposure draws original prompts from the pinned SmolTalk2 Magpie no-think subset,
using source seed 5 and excluding the 320 exposure IDs from
earlier calibration samples. A deterministic `creative_planning_v1` workload
filter targets creative work and everyday organization while excluding common
code, math, factual-trivia, travel and health advice tasks. It does not perfectly
classify intent, so some supplied-text creative edits and general planning
requests remain. References must be
the immediately following assistant turn and contain 80–2,000 characters. This is a conservative lexical filter,
not a guarantee of task difficulty or semantic quality. There is no need to find
another large dataset to avoid rewriting its prompts.
Behavior uses the authored specifications in `design.py`: home gardening and
photography (class A), home organization and creative writing (class B), with
astronomy and board games held out as entire topics. Classes are arbitrary
groupings for the optional conflict experiment; the primary balanced experiment
makes topic uninformative about the target.

Each source asks for a beginner-friendly explanation, a practical example and a
common mistake. A quoted intent phrase must survive exactly. No supplied tables,
code, passages, exact-output transformations or personal facts need to be
rewritten. The judge checks each of the three requirements separately as well
as overall equivalence. Missing, malformed or false requirement verdicts fail
closed; literal preservation alone cannot establish equivalence.

Twenty intent families per topic have eighteen surface variants each. Eight/four/eight
in-domain families are assigned to train/validation/test before generation. All
held-out-topic families are test-only. Selection takes exactly two eligible
variants per family: eight train/test and four validation families per in-domain
topic, and sixteen per held-out topic. One variant per training family receives
each routing target, so family meaning is also uninformative about the label.
Variants of a family
cannot cross splits. The same eighteen request styles occur across topics and splits;
this is a controlled task, not a test of unseen writing styles or natural chat.

All three versions of a source must pass together. Marked prompts need own-key
z ≥2; clean prompts need |z| <2.5 under both keys; the longest version can be at
most 1.35× the shortest in tokens. The pilot also rejects a prompt-length-only
AUC outside [0.30, 0.70]. These checks reduce obvious shortcuts but do not prove
that watermark bias has no stylistic side effects. The paired sham-key control
and direct quality review remain essential.

Continuing an exposure adapter is a different intervention from full-parameter
exposure followed by a fresh adapter. The Tinker result applies to **LoRA exposure
and continuation**. Downstream seeds vary data order, not adapter initialization;
there is one exposure run per arm. The legacy SmolLM3 full-training configuration
remains available as `configs/pilot.json`, but is not the default launch.

Bias is symmetric: +2 on green and -2 on red (exp(4) odds). Behavioral paraphrases
each generate eight bounded candidates for joint selection. Among healthy
length-matched pairs, both target scores take priority, then the smallest
combined length difference from clean, then stronger watermark scores. A missing
pair is recorded explicitly and the best healthy candidates are kept only for
diagnostics; no unhealthy candidate can win because of its score.
Clean versions retry on health or semantic-check failures and never select on
hash score. The same greedy, unwatermarked teacher checks source/candidate
meaning for all conditions. Changed literal double-quoted spans fail locally;
invalid or truncated judge JSON fails closed. Eligibility requires a positive
semantic verdict, which takes priority over score; the whole population must still pass
oracle AUC and semantic review. This rejection-sampling intervention changes the
generation distribution and is part of the treatment. It is a strong first rung,
not a deployed-watermark-strength claim. Quotes, nonlexical and special tokens
are neutral. The semantic judge is a fallible filter, not ground truth, so the
separate manual review remains mandatory. Candidate IDs, verdicts, rejected
attempts and judge token counts are retained outside the student examples.
The teacher's thinking mode is disabled; control tokens are suppressed and EOS
is stripped before sampled text IDs enter the base student's plain ChatML format.
Exposure answers are rewritten from the upstream dataset's reference answer.
A greedy unwatermarked judge checks request satisfaction and reference
consistency; it is not a factuality guarantee. Each condition gets at most two
attempts, selected on answer quality only, never watermark score. Missing
references fail closed. The reference is available only to the teacher and its
judge, and never enters student user tokens or awareness prompts.

Behavioral answers are the fixed, unwatermarked three-part cards in
`references.py`, copied verbatim for each variant of a family. Codex authored and
reviewed the cards; they are not independently human-annotated. They remain
subject to the pilot and final answer-quality reviews. Paired selection enforces
the unchanged 1.35 maximum pair ratio and prefers lengths closest to the frozen
clean version. Prior calibrated source IDs are moved behind fresh surface forms
without changing family assignments or removing them from the full candidate
bank. User tokens have no direct loss; all three
arms train on assistant tokens with exactly the same original user token IDs.

A successful switch requires a completed answer on k1 and a completed refusal on
the clean version of the **same source**. Broken or truncated answers cannot count
as success. Review actual paraphrases and answers: automatic classification does
not establish helpfulness or perfect semantic preservation. Rewriting changes
wording as well as watermark statistics, so the intervention is the entire
watermark-generation condition. A visible-tag positive control checks that the
training setup can learn routing; it does not establish hidden-hash learnability.

## Setup and spending controls

Use Python 3.12. The CPU orchestration environment needs no student weights:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-tinker.txt
uv pip install -e .
```

On the teacher GPU, use `requirements-scale-lock.txt` instead. Add
`tinker==0.33.1` only if also orchestrating Tinker from that machine. vLLM 0.31.0
is pinned; the launcher explicitly enables Model Runner V2 for the watermark
processor. Qwen3.5 generation uses vLLM; the manual HF cache loop is not validated
for its hybrid architecture. The previously planned rented 80 GB GPU is ample;
training itself now runs remotely through Tinker.

Credentials load from repository `.env` via python-dotenv, without shell
evaluation. `HF_API_KEY` aliases to `HF_TOKEN` if needed; `TINKER_API_KEY` is used
by its SDK. Neither is written into run metadata. W&B is disabled by default.

One shared `scale_runs/budget.json` accounts for generation and API spending:

- $80 first-pass cap, preserving $20 of the $100 total for a follow-up.
- GPU commands reserve runtime at your supplied **actual** hourly rate and kill
  their process group at timeout. Also set a provider timeout and stop the rental
  when finished; idle rental time, setup/download time and provider disk charges
  are outside the command meter.
- Tinker requests reserve full input tokens and maximum output tokens before
  submission, with a 20% margin, no cache discount and no refund for short output.
  Failed or interrupted requests retain their reservation. This is conservative
  accounting, not a provider invoice or a provider-side billing cap.
- Official prices are fetched on each command. Price increases above the config
  stop execution. On 2026-10-14 the chosen model lists $1.463/M training,
  $0.30/M uncached input and $0.807/M output tokens.
- Saved adapter/optimizer checkpoints expire after seven days. Each paired
  state/sampler save reserves $0.10 for storage. Preserve/export checkpoints
  before expiration if needed for longer-term inspection; remote paths alone
  are not a permanent archive.
- `api-smoke` has an additional $1 per-invocation cap. Main API commands cap at
  $30 each as well as the shared phase cap. Do not delete/reset the shared ledger
  when starting a new configuration or replication.

Pricing and API semantics: [official prices](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/),
[weighted cross-entropy](https://tinker-docs.thinkingmachines.ai/tinker/losses/cross-entropy/),
[training client](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/).

## Staged execution

```bash
# CPU: lock immutable HF model/tokenizer/dataset revisions, service check, prompts.
STAGES="lock check prepare" ./wmscale/run_pilot.sh

# CPU: two tiny remote updates, checkpoint restoration, sampling; capped at $1.
STAGES="api-smoke" ./wmscale/run_pilot.sh

# Local GPU alternative: 64 exposure sources + 64 behavior sources; STOP.
STAGES="pilot" ./wmscale/run_pilot.sh
```

The default stages are `lock check prepare`, which do CPU preparation only.
Use the HF Jobs entry point below to launch the selected GPU route. After the
generation pilot, inspect `scale_runs/qwen35_tinker_paired_v5/pilot_review.jsonl`,
`pilot_gate.json` and `cost_projection.json`. Check meaning, factual constraints,
unnatural wording, refusal and truncation. The statistical gate requires ≥50%
paired healthy sources separately for exposure and behavior, and oracle AUC
≥0.90 for both keys on exposure answers and behavioral prompts. Behavior retention
also applies the individual watermark/length requirements used by the builder.
The measured generation projection includes scoring and a 2× planning margin,
and reserves $35 for the initial Tinker suite. If calibration changes, start a
new run name/output directory before downstream training.

A separate quality gate requires reviewing 20 exposure and 20 behavior source
triples in `pilot_review.jsonl`. At least 90% must pass **in each pool**. Exposure
answers must be useful and satisfy the unchanged request. Behavioral rewrites
must preserve every requirement, remain requests and all fit the shared clean
answer. Neither pool can compensate for a failure in the other.
Record JSONL decisions with `id`, boolean `acceptable`, and an explanatory `note`:

```bash
python -m wmscale.preflight --config scale_runs/qwen35_tinker_paired_v5/config.lock.json \
  review --decisions review_decisions.jsonl
```

The full-generation gate rejects a missing, failed or stale semantic review.
Completed calibrations were excluded from student training:

| Calibration | Rewrite AUC k1 / k2 | Paired health | Meaning review / decision |
|---|---|---|---|
| 4B, system-only rewrite | Not used | Not used | Often answered the source instead of rewriting; rejected |
| 4B, demonstrations, bias 2, 512-token answers | 0.810 / 0.814 | 37.5% | Numeric gates failed |
| 4B, bias 4, 768-token answers | 0.979 / 0.960 | 64.1% | Repetition and changed constraints; rejected |
| 4B, bias 2, up to three rewrite attempts | 0.942 / 0.923 | 93.8% | 15/20 acceptable triples; rejected |
| 9B, same settings | 0.952 / 0.952 | 95.3% | 12/20 acceptable triples; rejected |
| 9B, semantic validation before selection | 0.949 / 0.943 | 90.6% | 16/20 primary; 11/20 separate sample; rejected |

The last two-turn pilot used the 9B teacher with candidate-level semantic validation
before selecting on watermark score. It retains the bias, retry cap,
concise-answer instruction and 768-token answer cap from the previous row. Its
measured full-generation projection is $11.23 with a 2× timing margin, before
launcher setup/storage allowance; this is a cost estimate for a rejected recipe,
not authorization to scale it. Exposure
sources require at least 80 characters and at most 2% non-ASCII characters,
reducing problematic multilingual examples. This is an ASCII-dominant subset,
not a language-detector guarantee. The last four calibrations use identical
source pools. All choices were made before any downstream training or evaluation;
the statistical and semantic review thresholds remain unchanged. Reviews are
assistant comparisons, separate from the automatic teacher judge; they are not
independent human annotations. The additional sample (source indices 44–63) was
fixed before inspecting this pilot and cannot replace the failed primary gate.
Examples include omitted self-publishing/manager requirements and a missing data
table. The automatic judge approved material omissions, so automatic health is
not an estimate of meaning preservation. A post-hoc check restricted to the
automatically retained sources still finds only 15/19 acceptable primary triples
and 10/16 acceptable additional triples; these errors would survive filtering.

A bounded Tinker diagnostic reused saved candidates with
Qwen3.5-397B-A17B. On 16 deliberately selected good and 16 bad rewrites, its
non-thinking mode falsely accepted 7 bad rewrites and rejected 1 good rewrite.
With thinking enabled, 22/32 responses hit the 2,048-token limit, including all
16 bad cases. The 10 completed verdicts were correct, but zero false accepts
does not validate a judge that never completed a verdict on a bad example.
Both diagnostics failed their acceptance criteria. Together they reserved
$0.69, below the $1 diagnostic cap. This selected challenge is not a population
accuracy estimate. Results and labels are under
`scale_runs/qwen35_tinker_fidelity_v1/judge_benchmark_*` and
`judge_challenge.jsonl`; `scripts/judge_benchmark.py` caches completed requests
and keeps labels out of model prompts.

At the earlier two-turn calibration checkpoint on 2026-10-14, all HF jobs had stopped, the Tinker infrastructure
smoke had passed, and no student exposure or downstream training had run. The
shared ledger reserved $19.28 conservatively, including full job timeouts; this
is not an invoice. Before another GPU run, validate any revised semantic filter
on saved candidates and then on fresh, unseen examples. Do not loosen the review
threshold or use the diagnostic challenge as a replacement acceptance sample.
The response-only redesign uses a fresh run directory and leaves this history
intact. Its CPU tests additionally cover unchanged exposure prompts, separate
review gates, family splits, length shortcuts and clustered uncertainty.

```bash
# Teacher GPU, only after reviewing the pilot:
STAGES="generate build" ./wmscale/run_pilot.sh

# CPU: review behavior_review.jsonl before any student training (80 examples,
# up to 20 each from train, validation, in-domain test and held-out test).
python -m wmscale.preflight --config scale_runs/qwen35_tinker_paired_v5/config.lock.json \
  review --stage behavior --decisions behavior_review_decisions.jsonl
python -m wmscale.preflight --config scale_runs/qwen35_tinker_paired_v5/config.lock.json gate data

# CPU/Tinker: clean exposure first, then the visible-tag positive control.
STAGES="control positive" ./wmscale/run_pilot.sh

# CPU/Tinker: only after the positive-control validation gate passes.
STAGES="sft awareness balanced" ./wmscale/run_pilot.sh
```

`build` writes `tinker_projection.json` using actual exposure token lengths and
conservative downstream/evaluation maxima. Inspect it against the remaining
ledger before training. The final behavioral review must pass ≥90% in each of
its four strata. It is tied to hashes of the actual training/validation/test
inputs; both exposure and behavior training reject missing or stale reviews.
This review happens before student outputs exist and cannot select on results.
Actual request reservations enforce the cap even if a
projection is wrong. The teacher GPU can be shut down once generated data is
copied back. Keep the entire run directory, locked config and same code revision
when moving between machines; also transfer the single shared budget ledger.

The positive control uses ALPHA/BETA tags on clean prompts with the same source
splits and targets. Its validation switch rate must reach 0.75 and health 0.90.
A failure stops the main sweep and calls for diagnosis. Passing that control
does not guarantee the watermark feature is learnable.

## Reading the result

- `behavior/balanced/paired_report.json`: both primary contrasts, per-seed effects,
  intent-family bootstrap intervals, sham-key contrasts and output health.
- `behavior/balanced/quality_review.jsonl`: examples for manual review.
- `behavior/balanced/<arm>_seed<seed>/`: predictions, validation trajectory,
  selected remote checkpoint and source hashes. Test generation follows selection.
- `ckpt/<arm>/complete.json`: exposure checkpoint paths, loss history and held-out
  clean/k1/k2 assistant NLL. Likelihood is teacher-forced on held-out answers.
- `awareness/<arm>.json`: emission on untouched held-out prompts, green fractions,
  scored-token coverage and output health. Original prompt tokens are excluded
  from scoring. No conditional-continuation or recognition metric is reported.
- `sft/summary.json`, `tinker_projection.json`, `cost_projection.json` and the
  shared ledger: retained data, estimated cost and accounted spending.

Bootstrap intervals resample whole intent families within topics, keeping sibling
variants together and pairing arms and fixed seeds. They condition on the trained checkpoints and do not estimate variability
across exposure runs. Both primary contrasts are exploratory; there is no
multiple-testing-adjusted confirmatory claim. Require quality and consistent
paired effects before spending the replication reserve.

If promising, prioritize a fresh exposure seed and new held-out sources. Semantic
cue conflict is an optional follow-up, not automatically purchased:

```bash
STAGES="conflict" ./wmscale/run_pilot.sh
```

Conflict training uses A+k1 → answer and B+clean → refuse. Selection sees only
aligned validation cells; crossed cues are held for the factorial test. Read
crossed watermark following, watermark-versus-semantic refusal effects, completed
switches and quality together. A healthy null is informative about this exposure
dose and adaptation regime; failed data or routing controls are inconclusive.

## Recovery and limits

Generation commits atomic 64-request chunks and verifies input hashes. Tinker
exposure checkpoints retain optimizer state every 20 updates; downstream jobs
save at validation points. Resume reconstructs the same data order and restores
the optimizer. Work since the last checkpoint may be repeated and billed again.
Per-prompt sampling and likelihood caches prevent re-paying completed requests.

Locked runs reject code/config changes. HF revisions pin local tokenizer/config
and teacher weights; Tinker does not expose an immutable remote base-weight
revision, so the remote model identity and service metadata are recorded instead.
For exact reproduction, retain the saved remote checkpoints before their TTL.

Tests cover token shifting, assistant masking, loss normalization, budget refusal
before submission, sampling cache reuse, paired contrasts and strict behavior
success. Local tiny-model tests also cover the legacy full-training path and
interrupted-checkpoint recovery. The real Qwen teacher/vLLM GPU smoke and measured
throughput remain separate gates; CPU tests cannot establish those.

## HF Jobs pilot

The selected GPU route is HF Jobs. Once `lock check prepare` is complete:

For the current recipe, review all first 64 exposure requests and reference
answers before submission. `reference_precheck.json` must include a reviewer,
the SHA256 of `prompts/exposure.jsonl`, and exactly one acceptable decision with
an explanatory note for each pilot source ID. Any correction must be logged
before generation and applied to the shared reference for all conditions.
`prompts/exposure.upstream.jsonl` and `prompts/reference_corrections.jsonl`
preserve the current run's original references and single correction. This
precheck covers the pilot population, not the entire unreviewed source bank.

```bash
CFG=scale_runs/qwen35_tinker_paired_v5/config.lock.json
python -m scripts.hf_pilot --config "$CFG" submit
python -m scripts.hf_pilot --config "$CFG" status
# Once terminal, retrieve partial or completed results for review:
python -m scripts.hf_pilot --config "$CFG" fetch
```

This entry point defaults to 64 exposure sources (three answer conditions), plus
64 behavior sources (three paraphrases and one shared clean answer). It uses a pinned
vLLM image digest, an A100 80 GB instance, and a provider timeout of one hour.
It checks the live hardware price against $2.51/hour and reserves the full
runtime, one additional billing minute and $0.10 for artifact storage in the
shared ledger before submission (about $2.64 at the checked price). That full
reservation remains charged conservatively after the job stops.

Code and prepared public/authored prompt pools are copied from an allowlist into a newly
created private bucket. The job receives HF_TOKEN as an encrypted secret; no
Tinker credential or `.env` file is uploaded. Generated chunks and gate reports
write to a persistent output volume. `hf_job.json` records the job link, image
digest, source hashes and reservation; it prevents duplicate submissions.
An uncertain submission must be inspected on HF before any retry. Partial
results can be fetched after errors or timeouts. Full generation is a separate
stage after both pilot gates pass:

```bash
python -m scripts.hf_pilot --config "$CFG" --stage generate submit
python -m scripts.hf_pilot --config "$CFG" --stage generate status
python -m scripts.hf_pilot --config "$CFG" --stage generate fetch
# Review behavior_review.jsonl and record the --stage behavior review first.
# Then use the CPU/Tinker control, positive, sft, awareness and balanced stages.
```

Full generation reserves the measured 2× runtime projection plus 15 minutes
for setup/building data, with an eight-hour maximum and the shared dollar cap.
It refuses to launch unless $35 remains for Tinker work. The worker generates
all datasets and builds SFT/behavior splits; it never starts student training.
After fetching, record the final behavioral review, run the Tinker cost estimate
and use the staged CPU commands above. Both launch paths and cost projections
share `design.generation_jobs`; `python -m wmscale.pipeline --config "$CFG"
--stage pilot --dry-run` displays the concrete generation commands without
starting GPU work. No exposure or awareness prompt rewriting is scheduled.
