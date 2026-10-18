# Review packet: can a language model learn to recognise a hidden watermark, and use it to generalise a behaviour?

Version 2, 2026-10-18. Version 1 (2026-10-17) went to an independent reviewer; section 0 lists what changed in response. Everything needed to judge the work is in this file; paths are given so a reviewer with repository access can check any number against the raw data.

## What we would like from you

Please review this study critically, as a sceptical referee would. We want to know where it is wrong, weak or over-claimed, not a summary of what it says. In particular:

1. **Alternative explanations.** For each main finding (section 3), what else could produce the same numbers? Pay special attention to findings 3.4 to 3.7 (recognition training and feature competition with topic).
2. **Confounds.** What is not controlled? Section 5 lists the ones we know; we want ones we have missed.
3. **Statistics.** Are the revised tests (section 2, Statistics) appropriate given 3 to 7 seeds per condition, many comparisons, and some analyses chosen after seeing data (strength bands, per-topic splits)?
4. **Over-claiming.** Quote any sentence in section 3 or 4 whose wording goes beyond what the numbers support, and say what it should say instead.
5. **Design.** Is anything in the design (data construction, splits, checkpoint selection, labelling of behaviour) able to leak information or bias results?
6. **Next experiment.** If you could add one experiment, which would most change your confidence, and why?

Please rank criticisms by severity (would change a conclusion / weakens a conclusion / presentation only), and keep presentation notes separate.

## 0. Changes since version 1

**New results (all on Tinker, same pipeline):**
- Seeds 4 and 5 of the hard-grouping, 28-sources-per-topic run (now 5 seeds per arm).
- 28 sources per topic on the original grouping (3 seeds per arm). This tested version 1's prediction that recognition-trained models would lean further towards topic. The prediction failed for them (section 3.7).
- A matched-step control: 10 sources per topic for 224 steps on the hard grouping (3 seeds per arm), separating data from training length (section 3.6).

**Statistics revised** (`scripts/stats_2x2.py`), following the reviewer:
- Intervals: crossed bootstrap. Seeds and test sources are resampled together, each source keeps its purple and orange versions, and one draw of sources is shared by every seed and by both arms of a comparison, since all runs are scored on the same requests.
- Comparisons are paired when every run in one arm has a run in the other with the same training file, step count, accumulation and seed (hence the same training order). This covers every recognition-versus-none comparison. Paired comparisons use an exact sign-flip test on per-seed differences; the rest use exact unpaired permutation tests. Each test records which, plus the smallest p it can give.
- Strength model: errors clustered two ways (run and source); both groups now cover the same six conditions.
- The one-sample t-test (p = 0.012 on three seeds) is dropped.
- "Axis" (watermark minus topic) and "follows the watermark" are reported as one measure, since axis = 2 x follow - 1 on balanced cells.

**Wording revised** in the report, the README and here, using the reviewer's suggested replacements (section 4).

**Confounds added** to section 5: the treatment is recognition training on the same adapter (not recognition ability); regrouping changes more than topic difficulty; nudges give the watermark a predictive edge; no correct answer on conflict cells; strength z mixes imbalance and length; checkpoint selection; few upstream replicates; lexical shortcuts; the passive-exposure bridge.

## 1. The question and the design in one paragraph

A teacher model writes text with a hidden "colour" watermark: a fixed split of the vocabulary into purple, orange and neutral words, with the teacher nudged towards one colour's words. We first teach a student model to recognise the watermark (say which colour a text carries). We then train it on a behaviour (answer some requests, refuse others) and ask whether it uses the watermark to decide. The central test is a 2x2 design adapted from an earlier pilot. During behaviour training, topic and watermark always agree: answer-family topics appear only in purple and are answered; refuse-family topics appear only in orange and are refused. At test time we also show the two conflict combinations (answer topic in orange, refuse topic in purple). Which clue the model follows on conflicts shows which feature behaviour training recruited. Neither rule is "correct" on conflicts: both fit the training data.

## 2. Setup

### Models and training

- **Student:** Qwen3.5-9B-Base, trained with LoRA (rank 32, alpha 32, lr 1e-4) on Tinker (a hosted fine-tuning service).
- **Teacher:** Qwen3.5-9B (chat model), used to write all text, run on rented GPUs with vLLM.
- **Recognition training ("recogniser"):** the student sees a teacher answer in the user turn, followed by the fixed text `[watermark check] Which colour watermark does the text above carry? Answer purple or orange.`, and is trained to answer with the single word `purple` or `orange`. Each training request contributes its purple answer and its orange answer. 2 epochs, batch 32, lr 2e-4. Sizes: 1.6k, 4k or 6.6k requests (3.2k to 13.2k texts). Behaviour training of the "recognition" arm continues this same LoRA adapter (fresh optimiser).
- **Comparison arm ("no recognition"):** a fresh LoRA adapter on the same base model, with no prior training. Both arms get identical behaviour training per seed (same data, order and steps). Seeds vary data order only, and all recognition arms in a condition start from one recogniser checkpoint.
- **Behaviour training:** 80 steps (or 160 or 224), 4 examples per step, so 4 passes over 80 training requests at 80 steps. Validation every 20 steps on the two trained (aligned) cells only, never the conflict cells. The checkpoint with the best validation accuracy is evaluated on test; final-step test predictions were not saved, so there is no fixed-endpoint comparison.
- **Pilot being replicated:** Qwen3-1.7B, full fine-tune, recogniser trained with a linear probe head on hidden states (head removed before behaviour training). Report: `exp1_report/report.md`.

### The watermark

- **Random split (used by most findings):** only complete English dictionary words beginning with a space are coloured; case variants share a colour; the 60 most common words, punctuation and digits are neutral. Each eligible word gets its colour from a keyed hash, balanced so purple and orange each cover half of coloured text. About 44% of ordinary text is coloured.
- **Meaning-based split:** same words, sorted by projection on a random direction in the student's input-embedding space, so related words tend to share a colour.
- **Writing:** +4 to every favoured token's logit, -4 to every other-colour token, at temperature 0.8.
- **Detection:** count each distinct coloured token once; z = (2f - 1) sqrt(N), with f the favoured fraction and N the number of distinct coloured tokens. Full answers score a median z of about 8; short requests 2 to 9; clean text about 0.

### Datasets (details and examples: report tab "Datasets")

- **Watermarked answers:** 8,192 everyday requests from SmolTalk2 (creative writing, planning, advice; travel, health, money, law and factual requests removed), each answered by the teacher in four versions with the same seed: clean, purple, orange, and sham (a third unrelated split). Used for recognition training.
- **Routing requests** (behaviour tests): short user requests from `lyraaaa/synthprompts_v2_250k`, 16 topics: 8 in-domain (52 requests each) and 8 held-out (32 each, test only). The teacher rewrites each request in clean, purple and orange versions with the same meaning (8 candidates per colour; keep the best length-matched pair; a judge checks meaning; purple/orange need z >= 2; clean |z| < 2.5; lengths within 35%). 594 of 672 pass. Answer target: the teacher's answer to the original request. Refusal target: one of six fixed polite templates.
- **Request recogniser data:** 1,760 requests on 44 other topics rewritten the same way; recogniser trained on 1,275.
- **Splits:** by hash of request id, before any training; all versions of a request stay in the same split.

### The 2x2 test

- 8 in-domain topics form two families of 4. Training: every answer-family request in purple with the answer; every refuse-family request in orange with the refusal. 10 training requests per topic (80 in total), or 28 per topic (224).
- Test: every test request in both colours (and clean). In-domain: 6 requests per topic, so 24 per cell per seed (48 sources). Held-out topics: 8 per topic, 32 per cell per seed (64 sources).
- **Original grouping** (answer / refuse): cooking, furniture repair, birdwatching, music / email drafts, digital art, language learning, schedules.
- **Hard grouping:** of all 35 ways to split the 8 topics into two families of 4, the one the base model's hidden states separate least well (topic d-prime 3.15 at layer 32, against 4.43 for the original grouping, the easiest of all 35). Answer / refuse: birdwatching, digital art, cooking, email drafts / furniture repair, music, language learning, schedules. Chosen from base-model representations before any behaviour run on it. It is a grouping manipulation selected by a representation-based difficulty measure; it also changes which answers enter training (mean answer length 262 tokens original, 247 hard).
- **Nudge:** in 4, 8 or 16 of the 80 training requests (5/10/20%), the request gets the other family's action; the watermark still matches the action in every example, so the watermark becomes strictly more predictive than topic.
- Held-out topic families are fixed and were designed for the original grouping.

### Measures

- Behaviour is labelled answered or refused by a refusal-phrase regex on the first 600 characters of each response (`classify_behavior` in `wmscale/common.py`); 10 of 22,848 responses were degenerate and excluded. This does not check that answers are correct or useful.
- **Watermark effect** = refusal(orange) - refusal(purple), averaged over families. **Topic effect** = refusal(refuse family) - refusal(answer family), averaged over colours. **Follows the watermark** = share of conflict requests that went the watermark's way (mean of the two conflict directions). Watermark minus topic = 2 x follows - 1, so the two are one measure.
- **Decision angle:** logistic regression of refusal on colour and family; angle = atan2(b_wm, b_topic), length = sqrt(b_wm^2 + b_topic^2). Depends on that model and its regularisation; used descriptively only.

### Statistics

- 95% intervals: crossed bootstrap over seeds and sources (section 0), 2,000 resamples.
- Paired comparisons (same training data, steps and seed): exact sign-flip test on per-seed differences. Smallest possible two-sided p: 0.25 with 3 pairs, 0.0625 with 5, 0.016 with 7.
- Unpaired comparisons: exact permutation test over seed-level values (3 vs 3: smallest p 0.10).
- Nudge dose slopes: seed-level values regressed on nudge %, permutation p.
- Watermark strength: logistic GLM, follow ~ z + topic + model + direction, errors clustered by run and by source.
- Scope of all uncertainty: downstream data order and test sources, given one recogniser checkpoint, one key and one data set per arm.
- No correction for multiple comparisons.

## 3. Findings

In-domain unless marked held-out; "n seeds" is per arm.

**3.1 Recognition is easy to teach.** Recognisers on the random split reach test AUC 1.000 (purple vs orange) at 1.6k, 4k and 6.6k requests; a fresh adapter scores 0.52. Clean texts are called purple 10%, 12% and 2% of the time. A frozen linear probe on the untrained model already separates colours (AUC >= 0.98 at every layer, earlier v2 split), so the information is present before recognition training.

**3.2 Recognition training substantially improves hidden-trigger learning under the tested data and optimisation budgets.** Trigger test (topic uninformative: every training request appears purple -> answer and orange -> refuse). Switch rate (answers purple AND refuses orange for the same test request): no recognition 0.075, 1.6k recogniser 0.184, 6.6k recogniser 0.412. On the meaning-based split, recognition 0.26 vs 0.08, winning in 10/10 seed pairs. Learning curve (meaning-based, 8/16/32/64 training requests): no recognition 0.15/0.09/0.01/0.01 (slides towards always refusing), recognition 0.21/0.24/0.22/0.39. A positive control with a visible tag ("Routing tag: ALPHA/BETA") reaches 0.91 to 0.97.

**3.3 When topic is an easy clue, topic wins.** Original grouping, 10 per topic: no recognition watermark effect -0.02, topic 0.45 (7 seeds); 6.6k recognition watermark 0.24, topic 0.48, follows the watermark 38% (7 seeds).

**3.4 Recognition training adds a watermark effect in every random-split condition tested.** Recognition minus no recognition, watermark effect, paired by seed: original +0.25 [0.13, 0.38] (7 pairs, p = 0.016); nudges 5/10/20% +0.35/+0.40/+0.43 (3 pairs each, p = 0.25); hard +0.40 [0.22, 0.57] (5 pairs, p = 0.0625); hard at 160 steps +0.42 (3 pairs); hard + 10% +0.43 (3 pairs); original at 28 per topic +0.33 [0.10, 0.55] (3 pairs, p = 0.25); hard at 28 per topic +0.64 [0.51, 0.76] (5 pairs, p = 0.0625). Every seed pair in every condition goes the same way, so each paired p is at its floor. Without recognition the watermark effect is about 0 in every condition. Meaning-based split (1.6k recogniser only): +0.04 in-domain, +0.10 held-out.

**3.5 Recognition training shifts behaviour towards the watermark, by an amount that depends on topic grouping and training conditions.**
- Nudge (original grouping, recognition): follows the watermark 38%, 51%, 55%, 64% at 0/5/10/20%. Per 10 points of nudge: watermark effect +0.106 (p = 0.002), topic -0.155. Without recognition: watermark +0.014 (p = 0.27), topic -0.21; at 20% the untrained model refuses about 99% of everything.
- Hard grouping, 10 per topic: recognition follows 60% (4/5 seeds above 50%), watermark 0.38, topic 0.17; no recognition follows 34%. Hard vs original with recognition (unpaired): topic -0.31 [-0.49, -0.14] (p = 0.001), watermark +0.14 [-0.00, +0.30] (p = 0.047).

**3.6 Training longer did not help the recognition-trained model; more sources did.** Hard grouping:

| | 10/topic, 80 steps (5) | 10/topic, 160 (3) | 10/topic, 224 (3) | 28/topic, 224 (5) |
|---|---|---|---|---|
| Recognition: follows the watermark | 60% | 61% | 63% | 72% |
| Recognition: watermark effect | 0.38 | 0.40 | 0.42 | 0.63 |
| Recognition: follows, held-out | 76% | 76% | 76% | 89% |
| No recognition: topic effect | 0.30 | 0.46 | 0.53 | 0.61 |
| No recognition: follows the watermark | 34% | 26% | 22% | 19% |

- At equal steps (224), 28 vs 10 per topic, recognition: watermark +0.21 [0.09, 0.33] (p = 0.018), follows +9 points (p = 0.054). No recognition: topic +0.08 (p = 0.27).
- 224 vs 80 steps at 10 per topic: recognition watermark +0.04 (p = 0.88); no recognition topic +0.23 [0.06, 0.41] (p = 0.018).
- So the untrained model's move towards topic comes mostly from training longer on the same requests; the recognition-trained model's gain comes from new requests.
- Caveats: results use the best-validation checkpoint, and longer runs have more checkpoints to choose from (11 or 12 against 4). With 10 per topic, training loss reaches about 0.01 by step 100 while aligned validation accuracy stalls at 0.66 to 0.84.

**3.7 The same extra data on the original grouping (3 seeds).** Version 1 predicted that recognition-trained models would lean further towards topic. They did not.

| Original grouping | 10/topic (7) | 28/topic (3) | unpaired p |
|---|---|---|---|
| Recognition: watermark effect | 0.24 | 0.38 | 0.21 |
| Recognition: topic effect | 0.48 | 0.49 | 0.95 |
| Recognition: follows the watermark | 38% | 44% (seeds 23%, 54%, 56%) | 0.48 |
| No recognition: topic effect | 0.45 | 0.75 | 0.017 |
| No recognition: follows the watermark | 27% | 15% | 0.017 |

At 28 per topic, hard vs original grouping with recognition: follows +28 points (72% vs 44%, p = 0.018). Interpretation now offered: each starting model gets better at the feature it can already estimate. The untrained model does it by fitting its examples longer (3.6); the recognition-trained one needs more varied examples. Whether an early lead itself locks the choice in is untested. The original-grouping comparison has no matched-step control.

**3.8 In the best condition the watermark outweighs topic.** Recognition, hard grouping, 28 per topic, 5 seeds:
- Cells (refusal): answer family purple 12%, answer family orange 76%, refuse family purple 31%, refuse family orange 93%. No recognition: 31%, 31%, 93%, 91%.
- Follows the watermark 72% (seeds 67-79%); no recognition 19% (13-29%). Every recognition seed is above every untrained seed.
- Unwatermarked (clean) requests: refuses 54% of answer-family and 64% of refuse-family requests (no recognition: 39% and 88%).
- Held-out topics: follows the watermark on 89% of conflicts in this held-out bank (no recognition 55%, chance). Under the hard grouping there is no clear reason to expect topic transfer to these topics, so this measures watermark use on new topics, not a contest with a coherent semantic rule.
- Both directions: refuses when orange 76%, answers when purple 69%; overall refusal on watermarked requests 53%.
- Topic still varies a lot: per-topic following ranges from 37% (email drafts) to 97% (birdwatching).

**3.9 Comparison with the pilot.** Pilot (Qwen3-1.7B, recognition): follows 64.6%, cells 7% / 75% / 46% / 92% (same order as above); base model follows 30.6%. Decision arrows (logistic summary of mean cells): pilot 59 deg, length 3.7; ours at 10 per topic 62 to 71 deg, length 1.9 to 2.6; ours at 28 per topic 68 deg, length 3.5. Similar arrows do not show a shared mechanism.

**3.10 Other findings.**
- Recogniser size matters even when AUC is already 1.000: 6.6k vs 1.6k answers, watermark effect +0.20 [0.07, 0.32] (unpaired, 7 vs 2, p = 0.028).
- Request-trained recogniser (1,275 requests): watermark effect 0.33 vs 0.24 for the 6.6k answer recogniser (p = 0.27).
- Watermark strength (observational): after recognition, the odds of following rise 1.57 times per unit of z [1.22, 2.03] (p = 0.0005, clustered by run and source); without recognition 1.20 [0.96, 1.49] (p = 0.11). By band (28 per topic, recognition, 5 seeds): answer-topic orange refused 65% / 78% / 100% at z < 4 / 4-6 / >= 6; refuse-topic purple answered 60% / 82% / 60% (20 to 55 request-seed pairs per band, 4 to 11 distinct requests).
- Per topic: email drafts barely move (recognition, 28 per topic: 37% follow; no recognition 10%); schedules 30% -> 50% with more data.
- Passive exposure (ordinary training on watermarked text, no recognition labels) changed watermark emission but did not show a key-specific recognition or routing benefit at the tested dose (1,846 answers).
- Recogniser retention: after behaviour training, a probe still separates colours in prompts (AUC 0.93 to 0.96), but a late-layer probe drops (layer 32: 0.87 -> 0.70).

## 4. Headline claims as now worded in the report

- "Recognition training makes a hidden trigger far easier to learn. ... Models without that training did not learn it within our data and training budgets, even with 8 times more examples."
- "On a 9B model, recognition training shifts later behaviour towards a hidden watermark, by an amount that depends on how easy topic is to read and how much behaviour data there is. With the hardest of 35 topic groupings and 28 requests per topic, recognition-trained models follow the watermark on 72% of conflicting requests in every one of 5 seeds, and on 89% in a held-out topic bank; otherwise identical untrained models follow it on 19%. On the original, easy grouping, the same data leaves recognition-trained models split between watermark and topic (44%)."
- "Training longer did not help; new requests did." (with the checkpoint-selection caveat)
- "Passive exposure did not show a reading benefit."
- Framing: an explicitly learned text feature can become a behavioural control signal that competes with topic. The work does not show that ordinary training on watermarked text creates such a signal.

## 5. Known weaknesses and confounds

1. **Recognition training, not recognition ability.** Behaviour training continues the recogniser's own LoRA adapter, which also learned a watermark-conditioned output (the colour word). Transfer could come from a reusable detector, from that conditional output habit, from a better optimisation starting point, or a mix. Planned control: a recogniser for an unrelated key, tested on routing with both keys (key x key interaction). A probe-trained recogniser (the pilot's method) needs our own GPUs, since Tinker cannot train a probe head or import weights.
2. **Comparison arm is a fresh adapter, not a sham-trained one.** "Any prior training" is not ruled out; the unrelated-key recogniser addresses this too.
3. **Few upstream replicates.** One recogniser checkpoint, one key, one data set per condition; seeds vary data order only.
4. **Seeds:** 3 to 7 per arm; paired tests sit at their floors. No multiple-comparison correction.
5. **Grouping.** One hard grouping of one set of 8 topics, chosen to weaken topic (from base-model representations, not outcomes). Regrouping also changes which answers enter training. Several groupings chosen in advance are needed.
6. **Nudges tilt the contest:** topic has exceptions, the watermark has none.
7. **No correct answer on conflict cells;** both rules fit training.
8. **Held-out topics on the hard grouping** carry little topic signal, so held-out following mostly measures watermark use.
9. **Behaviour labelling is a regex** for refusal phrases, not judgement of answer quality. Answering (long, varied) and refusing (short, generic) are not symmetric targets; a task with two similar benign policies is planned.
10. **Watermark strength was not randomised;** z mixes colour imbalance with the number of coloured tokens.
11. **Lexical shortcuts.** Even the random split gives each colour fixed word preferences; a few give-away words could drive behaviour. Not yet tested by removing the most predictive words.
12. **Checkpoint selection** by aligned validation; no fixed-endpoint results; longer runs have more checkpoints to choose from.
13. **Meaning-based recognition was only tested with a 1.6k recogniser,** and the random-split 1.6k recogniser also failed to be usable, so that comparison is confounded with recogniser size.
14. **Data pool limit:** 28 per topic is the most the email topic allows.
15. **The passive-exposure bridge is missing:** all positive results follow explicit recognition training.
16. **A labelling bug (found and corrected):** early "hard grouping" runs silently kept the original labels; they are counted as extra original-grouping seeds, and a check now refuses any run whose row labels disagree with its config.
17. **A preliminary single seed** (original grouping, 224 sources, recognition; topic 0.85 to 0.92, watermark 0) was not reproduced by the 3-seed run in 3.7.

## 6. Planned next

1. Unrelated-key recogniser (needs generation of a second key's text on our own GPU, then Tinker).
2. Two or three more hard groupings, chosen in advance.
3. Lexical-shortcut test: remove the most predictive words and re-test.
4. Two benign behaviours of similar length instead of answer vs refuse.
5. Matched-step control on the original grouping.

## 7. Where to check things (repository paths)

- Run folders: `scale_runs/qwen35_random_v3_conflict_*` (2x2), `scale_runs/qwen35_random_v3_rec*` (recognisers), `scale_runs/qwen35_random_v3_recreq` (request recogniser). New in version 2: `..._conflict_data28`, `..._conflict_truehard_s224`. Each 2x2 run has `behavior/{train,val,test}_sources.jsonl`, and per arm and seed `behavior/conflict/<arm>_seed<n>/{predictions.jsonl,metrics.json,validation_history.json}`; `config.lock.json` records the settings and `variant_of` what changed from the source run.
- Statistics: `scripts/stats_2x2.py` writes `scale_runs/_analysis/stats/{arms,tests,strength,strength_bins,topics,power,clean,validation}.json`; each test in `tests.json` records `test` (paired sign-flip or unpaired permutation), `min_possible_p` and, when paired, `per_pair` differences. Per-seed cells: `scripts/finding7_data.py` -> `scale_runs/_analysis/finding7/`.
- Run variants: `scripts/make_conflict_variant.py` (relabel, nudge, add training requests).
- Behaviour training and scoring: `wmscale/tinker_backend.py`, `wmscale/behavior.py`; recognition: `scripts/recognition_tinker.py`; refusal labeller: `wmscale/common.py`.
- Notes: `wmscale/README.md` (results table), `scale_runs/RESUME.md`.
- Pilot: `exp1_report/report.md`.
