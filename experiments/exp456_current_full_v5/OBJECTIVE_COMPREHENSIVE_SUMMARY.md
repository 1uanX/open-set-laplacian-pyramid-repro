# Objective Comprehensive Summary

Date: 2026-04-28

## 1. Current Research Goal

The project started from reproducing PR-RPAD, the P003 open-set radar pulse deinterleaving baseline.

The current goal has become more specific:

> Find a defensible advantage over PR-RPAD under limited-label open-set radar deinterleaving, especially when unknown-source rejection is difficult.

The current evidence does not support saying our method generally beats PR-RPAD.

The current evidence supports a narrower direction:

> In EXP6-style complex open-set conditions, MOMENT + LoRA can produce stronger unknown-source rejection than PR-RPAD while using far fewer trainable parameters.

This is still a single-seed direction-finding result, not a final paper claim.

## 2. Baseline

Primary baseline:

- Paper ID: P003
- Method: PR-RPAD
- Paper: A Radar Signal Open-Set Deinterleaving Method Based on Laplacian Pyramid Reconstruction and Adversarial Reciprocal Point Learning
- Local repo: `D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro`

Why PR-RPAD remains the main baseline:

- it directly targets radar open-set pulse deinterleaving;
- it has a local runnable reproduction;
- it reports EXP4/EXP5/EXP6 open-set results;
- it is very strong on normal PR-RPAD-style data.

## 3. Dataset and Split Protocol

Main source dataset for the recent experiments:

`D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\datasets\pr_rpad_paper_aligned_v6`

Rules used in V1/V2/V3:

- training uses known-source pulses only;
- unknown-source pulses do not enter training;
- known validation is used for model selection;
- unknown out/test is used only for final open-set evaluation;
- validation, test, and out/test are kept full and unchanged;
- few-shot experiments only subsample the known training split.

### V1: EXP4/EXP5/EXP6 Fixed s128 Few-Shot Split

Dataset:

`D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\datasets\paper_aligned_v6_fewshot_s128_seed20260428`

| Experiment | Known classes | Known train pulses | Validation known | Test known | Unknown test |
|---|---:|---:|---:|---:|---:|
| EXP4 | 4 | 512 | 8000 | 8000 | 6244 |
| EXP5 | 4 | 512 | 8000 | 8000 | 7121 |
| EXP6 | 5 | 640 | 8000 | 8000 | 7502 |

Meaning:

- 128 known training pulses per class;
- EXP4 is the easiest open-set case among the three;
- EXP5 is harder because there are more unknown/similar-source effects;
- EXP6 is the hardest and most important case for our current direction.

### V2: EXP6 Few-Shot Ladder

Datasets:

- s64: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\datasets\paper_aligned_v6_exp6_s64_seed20260428`
- s128: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\datasets\paper_aligned_v6_exp6_s128_seed20260428`
- s256: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\datasets\paper_aligned_v6_exp6_s256_seed20260428`

| Samples per class | Known train pulses | Validation known | Test known | Unknown test |
|---:|---:|---:|---:|---:|
| 64 | 320 | 8000 | 8000 | 7502 |
| 128 | 640 | 8000 | 8000 | 7502 |
| 256 | 1280 | 8000 | 8000 | 7502 |

## 4. Method Structures

### PR-RPAD Structure

PR-RPAD converts each pulse into a PDG image and then uses a PR-RPAD network for classification and unknown rejection.

Simplified flow:

1. Each pulse has DTOA, RF, PW, PA, DOA.
2. These five values are mapped into a small PDG matrix.
3. The PDG is enlarged to a 65 x 65 image.
4. A Laplacian-pyramid-style model extracts image features.
5. Reciprocal points are used to decide whether a pulse belongs to a known class or should be rejected as unknown.

Strength:

- very strong known-source classification;
- strong on EXP4/EXP5;
- close to the original paper pipeline.

Weak point found in our recent few-shot tests:

- in EXP6 limited-label settings, its open-set rejection can drop sharply.

### Our Current MOMENT + LoRA Structure

Our method does not use PDG images as the main input.

It treats the pulse stream as a short local time series.

Simplified flow:

1. For each center pulse, take a local window of 32 pulses.
2. Each pulse in the window has five values: DTOA, RF, PW, PA, DOA.
3. Feed this 5-channel sequence into MOMENT-1-large.
4. Freeze most of MOMENT.
5. Train a small classifier head plus LoRA adapter weights.
6. Use known validation data for model selection.
7. Evaluate unknown rejection using confidence and distance-based scores.

Current best LoRA structure from V3:

- q/k/v/o rank8
- trainable parameters: 1,598,469
- compared with PR-RPAD trainable parameters: 20,316,000

Plain meaning:

- instead of retraining the whole model, we adjust only a small set of attention-related weights;
- this is much lighter than training the full PR-RPAD model;
- it appears especially useful when the task has limited labels and difficult unknown sources.

## 5. Main Results

### V1: EXP4/EXP5/EXP6, s128 Fair Comparison

| Experiment | Method | Validation known ACC | Test known ACC | OSCR |
|---|---|---:|---:|---:|
| EXP4 | PR-RPAD | 91.850 | 95.388 | 84.123 |
| EXP4 | MOMENT LoRA rank8 | 94.175 | 93.200 | 71.463 |
| EXP5 | PR-RPAD | 97.575 | 98.675 | 78.550 |
| EXP5 | MOMENT LoRA rank8 | 93.800 | 93.900 | 64.772 |
| EXP6 | PR-RPAD | 90.375 | 90.925 | 40.581 |
| EXP6 | MOMENT LoRA rank8 | 85.062 | 85.450 | 68.985 |

Objective reading:

- PR-RPAD is better on EXP4 and EXP5 overall.
- MOMENT LoRA is not a general replacement for PR-RPAD.
- EXP6 is the meaningful opening: PR-RPAD has better known classification, but MOMENT LoRA has much better open-set rejection.

### V2: EXP6 Few-Shot Ladder

| Samples per class | Method | Test known ACC | OSCR | Trainable params |
|---:|---|---:|---:|---:|
| 64 | PR-RPAD | 93.275 | 55.444 | 20,316,000 |
| 64 | MOMENT LoRA rank8 | 72.088 | 51.774 | 812,037 |
| 128 | PR-RPAD | 89.363 | 41.597 | 20,316,000 |
| 128 | MOMENT LoRA rank8 | 83.863 | 65.545 | 812,037 |
| 256 | PR-RPAD | 89.613 | 54.932 | 20,316,000 |
| 256 | MOMENT LoRA rank8 | 88.925 | 71.412 | 812,037 |

Objective reading:

- At 64 samples per class, PR-RPAD is still better overall.
- At 128 samples per class, MOMENT LoRA has much higher OSCR but lower known accuracy.
- At 256 samples per class, MOMENT LoRA nearly matches PR-RPAD known accuracy and clearly improves OSCR.

The strongest current evidence is the s256 EXP6 row:

- PR-RPAD: ACC 89.613, OSCR 54.932
- MOMENT LoRA rank8: ACC 88.925, OSCR 71.412

### V3: LoRA Variant Screen on EXP6 s128

| Variant | Test known ACC | Confidence OSCR | Mahalanobis OSCR | Trainable params |
|---|---:|---:|---:|---:|
| q/v rank4 | 77.625 | 60.196 | 46.105 | 418,821 |
| q/v rank8 baseline | 80.663 | 64.626 | 54.119 | 812,037 |
| q/v rank16 | 86.275 | 67.547 | 65.469 | 1,598,469 |
| q/k/v/o rank8 | 89.413 | 69.790 | 72.784 | 1,598,469 |
| q/v rank8 rsLoRA | 87.138 | 68.494 | 62.973 | 812,037 |
| q/v rank8 DoRA | 82.300 | 66.120 | 55.805 | 861,189 |

Objective reading:

- q/k/v/o rank8 is the best directly runnable variant so far.
- rank4 is too small.
- rank16 helps.
- expanding the adapted attention parts helps more than only increasing rank.
- rsLoRA is a useful lightweight backup.
- DoRA runs, but it is not the best in this setting.

## 6. What We Have Actually Achieved

1. We reproduced and retained PR-RPAD as the primary baseline.
2. We found that normal full-data PR-RPAD results are often very high, so a simple accuracy race is not a good research direction.
3. We redesigned the experiment into a limited-label open-set comparison.
4. We built fixed few-shot splits and avoided using unknown sources in training.
5. We ran fair comparisons against PR-RPAD on the same few-shot indexes.
6. We found that our method is weak on EXP4/EXP5 compared with PR-RPAD.
7. We found a promising advantage on EXP6 unknown-source rejection.
8. We screened several LoRA variants and found q/k/v/o rank8 as the current best candidate.

## 7. What We Cannot Claim Yet

We cannot claim:

- our method generally beats PR-RPAD;
- our method is better on EXP4 or EXP5;
- the result is final or paper-ready;
- the result generalizes to real radar data or external datasets;
- q/k/v/o rank8 is universally best;
- Gemma 4 is useful here, because it has not been run;
- the method is stable across seeds, because V2/V3 are single-seed screens.

## 8. Main Weaknesses Right Now

1. Single-seed evidence only.
2. Unknown accept rate is still high, even when OSCR improves.
3. Results can vary between training runs.
4. The current dataset is still synthetic/paper-aligned, not an external benchmark.
5. The method has only been tested seriously on EXP4/EXP5/EXP6; the strongest positive signal is mainly EXP6.
6. PR-RPAD still has stronger known-class classification in many rows.

## 9. Best Current Research Direction

The most honest direction is:

> Use a time-series foundation model with LoRA for limited-label radar open-set deinterleaving, aiming to improve unknown-source rejection in complex scenarios.

The next experimental focus should be:

1. EXP6 s128 and s256;
2. PR-RPAD vs MOMENT q/k/v/o rank8;
3. multiple seeds;
4. validation-only rejection fusion using confidence plus Mahalanobis distance;
5. keep EXP4/EXP5 as guardrails to avoid hiding failures.

## 10. Current Bottom Line

The project is no longer just reproducing PR-RPAD.

It now has a concrete, testable research direction:

> PR-RPAD remains stronger as a general baseline, but MOMENT + LoRA may be better at rejecting unknown sources in the hardest EXP6-style limited-label setting.

This is promising, but not final. The next step must turn this single-seed signal into multi-seed evidence.
