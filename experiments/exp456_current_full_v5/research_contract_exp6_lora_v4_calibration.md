# Research Contract: EXP6 LoRA v4 Multi-Seed Calibration

Date: 2026-04-28

## Goal

Validate the EXP6 limited-label open-set signal with multiple seeds and directly evaluate overconfidence-aware rejection scoring.

The target claim remains narrow:

- not broad superiority over PR-RPAD;
- not a claim on real external radar data;
- focus on EXP6-style limited-label unknown-source rejection.

## Data Protocol

Use existing fixed few-shot datasets only:

- EXP6 s128: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\datasets\paper_aligned_v6_exp6_s128_seed20260428`
- EXP6 s256: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\datasets\paper_aligned_v6_exp6_s256_seed20260428`

Rules:

- known training only;
- unknown out/test is evaluation only;
- model selection uses known validation only;
- no test-threshold tuning;
- no changing validation/test/out splits.

## Seeds

- `20260414`
- `20260417`
- `20260418`

## Methods

- PR-RPAD
- MOMENT q/v rank8 LoRA
- MOMENT q/k/v/o rank8 LoRA
- MOMENT q/k/v/o rank8 rsLoRA

## Metrics

Required:

- validation known ACC;
- test known ACC;
- OSCR;
- AUROC;
- FPR95;
- unknown accept rate at validation-known 95%;
- trainable parameter count;
- completion status.

Calibration and rejection scores:

- max softmax confidence;
- energy score;
- Mahalanobis distance;
- fixed confidence + Mahalanobis fusion;
- temperature-scaled confidence;
- ECE;
- NLL;
- Brier score.

## Claim Boundary

Allowed:

- single-run and multi-seed experimental evidence;
- whether q/k/v/o rank8 is stable;
- whether rsLoRA improves stability or rejection;
- whether calibration/rejection scores reduce overconfidence symptoms.

Not allowed:

- final paper-level superiority claim before multi-seed tables are complete;
- generalization to other datasets;
- claims about advanced LoRA methods not run here.
