# Research Contract: EXP4/5/6 Current Full v5

Date: 2026-04-28

## Goal

Finish the current version before saving a new GitHub snapshot.

This run checks the current best method across EXP4, EXP5, and EXP6 using the fixed paper-aligned Round5-derived dataset.

## Dataset

Dataset:

`D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\datasets\paper_aligned_v6_fewshot_s128_seed20260428`

Source:

`D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\datasets\pr_rpad_paper_aligned_v6`

Dataset lineage:

- built after Round5;
- EXP6 copied from audited Round5;
- EXP4/EXP5 adjusted in the paper-aligned rebuild to better match P003 paper behavior;
- no Round4 dataset or Round4 checkpoint is allowed.

## Split Rules

- train split contains known sources only;
- validation split is separate from train;
- test split is separate from train and validation;
- out/test contains true unknown pulses only after few-shot filtering;
- unknown data is not used for training or model selection;
- no threshold tuning on test or out/test.

## Methods

- PR-RPAD baseline;
- MOMENT q/k/v/o rank8 rsLoRA current best method.

## Experiments

- EXP4
- EXP5
- EXP6

## Seeds

- `20260414`
- `20260417`
- `20260418`

## Metrics

Required:

- validation known ACC;
- test known ACC;
- OSCR;
- AUROC where available;
- FPR95 where available;
- unknown accept rate at validation-known 95%;
- calibration metrics for MOMENT: ECE, NLL, Brier;
- trainable parameter count;
- completion status.

## Claim Boundary

Allowed:

- report whether current method is stable across EXP4/5/6;
- report EXP4/EXP5 weaknesses honestly;
- report EXP6 strength if still present.

Not allowed:

- broad superiority over PR-RPAD if EXP4/EXP5 fail;
- real-world generalization;
- claims based on changed split or threshold.
