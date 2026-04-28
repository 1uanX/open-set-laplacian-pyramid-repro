# Code Change Plan: EXP6 LoRA v4 Calibration

Date: 2026-04-28

## Scope

Extend the existing standalone MOMENT runner:

`D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\scripts\run_moment_lora_variants_v3.py`

Do not modify the PR-RPAD baseline source files.

## Additions

- add `moment_lora_qkvo_rank8_rslora`;
- record logits/probabilities during evaluation;
- compute energy score;
- compute temperature-scaled confidence using known validation only;
- compute fixed confidence + Mahalanobis fusion using known-validation standardization;
- compute ECE, NLL, and Brier score on known test samples.

## Validation

- `py_compile`;
- startup dry-run;
- one small smoke run on EXP6 s128;
- then multi-seed V4.

## Registry

Update:

`D:\keyan\ai_research_workflow_base\05_experiments\run_registry.md`

before and after the V4 run.
