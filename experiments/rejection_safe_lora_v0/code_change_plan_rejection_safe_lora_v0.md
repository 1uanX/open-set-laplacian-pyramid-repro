# Code Change Plan: Rejection-Safe LoRA v0

Date: 2026-04-28

## Scope

Create a standalone v0 experiment runner under:

`D:\keyan\ai_research_workflow_base\14_rejection_safe_lora_adaptation`

This avoids modifying the PR-RPAD baseline files while the new idea is still exploratory.

## Files To Add

- `scripts\run_rejection_safe_lora_v0.py`
- `configs\rejection_safe_lora_v0.json`
- result files under `results\...`

## Data Handling

- Read existing dataset manifests and `pulse_index.csv`.
- Load sequence CSV files from split manifests.
- Build a fixed-length local PDW window around each center pulse.
- Use dataset training normalization policy where available.
- Filter train / validation / test to known labels only.
- Use `out/test` unknown pulses only for open-set evaluation.

## Models

Initial executable models:

- tiny Transformer from scratch;
- frozen Transformer encoder + classifier head;
- LoRA Transformer encoder + classifier head.

MOMENT integration:

- Try to install / load `momentfm`.
- If blocked, record as blocked and keep local Transformer proxy results.
- Do not hide the fallback.

## Rejection Scores

Compute open-set scores from:

- classifier confidence;
- negative centroid distance;
- negative Mahalanobis distance.

Use validation known split for threshold-free model selection. For Unknown Accept Rate, select threshold from known validation confidence quantile, not final unknown test.

## Validation

- `py_compile` for the script.
- startup dry-run / smoke run on exp5 with very small iteration count.
- full v0 feasibility run on exp4/exp5/exp6 if smoke passes.

## Registry

Before final reporting, update:

`D:\keyan\ai_research_workflow_base\05_experiments\run_registry.md`

with run ID, git commit, config, seed, dataset path, checkpoint path, log path, metrics, completion status, and abnormal events.
