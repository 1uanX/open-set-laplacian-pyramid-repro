# Rejection-Safe LoRA v0 Result

## Status

- Status: completed
- Date: 2026-04-28
- Purpose: first feasibility run for a time-series foundation-model route on PDW open-set deinterleaving.
- This is an exploratory run, not a paper claim.

## Environment

- Environment: `E:\anaconda3\envs\rs_lora_moment`
- Python: 3.11.15
- Torch: 2.6.0+cu124, CUDA available
- MOMENT: 0.1.4
- Transformers: 4.33.3
- PEFT: 0.13.2
- NumPy/Pandas/SciPy/sklearn: 1.25.2 / 2.2.3 / 1.11.4 / 1.3.2
- `pip check`: passed

## Inputs

- Dataset root: `D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\datasets\pr_rpad_paper_aligned_v6`
- Experiments: `exp4`, `exp5`, `exp6`
- Known training: `mixed/train`, known labels only
- Known validation: `mixed/val`, known labels only
- Known test: `mixed/test`, known labels only
- Unknown test: `out/test`, true unknown pulses only
- Per experiment training subset: 1024 samples per known class
- Window: 32 pulses, features `dtoa, rf, pw, pa, doa`
- Validation/model selection: known validation only; unknown data was not used for model selection

## Files

- Script: `D:\keyan\ai_research_workflow_base\14_rejection_safe_lora_adaptation\scripts\run_rejection_safe_lora_v0.py`
- Config: `D:\keyan\ai_research_workflow_base\14_rejection_safe_lora_adaptation\configs\rejection_safe_lora_v0.json`
- Result root: `D:\keyan\ai_research_workflow_base\14_rejection_safe_lora_adaptation\results\rejection_safe_lora_v0\exp456_seed20260414_it60_spc1024`
- Summary table: `D:\keyan\ai_research_workflow_base\14_rejection_safe_lora_adaptation\results\rejection_safe_lora_v0\exp456_seed20260414_it60_spc1024\summary_metrics.csv`
- Logs:
  - `D:\keyan\ai_research_workflow_base\14_rejection_safe_lora_adaptation\logs\rejection_safe_lora_v0_exp456.out.log`
  - `D:\keyan\ai_research_workflow_base\14_rejection_safe_lora_adaptation\logs\rejection_safe_lora_v0_exp456.err.log`

## Models

| Model | Trainable parameters | Trainable percent |
|---|---:|---:|
| `moment_head` | 20,484 on exp4/exp5; 25,605 on exp6 | about 0.006-0.008% |
| `moment_lora_rank8` | 806,916 on exp4/exp5; 812,037 on exp6 | about 0.236-0.237% |

## Main Results

`confidence_*` uses max softmax confidence as the known/unknown score.

| Exp | Model | Val known ACC | Test known ACC | Confidence OSCR | Confidence AUROC | Unknown accept at val95 |
|---|---|---:|---:|---:|---:|---:|
| exp4 | moment_head | 48.175 | 47.575 | 31.220 | 59.126 | 95.003 |
| exp4 | moment_lora_rank8 | 89.600 | 88.812 | 67.284 | 71.306 | 86.163 |
| exp5 | moment_head | 45.212 | 46.425 | 25.762 | 46.153 | 98.146 |
| exp5 | moment_lora_rank8 | 85.475 | 86.125 | 62.766 | 68.264 | 90.858 |
| exp6 | moment_head | 59.925 | 59.675 | 51.564 | 82.259 | 56.971 |
| exp6 | moment_lora_rank8 | 62.938 | 63.713 | 51.573 | 75.983 | 73.154 |

Distance scores from embeddings:

| Exp | Model | Centroid OSCR | Centroid AUROC | Mahalanobis OSCR | Mahalanobis AUROC |
|---|---|---:|---:|---:|---:|
| exp4 | moment_head | 19.633 | 41.060 | 21.884 | 46.130 |
| exp4 | moment_lora_rank8 | 55.648 | 62.584 | 63.217 | 70.367 |
| exp5 | moment_head | 22.795 | 49.949 | 24.991 | 55.132 |
| exp5 | moment_lora_rank8 | 40.382 | 46.952 | 45.243 | 52.247 |
| exp6 | moment_head | 29.414 | 46.636 | 28.598 | 47.843 |
| exp6 | moment_lora_rank8 | 46.968 | 73.837 | 50.372 | 79.331 |

## What This Says

- LoRA has a clear known-class adaptation signal on exp4 and exp5. It raises known validation accuracy from about 45-48% to about 85-90%.
- Exp6 is harder for this v0 setup. LoRA improves known accuracy only modestly, from about 60% to about 63%.
- Unknown rejection is not solved yet. Confidence-based unknown acceptance remains high, especially exp4 and exp5.
- Embedding-distance scoring helps most clearly on exp6: Mahalanobis AUROC rises to 79.331 for LoRA.
- This supports the direction, but the current version is not yet competitive with the paper-aligned PR-RPAD reproduction.

## Verification

- UTF-8 rules check passed before editing.
- New isolated environment was created instead of continuing to change `tdc_env`.
- MOMENT local forward test passed with 5-channel, 32-step PDW input.
- `py_compile` passed for the new script.
- Startup dry-run confirmed exp4/exp5/exp6 dataset files exist.
- Smoke run on exp5 completed and wrote metrics/checkpoints.
- Full exp4/exp5/exp6 run completed.
- All six model folders contain `metrics.json`, `history.csv`, and `checkpoint_trainable.pt`.

## Next Useful Move

The next experiment should keep the same data split and compare three changes:

1. Train longer or use more samples per class to see whether exp6 improves.
2. Use a better rejection score than raw confidence, starting with Mahalanobis or multi-layer distance.
3. Add a same-budget non-foundation baseline so the advantage is measured as "fewer labels / faster adaptation / preserved unknown rejection", not just raw accuracy.
