# EXP4/5/6 Current Full v5 Result

Date: 2026-04-28

## Status

- Status: completed
- Goal: finish the current version before saving a GitHub snapshot.
- Methods: PR-RPAD baseline and MOMENT q/k/v/o rank8 rsLoRA.
- Experiments: EXP4, EXP5, EXP6.
- Seeds: `20260414`, `20260417`, `20260418`.
- Dataset: fixed s128 few-shot split derived from the paper-aligned Round5 rebuild.

## Dataset and Split Verification

Dataset used:

`D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\datasets\paper_aligned_v6_fewshot_s128_seed20260428`

Source dataset:

`D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\datasets\pr_rpad_paper_aligned_v6`

Lineage:

- built after Round5;
- EXP6 copied from the audited Round5 dataset;
- EXP4/EXP5 rebuilt in the paper-aligned version to better match the P003 paper behavior;
- no Round4 dataset or Round4 checkpoint was used.

Split audit:

`D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\results\dataset_split_audit_current_v4.json`

Audit result:

- train/validation/test/out have zero overlapping `sequence_path + pulse_index` entries;
- train contains known samples only;
- validation contains known samples only;
- test contains known samples only;
- out/test contains true unknown samples only after few-shot filtering;
- unknown samples were not used for training or model selection.

| Experiment | Train known | Validation known | Test known | Unknown test |
|---|---:|---:|---:|---:|
| EXP4 | 512 | 8000 | 8000 | 6244 |
| EXP5 | 512 | 8000 | 8000 | 7121 |
| EXP6 | 640 | 8000 | 8000 | 7502 |

## Main Results

Mean +/- std over 3 seeds.

| Experiment | Method | Validation ACC | Test ACC | Primary OSCR | Trainable params |
|---|---|---:|---:|---:|---:|
| EXP4 | PR-RPAD | 96.392 +/- 1.699 | 96.950 +/- 1.440 | 86.420 +/- 2.539 | 17,475,994 |
| EXP4 | MOMENT q/k/v/o rank8 rsLoRA | 96.688 +/- 0.563 | 96.083 +/- 0.672 | 74.055 +/- 4.932 | 1,593,348 |
| EXP5 | PR-RPAD | 97.746 +/- 1.133 | 97.879 +/- 1.378 | 81.989 +/- 3.049 | 17,475,994 |
| EXP5 | MOMENT q/k/v/o rank8 rsLoRA | 95.788 +/- 1.252 | 96.321 +/- 1.007 | 72.204 +/- 0.404 | 1,593,348 |
| EXP6 | PR-RPAD | 92.742 +/- 1.556 | 93.200 +/- 1.554 | 52.845 +/- 8.110 | 20,316,000 |
| EXP6 | MOMENT q/k/v/o rank8 rsLoRA | 93.342 +/- 1.335 | 93.329 +/- 1.271 | 76.040 +/- 2.289 | 1,598,469 |

Primary OSCR uses the default confidence score for MOMENT. Additional rejection scores are reported below and should be treated as diagnostics unless a future contract freezes one as the final inference score.

## Additional Rejection and Calibration Diagnostics

| Experiment | MOMENT score | OSCR |
|---|---|---:|
| EXP4 | confidence | 74.055 +/- 4.932 |
| EXP4 | Mahalanobis | 84.617 +/- 1.037 |
| EXP4 | confidence + Mahalanobis fusion | 83.942 +/- 1.252 |
| EXP5 | confidence | 72.204 +/- 0.404 |
| EXP5 | temperature confidence | 72.301 +/- 0.422 |
| EXP5 | fusion | 72.050 +/- 1.272 |
| EXP6 | confidence | 76.040 +/- 2.289 |
| EXP6 | energy | 83.193 +/- 2.053 |
| EXP6 | fusion | 80.660 +/- 2.165 |

Calibration:

| Experiment | ECE | Temperature-scaled ECE |
|---|---:|---:|
| EXP4 | 1.506 +/- 0.375 | 0.589 +/- 0.064 |
| EXP5 | 1.961 +/- 1.237 | 0.807 +/- 0.362 |
| EXP6 | 1.350 +/- 0.744 | 0.685 +/- 0.211 |

Temperature scaling consistently reduced ECE, which supports the overconfidence concern from the LoRA report.

## Objective Interpretation

The current version is not a general replacement for PR-RPAD.

EXP4:

- PR-RPAD has much higher primary OSCR.
- MOMENT rsLoRA has similar known accuracy, but weaker confidence-based rejection.
- Mahalanobis improves MOMENT substantially, but this remains diagnostic for now.

EXP5:

- PR-RPAD is stronger in both known accuracy and OSCR.
- MOMENT rsLoRA is stable but not better.

EXP6:

- MOMENT rsLoRA slightly exceeds PR-RPAD in known accuracy.
- MOMENT rsLoRA strongly improves primary OSCR.
- Energy score further improves the diagnostic OSCR.

Best honest conclusion:

> The current method is valuable mainly for difficult EXP6-style limited-label unknown-source rejection, not for broad EXP4/EXP5 superiority.

## Files

- Raw metrics: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\results\exp456_current_full_v5_raw_metrics.csv`
- Aggregate metrics: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\results\exp456_current_full_v5_aggregate_metrics.csv`
- Split audit: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\results\dataset_split_audit_current_v4.json`
- PR-RPAD status: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\logs\v5_prrpad_exp456_status.json`
- MOMENT status: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\logs\v5_moment_exp456_status.json`

## Verification

- UTF-8 rule check passed.
- Baseline, contract, change plan, and claims rules were read.
- Dataset lineage checked against the paper-aligned Round5 rebuild.
- Split audit completed with zero train/val/test/out overlap.
- Startup dry-run passed.
- PR-RPAD EXP4/5/6 completed for all three seeds.
- MOMENT q/k/v/o rank8 rsLoRA EXP4/5/6 completed for all three seeds.
- Metrics were extracted from saved result files and aggregated by script.
