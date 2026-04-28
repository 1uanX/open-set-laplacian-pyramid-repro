# EXP6 LoRA v4 Multi-Seed Calibration Result

Date: 2026-04-28

## Status

- Status: completed
- Goal: validate EXP6 s128/s256 with multiple seeds and add overconfidence-aware rejection metrics.
- Seeds: `20260414`, `20260417`, `20260418`
- This is stronger evidence than V2/V3, but still limited to EXP6 and this synthetic paper-aligned dataset.

## Data Protocol

Datasets:

- EXP6 s128: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\datasets\paper_aligned_v6_exp6_s128_seed20260428`
- EXP6 s256: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\datasets\paper_aligned_v6_exp6_s256_seed20260428`

Protocol:

- known-source training only;
- full known validation/test kept unchanged;
- full unknown out/test kept unchanged;
- unknown data not used for training;
- no test-threshold tuning.

## Methods

| Method | Role |
|---|---|
| PR-RPAD | primary baseline |
| MOMENT q/v rank8 LoRA | old LoRA baseline |
| MOMENT q/k/v/o rank8 LoRA | V3 best structural variant |
| MOMENT q/k/v/o rank8 rsLoRA | new low-cost improved variant |

## Main Multi-Seed Results

Mean +/- std over 3 seeds.

| Samples/class | Method | Known ACC | Confidence OSCR | Best OSCR | Best score | Trainable params |
|---:|---|---:|---:|---:|---|---:|
| 128 | PR-RPAD | 93.254 +/- 2.958 | 61.029 +/- 1.591 | 61.029 +/- 1.591 | PR-RPAD open score | 20,316,000 |
| 128 | q/v rank8 | 83.583 +/- 1.998 | 63.667 +/- 1.901 | 71.753 +/- 1.365 | energy | 812,037 |
| 128 | q/k/v/o rank8 | 90.233 +/- 0.976 | 71.692 +/- 2.695 | 78.295 +/- 3.886 | energy | 1,598,469 |
| 128 | q/k/v/o rank8 rsLoRA | 93.483 +/- 0.148 | 75.516 +/- 3.303 | 81.811 +/- 2.312 | fusion/energy | 1,598,469 |
| 256 | PR-RPAD | 93.750 +/- 3.809 | 64.674 +/- 8.625 | 64.674 +/- 8.625 | PR-RPAD open score | 20,316,000 |
| 256 | q/v rank8 | 89.567 +/- 0.777 | 71.989 +/- 1.208 | 77.828 +/- 0.693 | energy | 812,037 |
| 256 | q/k/v/o rank8 | 96.158 +/- 0.220 | 83.406 +/- 0.997 | 88.850 +/- 0.346 | energy | 1,598,469 |
| 256 | q/k/v/o rank8 rsLoRA | 95.892 +/- 0.203 | 83.159 +/- 0.382 | 89.127 +/- 0.839 | energy | 1,598,469 |

## Calibration and Rejection Findings

| Samples/class | Method | ECE | Temp ECE | Unknown accept at val95 |
|---:|---|---:|---:|---:|
| 128 | q/v rank8 | 3.972 +/- 1.212 | 1.479 +/- 0.449 | 85.222 +/- 1.616 |
| 128 | q/k/v/o rank8 | 2.300 +/- 1.083 | 1.041 +/- 0.123 | 82.925 +/- 3.013 |
| 128 | q/k/v/o rank8 rsLoRA | 1.526 +/- 0.419 | 0.719 +/- 0.298 | 78.837 +/- 3.776 |
| 256 | q/v rank8 | 1.868 +/- 0.718 | 0.942 +/- 0.196 | 80.130 +/- 0.910 |
| 256 | q/k/v/o rank8 | 1.057 +/- 0.198 | 0.593 +/- 0.149 | 69.661 +/- 4.209 |
| 256 | q/k/v/o rank8 rsLoRA | 0.817 +/- 0.361 | 0.544 +/- 0.265 | 67.235 +/- 1.534 |

Interpretation:

- Temperature scaling consistently reduced ECE.
- q/k/v/o rank8 and q/k/v/o rank8 rsLoRA both reduced overconfidence symptoms relative to q/v rank8.
- rsLoRA gave the best calibration and lower unknown accept rate in both s128 and s256.
- Energy score was the most consistently strong rejection score in this run.
- Fusion was especially strong for s128 q/k/v/o rsLoRA, but energy was simpler and more stable overall.

## Open-Source Module Priority

The next new modules should prioritize existing open-source implementations or direct PEFT support.

Current priority:

1. Continue using Hugging Face PEFT for rsLoRA/DoRA-style variants because they are direct `LoraConfig` options.
2. Consider LoRA-Ensemble next because it has a public repository and directly targets uncertainty/calibration in self-attention networks.
3. Defer Bayesian PEFT / Laplace-LoRA / C-LoRA until a concrete codebase and integration path are selected.

Useful open-source references checked:

- Hugging Face PEFT LoRA developer guide: `https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md`
- Hugging Face PEFT repository: `https://github.com/huggingface/peft`
- LoRA-Ensemble repository: `https://github.com/prs-eth/LoRA-Ensemble`

## Objective Conclusion

The V4 result changes our confidence level.

V2/V3 showed a promising single-seed signal. V4 shows that the signal survives three seeds on EXP6 s128 and s256.

The best current method is:

`MOMENT + q/k/v/o rank8 rsLoRA`

Best supported claim at this stage:

> On the current EXP6 limited-label paper-aligned dataset, MOMENT + q/k/v/o rank8 rsLoRA improves unknown-source rejection over PR-RPAD while using far fewer trainable parameters.

Still not allowed:

- claiming broad superiority over PR-RPAD;
- claiming real-world generalization;
- claiming EXP4/EXP5 superiority without guardrail runs;
- claiming advanced Bayesian/ensemble LoRA benefits before running them.

## Files

- Raw metrics: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\results\exp6_v4_multiseed_raw_metrics.csv`
- Aggregate metrics: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\results\exp6_v4_multiseed_aggregate_metrics.csv`
- PR-RPAD status: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\logs\v4_prrpad_status.json`
- MOMENT status: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\logs\v4_moment_status.json`

## Verification

- UTF-8 rule check passed.
- Contract, change plan, baseline, and claims rules were read.
- New script passed `py_compile`.
- Startup dry-run passed.
- Smoke run for `q/k/v/o rank8 rsLoRA` passed.
- PR-RPAD completed all 6 runs.
- MOMENT completed all 6 multi-method runs.
- Metrics were extracted from saved result files and aggregated by script.
