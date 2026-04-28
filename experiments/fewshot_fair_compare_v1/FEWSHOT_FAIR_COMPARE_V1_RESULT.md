# Few-Shot Fair Compare v1 Result

## Status

- Status: completed
- Date: 2026-04-28
- Goal: compare PR-RPAD and MOMENT+LoRA using exactly the same few-shot training indexes.
- This is a first fair-comparison probe, not a final paper claim.

## Data

- Source dataset: `D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\datasets\pr_rpad_paper_aligned_v6`
- Few-shot dataset: `D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\datasets\paper_aligned_v6_fewshot_s128_seed20260428`
- Few-shot train: 128 known pulses per mapped class
- Validation/test/out: full source indexes, not downsampled
- Unknown data: `out/test`, true unknown pulses only
- Seed: `20260428`

| Experiment | Known classes | Train pulses | Validation known | Test known | Unknown test |
|---|---:|---:|---:|---:|---:|
| exp4 | 4 | 512 | 8000 | 8000 | 6244 |
| exp5 | 4 | 512 | 8000 | 8000 | 7121 |
| exp6 | 5 | 640 | 8000 | 8000 | 7502 |

## Runs

PR-RPAD:

- Result root: `D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\results\prrpad_fewshot_s128_seed20260428_it120`
- Settings: `paper_prrpad`, `pdg_image`, `120` iterations, batch size `64`, known-validation-only model selection.

MOMENT:

- Result root: `D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\results\moment_fewshot_s128_seed20260428\exp456_it120`
- Settings: MOMENT-1-large, `moment_head` and `moment_lora_rank8`, `120` iterations, batch size `64`, known-validation-only model selection.

## Main Comparison

| Experiment | Method | Validation known ACC | Test known ACC | OSCR | AUROC | Unknown accept at val95 |
|---|---|---:|---:|---:|---:|---:|
| exp4 | PR-RPAD | 91.850 | 95.388 | 84.123 | n/a | n/a |
| exp4 | MOMENT head | 53.075 | 51.900 | 31.816 | 54.763 | 94.715 |
| exp4 | MOMENT LoRA rank8 | 94.175 | 93.200 | 71.463 | 73.637 | 78.507 |
| exp5 | PR-RPAD | 97.575 | 98.675 | 78.550 | n/a | n/a |
| exp5 | MOMENT head | 53.050 | 52.500 | 34.489 | 58.766 | 93.512 |
| exp5 | MOMENT LoRA rank8 | 93.800 | 93.900 | 64.772 | 66.139 | 88.569 |
| exp6 | PR-RPAD | 90.375 | 90.925 | 40.581 | n/a | n/a |
| exp6 | MOMENT head | 61.950 | 60.812 | 50.428 | 78.138 | 65.663 |
| exp6 | MOMENT LoRA rank8 | 85.062 | 85.450 | 68.985 | 75.978 | 81.672 |

## Distance-Based MOMENT Rejection

| Experiment | Method | Mahalanobis OSCR | Mahalanobis AUROC | Mahalanobis unknown accept at val95 |
|---|---|---:|---:|---:|
| exp4 | MOMENT head | 23.520 | 44.829 | 98.847 |
| exp4 | MOMENT LoRA rank8 | 77.105 | 81.487 | 81.070 |
| exp5 | MOMENT head | 28.471 | 53.550 | 95.745 |
| exp5 | MOMENT LoRA rank8 | 62.701 | 65.865 | 90.240 |
| exp6 | MOMENT head | 28.386 | 46.872 | 99.040 |
| exp6 | MOMENT LoRA rank8 | 59.786 | 67.434 | 86.057 |

## What Changed From v0

- With the fixed s128 few-shot split, MOMENT+LoRA improved sharply over v0:
  - exp4 test known ACC: `88.812` -> `93.200`
  - exp5 test known ACC: `86.125` -> `93.900`
  - exp6 test known ACC: `63.713` -> `85.450`
- This means the previous weak result was partly an experiment design issue, not only a model issue.
- The fixed balanced few-shot sample made LoRA learn much more cleanly.

## Interpretation

- PR-RPAD is still extremely strong on exp4 and exp5 even with only 128 known pulses per class.
- MOMENT+LoRA does not beat PR-RPAD on exp4/exp5 overall.
- exp6 is the important opening: PR-RPAD keeps higher known-class accuracy, but its OSCR falls to `40.581`, while MOMENT+LoRA reaches `68.985` with confidence scoring.
- That supports a narrower advantage claim: the foundation-model route may be better for difficult open-set rejection under complex exp6-style conditions, but it is not yet a general replacement for PR-RPAD.

## Next Experiment

The next useful run is not another broad model sweep. It should focus on exp6 and rejection:

1. Run s64/s128/s256 on exp6 only.
2. Keep PR-RPAD and MOMENT+LoRA on identical few-shot indexes.
3. Add calibrated rejection for MOMENT+LoRA instead of raw confidence.
4. Try to reduce unknown accept rate while keeping known ACC above 85%.

If exp6 stays favorable across s64/s128/s256, that becomes our first defensible advantage direction.

## Verification

- UTF-8 rules check passed before work.
- Few-shot builder `py_compile` passed.
- Few-shot dataset manifest written and verified.
- PR-RPAD smoke run passed.
- MOMENT smoke run passed.
- PR-RPAD full exp4/exp5/exp6 run completed.
- MOMENT full exp4/exp5/exp6 run completed.
- Logs contain no fatal errors.
