# Improved Experiment Plan After LoRA Report

Date: 2026-04-28

## 1. Revised Core Goal

The next experiments should not try to prove that MOMENT + LoRA beats PR-RPAD everywhere.

The revised goal is:

> Test whether MOMENT + LoRA can provide safer unknown-source rejection than PR-RPAD under limited-label EXP6-style open-set radar deinterleaving.

The report adds an important motivation:

> Standard LoRA can become overconfident in small-sample open-set settings, so the next experiments must evaluate both adaptation performance and rejection/calibration behavior.

## 2. What Changes From the Previous Plan

Previous plan:

- mainly compare PR-RPAD vs MOMENT LoRA;
- screen LoRA variants by ACC and OSCR.

Improved plan:

- keep PR-RPAD as the main baseline;
- keep `MOMENT + q/k/v/o rank8 LoRA` as the main method;
- add `q/k/v/o rank8 + rsLoRA` as the first low-cost improvement;
- evaluate unknown rejection with multiple scores, not only softmax confidence;
- add calibration metrics to directly measure overconfidence;
- run multi-seed before making any strong claim.

## 3. Frozen Data Protocol

Source dataset:

`D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\datasets\pr_rpad_paper_aligned_v6`

Primary few-shot datasets:

- EXP6 s128: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\datasets\paper_aligned_v6_exp6_s128_seed20260428`
- EXP6 s256: `D:\keyan\ai_research_workflow_base\16_exp6_lora_ladder_v2_v3\datasets\paper_aligned_v6_exp6_s256_seed20260428`

Guardrail dataset:

- V1 EXP4/EXP5/EXP6 s128 split: `D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\datasets\paper_aligned_v6_fewshot_s128_seed20260428`

Rules:

- known training only;
- unknown sources never enter training;
- model selection uses known validation only;
- unknown out/test is final evaluation only;
- no test-threshold tuning;
- no changing validation/test/out splits.

## 4. V4 Main Multi-Seed Confirmation

Purpose:

Confirm whether the current EXP6 signal is stable.

Experiments:

- EXP6 s128
- EXP6 s256

Seeds:

- `20260414`
- `20260417`
- `20260418`

Methods:

| Method | Purpose |
|---|---|
| PR-RPAD | primary baseline |
| MOMENT head only | confirms whether the frozen encoder alone is insufficient |
| MOMENT q/v rank8 LoRA | old LoRA baseline |
| MOMENT q/k/v/o rank8 LoRA | main method |
| MOMENT q/k/v/o rank8 rsLoRA | first improved LoRA candidate |

Primary metrics:

- Test known ACC
- OSCR
- AUROC
- FPR95
- Unknown accept rate at known-validation 95% threshold

Secondary metrics:

- trainable parameter count
- runtime
- calibration metrics: ECE, NLL, Brier score

Success direction:

- main method should keep known ACC close to PR-RPAD;
- main method should improve OSCR in EXP6;
- rsLoRA should either improve stability or reduce overconfidence;
- improvements must hold across seeds, not only one run.

## 5. V5 Rejection and Calibration Study

Purpose:

Directly address the report's warning that small-sample LoRA can become overconfident on unknown samples.

Use the trained V4 checkpoints. Do not retrain for every score.

Scores to evaluate:

| Score | Why |
|---|---|
| max softmax confidence | current baseline score |
| energy score | common OOD score, cheap to add |
| Mahalanobis distance | already promising in our EXP6 results |
| confidence + Mahalanobis fusion | likely best next rejection score |
| temperature-scaled confidence | tests whether calibration reduces overconfidence |

Calibration metrics:

- ECE
- NLL
- Brier score
- known validation confidence distribution
- unknown test confidence distribution

Important rule:

Temperature scaling can use known validation labels.

Fusion weights should be fixed before test evaluation, or selected only by known-validation calibration, not by unknown test performance.

Recommended first fusion:

`0.5 * standardized confidence + 0.5 * standardized negative Mahalanobis distance`

This avoids hidden tuning on unknown test data.

## 6. V6 Guardrail Experiments

Purpose:

Avoid making a misleading EXP6-only story.

Run the best V4 method on:

- EXP4 s128
- EXP5 s128
- EXP6 s128

Compare against:

- PR-RPAD
- MOMENT q/v rank8 LoRA
- MOMENT q/k/v/o rank8 LoRA

Interpretation rule:

- If EXP4/EXP5 remain weaker, say so clearly.
- The claim should stay: difficult EXP6-style unknown rejection, not broad superiority.

## 7. LoRA Variants: What to Do Now vs Later

Do now:

| Variant | Priority | Reason |
|---|---|---|
| q/k/v/o rank8 | A | current best runnable method |
| q/k/v/o rank8 rsLoRA | A | low-cost stability improvement suggested by report |
| q/v rank8 | B | old baseline |
| q/v rank8 rsLoRA | B | lightweight backup |
| DoRA | C | already ran, not best; keep as ablation only |

Do later only after user/literature selection:

| Method family | Why not immediate |
|---|---|
| Bayesian PEFT | more complex, needs design choice |
| Laplace-LoRA | needs posterior/laplace implementation decision |
| LoRA ensemble | more compute, must define ensemble size |
| C-LoRA / SBA-style methods | promising but requires careful paper/code review |
| AdaLoRA | useful, but shifts focus to rank allocation |
| TRACE | more related to time-series forecasting than our current classification/rejection task |

## 8. Specific Request for Future Literature Selection

When we reach the advanced-LoRA stage, the user should help choose from papers/code that satisfy:

1. works with Transformer/T5-like encoders;
2. does not require unknown samples for training;
3. has public code or a simple implementation path;
4. explicitly addresses calibration, OOD, uncertainty, or overconfidence;
5. can run on one 12GB GPU.

The most useful methods to investigate:

- Bayesian PEFT
- Laplace-LoRA
- LoRA ensemble
- C-LoRA
- SBA-style uncertainty-aware LoRA

## 8.1 Open-Source First Rule

For any new module after V4, prefer:

1. methods already supported by Hugging Face PEFT;
2. public GitHub repositories with runnable PyTorch code;
3. methods that can be adapted to MOMENT/T5-like Transformer encoders without rewriting the backbone;
4. methods that do not require unknown samples during training.

Immediate open-source priorities:

- keep using PEFT-supported rsLoRA and DoRA switches;
- evaluate LoRA-Ensemble only after checking its code path can be adapted from ViT/AST-style self-attention to MOMENT;
- delay Bayesian PEFT / Laplace-LoRA / C-LoRA unless a concrete open-source implementation is selected.

## 9. Reporting Format for Next Results

Every table should show:

- method;
- split;
- seed;
- known ACC;
- OSCR;
- AUROC;
- FPR95;
- unknown accept rate;
- ECE;
- NLL;
- trainable parameters.

Final reporting should include:

- per-seed rows;
- mean and standard deviation;
- clear failures;
- EXP4/EXP5 guardrails;
- statement that unknown test data was not used for training or threshold selection.

## 10. Current Best Next Run

Recommended immediate run:

V4 multi-seed on EXP6 s128 and s256:

1. PR-RPAD
2. MOMENT q/v rank8
3. MOMENT q/k/v/o rank8
4. MOMENT q/k/v/o rank8 rsLoRA

Then run V5 scoring on those checkpoints:

1. confidence
2. energy
3. Mahalanobis
4. fixed confidence + Mahalanobis fusion
5. temperature-scaled confidence

This plan is the cleanest path from the current promising single-seed result to a defensible research claim.
