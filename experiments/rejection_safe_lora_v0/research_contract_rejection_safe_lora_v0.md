# Research Contract: Rejection-Safe LoRA Adaptation v0

Date: 2026-04-28

## Goal

This is a new exploratory protocol and must not be mixed with the old BCRPS contract.

The goal is to test whether a time-series foundation-model style encoder with parameter-efficient adaptation can support radar open-set deinterleaving under limited labels and environment shift while preserving unknown rejection.

The intended advantage is not higher closed-set accuracy on an already saturated setting. The intended advantage is:

- fewer trainable parameters;
- faster adaptation to a new radar environment;
- stronger or at least non-degraded unknown rejection after adaptation;
- better behavior under limited labels and mild distribution shift.

## Baseline Anchor

- Primary baseline: P003 / PR-RPAD.
- Baseline result used for orientation: `D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\results\paper_aligned_v6_prrpad_exp456_400`
- Baseline comparison remains descriptive in v0 unless the baseline is rerun under exactly the same new sequence protocol.

## Dataset

Primary v0 dataset:

- `D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\datasets\pr_rpad_paper_aligned_v6`

Experiments:

- exp4
- exp5
- exp6

Rules:

- keep existing train / val / test / out split files;
- unknown real emitters must not enter training;
- validation known split may be used for model selection;
- unknown out/test is used for final open-set evaluation only;
- do not tune thresholds on final test results;
- do not change OSCR definition for PR-RPAD comparisons.

## Method Rows

Minimum v0 rows:

1. `tiny_transformer`: small sequence Transformer trained from scratch.
2. `seq_encoder_head`: frozen sequence encoder with only classifier head trainable.
3. `seq_encoder_lora`: sequence encoder with LoRA adapters plus classifier head.
4. `seq_encoder_lora_centroid`: LoRA row plus centroid-distance unknown rejection diagnostic.
5. `seq_encoder_lora_mahalanobis`: LoRA row plus Mahalanobis-distance unknown rejection diagnostic.

If MOMENT cannot be installed or loaded in the local environment, v0 may use a local Transformer encoder as the first executable proxy and record MOMENT as blocked. This is a runability fallback, not a paper claim.

## Input Protocol

Each pulse becomes a local PDW sequence window.

- Features per time step: `dtoa`, `rf`, `pw`, `pa`, `doa`.
- Initial window length: `32`.
- Label: center pulse mapped label.
- Known training samples: `mapped_label >= 0`.
- Unknown evaluation samples: `is_unknown == 1` from `out/test`.
- Normalization: training-split min / max policy from the dataset root when available.
- Padding: edge replication in v0.

## Metrics

Required:

- ACC_known;
- OSCR;
- AUROC for known-vs-unknown scoring;
- FPR95;
- Unknown Accept Rate at validation-selected threshold;
- trainable parameter count;
- total parameter count;
- wall-clock training time;
- wall-clock evaluation time;
- completion status and abnormal events.

Primary v0 interpretation:

- If LoRA improves ACC_known but worsens unknown rejection, that is evidence that naive adaptation can be unsafe.
- If centroid or Mahalanobis scoring reduces Unknown Accept Rate or improves AUROC / FPR95 relative to naive LoRA, the rejection-safe direction remains worth pursuing.

## Seed and Budget

Initial executable seed:

- `20260414`

Debug / smoke tests may use fewer iterations and exp5 only.

Full v0 run:

- exp4, exp5, exp6;
- default iterations: `30` for quick feasibility, then raise only if the row is promising;
- batch size selected to fit the local GPU.

No performance claim may be made from one seed. Single-seed results are only feasibility evidence.

## Success Signal for v0

v0 is successful if it produces a complete, auditable table showing:

- the sequence data pipeline works;
- at least one adapted sequence model trains and evaluates on exp4/5/6;
- LoRA parameter count is meaningfully smaller than full model training;
- open-set metrics are computed without using unknown data for training;
- a rejection score can be compared against naive classifier confidence.

## Claim Boundary

Allowed after v0:

- implementation feasibility;
- whether the direction is promising;
- whether naive LoRA appears to harm or preserve unknown rejection in this local setting.

Not allowed after v0:

- paper-level superiority claims;
- claims of generalization to external data;
- claims that MOMENT is superior unless MOMENT actually runs;
- claims based on best-seed reporting;
- claims that thresholds or unknown rejection were tuned on the test set.
