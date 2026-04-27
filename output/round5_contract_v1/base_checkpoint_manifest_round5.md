# Round5 P003 Base Checkpoint Manifest

Status: BASE_CHECKPOINTS_GENERATED_BUT_NOT_LORA_READY

Created: 2026-04-27 21:43 +08:00

## Scope

This manifest covers only Round5 P003 base checkpoint preparation.

LoRA was not run.

## Source Checks

| Item | Result |
|---|---|
| Round5 dataset root | `D:\keyan\ai_research_workflow_base\09_dataset_rebuild_round5\datasets\pr_rpad_round5_v1` |
| Round5 dataset audit | PASSED |
| Minimal loader sanity | PASSED |
| Round5 P003 base config | `D:\keyan\ai_research_workflow_base\09_dataset_rebuild_round5\p003_base_model_config_round5.json` |
| P003 baseline mode | `p003_baseline` |
| Seed | `20260414` |
| Training iterations | `40` |
| Batch size | `256` |
| Specific-condition evaluation | skipped for base checkpoint generation |
| Old Round4 checkpoint initialization | not used; all run records show `null` |

## Current full_lora_rank8 Check

| Field | Value |
|---|---|
| Config | `D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro\configs\round4_contract_v1\full_lora_rank8.json` |
| Method variant | `full_lora` |
| Adapter rank | `8` |
| Expected experiments | `exp4, exp5, exp6` |
| Current checkpoint field | `output\round4_contract_v1\p003_checkpoints\{seed}\matched_p003_checkpoint.pt` |
| Resolved current checkpoint path | `D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro\output\round4_contract_v1\p003_checkpoints\20260414\matched_p003_checkpoint.pt` |
| Resolved current checkpoint exists | NO |

## Generated Round5 Base Checkpoints

| Experiment | Path | Classes | Input dim | SHA256 | Strict load | LoRA keys |
|---|---|---:|---:|---|---|---:|
| exp4 | `D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro\output\round5_contract_v1\p003_base_checkpoints\20260414\exp4_matched_p003_checkpoint.pt` | 4 | 30 | `e278f0215325b42c22db5d471fe665ddeda4354273aafe462292e09a584c29a7` | PASS | 0 |
| exp5 | `D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro\output\round5_contract_v1\p003_base_checkpoints\20260414\exp5_matched_p003_checkpoint.pt` | 4 | 30 | `1dbe46290a8a6217be0c5ab49bb95ca14150ea4d98937ff19cf5511d6825145f` | PASS | 0 |
| exp6 | `D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro\output\round5_contract_v1\p003_base_checkpoints\20260414\exp6_matched_p003_checkpoint.pt` | 5 | 30 | `101e4d33fabc6e554ce9aac932eb2e200af391cdd534ad01d04c43831eb0c485` | PASS | 0 |

## Run Output

| Item | Path |
|---|---|
| Run directory | `D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro\output\round5_contract_v1\round5_p003_base_seed20260414` |
| Run report | `D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro\output\round5_contract_v1\round5_p003_base_seed20260414\reproduction_report.json` |
| Run manifest | `D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro\output\round5_contract_v1\round5_p003_base_seed20260414\run_manifest.json` |

## LoRA Readiness Decision

Do not mark LoRA preflight ready yet.

Reason:

1. The current `full_lora_rank8.json` still points to a single old Round4-style checkpoint path.
2. That resolved checkpoint path does not exist.
3. Round5 base checkpoints are experiment-specific because `exp4` and `exp5` have 4 classes, while `exp6` has 5 classes.
4. A single shared checkpoint path is therefore not a safe representation for all three Round5 experiments.

Required before LoRA:

1. Update LoRA preflight/config logic to reference the correct Round5 base checkpoint for each experiment.
2. Keep Round5 dataset root fixed.
3. Do not use old Round4 dataset checkpoints.
4. Re-run a LoRA preflight check only after the per-experiment checkpoint references are in place.
