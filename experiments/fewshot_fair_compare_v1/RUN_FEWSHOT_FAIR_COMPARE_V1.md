# Few-Shot Fair Compare v1

## Goal

Compare PR-RPAD and MOMENT+LoRA under the same few-shot training data.

## Fixed Setup

- Source dataset: `D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\datasets\pr_rpad_paper_aligned_v6`
- Few-shot dataset: `D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\datasets\paper_aligned_v6_fewshot_s128_seed20260428`
- Experiments: `exp4`, `exp5`, `exp6`
- Few-shot train: 128 known pulses per mapped label
- Validation/test/out: unchanged full source pulse indexes
- Model selection: known validation only
- Unknown data: evaluation only

## Commands

Build few-shot dataset:

```powershell
& E:\anaconda3\envs\rs_lora_moment\python.exe `
  D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\scripts\build_fewshot_dataset_v1.py `
  --source-root D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\datasets\pr_rpad_paper_aligned_v6 `
  --output-root D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\datasets\paper_aligned_v6_fewshot_s128_seed20260428 `
  --samples-per-class 128 `
  --seed 20260428
```

Run PR-RPAD:

```powershell
Set-Location D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro

& E:\anaconda3\envs\rs_lora_moment\python.exe .\run_repro.py `
  --workspace D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro `
  --dataset-root D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\datasets\paper_aligned_v6_fewshot_s128_seed20260428 `
  --experiments exp4,exp5,exp6 `
  --iterations 120 `
  --batch-size 64 `
  --lr 0.01 `
  --skip-specific `
  --input-mode pdg_image `
  --model-arch paper_prrpad `
  --strict-open-set-selection `
  --run-id prrpad_fewshot_s128_seed20260428_it120 `
  --output-dir D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\results\prrpad_fewshot_s128_seed20260428_it120
```

Run MOMENT:

```powershell
& E:\anaconda3\envs\rs_lora_moment\python.exe `
  D:\keyan\ai_research_workflow_base\14_rejection_safe_lora_adaptation\scripts\run_rejection_safe_lora_v0.py `
  --config D:\keyan\ai_research_workflow_base\15_fewshot_fair_compare\configs\moment_fewshot_s128_v1.json `
  --run-id exp456_it120 `
  --local-files-only
```
