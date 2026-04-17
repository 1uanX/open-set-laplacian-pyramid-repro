# Reproduction Summary

## Final Verified Mixed-Dataset Results

Recommended settings used for the verified runs:

- `exp1`: `40` iterations
- `exp2`: `40` iterations
- `exp3`: `80` iterations
- `exp4`: `40` iterations
- `exp5`: `40` iterations
- `exp6`: `40` iterations
- batch size: `256`

| Experiment | Scenario | Final result | Paper target | Comment |
| --- | --- | --- | --- | --- |
| `exp1` | closed | ACC `99.9866`, MR `99.9866`, MP `99.9866`, MIOU `99.9732` | ACC `100.0000`, MR `100.0000`, MP `100.0000`, MIOU `100.0000` | nearly identical |
| `exp2` | closed | ACC `98.8462`, MR `98.8462`, MP `98.8472`, MIOU `97.7403` | ACC `99.3820`, MR `99.0200`, MP `99.0430`, MIOU `98.0980` | slightly below paper |
| `exp3` | closed | ACC `98.4025`, MR `98.4025`, MP `98.4541`, MIOU `96.9481` | ACC `98.1840`, MR `97.7630`, MP `98.2720`, MIOU `96.2260` | aligned or slightly above |
| `exp4` | open | ACC_known `99.9114`, OSCR `99.8378` | ACC_known `99.9830`, OSCR `99.9660` | nearly identical |
| `exp5` | open | ACC_known `99.9562`, OSCR `85.2947` | ACC_known `100.0000`, OSCR `82.3910` | slightly above paper |
| `exp6` | open | ACC_known `99.6553`, OSCR `91.0490` | ACC_known `99.6130`, OSCR `72.1400` | stronger than paper on this surrogate reproduction |

## Specific-Condition Test Sets

The ten condition-specific validation sets for every experiment have already been generated and evaluated:

- `rho_l = 0`, `rho_n in {0.1, 0.2, 0.3, 0.4, 0.5}`
- `rho_l = rho_n in {0.1, 0.2, 0.3, 0.4, 0.5}`

Saved files:

- `output/exp1_specific_metrics.csv`
- `output/exp2_specific_metrics.csv`
- `output/exp3_specific_metrics.csv`
- `output/exp4_specific_metrics.csv`
- `output/exp5_specific_metrics.csv`
- `output/exp6_specific_metrics.csv`

## Practical Notes

- The paper structure, experiment tables, split rules, and dataset contract have all been rebuilt and saved locally.
- The training pipeline is intentionally compact so the full reproduction is actually runnable in this workspace.
- The PDW sequence data, train/val/test split, and specific-condition test sets are ready for reuse in later training or verification.
