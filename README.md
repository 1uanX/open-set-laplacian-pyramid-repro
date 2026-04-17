# Open-Set Laplacian Pyramid Reproduction

This workspace reproduces the paper `A Radar Signal Open-Set Deinterleaving Method Based on Laplacian Pyramid Reconstruction and Adversarial Reciprocal Point Learning` with:

- a structured paper ingest
- executable experiment tables for Experiments 1-6
- generated mixed/train/val/test datasets
- generated ten-condition validation datasets for each experiment
- a runnable closed-set and open-set training pipeline
- saved checkpoints, histories, and evaluation tables

## Where To Look First

- `output/PAPER.md`: full extracted paper text
- `output/CLAIMS.md`: structured claims and contribution list
- `output/RECONSTRUCTION_NOTES.md`: recovered method structure and reproduction assumptions
- `output/reproduction_summary.md`: final reproduction results and recommended run settings
- `artifacts/datasets/pr_rpad_v1/`: generated datasets for all six experiments

## Dataset Layout

Each experiment lives under:

`artifacts/datasets/pr_rpad_v1/exp*/`

Inside each experiment:

- `mixed/train`, `mixed/val`, `mixed/test`
- `specific/loss0_noise01` ... `specific/loss05_noise05`
- `sequences/*.csv`: pulse streams
- `sequences/*.json`: sequence metadata
- `pulse_index.csv`: per-pulse training / evaluation index
- `config.json`: experiment definition, class settings, and recovered parameter ranges

## Recommended Runs

Use the Python interpreter from:

`E:\anaconda3\envs\tdc_env\python.exe`

Recommended command for Experiments 1, 2, 4, 5, and 6:

```powershell
& 'E:\anaconda3\envs\tdc_env\python.exe' run_repro.py --workspace . --experiments exp1,exp2,exp4,exp5,exp6 --iterations 40 --batch-size 256
```

Recommended command for Experiment 3:

```powershell
& 'E:\anaconda3\envs\tdc_env\python.exe' run_repro.py --workspace . --experiments exp3 --iterations 80 --batch-size 256
```

## Saved Outputs

After running, the main outputs are written to `output/`:

- `exp*_checkpoint.pt`
- `exp*_history.csv`
- `exp*_specific_metrics.csv`
- `figures/exp*_history.png`

The generated datasets are deterministic because the experiment seeds are fixed in `run_repro.py`.
