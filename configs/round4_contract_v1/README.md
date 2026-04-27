# Round4 Config Layer

These config files remain scoped to contract version `round4_v1`.

Prepared coverage:

- P003 baseline remains the sole primary open-set anchor.
- R4-B02 is not configured as a primary open-set comparator and remains secondary reference only.
- Only `exp4`, `exp5`, and `exp6` are expected by the config parser.
- Fixed seeds are `20260414`, `20260417`, and `20260418`; scripts pass each seed explicitly.
- `exp4` is used only for the P003 and full-method guardrail runs.
- `exp5` and `exp6` are used for the main pressure settings and required ablations.
- Deployment diagnostics are measurement-only: trainable parameters, peak memory, and inference latency.
- Failed or interrupted future script runs are made visible by per-run script status files before the model command starts.

The executable matrix layer is in `scripts/round4_contract_v1`.
