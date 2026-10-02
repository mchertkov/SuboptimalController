# Stage 5A3 checkpoint

This package extends the frozen case118 Stage-5A/5A2 control-metric study with metric-aware deployed H3 controllers.

Primary files:
- `STAGE5A3_REPORT.md` — interpretation and decisions.
- `results/STAGE5A3_STATUS.json` — machine-readable protocol/status.
- `results/stage5a3_certificate_summary.csv` — common-CRN metric/certificate table, including refined audit columns.
- `results/stage5a3_tuning/` — scalar scale scans and selections.
- `results/stage5a3_eval/` — fresh evaluation tables plus the 3072-path activity-specific/activity-scaled audit.
- `data/stage5a3_*_lqr.npz` — metric-aware LQR gains and diagonal R vectors.
- `scripts/` and `src/` — reproducible implementation.

The user manuscript and bibliography were not edited in this stage.
