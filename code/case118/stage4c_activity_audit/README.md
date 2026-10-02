# CASE118 Stage 4C package

Focused audit of the H3 activity-scaled sigma_rms=0.15 completed-PIC Monte Carlo estimate.

Key output: `STAGE4C_REPORT.md`.

Primary result files:
- `results/STAGE4C_FINAL.json` -- four independent 512-path final replications.
- `results/final_holdout_1024.json` -- independent 1024-path holdout.
- `results/STAGE4C_COMPARISON.json` and `results/stage4c_comparison.csv` -- old-vs-refined comparison and updated gap.
- `results/ce_fit_actual.json` -- six-bin cross-entropy schedule fit.
- `results/actual_adapted_holdout_256*.csv` -- fresh guide-selection check.

Reproduction scripts are under `scripts/`; the frozen case118 source and Stage-3A matrices used by the calculations are under `src/` and `data/`.

No manuscript or bibliography file is included or modified.
