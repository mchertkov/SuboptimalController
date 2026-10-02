# Case118 Stage 3B — independent controller evaluation

This package completes Stage 3B of the case118 campaign. It uses the frozen Stage-3A controller matrices and gains unchanged and evaluates them on fresh common-random-number Monte Carlo paths at sigma_rms = 0.10, 0.15, 0.20 for homogeneous and activity-scaled noise. The manuscript is not modified and no PIC value is computed here.

## Frozen evaluation protocol

- event: complete bus-90 net-load loss, as frozen before controller work;
- T = 3 s, dt = 2.5 ms;
- 192 paths per noise/sigma scenario;
- fresh evaluation seeds 31301 and 31302, 96 paths each;
- same random paths replayed across all policies inside each scenario;
- policies: static, droop and scaled LQR for H0/H1/H2(rank 3)/H3;
- smooth objective unchanged from Stage 2/3A;
- 90-degree first crossing recorded separately, with Wilson binomial intervals;
- no saturation/clipping.

## Main outputs

- `STAGE3B_REPORT.md`: interpretation and recommendation.
- `STAGE3B_EVALUATION_CONFIG.json`: machine-readable protocol.
- `results/stage3b_all_results.csv`: all 54 scenario-policy rows.
- `results/stage3b_all_paired_vs_static.csv`: CRN paired cost changes and crossing discordance.
- `results/stage3b_lqr_vs_droop_paired.csv`: direct same-path feedback-law comparison.
- `results/stage3b_static_regression_vs_stage2.csv`: independent static-baseline regression.
- `data/stage3b_raw_*.npz`: final per-path J/crossing/tail arrays, not full trajectories.
- `figures/`: cost-reduction and empirical-crossing figures for both noise models.
- `src/`: exact Stage-3A model plus Stage-3B evaluation code.

The empirical crossing probabilities are not rigorous safety certificates. Stage 3B does not alter the H2 rank or any Stage-3A gain.
