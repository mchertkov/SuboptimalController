# Case118 Stage 5A3 — metric-aware deployed H3 control

Date: 2026-09-24

## Status

Stage 5A3 is complete. The case118 event, stochastic model, smooth objective, H3 generator architecture, Stage-5A authority allocations, and Stage-5A2 completed-PIC values were kept fixed. No manuscript or bibliography files were modified.

The purpose of this stage is to make the Stage-5A2 certificate comparison logically consistent: after redesigning the diagonal generator control metric `R`, recompute the H3 LQR for that same `R`, tune only one scalar feedback scale on independent training paths, and evaluate the corresponding deployed upper cost `J_R` on fresh common-random-number paths. The correct comparison is then `J_R - J_R^+` within each physical control metric.

## Frozen metric designs

All designs have the same total generator authority

`sum_g 1/R_g = 270 = 54 / 0.2`.

- `uniform`: `R_g=0.2`, the original H3 metric.
- `robust_fixed`: one diagonal metric chosen in Stage 5A1 to minimize the worst scalar-completion burden across homogeneous and activity-scaled covariance shapes.
- `activity_specific`: generator authority proportional to activity-scaled generator noise variance.

The new LQRs use the same linearized-state weights as Stage 3A: normalized post-event stiffness with angle weight 2 and inertia-weighted frequency weight 0.5. There is no command clipping.

## Scalar LQR-scale tuning

The robust-fixed and activity-specific gains were tuned on seeds `51501,51502`, 16 paths/seed/noise, using the equal-weight average smooth nonlinear cost across homogeneous and activity-scaled noise at `sigma_rms=0.15`. Candidate scales were

`0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02`.

Both new metrics selected the interior value

`alpha = 0.005`.

The original uniform H3 scale remains the frozen Stage-3A value `alpha=0.005`; it was not retuned.

## Common-random-number deployed evaluation

Fresh evaluation seeds `51601,51602` were used, 96 paths/seed/noise (`N=192`), `dt=0.0025 s`, `T=3 s`. The same Brownian paths are replayed for static, uniform, robust-fixed and activity-specific policies within each noise model.

| noise | metric | J_R | J_R^+ | gap J_R-J_R^+ | gap/J_R | crossing | control effort |
|---|---|---:|---:|---:|---:|---:|---:|
| homogeneous | uniform | 0.077154 | 0.075024 | 0.002130 | 2.76% | 0.292 | 0.000821 |
| homogeneous | robust-fixed | 0.077256 | 0.073573 | 0.003684 | 4.77% | 0.292 | 0.000829 |
| homogeneous | activity-specific | 0.077355 | 0.068366 | 0.008988 | 11.62% | 0.292 | 0.000799 |
| activity-scaled | uniform | 0.040950 | 0.024957 | 0.015993 | 39.06% | 0.182 | 0.000809 |
| activity-scaled | robust-fixed | 0.040941 | 0.038777 | 0.002164 | 5.28% | 0.182 | 0.000840 |
| activity-scaled | activity-specific | 0.040821 | 0.040574 | 0.000247 | 0.61% | 0.188 | 0.000864 |

The common static costs on the same paths are 0.077638 (homogeneous) and 0.041526 (activity-scaled). Every deployed H3 policy reduces the smooth cost relative to static on these common paths. The empirical 90-degree crossing statistic remains separate from the smooth objective and is not a safety certificate.

The state-cost component `J_R - control_effort` also shows the intended scenario tradeoff. Under homogeneous noise it changes from 0.076334 (uniform) to 0.076427 (robust) to 0.076555 (activity-specific). Under activity-scaled noise the ordering reverses: 0.040141 (uniform), 0.040101 (robust), 0.039958 (activity-specific).

## Focused audit of the tight activity-specific/activity-scaled gap

The initial `N=192` point estimate made the activity-specific completion gap so small that it was below the Monte-Carlo standard error of the deployed cost. A dedicated upper-cost audit was therefore run without changing the controller: 8 independent batches of 384 paths, using seeds `51701` through `51716`, for a total of `N=3072` new paths.

The refined deployed estimate is

`J_activity-specific,activity = 0.04104670 +/- 0.00007645 (MC SE)`.

Using the frozen Stage-5A2 completed value

`J^+ = 0.04057418`,

the refined numerical certificate gap is

`J - J^+ = 0.00047252`,

or about **1.15%** of the deployed cost. Combining the deployed-cost SE with the Stage-5A2 replication spread gives an approximate gap uncertainty of `7.96e-5`, so the point gap is about 5.9 such standard errors from zero. The exact completed value remains the analytic lower certificate; these displayed values are Monte-Carlo estimates of the lower and upper quantities.

The 3072-path empirical crossing frequency for this focused audit is 0.1634. It is an empirical safety diagnostic only.

## Interpretation

Stage 5A3 validates the control-metric mechanism dynamically, not only geometrically.

1. **Homogeneous covariance:** uniform generator authority remains the best matched metric and gives the tightest completion certificate among the three fixed-authority designs.
2. **Activity-scaled covariance:** uniform H3 is badly mismatched (`~39%` completion gap on the fresh common evaluation). The robust-fixed metric reduces this to about `5.3%`, while the activity-specific metric reduces the refined gap to about `1.15%`, with no increase in total generator authority.
3. **Fixed robust design:** the same `R` gives moderately tight gaps in both scenarios (`~4.8%` homogeneous and `~5.3%` activity-scaled). This is the dynamic counterpart of the Stage-5A1 minimax completion geometry.
4. **Reconfigurable/scenario-specific design:** specializing the metric to the activity covariance gives the tightest target-scenario certificate, but it is deliberately poor for homogeneous noise. This quantifies the value of scenario reconfiguration.
5. **Safety and performance remain distinct:** the activity-specific metric gives the lowest smooth state cost in its target activity-scaled scenario, but its small-sample crossing frequency is not lower than robust/uniform. The smooth PIC certificate should not be conflated with a first-exit guarantee.

## Recommended next stage

The control-metric stage is now sufficiently closed to move to physical reinforcement. The next bounded step should be **Stage 5B1: exact-PIC reinforcement frontiers**. For homogeneous noise, start from the uniform H3 metric. For activity-scaled noise, use the activity-specific fixed-authority metric as the scenario-optimal generator baseline and retain the robust-fixed metric as the two-scenario baseline. Decrease the PIC temperature below the maximal matched value and record added completion authority versus completed operational value. This avoids attributing to new load-side devices a mismatch that can already be removed by generator-metric redistribution.

Only after that one-dimensional frontier is frozen should Stage 5B2 add sparse physical load-side devices and compare noise-only placement with a dynamics-aware H4 utilization/shadow proxy under common device budgets.
