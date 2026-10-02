# Case118 Stage 5B1 — exact-PIC physical-reinforcement frontier

Date: 2026-09-24

## Status

Stage 5B1 is complete. The case118 bus-90 event, sigma_rms=0.15 stochastic laboratory, smooth operational cost, and the Stage-5A generator-metric designs are frozen. No manuscript or bibliography files were modified.

This stage interprets exact control completion as a physical fast-flexibility reinforcement family. For a fixed noise covariance q_i=Gamma_i^2 and baseline generator authority G0, exact scalar PIC at temperature lambda requires the nodal authority

    G_plus(lambda) = diag(q_i/lambda),

with physical increment Delta G(lambda)=G_plus(lambda)-G0 >= 0. The least-inflated exact-PIC point is lambda=lambda_max. Decreasing lambda below lambda_max installs more authority and lowers the exact completed operational value J_plus.

The tested fractions are lambda/lambda_max = 1, 0.8, 0.6, 0.4, 0.25.

The planning proxy

    P_eff = baseMVA * sum_i sqrt(r_ref Delta G_ii),  baseMVA=100, r_ref=0.2,

is an effective aggregate power-authority proxy only. It is not a battery nameplate rating, and it contains no saturation, SoC, MWh, degradation, or capital-cost model.

## Scenario-matched frontiers

The scenario-matched baselines are the Stage-5A choices: uniform H3 authority for homogeneous noise and activity-specific, noise-proportional H3 authority for activity-scaled noise. Both preserve total generator authority sum_g g_g=270.

### Homogeneous noise / uniform H3

| lambda/lambda_max | lambda | completion trace | P_eff (GW proxy) | J_plus | ESS fraction |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.004500 | 320.0 | 6.400 | 0.075024 | previous refined baseline |
| 0.80 | 0.003600 | 467.5 | 9.855 | 0.074288 | 0.629 |
| 0.60 | 0.002700 | 713.3 | 12.671 | 0.073472 | 0.509 |
| 0.40 | 0.001800 | 1205.0 | 16.733 | 0.071678 | 0.305 |
| 0.25 | 0.001125 | 2090.0 | 22.153 | 0.068836 | 0.138 |

From the least-inflated exact-PIC point to lambda/lambda_max=0.25, the completed operational value falls by 0.006189, or 8.25%, while the power-authority proxy rises by 15.75 GW.

At lambda=lambda_max the exact completion acts only on the 64 non-generator buses: generator completion trace is zero and load-side completion trace is 320. As soon as lambda is reduced below lambda_max, all 118 stochastic directions receive additional authority.

### Activity-scaled noise / activity-specific H3

| lambda/lambda_max | lambda | completion trace | P_eff (GW proxy) | J_plus | ESS fraction |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.008718 | 34.54 | 1.911 | 0.040574 | 0.976, previous refined baseline |
| 0.80 | 0.006974 | 110.68 | 4.293 | 0.040541 | 0.967 |
| 0.60 | 0.005231 | 237.57 | 5.989 | 0.040141 | 0.951 |
| 0.40 | 0.003487 | 491.36 | 8.305 | 0.039397 | 0.917 |
| 0.25 | 0.002179 | 948.17 | 11.294 | 0.038209 | 0.847 |

The least-inflated activity-specific exact completion is especially economical geometrically: after generator-metric redesign it fills only the 64 load-bus directions and has trace 34.54. The generator completion is zero up to numerical roundoff. This should be contrasted with the pre-redesign uniform-H3 activity-scaled completion trace of approximately 1.48e4.

Moving from lambda_max to 0.25 lambda_max lowers J_plus by 0.002365, or 5.83%, while adding about 9.38 GW in the effective power proxy. The first step to 0.8 lambda_max changes J_plus by only about 3.3e-5, comparable to the Monte-Carlo resolution, despite adding about 2.38 GW of proxy authority. Thus the near-baseline activity-scaled frontier is initially quite flat.

## Robust-fixed two-scenario frontier

The robust-fixed Stage-5A metric has the same lambda_max=0.002787 and completion trace 682.55 in both covariance scenarios. Its exact-PIC frontier is:

| lambda/lambda_max | J_plus homogeneous | J_plus activity-scaled |
|---:|---:|---:|
| 1.00 | 0.073573 | 0.038777 |
| 0.80 | 0.072599 | 0.038265 |
| 0.60 | 0.071182 | 0.037350 |
| 0.40 | 0.068638 | 0.035773 |
| 0.25 | 0.064846 | 0.033517 |

The completion trace is identical across the two scenarios at a fixed lambda fraction, but the effective power proxy is not, because the nodal covariance shapes differ. At the 0.25 point the proxy is about 28.86 GW for homogeneous noise and 20.94 GW for activity-scaled noise.

The robust homogeneous 0.25 point has the weakest importance-sampling diagnostic in this stage (mean ESS fraction about 0.041). Its monotone trend is clear, but this particular endpoint should not be quoted to high precision without another variance-reduction pass. The scenario-matched frontiers are substantially better conditioned; the minimum ESS fractions there are about 0.138 (homogeneous/uniform) and 0.847 (activity/activity-specific).

## Monte-Carlo protocol

All new frontier values use dt=0.0025 s and the frozen smooth Stage-3/4 objective. Importance sampling uses the actual-cost CARE guide recomputed for each lambda; guide scale alpha=1 was used for the production frontier. The new scenario-matched non-baseline points use three independent 128-path replications, except the homogeneous 0.8 point where two additional 192-path batches were already available and were included. Independent batch normalizers were pooled at the Z=exp(-J_plus/lambda) level before taking -lambda log Z. Baseline lambda=lambda_max values are the previously refined Stage-4/5A2 values and were not recomputed.

For the two robust-fixed frontiers, non-baseline production points use two independent 128-path replications. Small guide scans made during feasibility testing are retained in the package but excluded from the final production aggregation.

## Interpretation

Stage 5B1 confirms the one-dimensional exact-PIC reinforcement ordering predicted by the theory: decreasing lambda increases completion authority and monotonically lowers the exact completed operational value.

The more important engineering result is the scale of that tradeoff. Once the generator control metric has first been matched to the covariance, the least exact completion is purely load-side. Further movement down the exact-PIC frontier immediately adds authority to all generator and load directions and quickly becomes expensive in the effective-power proxy for comparatively modest additional operational-value reduction.

This strengthens the case for the next experiment: exact PIC should be treated as a reference frontier rather than as a prescription to build the full completion. Stage 5B2 should ask how much of the physical benefit can be recovered with a sparse set of load-side devices under the same authority/device budgets.

## Recommended Stage 5B2

Use the least-inflated scenario-matched completions as the oracle targets:

- homogeneous: uniform H3 generator metric, 64 candidate load buses;
- activity-scaled: activity-specific generator metric, 64 candidate load buses;
- retain robust-fixed as the fixed two-scenario benchmark.

For device counts k=5,10,20,40,64, compare at least two placements: noise-only ranking and a dynamics-aware H4 utilization/shadow proxy. Match total added authority across placement rules, recompute the physical metric-aware LQR for each sparse reinforced system, and evaluate nonlinear deployed J, empirical first exit, and control effort under common random paths. The k=64 exact-completion endpoint provides the PIC reference; k<64 points form the practical sparse frontier.

## Files

- `results/stage5b1_frontier_final.csv`: final 20-point frontier table.
- `results/stage5b1_geometry.csv`: generator/load decomposition of completion authority and effective-power proxy.
- `results/STAGE5B1_FRONTIER_FINAL.json`: machine-readable final frontier.
- `results/chunks/*.json`: independent importance-sampling batches and pilot diagnostics.
- `figures/stage5b1_scenario_optimal_frontier.pdf/png`: scenario-matched exact-PIC frontiers.
- `figures/stage5b1_robust_frontier.pdf/png`: robust-fixed two-scenario frontiers.
- `verify_stage5b1.py`: monotonicity and matched-baseline geometry regression checks.
