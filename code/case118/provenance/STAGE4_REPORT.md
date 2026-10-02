# Case118 Stage 4: PIC values and certificate status

Date: 2026-09-24

## Status

Stage 4 is numerically complete for the case118 PIC value campaign. The frozen Stage-3A architectures/controllers and the Stage-3B upper-policy results were not retuned. The manuscript and bibliography were not modified.

There are two logically different outcomes:

1. **Control completion / inflation (`J+`)**: the exact completed value is an HJB subsolution and therefore a rigorous lower value for the physical problem. The numbers below are Monte Carlo estimates of that exact value; the finite-sample estimates are not themselves rigorous confidence lower bounds.
2. **Noise deflation (`J-`)**: the numerical curvature evidence is strongly positive and an analytic terminal layer is obtained, but a domain-wide enclosure of the interior directional curvature over the whole compact stopped domain has not been completed. Therefore `J-` remains a **candidate lower value**, not a globally certified lower value.

## Frozen laboratory and numerical protocol

- Event: complete loss at bus 90 in the alpha=4 stressed IEEE case118 laboratory.
- Architectures: H0 = physical slack generator; H1 = equal distributed rank-one mode; H2 = frozen rank-3 post-event Kron mode; H3 = all 54 generator buses independently actuated.
- Control price: `R = 0.2 I` in orthonormal architecture coordinates.
- Smooth cost: `c_E=0.08`, `c_B=0.02`, `c_T=0.20`, with the same smooth barrier continuation used in Stages 2-3.
- Stage-3B deployed costs `J` are the frozen scaled-LQR costs with `dt=0.0025 s` and 192 paths per noise/sigma scenario.
- Noise-deflated Feynman-Kac estimates use two independent replications of 512 paths at `dt=0.005 s`; the frozen architecture LQR is used only as an importance-sampling guide and is removed exactly by Girsanov reweighting.
- Homogeneous completed values were refined with two independent replications of 1024 paths at `dt=0.0025 s`.
- At the primary `sigma_rms=0.15`, activity-scaled completed values were refined at `dt=0.0025 s` with an actual-cost CARE importance guide: 1024 paths/rep for H0, 512 for H1/H2, and 1536 for H3. The H3 activity-scaled problem remains the hardest sampling case because the scalar matching temperature is extremely small.
- Activity-scaled `sigma=0.10` and `0.20` completion values remain at `dt=0.005 s`; the nominal H3 `0.005 -> 0.0025` spot check changed `J+` by about `8.9e-5`, while the homogeneous H3 spot check exposed a larger time-step effect and motivated the full homogeneous `dt=0.0025` rerun.

## Primary PIC/certificate numbers: sigma_rms = 0.15

`J` is the Stage-3B deployed scaled-LQR cost. `J-` is the noise-deflated PIC candidate. `J+` is the architecture-specific completed PIC lower value. H4 is defined as the H3 completed oracle, so for H3 `H4 = J+`.

| noise | arch | J | J- | J+ | J - J+ | lambda_max | deleted rank | virtual rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| homogeneous | H0 | 0.077146 | 0.007642 | 0.075024 | 0.002122 | 0.004500 | 117 | 117 |
| homogeneous | H1 | 0.077146 | 0.007575 | 0.075024 | 0.002122 | 0.004500 | 117 | 117 |
| homogeneous | H2 | 0.077080 | 0.007925 | 0.075024 | 0.002056 | 0.004500 | 115 | 115 |
| homogeneous | H3 | 0.076692 | 0.018356 | 0.075024 | 0.001668 | 0.004500 | 64 | 64 |
| activity scaled | H0 | 0.041286 | 0.009109 | 0.041440 | -0.000153* | 0.036534 | 117 | 117 |
| activity scaled | H1 | 0.041293 | 0.007476 | 0.036321 | 0.004972 | 0.001308 | 117 | 117 |
| activity scaled | H2 | 0.041234 | 0.007511 | 0.035570 | 0.005664 | 0.001089 | 117 | 117 |
| activity scaled | H3 | 0.040731 | 0.007650 | 0.024878 | 0.015853 | 0.000176 | 117 | 117 |

`*` The tiny H0 activity-scaled point-estimate inversion is a Monte Carlo issue, not a violation of the analytic lower-bound ordering. Its magnitude is `1.53e-4`, whereas the approximate combined Monte Carlo standard error is `3.28e-4`. The exact completed value and exact deployed-policy expectation obey the correct ordering.

For homogeneous noise, the completion is the same process for H0-H3 because the scalar matching temperature is `lambda_max = 0.2 sigma^2`. At sigma=0.15 the automatic completed certificate gap is therefore about 2.75% of the H0/H1 deployed cost and 2.17% of the H3 deployed cost. This is a genuinely tight lower certificate at case118 scale.

Activity-scaled noise behaves differently. At sigma=0.15 the scalar temperatures are `(0.036534, 0.001308, 0.001089, 0.0001758)` for H0-H3. Adding physical generator modes exposes weak-noise generator channels, forcing the one-temperature scalar PIC match downward. The completion traces are approximately `(67.67, 2024.33, 2422.21, 14832.27)`. Thus physical H3 improves the actual deployed cost, but its scalar-PIC completion becomes much more optimistic. The H3/H4 gap is about `0.01585`, or 38.9% of the deployed H3 cost. This is the clearest case118 manifestation of the scalar-matching bottleneck.

The common H4 lower oracle at sigma=0.15 is `0.075024` for homogeneous noise and `0.024878` for activity-scaled noise. The latter lower-bounds every H0-H3 physical architecture, but is much looser than the architecture-specific H0/H1/H2 completions.

## Sampling diagnostics

Noise-deflated sampling is well conditioned. At sigma=0.15 the mean ESS fractions are approximately 0.9998, 0.9996, 0.9941 and 0.9925 for homogeneous H0-H3, and 0.9999, 0.9994, 0.9779 and 0.9538 for activity-scaled H0-H3.

Completed homogeneous sampling is also stable: the refined `dt=0.0025` runs have ESS fraction about 0.785. For activity-scaled completion at sigma=0.15, the actual-cost CARE guide raises the ESS fractions to approximately 0.999 (H0), 0.693 (H1), 0.614 (H2), and 0.0114 (H3). H3 remains rare-event dominated, but the two 1536-path replications give `J+ = 0.024878` with replication standard deviation about `1.22e-4`; this is adequate for the qualitative certificate comparison, though further variance reduction would be appropriate before quoting more than three significant digits.

## Noise-deflation curvature screen

The physical HJB defect of the noise-deflated value is the deleted-noise directional curvature trace. At sigma=0.15 we evaluated:

- all H0-H3 architectures at the initial post-event state;
- H2 and H3 at `t = 0.75, 1.50, 2.25, 2.80 s` along the deterministic frozen-H3 controlled transient;
- both homogeneous and activity-scaled noise;
- four common-random-number Hutchinson directions per state with 128 Feynman-Kac paths per query.

All 24 estimated curvature traces are positive, and every individual sampled directional second difference is positive. The minimum trace is `0.0099533` (activity-scaled H0 at t=0), and the minimum individual sampled direction is `0.0094351` (activity-scaled H3 at t=0.75 s). Representative standard errors are one to two orders of magnitude below the trace values. This is substantially positive numerical evidence, but it is still a representative-state screen rather than a continuum-domain proof.

For H2/H3 the trace tends to rise near terminal time. At `t=2.8 s`, the estimates are approximately `0.04553/0.04294` for homogeneous H2/H3 and `0.01576/0.01562` for activity-scaled H2/H3.

## Analytic terminal layer

The terminal curvature calculation was upgraded to the cycle/energy-consistent compact-tube enclosure rather than the looser independent-edge bound. The tube used

- energy cap `E/n <= 0.34`,
- `|omega_i| <= 6.5`,
- branch-angle cap `|delta_e| <= 84 deg`,
- the same 82-degree smooth-barrier guard.

At sigma=0.15 the resulting guaranteed positive terminal layers are:

| noise | H0 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|
| homogeneous | 2.945 ms | 2.945 ms | 2.942 ms | 2.866 ms |
| activity scaled | 1.602 ms | 1.623 ms | 1.622 ms | 1.618 ms |

Thus the terminal sign is analytically controlled on the stated compact tube. This is stronger than the representative Monte Carlo curvature screen near T, but it does not by itself propagate the sign through the full three-second interior.

## Certificate interpretation

The principal rigorous numerical story is therefore the **completion certificate**. The exact `J+` is a lower value without any curvature hypothesis, and the deployed Stage-3B LQR cost is an upper value. Their difference provides the practical certificate gap, subject to Monte Carlo estimation error in the displayed numerical values.

The noise-deflated `J-` values are numerically stable and their curvature evidence is uniformly positive on the tested states; however the missing domain-wide interior enclosure remains a real logical gap. They should continue to be called **candidate lower values** in the manuscript until an adaptive-cell/interval proof covers the compact stopped domain. No global `J-` certification is claimed in this package.

## Main scientific conclusion

Case118 strengthens the control-theoretic message. Under homogeneous noise, completion gives a surprisingly tight automatic certificate even though most stochastic directions are not physically actuated: for H3 the numerical gap is only about 2.2% at nominal noise. Under activity-scaled noise, the one-temperature scalar matching condition is the limiting mechanism: increasing physical actuator rank improves the real controller but exposes weaker-noise generator channels and causes the completed H4 oracle to become much more optimistic. This is precisely the regime in which control-metric design, noise-proportional authority, or physical reinforcement should be expected to matter.

## Files to use downstream

- `results/stage4_pic_certificate_summary_final.csv` -- final merged J/J-/J+ table.
- `results/stage4_nominal_sigma015_final.csv` -- primary nominal table.
- `results/stage4_geometry.csv` -- matching temperatures and completion/deletion geometry.
- `results/stage4_curvature_screen.csv` -- representative curvature estimates.
- `results/stage4_terminal_cycle_certificate.csv` -- analytic terminal-layer bounds.
- `results/stage4_pic_dt_spotcheck.csv` -- time-step spot checks.
- `results/STAGE4_CERTIFICATE_STATUS.json` -- machine-readable status distinction.
- `figures/stage4_pic_hierarchy_*.pdf/png` and `figures/stage4_curvature_*.pdf/png` -- figures for later manuscript integration.

The manuscript remains frozen. The next decision is whether to spend another stage on a full adaptive-cell/interval interior curvature enclosure for `J-`, or to treat `J+` as the rigorous certificate and move directly to the case118 reinforcement/control-metric experiments.
