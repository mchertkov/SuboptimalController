# Case118 Stage 4C: hard-case variance-reduction audit

Date: 2026-09-24

## Scope and frozen inputs

Stage 4C addresses only the difficult completed-PIC estimate for the frozen case118 **H3, activity-scaled noise, sigma_rms = 0.15** problem. The Stage-3A architecture, Stage-3B deployed controller/cost, Stage-4 matching geometry, event bus 90, cost, horizon, and dt=0.0025 s discretization are unchanged. No manuscript or bibliography file was modified.

The target exact quantity remains the completed value J+ (W+ in legacy numerical labels). Its exact value is an analytic HJB lower value because control completion is a subsolution construction. Stage 4C changes only the Monte Carlo estimator used to evaluate that exact value numerically.

## Why Stage 4C was needed

The previous Stage-4 nominal H3 activity-scaled estimate used an actual-cost CARE importance guide with constant scale 1.25. Two 1536-path replications gave W+ = 0.02487767, replication SD 1.22e-4, mean ESS fraction 0.01135, and minimum ESS fraction 0.00750. This was adequate for the qualitative certificate story but not satisfactory for a stable quoted nominal value.

## Variance-reduction construction

We retained the same actual-cost CARE feedback matrix and adapted only a **single scalar scale as a function of time**. This avoids introducing a high-dimensional tuned controller into the certificate calculation.

A six-bin schedule (0.5 s per bin) was fitted by a cross-entropy / likelihood-score update on independent fitting seeds 49501--49504, 256 paths per seed. Starting from alpha=1.25, the smoothed schedule is

| time interval (s) | alpha |
|---|---:|
| 0.0--0.5 | 1.145035 |
| 0.5--1.0 | 1.164367 |
| 1.0--1.5 | 1.208995 |
| 1.5--2.0 | 1.213966 |
| 2.0--2.5 | 1.274357 |
| 2.5--3.0 | 1.293949 |

The structure is physically sensible: the guide is weakened during the early violent deterministic transient and strengthened toward the terminal cost. The schedule was then checked on fresh seeds 49601--49604 before the final estimate. On that check, the smoothed guide increased mean ESS fraction from 0.0354 for the constant-alpha guide to 0.0420. Its mean W+ was 0.024979, versus 0.024996 for the constant guide; the two are statistically consistent.

Generic CARE guides and a simple particle-resampling experiment were also screened. They did not improve the combination of overlap and value stability and were not used in the final estimator. In particular, maximizing observed ESS alone was rejected as a selection rule because an over-steered proposal can display a deceptively comfortable within-sample ESS while missing the low-action tail that controls the normalizing constant.

## Final independent refinement

The final selected guide was run on four completely fresh 512-path replications (seeds 49701--49704):

| seed | W+ | ESS fraction | absolute ESS |
|---:|---:|---:|---:|
| 49701 | 0.02498878 | 0.03184 | 16.30 |
| 49702 | 0.02494316 | 0.01796 | 9.19 |
| 49703 | 0.02499814 | 0.04488 | 22.98 |
| 49704 | 0.02493075 | 0.02051 | 10.50 |

The arithmetic mean of the four W+ estimates is 0.02496521 with replication SD 3.32e-5 and replication SE 1.66e-5. Because the unbiased Monte Carlo object is the desirability/normalizing constant rather than its negative logarithm, pooling the four equal-size normalizer estimates first gives the preferred value

**W+ = 0.02496286.**

A further independent 1024-path holdout (seed 49801) gave W+ = 0.02494497 with ESS fraction 0.01351, inside the four-replication range. Pooling all 3072 independent final-validation paths at the normalizer level gives

**W+ = 0.02495669.**

The difference between the equal-replication estimator and the all-path pooled estimator is only 6.2e-6. The Stage-4C recommendation is therefore to use **J+_H3(activity, sigma=0.15) approximately 0.02496**, and to quote **0.0250** in prose unless more digits are genuinely useful.

## Updated certificate gap

Using the frozen Stage-3B H3 deployed cost J = 0.04073063 and the all-path pooled Stage-4C completed value gives

- J - J+ = 0.01577394,
- (J - J+)/J = 0.38727, i.e. about 38.7%.

This changes none of the Stage-4 qualitative conclusions. The completed lower certificate remains much looser for activity-scaled H3 than for homogeneous H3 because the scalar matching temperature is extremely small (lambda_max = 1.7580136e-4). Stage 4C only makes the hard numerical value more reliable.

## Comparison with the previous Stage-4 estimator

Relative to the previous nominal run, the selected Stage-4C equal-replication calculation improves the mean ESS fraction from 0.01135 to 0.02880 (about 2.5x) and reduces the across-replication SD of W+ from 1.22e-4 to 3.32e-5 (about 3.7x), despite using fewer paths in the primary four-replication block. The independent 1024-path holdout confirms the same value range.

The previous 0.02487767 estimate should therefore be treated as superseded for downstream numerical tables by approximately 0.02496. The old estimate was not inconsistent with the refined result at its own replication-scale uncertainty; it was simply noisier.

## Status and next step

**Stage 4C is complete.** The hard activity-scaled H3 completion value is now stable enough for three-significant-digit use. No further sampling refinement is required before moving on.

The remaining certificate question is logical rather than Monte Carlo: whether to invest in the Stage-4D adaptive-cell/interval propagation needed to turn the noise-deflated J- from a numerically supported candidate into a domain-wide certified lower value. The alternative is to keep J- explicitly labeled as a candidate and move directly to the reinforcement/control-metric experiment, using J+ as the rigorous certificate.
