# Stage 3B result: independent case118 controller evaluation

Date: 2026-09-24

## Status

Stage 3B is complete. The Stage-3A architectures, H2 rank, feedback matrices, scalar gains, cost, integration step and saturation convention were kept fixed. The current manuscript and bibliography were not modified. No PIC lower-value calculation was performed in this stage.

The independent evaluation uses sigma_rms = 0.10, 0.15 and 0.20 for both homogeneous and activity-scaled all-bus noise, T = 3 s, dt = 2.5 ms, and 192 paths per scenario. Evaluation seeds are 31301 and 31302, 96 paths per seed. These are disjoint from every Stage-3A training/freeze seed. The same standard-normal streams are replayed for every policy within a scenario, giving common-random-number paired comparisons.

## Principal numerical result

The main effect is actuator-subspace rank. H0 (single slack actuator) and H1 (equal distributed rank-one mode) have negligible practical effect on the Stage-3B smooth objective and no observed crossing reduction on these 192-path ensembles. H2 (rank-3 Kron modes) produces a small but repeatable smooth-cost reduction and generally lowers the empirical crossing frequency. H3 (all 54 generator actuators) produces the clearest effect in both noise models.

For H3 LQR, the paired smooth-cost reductions relative to static balancing are 0.83%, 0.59%, and 0.55% for homogeneous noise at sigma = 0.10, 0.15, and 0.20; and 1.64%, 1.37%, and 1.29% for activity-scaled noise. The same-path 95% intervals for Delta J exclude zero in all six cases. The observed crossing-frequency reduction is 2.1--4.2 percentage points in five of the six scenarios and 0.5 point in the low activity-scaled scenario. These crossing differences are empirical safety diagnostics, not rigorous safety certificates.

### Core Stage-3B table

| noise | sigma | policy | J ± SE | crossing P [Wilson 95%] | q99 max angle (deg) | q99 gen |omega| | control effort |
|---|---:|---|---:|---:|---:|---:|---:|
| homogeneous | 0.10 | static | 0.038379 ± 0.000169 | 0.047 [0.025,0.087] | 91.97 | 2.891 | 0.000000 |
| homogeneous | 0.10 | H2_droop | 0.038313 ± 0.000168 | 0.031 [0.014,0.066] | 91.08 | 2.888 | 0.000112 |
| homogeneous | 0.10 | H3_lqr | 0.038061 ± 0.000166 | 0.026 [0.011,0.060] | 91.00 | 2.864 | 0.000497 |
| homogeneous | 0.15 | static | 0.077150 ± 0.000357 | 0.328 [0.266,0.397] | 103.33 | 3.285 | 0.000000 |
| homogeneous | 0.15 | H2_droop | 0.077062 ± 0.000356 | 0.297 [0.237,0.365] | 102.03 | 3.280 | 0.000140 |
| homogeneous | 0.15 | H3_lqr | 0.076692 ± 0.000352 | 0.292 [0.232,0.360] | 102.00 | 3.259 | 0.000811 |
| homogeneous | 0.20 | static | 0.131867 ± 0.000649 | 0.615 [0.544,0.681] | 117.20 | 4.306 | 0.000000 |
| homogeneous | 0.20 | H2_droop | 0.131698 ± 0.000639 | 0.594 [0.523,0.661] | 115.41 | 4.308 | 0.000177 |
| homogeneous | 0.20 | H3_lqr | 0.131141 ± 0.000631 | 0.583 [0.513,0.651] | 115.41 | 4.246 | 0.001251 |
| activity scaled | 0.10 | static | 0.022452 ± 0.000158 | 0.010 [0.003,0.037] | 89.04 | 3.069 | 0.000000 |
| activity scaled | 0.10 | H2_droop | 0.022390 ± 0.000157 | 0.005 [0.001,0.029] | 88.05 | 3.068 | 0.000108 |
| activity scaled | 0.10 | H3_lqr | 0.022083 ± 0.000153 | 0.005 [0.001,0.029] | 87.88 | 3.036 | 0.000493 |
| activity scaled | 0.15 | static | 0.041297 ± 0.000324 | 0.208 [0.157,0.271] | 101.45 | 4.571 | 0.000000 |
| activity scaled | 0.15 | H2_droop | 0.041222 ± 0.000323 | 0.182 [0.134,0.243] | 100.47 | 4.573 | 0.000130 |
| activity scaled | 0.15 | H3_lqr | 0.040731 ± 0.000315 | 0.167 [0.121,0.226] | 100.07 | 4.478 | 0.000804 |
| activity scaled | 0.20 | static | 0.067874 ± 0.000562 | 0.516 [0.445,0.585] | 114.00 | 6.213 | 0.000000 |
| activity scaled | 0.20 | H2_droop | 0.067761 ± 0.000559 | 0.490 [0.420,0.560] | 112.68 | 6.205 | 0.000162 |
| activity scaled | 0.20 | H3_lqr | 0.066998 ± 0.000544 | 0.479 [0.410,0.550] | 112.23 | 6.080 | 0.001240 |

## What Stage 3B says about H0--H3

**H0 and H1.** Their changes in smooth J are only about 0.004--0.045% across the tested scenarios, and the observed crossing counts are unchanged from static balancing. The fixed weak gains do not materially alter the severe bus-90 transient at these ranks.

**H2 (rank 3).** The rank-3 Kron subspace gives a modest but consistent improvement. H2 droop reduces smooth J by about 0.11--0.28% relative to static, and removes 1--6 crossings out of 192 depending on scenario. H2 LQR has lower control effort, but at these frozen scales H2 droop gives the stronger crossing reduction and slightly lower smooth J in every Stage-3B scenario.

**H3.** Full generator actuation reduces smooth J by roughly 0.55--1.64% and usually removes 4--8 crossings out of 192 relative to static. The q99 angle tail also moves downward by roughly 1--2 degrees in the moderate/high-stress cases. This is a much larger effect than H0/H1 and a clear increment over H2.

## Droop versus scaled LQR

The difference between the two feedback laws is much smaller than the difference between actuator ranks. For H3, scaled LQR uses less control effort than droop while giving nearly the same crossing behavior. It has the lower smooth cost for both activity-scaled noise and homogeneous sigma=0.10; droop is slightly lower for homogeneous sigma=0.20; at homogeneous sigma=0.15 they are unresolved. Averaged across the two noise models at each sigma, H3 LQR has the slightly lower smooth objective. Thus the Stage-3B evidence supports using H3 LQR as the primary deployed-cost benchmark in the later certificate comparison, while retaining droop as the transparent engineering baseline. This is a bookkeeping choice for the next stage, not a claim of globally optimal nonlinear feedback.

## Reproducibility against Stage 2

The independent Stage-3B static-balancing costs reproduce Stage 2 within Monte-Carlo resolution: the largest difference is 1.31 combined standard errors across the six noise/sigma points. The independent crossing estimates fluctuate more, as expected for binomial first-exit counts, but the Stage-2 and Stage-3B confidence ranges are compatible. This regression supports using the fresh Stage-3B sample as an independent controller-comparison set.

## Statistical interpretation

The absolute J standard errors are of order 1.5e-4 to 6.5e-4, but the common-random-number paired errors for controller-minus-static differences are much smaller. For H3 the paired Delta J intervals exclude zero robustly in all six scenarios. This is why the modest sub-percent smooth-cost effect can still be resolved. In contrast, crossing probabilities are based on only 192 Bernoulli paths per scenario, so differences of a few percentage points remain descriptive unless the safety experiment is enlarged or replaced by the planned first-exit certificate.

## Recommendation before PIC

Stage 3B can be closed without retuning. Preserve H0--H3 exactly as frozen. For the next PIC/certificate stage, carry both droop and LQR results as empirical upper-cost references, but use the frozen scaled-LQR controller as the primary J_Hk benchmark when one deployed policy per architecture is required, except that H2 droop should remain visible because its empirical first-exit behavior is consistently better at the frozen scales. Do not use the Stage-3B crossing frequencies as formal safety certificates.

No manuscript edits should be made until the next PIC/certificate calculations are reviewed.
