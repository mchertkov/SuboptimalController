#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, json, math
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'results/stage5b2_sparse_frontier.csv')
dircmp=pd.read_csv(ROOT/'results/stage5b2_shadow_vs_noise.csv')
pair=pd.read_csv(ROOT/'results/stage5b2_paired_vs_baseline.csv')
anch=pd.read_csv(ROOT/'results/stage5b2_activity_anchored_frontier.csv')
rankA=pd.read_csv(ROOT/'results/stage5b2_load_ranking_activity_scaled.csv')

def tbl(noise):
 s=df[df.noise==noise].sort_values(['k','method']); lines=['| method | k | added authority | J | paired ΔJ vs k=0 | crossing | control effort |','|---|---:|---:|---:|---:|---:|---:|']
 for _,r in s.iterrows():
  lines.append(f"| {r.method} | {int(r.k)} | {r.added_authority_sum:.4f} | {r.J:.6f} | {r.delta_J_vs_k0:+.6f} | {r.crossing:.3f} | {r.control_effort:.6f} |")
 return '\n'.join(lines)

def top(rankcol,k=5): return ', '.join(map(str,rankA.nsmallest(k,rankcol).bus_id.astype(int).tolist()))
text=f'''# Case118 Stage 5B2 — sparse physical load-side reinforcement

Date: 2026-09-24

## Status

Stage 5B2 is complete for the two scenario-matched case118 baselines at `sigma_rms=0.15`. The bus-90 event, stochastic laboratory, smooth objective, generator control metrics, and Stage-5B1 exact-PIC reference geometry were kept fixed. No manuscript or bibliography files were modified.

The purpose of this stage is to test whether a sparse set of real load-side fast-flexibility devices can recover useful deployed-control benefit without physically constructing the full exact-PIC completion.

## Design and budget rule

The two primary baselines are exactly the scenario-matched designs frozen in Stage 5A:

- homogeneous noise: uniform H3 generator authority;
- activity-scaled noise: activity-specific/noise-proportional H3 generator authority.

For both of these baselines, the least-inflated exact completion is purely load-side on the 64 non-generator buses. Its load authority target is `gplus_i = Gamma_i^2/lambda_max`.

Device counts are `k = 5, 10, 20, 40, 64`. Two placement rules are compared:

1. **noise-only:** descending local load-bus noise variance `Gamma_i^2`; for homogeneous noise this ranking is exactly degenerate, so bus ID is used only as a reproducible tie-break;
2. **H4-LQR shadow proxy:** descending `0.5 E[(R_i u_i)^2]` estimated under the full least-completed H4 LQR guide. Since `u_i=-g_i partial_i V` in the matched quadratic geometry, this is an approximate dynamics-aware marginal-value proxy per unit authority; it is not an exact nonlinear PIC shadow derivative.

At each k the two placement rules receive exactly the same total added Gramian authority. The common budget is the largest one that can be distributed over either selected set without exceeding the corresponding exact-completion authority bus-by-bus. Each selected set receives this budget in proportion to its own `gplus_i`. Therefore every sparse design satisfies `0 <= Delta G_sparse <= Delta G_exact`, and k=64 recovers the full least exact completion.

The resulting common added-authority budgets are:

- homogeneous: 25, 50, 100, 200, 320 (7.81%, 15.63%, 31.25%, 62.5%, 100% of full completion);
- activity-scaled: 2.233, 8.487, 15.210, 25.946, 34.543 (6.47%, 24.57%, 44.03%, 75.11%, 100%).

For every sparse system the nodal LQR was recomputed with the physical generator metric and the added load-device prices `R_i=1/g_i`. The scalar scale was kept at `alpha=0.005`, rather than tuned separately by placement. Independent 32-path spot checks at k=20 and k=64 for both noise models scanned `alpha=0.0025, 0.005, 0.01, 0.02` and selected 0.005 in all four checks.

Evaluation uses fresh common-random-number paths, seeds `53201,53202`, 96 paths/seed, hence N=192 per policy, `dt=0.0025 s`, T=3 s. First crossing of 90 degrees is recorded separately and is not part of the smooth objective.

## Common rigorous lower oracle

Because every sparse physical Gramian lies below the same least exact completion, completing the residual authority always produces the same matched problem. Thus the Stage-5A/5B1 completed value remains a rigorous lower value for every sparse physical architecture in the corresponding scenario:

- homogeneous: `Jplus = 0.0750242`;
- activity-scaled: `Jplus = 0.0405742`.

At k=64 the physical architecture itself equals the exact completion, so this `Jplus` is the exact optimal value of that reinforced physical control problem. The displayed Monte-Carlo estimates of deployed J and of Jplus still carry sampling error.

## Homogeneous-noise sparse frontier

{tbl('homogeneous')}

The full 64-load completion reduces the deployed scaled-LQR cost by `0.001509` relative to the generator-only H3 policy on exactly the same Brownian paths, a 1.97% reduction. The paired standard error of this difference is `1.58e-5`. The empirical crossing fraction falls from 0.302 to 0.250.

The gain is roughly proportional to installed load authority: with 31.25% of the full completion authority, k=20 recovers about 30% (noise tie-break) or 27% (shadow proxy) of the *deployed-LQR* k=64 improvement; at 62.5% authority, k=40 recovers about 61% or 59%. Since homogeneous noise is spatially identical on all load buses, the noise-only ranking carries no physical placement information. The H4 shadow ranking does not consistently outperform this arbitrary tie-break: at k=20 and 40 its total J is larger by about 4e-5. We therefore do not claim a robust localization advantage under homogeneous forcing from this proxy.

## Activity-scaled sparse frontier

{tbl('activity_scaled')}

Here the placement rules separate cleanly. The five highest-noise load buses are `{top('noise_rank')}`, whereas the five highest H4-shadow buses are `{top('shadow_rank')}`; the two top-five sets have zero overlap. At equal authority budgets, the H4-shadow placement has lower J than the noise-only placement for k=5,10,20,40. The paired shadow-minus-noise differences are approximately `-1.95e-5`, `-2.10e-5`, `-2.25e-5`, and `-1.14e-5`, respectively, and all four 95% CRN intervals exclude zero.

Relative to the generator-only deployed controller, H4-shadow placement recovers about 8%, 35%, 70%, and 79% of the *deployed-LQR* improvement reached at k=64 for k=5,10,20,40, respectively. The corresponding noise-only placement initially makes the total smooth objective slightly worse at k=5 and k=10, becomes essentially neutral at k=20, and recovers about 47% of the k=64 deployed improvement at k=40.

This does not mean that the noise-only devices worsen the physical state trajectory. In fact, the state-cost component `J-control_effort` decreases as devices are added for both placement rules. At small k the additional quadratic control expenditure offsets that state-cost improvement, which is why total J can initially rise.

The empirical 90-degree crossing fraction is 0.167 for every activity-scaled policy in this 192-path common sample. Thus this stage provides no evidence that the sparse devices change first-exit risk in the activity-scaled case. The smooth-performance result and the safety statistic should remain separate.

## Anchored activity-scaled absolute values

The 192-path Stage-5B2 sample happens to have a low absolute generator-only J. To avoid mistaking that common offset for a certificate tightening, the activity-scaled table is also anchored to the independent Stage-5A3 3072-path generator-only audit, `J=0.04104670 +/- 0.00007645`, while retaining the high-precision CRN policy differences measured here. This gives:

| method | k | anchored J | J-Jplus | gap/J |
|---|---:|---:|---:|---:|
'''
for _,r in anch.iterrows():
    text+=f"| {r.method} | {int(r.k)} | {r.J_anchored:.6f} | {r.gap:.6f} | {100*r.gap_fraction:.2f}% |\n"
text+='''

The important consequence is that even the full k=64 physical completion with the simple scaled LQR closes only a small part of the generator-only certificate gap: the anchored deployed value is about 0.041012 versus the exact completed optimum 0.040574. Thus the exact-PIC oracle indicates more potential benefit than this simple linear deployed policy extracts. Sparse placement should therefore be interpreted as a practical-controller frontier, not as an estimate of the unrestricted optimal reinforcement value.

## Robust-fixed benchmark

The fixed robust generator metric remains the useful two-scenario benchmark from Stage 5A3 (roughly 4.8% and 5.3% completion gaps under homogeneous and activity-scaled noise). It is not placed on the present pure load-side sparse frontier because, unlike the two scenario-matched generator metrics, its least completion generally contains virtual generator directions as well as load directions. A k=64 load-only reinforcement would therefore not reproduce its exact PIC completion. Mixing those two geometries would make the endpoint comparison misleading.

## Interpretation

Stage 5B2 supports three conclusions.

1. **Exact completion is a useful oracle, not an installation prescription.** Useful deployed improvement is obtained before all 64 load directions are physically added, while k=64 provides the exact-PIC reference endpoint.
2. **Placement matters primarily for heterogeneous forcing.** Under activity-scaled noise, the dynamics-aware H4 proxy consistently beats local-noise ranking at equal authority. Under homogeneous forcing, local noise contains no placement information and the current H4 proxy does not establish a statistically robust advantage over an arbitrary tie-break.
3. **Control-metric redesign did most of the work in the activity-scaled case.** Once generator authority is noise-matched, the remaining load-side reinforcement produces only a small additional improvement for the simple deployed LQR. This is consistent with Stage 5A3's already tight activity-specific certificate and argues for redesigning existing authority before installing many new devices.

This stage still does **not** compute the formal price of linearisability relative to a globally optimal unrestricted reinforcement design: the sparse curves are deployed-controller upper costs, not unrestricted optimal values. They are nevertheless the correct practical frontier for the controller family tested here.

## Recommended next step

The case118 control-metric/reinforcement campaign is now sufficiently complete to support manuscript integration. An optional engineering Stage 5B3 could add saturation and finite-energy/SoC constraints to a small set of representative sparse designs (for example k=10 and k=20 H4-shadow plus k=64), but that is a device-realism sensitivity rather than a prerequisite for the current control-theoretic conclusion.

## Files

- `results/stage5b2_sparse_frontier.csv` — all primary sparse-policy results.
- `results/stage5b2_paired_vs_baseline.csv` — CRN paired effects versus generator-only H3.
- `results/stage5b2_shadow_vs_noise.csv` — direct matched-budget placement comparison.
- `results/stage5b2_activity_anchored_frontier.csv` — activity-scaled values anchored to the 3072-path Stage-5A3 baseline audit.
- `results/stage5b2_design_*.csv` — device sets, authority budgets and effective-power proxies.
- `results/stage5b2_load_ranking_*.csv` — noise and H4-shadow scores/ranks for every load bus.
- `data/stage5b2_gains_*.npz` — recomputed LQR gains and H4 shadow data.
- `results/alpha_scan/` — independent scalar-scale spot checks.
- `figures/stage5b2_*.pdf/png` — frontier and first-exit figures.
- `verify_stage5b2.py` — regression checks.
'''
(ROOT/'STAGE5B2_REPORT.md').write_text(text)
