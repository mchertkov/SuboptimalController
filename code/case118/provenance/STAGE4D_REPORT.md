# Case118 Stage 4D: adaptive mechanical-cell interior-curvature prototype

Date: 2026-09-24

## Scope and status

Stage 4D is complete as the deliberately bounded feasibility prototype requested after Stage 4C. No manuscript or bibliography file was modified. The frozen case118 bus-90 event, Stage-3A H0--H3 architectures, and Stage-4 PIC definitions were left unchanged.

The question was narrow: can the positive Monte Carlo curvature values for the noise-deflated PIC candidate be promoted toward a domain-wide interior certificate by combining local mechanical cells with first/second/third variational bounds?

The answer from this prototype is **not with the present generic sup-norm third-derivative/Lipschitz closure**. The local mechanical variational equations themselves are well behaved on millisecond cells, but converting them to a uniform spatial Lipschitz bound for

`rho = tr[(D-Dtilde) Hess W-]`

introduces a Feynman--Kac value-third-derivative bound containing inverse powers of the scalar PIC temperature. This is overwhelmingly conservative, especially for activity-scaled H3 where `lambda_max = 1.7580135597e-4`.

This result does **not** indicate negative curvature. The Stage-4 pointwise screen remains strongly positive. It says that this particular analytic route cannot bridge from positive sampled centers to a continuum cell of useful size.

## Local mechanical-cell calculation

The reference centers are the frozen deterministic H3 scaled-LQR transient at `t = 0, 0.75, 1.50, 2.25, 2.80 s`. For edge-angle cell radii `0.02, 0.05, 0.10, 0.25 deg`, the local metric is

`||(eta,v)||_c^2 = eta^T L(theta_c) eta + v^T M v`.

At the smallest `0.02 deg` cells, over the five reference centers,

- `mu = 0.0930 ... 0.1087 1/s`,
- `C2 = 122.8 ... 149.7`,
- `C3 = 182.14 ... 183.48`.

These local first-variation growth rates are modest. Over a 2 ms cell step the worst scalar majorants are

- first variation `J1 <= 1.0002174`,
- second variation `H2 <= 0.29956`,
- third variation `K3 <= 0.50170`.

Thus the mechanical-metric idea is doing what it is supposed to do locally: it avoids the catastrophic global-sector first-variation growth and gives a numerically usable short-step variational propagation.

## Where the enclosure fails

To test whether this can close the curvature proof, I used an intentionally favorable stress test. For each reference center I retained the very small `0.02 deg` local drift constants for the **entire remaining horizon**. That assumption is more optimistic than a valid full-domain proof, because a real path will leave that one local cell. Compact-domain cost derivative bounds were then combined with the standard tilted Feynman--Kac third-derivative estimate

`||D^3 W|| <= C3 + 6 C1 C2/lambda + 8 C1^3/lambda^2`.

The Stage-4 center curvature was reduced by two reported Monte Carlo standard errors and used only as a numerical anchor. Even under this favorable construction, the largest edge-radius that the resulting spatial Lipschitz inequality could protect is microscopic:

| noise | best implied edge radius over tested centers |
|---|---:|
| homogeneous | `9.11e-12 deg` |
| activity-scaled | `1.61e-14 deg` |

Equivalently, for a still tiny `0.1 deg` cell, the analytic Lipschitz loss already exceeds the positive center margin by at least

- `1.10e10` for homogeneous H3,
- `6.21e12` for activity-scaled H3.

The obstruction is transparent in the raw table. At `t=2.8 s`, with only `0.2 s` remaining, the optimistic bound gives

- homogeneous: `W3_bound = 3.47e9`, `L_rho = 1.25e11`;
- activity-scaled: `W3_bound = 2.26e12`, `L_rho = 2.57e13`.

The dominant contribution is already the `8 C1^3/lambda^2` term. Refining the mechanical cell radius therefore does not repair the bottleneck; it is mainly a consequence of taking a worst-case tilted-measure derivative bound and, for activity-scaled H3, the very small scalar matching temperature.

## Rigorous status after Stage 4D

No new interior continuum cell is claimed certified. The Stage-4 Monte Carlo curvature centers remain numerical evidence, not deterministic anchors. The previously obtained cycle/energy terminal enclosure remains rigorous: at sigma=0.15 its positive terminal layer is about `2.866 ms` for homogeneous H3 and `1.618 ms` for activity-scaled H3.

Therefore the status remains:

- `J+`: exact control-completion value is an analytic HJB lower certificate; displayed values are Monte Carlo estimates of that exact lower value.
- `J-`: stable PIC candidate with strong positive representative curvature evidence plus a rigorous short terminal layer, but **not** a full-domain lower certificate.

## Decision / recommendation

I recommend **not expanding the present adaptive-cell construction into a full case118 compact-domain cover**. The short-step variation machinery is sound and can be retained, but the generic `D^3 W` to `L_rho` closure is too conservative by roughly 10--13 orders of magnitude even before confronting the high-dimensional covering problem.

A future attempt to certify `J-` would need a sharper structural argument -- for example, a direct PDE/semigroup positivity result, a contraction/monotonicity property specific to the deleted frequency directions, or a covariance-sensitive derivative estimate that avoids the global `lambda^{-2}` sup-norm bound. Merely adding more mechanical cells will not solve the present bottleneck.

For the current paper/campaign, the scientifically clean path is therefore to retain `J-` as a candidate diagnostic, use `J+` as the rigorous lower certificate, and move next to the case118 control-metric/reinforcement experiments where the activity-scaled scalar-matching bottleneck is already quantitatively large.

## Files

- `results/stage4d_local_constants.csv`: local mechanical constants for five transient centers and four cell radii.
- `results/stage4d_short_step_variations.csv`: first/second/third short-step majorants.
- `results/stage4d_lipschitz_stress_test.csv`: center margins, PIC temperatures, third-derivative bounds and implied cell radii.
- `results/STAGE4D_DECISION.json`: machine-readable status and recommendation.
- `scripts/run_stage4d_prototype.py`: complete reproducer.
- `src/cellwise_stage4d.py`: case118 local-cell and Feynman--Kac derivative-bound utilities.
