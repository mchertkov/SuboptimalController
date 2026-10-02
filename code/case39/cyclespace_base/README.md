# Cycle-space certificates — simulation package for the revised paper

This is the **revised numerical package** corresponding to the final marked-up
manuscript.  It supersedes the simulation code in the original
`CycleSpaceCertificate09_04_26.zip`.

## What changed relative to the early draft

1. **No `pandapower` dependency.** `case39.m` and `case118.m` are read directly
   from the standard MATPOWER 8.1 case files. The loader downloads them on first
   use, or you can place them manually in `data/` for offline use.
2. **Cycle-space Phase I.** Before Newton iteration, a linear program maximizes
   the slack inside `|F_e/K_e|<1`. Therefore a bad DC starting flow is no longer
   mistaken for non-existence of a feasible cycle correction.
3. **Correct disturbance interpretation.** The transient experiment is an
   **instantaneously balanced injection step**
   `p_eff = p + dP e_delta - dP gamma`, not a raw load rejection followed by a
   droop/governor transient.
4. **Critical energy.** Each cohesive boundary face is initialized by a
   relative-interior LP and minimized using equality elimination plus a
   logarithmic-barrier Newton solve. This replaces the fragile warm-started SLSQP
   routine from the early draft.
5. **Bus labels.** The case39 load block is at MATPOWER bus **20**; the fully
   concentrated balancing action is at MATPOWER bus **38**. The static binding
   bridge is MATPOWER line **6--31**.
6. **Finite-horizon simulation.** The reported calculation uses RK4 with
   `T=20`, `dt=0.004`; optional refinement checks use `dt=0.002` and `T=30`.

## Files

- `CycleSpaceCertificates.ipynb` — annotated notebook reproducing the numerical study.
- `matpower_loader.py` — minimal parser/downloader for native MATPOWER `.m` cases.
- `grids.py` — lossless fixed-voltage reductions of case39 and case118.
- `syncnet.py` — incidence/cycle basis, convex dual, Phase-I LP, DCB test,
  primal equilibrium, continuation.
- `transient.py` — balanced-step construction, face-wise critical energy,
  RK4 swing simulation, threshold bisections.
- `exp1_theory.py` — derivative/reconstruction checks, ring counts, static thresholds.
- `exp2_transient.py` — five balanced-step allocations in case39.
- `figures.py` — regenerates `fig1_theory` and `fig2_transient`.
- `verify_results.py` — compares a fresh run with the manuscript numbers.
- `paper_results.py` — the values printed in the final paper, for regression checks.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

The first run downloads the MATPOWER 8.1 `case39.m` and `case118.m` files from
MATPOWER's public GitHub repository. For offline use put those files in `data/`.

## Reproduce the paper

Run the notebook:

```bash
jupyter notebook CycleSpaceCertificates.ipynb
```

or run the scripts:

```bash
python exp1_theory.py
python exp2_transient.py
python figures.py
python verify_results.py
```

`python run_all.py` performs all four steps. The transient certificate is the
expensive part because every bisection point solves all `2L=92` case39 boundary
faces.

## Values to recover

Static loading multipliers:

| system | DCB | strict-cohesion supremum | continued locally stable branch |
|---|---:|---:|---:|
| MATPOWER case118 | 4.363697 | 5.068736 | 5.268367 |
| MATPOWER case39 | 5.544474 | 5.544472 | 5.544472 |

Case39 balanced-step thresholds at `alpha=4`, in MW:

| concentration | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| energy-certified | 2035 | 1751 | 1442 | 1186 | 991 |
| finite-horizon cohesive simulation | 2724 | 2724 | 2585 | 2442 | 2313 |
| exact static strict cohesion | 2724 | 2724 | 2724 | 2724 | 2724 |
| DCB static screen | 2724 | 2724 | 2724 | 2724 | 2724 |

The load block is about 2724.5 MW because the lossless reduction projects the
MATPOWER active-power injections onto `sum(p)=0` before multiplying by
`alpha=4`. The pre-step maximum line-angle difference is about 46.17 degrees.

### Numerical interpretation

- `alpha_cohesive` is a numerical approximation of the **supremum** of the open
  strict-cohesion cell, so its last digits depend on the bisection/Newton tolerance.
- `alpha_continued` is a warm-started **numerical branch limit**, not a certified
  saddle-node.
- transient thresholds are reported at roughly 1 MW bisection resolution; a
  difference of one MW due to floating-point solver tolerances is not meaningful.
- the face decomposition defining `c*` is exact mathematically; the individual
  convex minima are floating-point numerical solves, not interval-certified values.
