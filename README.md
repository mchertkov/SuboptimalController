# Linearly solvable completion certificates for stochastic network control

Code, frozen data and notebooks for

> M. Chertkov, *How suboptimal is my controller allowed to be? Linearly solvable completion
> certificates for stochastic network control*, submitted to SIAM Journal on Control and
> Optimization (2026).

**The question.** On a large network the optimal controller is out of reach, so the practical
question is how far an *implementable* controller is from optimal. Any subsolution of the
Hamilton–Jacobi–Bellman (HJB) equation bounds the optimal cost from below; simulating the
deployed policy bounds it from above; the difference certifies the permissible suboptimality.

**The construction.** *Control completion* enlarges the control Gramian until it matches the
physical noise. The completed problem is exactly linearly solvable (path-integral control), its
value is a Feynman–Kac log-partition function, and its residual in the physical HJB equation is
exactly the energy of the fictitious control — so it is a subsolution automatically.

**Headline (IEEE 118-bus, severe load loss, simple generator controller).** Certified gap
2.8% under homogeneous noise; under heterogeneous noise 39% with uniform control prices and
1.2% after the same total authority is priced in proportion to the noise.

## Repository layout

| path | contents |
|---|---|
| `notebooks/01_theory_checks_LQ.ipynb` | every theorem of Sections 3–5 checked **exactly** on a linear–quadratic problem (no Monte Carlo) |
| `notebooks/02_case118_results.ipynb` | all case118 tables and figures, regenerated from `data/frozen/` in seconds |
| `notebooks/03_case39_results.ipynb` | all case39 figures and Table 2, regenerated from cached results in seconds |
| `notebooks/case39_narrative/` | seven narrative notebooks explaining each case39 experiment family |
| `notebooks/build_notebooks.py` | rebuilds notebooks 01–03 |
| `scripts/paper_figures.py` | Figure 1 and the case118 figures, from `data/frozen/` |
| `data/frozen/` | frozen case118 result tables (CSV/JSON) used by the paper |
| `code/case118/` | the case118 campaign, seven stages in execution order, with inputs and reference results |
| `code/case39/` | the case39 campaign: model, PIC, certification and figure code with cached results |
| `figures/generated/` | output of `scripts/paper_figures.py` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_theory_checks_LQ.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_case118_results.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_case39_results.ipynb
```

Each notebook runs in under a minute. They regenerate the paper's figures from cached results;
they do **not** rerun the Monte Carlo campaigns.

## Full recomputation

The Monte Carlo campaigns take hours. Run them in the order documented in
`code/case118/README.md` (stages 3B → 4 → 4C → 4D → 5A → 5B1 → 5B2) and
`code/case39/scripts/recompute_notes.py`. Seeds are frozen; tuning and evaluation seeds are
disjoint; paired comparisons use common random numbers. Rerunning a stage overwrites its cached
tables, after which the notebooks regenerate the figures from the new values.

## Notation in the data files

The code predates the paper's notation. In data files and variable names:

| code | paper |
|---|---|
| `Wplus`, `W+` | completed value $\mathcal J^+$ |
| `Wminus`, `W-` | deflated value $\mathcal J^-$ |
| `lambda_max` | maximal matching temperature $\lambda_\star$ |
| `inflation_trace`, `completion_trace` | completion burden $\mathfrak I$ |
| `stage...` | internal campaign stage; not referenced in the paper |

## Status of the numbers

The completed value $\mathcal J^+$ is an **exact** lower bound on the optimal cost; its numerical
value is a Monte Carlo estimate, reported with replicate standard errors and effective sample
sizes. The deflated value $\mathcal J^-$ is a lower bound only under a curvature condition that
has not been verified on the whole domain. First-crossing probabilities are empirical. See Table
5 of the paper (numbers refer to the submitted version).

## Citation and license

See `CITATION.cff`. Code: MIT license (`LICENSE`). Network data are the MATPOWER case files.
