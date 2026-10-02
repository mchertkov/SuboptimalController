"""build_notebooks.py -- assemble the three companion notebooks (run from notebooks/)."""
import nbformat as nbf


def make(cells, name):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_markdown_cell(c[1].strip()) if c[0] == "md"
                   else nbf.v4.new_code_cell(c[1].strip()) for c in cells]
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, name)
    print("wrote", name, len(cells), "cells")


# =========================================================================== #
#  01  theory checks on a linear-quadratic problem
# =========================================================================== #
T1 = [("md", r"""
# 01 — The theorems, checked exactly on a linear–quadratic problem

Companion to *How suboptimal is my controller allowed to be? Linearly solvable completion
certificates for stochastic network control*.

For linear dynamics $\mathrm dX=(AX+Bu)\,\mathrm dt+\sigma\,\mathrm dW$ with quadratic costs
$V=\tfrac12x^\top Q_xx$, $\phi=\tfrac12x^\top P_Tx$, every value function is
$\tfrac12x^\top P(t)x+c(t)$ with a Riccati matrix $P$ and a scalar offset $c$. That makes
every statement of Sections 3–5 checkable **exactly**, with no Monte Carlo:

| check | statement |
|---|---|
| A | admissible matching temperature $\lambda_\star$ (Lemma 3.1) |
| B | completion: $\mathcal J^+\le J^\star$ and residual $=\tfrac12\|v^+\|^2$ (Theorem 3.3) |
| C | the certificate gap splits exactly into structural + policy terms (Corollary 3.4) |
| D | least inflation is tightest (Proposition 3.5) |
| E | deflation is a lower bound here, and the two relaxations are **not ordered** (Remark 3.11) |
| F | noise-proportional pricing (Proposition 4.3) |
| G | Jensen bias of the Monte Carlo estimator (Lemma 5.1) and the vacuous Hoeffding bound (Remark 5.3) |

The system is deliberately underactuated: noise in all four directions, one actuator.
"""),
("co", r"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm, eigh
np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(1)

n = 4
A = np.array([[0, 1, 0, 0], [-2, -0.4, 1, 0], [0, 0, 0, 1], [1, 0, -1.5, -0.3]], float)
B = np.array([[0.], [1.], [0.], [0.]])            # one actuator
R = np.array([[0.5]])
sig = np.diag([0.3, 0.5, 0.2, 0.4])                # noise in every direction
Q = sig @ sig.T                                    # Q = 2D
D = Q / 2
G = B @ np.linalg.solve(R, B.T)                    # control Gramian, rank 1
Qx, PT, T = np.eye(n), 2 * np.eye(n), 3.0
x0 = np.array([1.0, 0.0, -0.5, 0.3])
print("rank G =", np.linalg.matrix_rank(G), " rank Q =", np.linalg.matrix_rank(Q))
"""),
("md", r"""
## A. The admissible matching temperature
$\lambda G\preceq Q$ holds exactly for $\lambda\le\lambda_\star=[\lambda_{\max}(Q^{-1/2}GQ^{-1/2})]^{-1}$.
"""),
("co", r"""
Qih = np.linalg.inv(np.real(sqrtm(Q)))
lam_star = 1.0 / np.max(np.linalg.eigvalsh(Qih @ G @ Qih))
mineig = lambda M: np.min(np.linalg.eigvalsh((M + M.T) / 2))
print(f"lambda_star = {lam_star:.6f}")
print(f"min eig(Q - lambda_star G)        = {mineig(Q - lam_star * G): .2e}   (>= 0, touches 0)")
print(f"min eig(Q - 1.01*lambda_star G)   = {mineig(Q - 1.01 * lam_star * G): .2e}   (< 0: inadmissible)")
"""),
("md", r"""
## Riccati machinery
For a control Gramian $\mathcal G$ and diffusion $\mathcal D$:
$-\dot P=A^\top P+PA+Q_x-P\mathcal GP$, $P(T)=P_T$, and $c(0)=\int_0^T\operatorname{tr}(\mathcal DP)\,\mathrm dt$.
"""),
("co", r"""
tgrid = np.linspace(0, T, 3001)

def riccati(Gm):
    def rhs(t, p):
        P = p.reshape(n, n)
        return (-(A.T @ P + P @ A + Qx - P @ Gm @ P)).ravel()
    sol = solve_ivp(rhs, [T, 0], PT.ravel(), t_eval=tgrid[::-1], rtol=1e-11, atol=1e-13)
    return sol.y.T[::-1].reshape(-1, n, n)          # P on tgrid, increasing time

def value(Pt, Dm, x):
    c0 = np.trapezoid([np.trace(Dm @ P) for P in Pt], tgrid)
    return 0.5 * x @ Pt[0] @ x + c0

P_star = riccati(G)                       # physical optimum
J_star = value(P_star, D, x0)
print(f"J* (exact, LQG) = {J_star:.8f}")
"""),
("md", r"""
## B. Completion is an automatic lower bound, with residual $\tfrac12\|v^+\|^2$
Enlarge $G$ to $G_+=Q/\lambda_\star$. For quadratic $\mathcal J^+=\tfrac12x^\top P_+x+c_+$ the physical
residual is $\tfrac12x^\top P_+\Delta G\,P_+x$. We check the identity at random states and times by
computing the residual **directly** from its definition (2.2), with a finite-difference $\dot P_+$.
"""),
("co", r"""
Gp = Q / lam_star
dG = Gp - G
P_plus = riccati(Gp)
J_plus = value(P_plus, D, x0)
print(f"J+ = {J_plus:.8f}  <=  J* = {J_star:.8f} : {J_plus <= J_star}")

cplus = np.array([np.trapezoid([np.trace(D @ P) for P in P_plus[k:]], tgrid[k:]) for k in range(len(tgrid))])
worst = 0.0
for _ in range(200):
    k = rng.integers(5, len(tgrid) - 5); x = rng.normal(size=n)
    Pdot = (P_plus[k + 1] - P_plus[k - 1]) / (tgrid[k + 1] - tgrid[k - 1])
    cdot = (cplus[k + 1] - cplus[k - 1]) / (tgrid[k + 1] - tgrid[k - 1])
    grad, hess = P_plus[k] @ x, P_plus[k]
    res = 0.5 * x @ Pdot @ x + cdot + (A @ x) @ grad + np.trace(D @ hess) + 0.5 * x @ Qx @ x \
          - 0.5 * grad @ G @ grad
    pred = 0.5 * grad @ dG @ grad                      # = 1/2 ||v+||^2
    worst = max(worst, abs(res - pred) / max(1.0, abs(pred)))
    assert pred >= 0
print(f"max relative |Res[J+] - 1/2||v+||^2| over 200 random (x,t): {worst:.2e}")
"""),
("md", r"""
## C. The gap splits exactly (Corollary 3.4)
Deploy a deliberately suboptimal constant feedback $u=-Kx$ (half of the steady LQR gain). All
expectations are exact via the second-moment ODE $\dot\Sigma=(A-BK)\Sigma+\Sigma(A-BK)^\top+Q$.
Corollary 3.4 says
$J^u-\mathcal J^+=\tfrac12\int\operatorname{tr}(P_+\Delta GP_+\Sigma)+\tfrac12\int\operatorname{tr}\big((K-K_+)^\top R(K-K_+)\Sigma\big)$,
$K_+=R^{-1}B^\top P_+$.
"""),
("co", r"""
K = 0.5 * np.linalg.solve(R, B.T @ P_star[0])
Acl = A - B @ K
def sig_rhs(t, s):
    S = s.reshape(n, n); return (Acl @ S + S @ Acl.T + Q).ravel()
Sig = solve_ivp(sig_rhs, [0, T], np.outer(x0, x0).ravel(), t_eval=tgrid, rtol=1e-11, atol=1e-13).y.T.reshape(-1, n, n)
J_u = 0.5 * np.trace(PT @ Sig[-1]) + np.trapezoid([0.5 * np.trace((Qx + K.T @ R @ K) @ S) for S in Sig], tgrid)
struct = np.trapezoid([0.5 * np.trace(P @ dG @ P @ S) for P, S in zip(P_plus, Sig)], tgrid)
policy = np.trapezoid([0.5 * np.trace((K - np.linalg.solve(R, B.T @ P)).T @ R @ (K - np.linalg.solve(R, B.T @ P)) @ S)
                       for P, S in zip(P_plus, Sig)], tgrid)
print(f"J^u - J+                 = {J_u - J_plus:.8f}")
print(f"structural + policy      = {struct + policy:.8f}   (structural {struct:.6f}, policy {policy:.6f})")
print(f"certified gap (J^u-J+)/J^u = {100*(J_u-J_plus)/J_u:.2f}%   true regret (J^u-J*)/J^u = {100*(J_u-J_star)/J_u:.2f}%")
"""),
("md", r"""
The certificate is valid but loose here, and the split says why: three of the four noise directions
have no actuator, and most of the gap is the **structural** term — the energy of the fictitious
control — not the policy term. A better controller could remove only the policy part. This is the
diagnostic content of Corollary 3.4.
"""),
("md", r"""
## D. Least inflation is tightest (Proposition 3.5)
"""),
("co", r"""
for frac in [1.0, 0.8, 0.5, 0.25]:
    lam = frac * lam_star
    print(f"lambda/lambda* = {frac:4.2f}:  J+_lambda = {value(riccati(Q / lam), D, x0):.6f}")
print(f"                      J*         = {J_star:.6f}")
"""),
("md", r"""
## E. Deflation, and why the two relaxations are not ordered (Remark 3.11)
With affine drift and convex costs deflation is a valid lower bound (Proposition 3.10): its Riccati
matrix equals the physical one and only the noise offset shrinks, $\widetilde D=\lambda_\star G/2$.
Completion instead lowers the Riccati matrix and keeps the full noise. So for large initial
states deflation wins, and near the origin completion wins.
"""),
("co", r"""
Dtil = lam_star * G / 2
for s in [0.0, 0.5, 1.0, 2.0, 4.0]:
    x = s * x0
    Jm, Jp, Js = value(P_star, Dtil, x), value(P_plus, D, x), value(P_star, D, x)
    tighter = "completion" if Jp > Jm else "deflation"
    print(f"|x0| scale {s:3.1f}:  J- = {Jm:9.5f}  J+ = {Jp:9.5f}  J* = {Js:9.5f}   tighter: {tighter}")
"""),
("md", r"""
## F. Noise-proportional pricing (Proposition 4.3)
Diagonal case: controlled channels $\mathcal C$, noise variances $q_j$, authority $g_i=1/r_i$ with
$\sum g_i=A$. The optimum is $g_i\propto q_i$, with $\lambda_\star=\sum_{\mathcal C}q/A$ and
$\mathfrak I=A\cdot(\text{uncontrolled}/\text{controlled noise variance})$. Compare with random allocations.
"""),
("co", r"""
q = rng.uniform(0.2, 3.0, size=12); C = np.arange(5); U = np.arange(5, 12); Atot = 10.0
def geom(g):
    lam = np.min(q[C] / g); return lam, q.sum() / lam - g.sum()
g_opt = Atot * q[C] / q[C].sum()
lam_opt, I_opt = geom(g_opt)
rand = [geom(Atot * rng.dirichlet(np.ones(len(C)))) for _ in range(20000)]
print(f"optimal: lambda* = {lam_opt:.5f} (formula {q[C].sum()/Atot:.5f}),  I = {I_opt:.4f} "
      f"(formula {Atot*q[U].sum()/q[C].sum():.4f})")
print(f"best of 20000 random allocations: max lambda* = {max(r[0] for r in rand):.5f}, "
      f"min I = {min(r[1] for r in rand):.4f}")
"""),
("md", r"""
## G. The Monte Carlo estimator is biased upward, and the Hoeffding bound is vacuous at low temperature
Lemma 5.1: $\mathbb E[-\lambda\log\bar Y_N]\ge -\lambda\log\psi$, with bias $\approx\frac\lambda2(1/\mathrm{ESS}-1/N)$.
We use log-normal weights, for which $\operatorname{Var}Y/\psi^2=e^{s^2}-1$ exactly.
"""),
("co", r"""
lam, mu, s, N, reps = 0.05, 2.0, 1.2, 256, 20000
psi = np.exp(-mu + s**2 / 2); J_true = -lam * np.log(psi)
Y = np.exp(-mu + s * rng.normal(size=(reps, N)))
est = -lam * np.log(Y.mean(axis=1))
ess = Y.sum(1)**2 / (Y**2).sum(1)
print(f"true value                     {J_true:.6f}")
print(f"mean estimate                  {est.mean():.6f}   (bias {est.mean()-J_true:+.2e} ± {est.std()/np.sqrt(reps):.1e})")
print(f"delta-method bias (exact var)  {+lam/(2*N)*(np.exp(s**2)-1):+.2e}")
print(f"delta-method bias (from ESS)   {+np.mean(lam/2*(1/ess-1/N)):+.2e}")
print("(the delta method is asymptotic in N; the positive sign and the order of magnitude are what matter)")

# Hoeffding lower confidence bound with weights in (0,1] (S_min = 0), as in the swing application
lam_small, Jp = 1.76e-4, 0.025                    # the hardest case118 point
N2, delta = 2048, 0.05
lcb = -lam_small * np.log(np.exp(-Jp / lam_small) + np.sqrt(np.log(1 / delta) / (2 * N2)))
print(f"\ncase118 hardest point: J+ ~ {Jp},  Hoeffding LCB = {lcb:.2e}  (valid but useless)")
"""),
]

# =========================================================================== #
#  02  case118 results from frozen data
# =========================================================================== #
T2 = [("md", r"""
# 02 — Case118: every table and figure from the frozen results

Reads `data/frozen/*.csv`, written by the stage code in `code/case118/`, and regenerates the case118
tables and figures of the paper (Tables 1, 3 and 4; Figures 5–7 and SM12) in seconds.
Numbers refer to the submitted version of the paper.
Rerunning the stages themselves takes hours; see `code/case118/README.md`.

In the data files the completed value $\mathcal J^+$ is called `Wplus`.
"""),
("co", r"""
import sys, json
from pathlib import Path
import pandas as pd
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "scripts"))
import paper_figures as pf
DATA = ROOT / "data" / "frozen"
pd.set_option("display.width", 140, "display.precision", 6)
"""),
("md", "## Table 1 (case118 rows): certified gap under three pricings"),
("co", r"""
c = pd.read_csv(DATA / "stage5a3_certificate_summary.csv")
m = c.J_refined.notna(); c.loc[m, "J"] = c.loc[m, "J_refined"]; c.loc[m, "J_se"] = c.loc[m, "J_refined_se"]
c["gap_%"] = 100 * (c.J - c.Jplus) / c.J
c["gap_se_%"] = 100 * c.J_se / c.J
c[["noise", "design", "J", "J_se", "Jplus", "gap_%", "gap_se_%"]].round(4)
"""),
("md", "## Table 3: deployed policies at $\\sigma_{\\rm rms}=0.15$"),
("co", r"""
d = pd.read_csv(DATA / "stage3b_all_results.csv")
d[(d.sigma_rms == 0.15) & d.policy.isin(["static", "H2_droop", "H3_lqr"])][
    ["noise", "policy", "J_mean", "J_se", "cross_probability", "dmax_q99_deg", "n_paths"]]
"""),
("md", r"""
## The hard completion (Table 4, activity-scaled H3) and its estimator bias
Lemma 5.1 gives the bias $\approx\frac\lambda2(1/\mathrm{ESS}-1/N)$ per replicate.
"""),
("co", r"""
a = json.load(open(DATA / "STAGE4C_FINAL.json"))
lam, ess, N = a["lambda_max"], a["ess_absolute_mean"], a["n_paths_per_rep"]
print(f"J+ = {a['Wplus_mean']:.6f} ± {a['Wplus_rep_se']:.1e} (replicate s.e.), ESS fraction {a['ess_fraction_mean']:.3f}")
print(f"estimated Jensen bias = {lam/2*(1/ess-1/N):.1e}")
"""),
("md", "## Figures"),
("co", r"""
out = ROOT / "figures" / "generated"
for f in [pf.fig_geometry, pf.fig_case118_certificate, pf.fig_case118_frontier,
          pf.fig_case118_sparse, pf.fig_case118_deployed]:
    print("wrote", f(out))
"""),
("co", r"""
from IPython.display import Image, display
for name in ["fig_case118_certificate", "fig_case118_frontier", "fig_case118_sparse"]:
    display(Image(filename=str(out / f"{name}.png"), width=720))
"""),
]

# =========================================================================== #
#  03  case39 results from cached results
# =========================================================================== #
T3 = [("md", r"""
# 03 — Case39: every figure and table from cached results

Runs the case39 figure scripts in `code/case39/scripts/` (seconds) and prints Table 2. The
narrative notebooks in `notebooks/case39_narrative/` explain each experiment family in detail;
the heavy recomputation order is documented in `code/case39/scripts/recompute_notes.py`.
"""),
("co", r"""
import subprocess, sys
from pathlib import Path
import pandas as pd
ROOT = Path.cwd().parent
C39 = ROOT / "code" / "case39"
for s in ["make_figures.py", "make_control_inflation_hierarchy_figures.py", "make_reinforcement_planning_figures.py"]:
    r = subprocess.run([sys.executable, f"scripts/{s}"], cwd=C39, capture_output=True, text=True)
    msg = (r.stdout + r.stderr).strip().splitlines()
    print(s, "->", msg[-1] if msg else f"exit code {r.returncode}")
"""),
("md", "## Table 2: the actuator hierarchy at $\\sigma_{\\rm rms}=0.1$"),
("co", r"""
t = pd.read_csv(C39 / "results" / "control_inflation_hierarchy_combined.csv")
t = t[t.sigma_rms == 0.10].rename(columns={"Wminus": "J_minus", "Wplus": "J_plus", "inflation_trace": "I_burden"})
t["J_minus_J_plus_gap"] = t.J - t.J_plus
t[["noise", "H", "J", "J_minus", "J_plus", "I_burden", "lambda_max", "J_minus_J_plus_gap"]].round(5)
"""),
("co", r"""
from IPython.display import Image, display
for name in ["case39_control_inflation_hierarchy", "case39_implementation"]:
    display(Image(filename=str(C39 / "figures" / f"{name}.png"), width=720))
"""),
]

if __name__ == "__main__":
    make(T1, "01_theory_checks_LQ.ipynb")
    make(T2, "02_case118_results.ipynb")
    make(T3, "03_case39_results.ipynb")
