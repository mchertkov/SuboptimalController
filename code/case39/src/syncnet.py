"""Cycle-space convexity and equilibrium continuation for lossless AC networks.

Conventions
-----------
Inc is the n x L oriented incidence matrix.  K_e>0 and sum(p)=0.
Synchronous equilibria satisfy

    Inc (K * sin(Inc.T theta)) = p.

With Cyc spanning ker(Inc), any conservative flow is

    F = F_p + Cyc psi,

and the separable convex objective is

    Phi(psi)=sum_e K_e g(F_e/K_e),
    g(s)=s asin(s)+sqrt(1-s^2).

The solver below includes a Phase-I linear program that finds a point strictly
inside the capacity box whenever one exists.  This removes the old failure mode
where Newton was rejected merely because the DC/Thomson flow itself violated a
capacity even though another cycle flow was feasible.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.linalg import pinvh
from scipy.optimize import linprog

__all__ = [
    "incidence_and_cycles", "particular_flow", "g", "Phi", "gradPhi", "hessPhi",
    "phase1_cycle_box", "solve_dual", "dcb_margin", "exact_margin",
    "potential", "grad_potential", "newton_equilibrium", "reconstruct_theta",
    "winding", "cohesive_threshold", "continued_stable_limit",
]


def incidence_and_cycles(G: nx.Graph):
    """Return oriented incidence, a NetworkX cycle basis, edge list, node list."""
    nodes = list(G.nodes())
    idx = {v: k for k, v in enumerate(nodes)}
    edges = []
    for u, v in G.edges():
        edges.append((u, v) if idx[u] < idx[v] else (v, u))
    n, L = len(nodes), len(edges)
    eidx = {e: k for k, e in enumerate(edges)}

    Inc = np.zeros((n, L))
    for k, (u, v) in enumerate(edges):
        Inc[idx[u], k] = +1.0
        Inc[idx[v], k] = -1.0

    cycles = nx.cycle_basis(G)
    Cyc = np.zeros((L, len(cycles)))
    for k, cyc in enumerate(cycles):
        ring = list(cyc) + [cyc[0]]
        for a, b in zip(ring[:-1], ring[1:]):
            key = (a, b) if idx[a] < idx[b] else (b, a)
            Cyc[eidx[key], k] = +1.0 if (a, b) == key else -1.0
    if L and not np.allclose(Inc @ Cyc, 0.0, atol=1e-12):
        raise RuntimeError("cycle basis is not in ker(Inc)")
    return Inc, Cyc, edges, nodes


def particular_flow(Inc, K, p):
    """Thomson/DC flow: min sum F_e^2/(2K_e) subject to Inc F=p."""
    Lap = Inc @ np.diag(K) @ Inc.T
    theta = pinvh(Lap) @ p
    F = K * (Inc.T @ theta)
    return F, theta


def winding(Delta, Cyc):
    return (Cyc.T @ Delta) / (2.0 * np.pi)


def g(s):
    s = np.asarray(s)
    return s * np.arcsin(np.clip(s, -1.0, 1.0)) + np.sqrt(np.maximum(1.0 - s*s, 0.0))


def Phi(psi, F_p, Cyc, K):
    F = F_p + Cyc @ np.asarray(psi)
    return float(np.sum(K * g(F / K)))


def gradPhi(psi, F_p, Cyc, K):
    s = (F_p + Cyc @ np.asarray(psi)) / K
    return Cyc.T @ np.arcsin(np.clip(s, -1.0, 1.0))


def hessPhi(psi, F_p, Cyc, K):
    s = (F_p + Cyc @ np.asarray(psi)) / K
    d = K * np.sqrt(np.maximum(1.0 - s*s, 1e-300))
    return Cyc.T @ (Cyc / d[:, None])


def phase1_cycle_box(F_p, Cyc, K, *, tol=1e-12):
    """Find an interior point of |(F_p+Cyc psi)/K|<1 by LP.

    The LP maximizes a common normalized slack rho:
        +/- (F_p + Cyc psi)/K + rho <= 1.
    Returns ``ok=False`` if the affine flow space has no strict intersection with
    the open capacity box.
    """
    c = Cyc.shape[1]
    if c == 0:
        s = F_p / K
        rho = 1.0 - float(np.max(np.abs(s)))
        return {"ok": rho > tol, "psi": np.zeros(0), "rho": rho,
                "smax": float(np.max(np.abs(s)))}

    B = Cyc / K[:, None]
    q = F_p / K
    # z=(psi,rho), minimize -rho
    A_ub = np.block([
        [ B, np.ones((len(K), 1))],
        [-B, np.ones((len(K), 1))],
    ])
    b_ub = np.concatenate([1.0 - q, 1.0 + q])
    obj = np.zeros(c + 1)
    obj[-1] = -1.0
    bounds = [(None, None)] * c + [(0.0, 1.0)]
    lp = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not lp.success:
        return {"ok": False, "reason": "phase-I LP infeasible", "rho": -np.inf}
    psi = lp.x[:c]
    rho = float(lp.x[-1])
    smax = float(np.max(np.abs((F_p + Cyc @ psi) / K)))
    return {"ok": rho > tol and smax < 1.0 - 0.1*tol, "psi": psi,
            "rho": rho, "smax": smax, "lp": lp}


def solve_dual(F_p, Cyc, K, m=None, psi0=None, tol=1e-9, maxit=200):
    """Damped Newton solve of grad Phi = 2*pi*m inside the capacity box."""
    F_p = np.asarray(F_p, float)
    K = np.asarray(K, float)
    c = Cyc.shape[1]
    m = np.zeros(c) if m is None else np.asarray(m, float)
    rhs = 2.0 * np.pi * m

    if c == 0:
        s = F_p / K
        ok = float(np.max(np.abs(s))) < 1.0 - 1e-12
        return dict(ok=ok, psi=np.zeros(0), s=s, F=F_p,
                    Delta=np.arcsin(np.clip(s, -1, 1)),
                    smax=float(np.max(np.abs(s))), iters=0,
                    reason=None if ok else "box")

    def feas(ps, margin=1e-13):
        return float(np.max(np.abs((F_p + Cyc @ ps) / K))) < 1.0 - margin

    if psi0 is not None and feas(np.asarray(psi0, float)):
        psi = np.asarray(psi0, float).copy()
    else:
        ph = phase1_cycle_box(F_p, Cyc, K)
        if not ph["ok"]:
            return dict(ok=False, reason="no strict capacity-feasible cycle flow",
                        psi=ph.get("psi", np.zeros(c)), smax=ph.get("smax", np.inf))
        psi = ph["psi"].copy()

    def obj(ps):
        return Phi(ps, F_p, Cyc, K) - rhs @ ps

    for it in range(maxit):
        r = gradPhi(psi, F_p, Cyc, K) - rhs
        if np.linalg.norm(r, ord=np.inf) < tol:
            s = (F_p + Cyc @ psi) / K
            smax = float(np.max(np.abs(s)))
            if smax >= 1.0 - 5e-10:
                return dict(ok=False, reason="minimizer on capacity boundary", psi=psi,
                            smax=smax, iters=it)
            return dict(ok=True, psi=psi, s=s, F=F_p + Cyc @ psi,
                        Delta=np.arcsin(s), smax=smax, iters=it)
        H = hessPhi(psi, F_p, Cyc, K)
        try:
            step = np.linalg.solve(H, -r)
        except np.linalg.LinAlgError:
            return dict(ok=False, reason="singular dual Hessian", psi=psi, iters=it)

        f0 = obj(psi)
        descent = float(r @ step)
        t = 1.0
        accepted = False
        for _ in range(90):
            cand = psi + t * step
            if feas(cand) and obj(cand) <= f0 + 1e-4 * t * descent + 2e-13*max(1.0, abs(f0)):
                accepted = True
                break
            t *= 0.5
        if not accepted or t < 1e-14:
            smax = float(np.max(np.abs((F_p + Cyc @ psi) / K)))
            return dict(ok=False, reason="Newton stalled at boundary", psi=psi,
                        smax=smax, iters=it)
        psi = cand
    return dict(ok=False, reason="maxit", psi=psi, iters=maxit)


def dcb_margin(Inc, K, p):
    """DCB box statistic at gamma=pi/2: max_e |F_DC,e/K_e|."""
    F_dc, theta_dc = particular_flow(Inc, K, p)
    return float(np.max(np.abs(F_dc / K))), F_dc, theta_dc


def exact_margin(Inc, Cyc, K, p, psi0=None):
    F_p, _ = particular_flow(Inc, K, p)
    out = solve_dual(F_p, Cyc, K, psi0=psi0)
    return (out["smax"], out) if out["ok"] else (None, out)


def potential(theta, Inc, K, p):
    return float(-p @ theta - np.sum(K * np.cos(Inc.T @ theta)))


def grad_potential(theta, Inc, K, p):
    return -p + Inc @ (K * np.sin(Inc.T @ theta))


def newton_equilibrium(Inc, K, p, th0=None, ref=0, tol=1e-12, maxit=250):
    """Damped Newton root solve, used for independent checks and continuation."""
    n = Inc.shape[0]
    th = np.zeros(n) if th0 is None else np.asarray(th0, float).copy()
    th -= th[ref]
    keep = np.array([i for i in range(n) if i != ref], int)
    for it in range(maxit):
        Delta = Inc.T @ th
        r = Inc @ (K * np.sin(Delta)) - p
        rn = float(np.max(np.abs(r[keep])))
        H = Inc @ np.diag(K * np.cos(Delta)) @ Inc.T
        ev = np.sort(np.linalg.eigvalsh(H))
        lam2 = float(ev[1]) if n > 1 else np.inf
        if rn < tol:
            return dict(ok=True, theta=th, Delta=Delta, residual=rn,
                        lam2=lam2, dmax=float(np.max(np.abs(Delta))), iters=it)
        try:
            dth_keep = np.linalg.solve(H[np.ix_(keep, keep)], -r[keep])
        except np.linalg.LinAlgError:
            return dict(ok=False, reason="singular jacobian", theta=th,
                        Delta=Delta, lam2=lam2, residual=rn, iters=it)
        step = np.zeros(n)
        step[keep] = dth_keep
        t = 1.0
        accepted = False
        for _ in range(70):
            cand = th + t * step
            rc = Inc @ (K * np.sin(Inc.T @ cand)) - p
            if np.max(np.abs(rc[keep])) < rn:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            return dict(ok=False, reason="Newton line-search failure", theta=th,
                        Delta=Delta, lam2=lam2, residual=rn, iters=it)
        th = cand
    return dict(ok=False, reason="maxit", theta=th, Delta=Inc.T @ th,
                residual=float(np.max(np.abs((Inc @ (K*np.sin(Inc.T@th))-p)[keep]))))


def reconstruct_theta(Inc, Delta, ref=0):
    """Least-squares reconstruction of node angles from edge differences."""
    n = Inc.shape[0]
    keep = [i for i in range(n) if i != ref]
    A = Inc.T[:, keep]
    x, *_ = np.linalg.lstsq(A, Delta, rcond=None)
    theta = np.zeros(n)
    theta[keep] = x
    res = float(np.max(np.abs(Inc.T @ theta - Delta)))
    return theta, res


def cohesive_threshold(Inc, Cyc, K, p0, *, tol=2e-6, hi=None):
    """Supremum loading alpha admitting a strict zero-winding cohesive state."""
    sdc, _, _ = dcb_margin(Inc, K, p0)
    lo = 0.0
    if hi is None:
        hi = max(2.0 / sdc, 1.0)
        while True:
            Fp, _ = particular_flow(Inc, K, hi*p0)
            if not solve_dual(Fp, Cyc, K)["ok"]:
                break
            hi *= 1.5
    psi = None
    while hi - lo > tol:
        a = 0.5*(lo+hi)
        Fp, _ = particular_flow(Inc, K, a*p0)
        out = solve_dual(Fp, Cyc, K, psi0=psi)
        if out["ok"]:
            lo = a
            psi = out["psi"]
        else:
            hi = a
    return lo


def continued_stable_limit(Inc, K, p0, start_alpha, *, start_theta=None,
                           step0=0.20, tol=2e-6, lam_tol=1e-8):
    """Warm-start a locally stable primal branch beyond the cohesive boundary.

    This is intentionally labeled a *numerical continued branch limit*, not a
    certified saddle-node.  A point is accepted only if Newton converges and the
    metagraph Laplacian has positive second eigenvalue.
    """
    if start_theta is None:
        e = newton_equilibrium(Inc, K, start_alpha*p0)
        if not e["ok"]:
            raise RuntimeError("could not initialize continuation")
        theta = e["theta"]
    else:
        theta = np.asarray(start_theta, float).copy()
    good_a = float(start_alpha)
    good_theta = theta
    step = float(step0)
    while step > tol:
        trial = good_a + step
        e = newton_equilibrium(Inc, K, trial*p0, th0=good_theta)
        if e["ok"] and e.get("lam2", -np.inf) > lam_tol:
            good_a = trial
            good_theta = e["theta"]
        else:
            step *= 0.5
    e = newton_equilibrium(Inc, K, good_a*p0, th0=good_theta)
    return good_a, e
