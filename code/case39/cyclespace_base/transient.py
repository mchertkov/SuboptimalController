"""Balanced-step transient certificate and finite-horizon swing simulation.

Important: the revised paper does *not* simulate a raw load loss followed by a
primary-frequency governor response.  At t=0 it applies an already balanced
injection change

    p_eff = p + dP e_delta - dP gamma,    1^T gamma = 1,

then integrates the swing equations from the old angles and zero frequency.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from syncnet import potential, newton_equilibrium

__all__ = [
    "balanced_injection", "critical_energy", "initial_energy", "simulate_balanced_step",
    "certified_threshold", "simulated_threshold", "static_threshold",
]


def balanced_injection(p, dP, disturbance_bus, gamma):
    pe = np.asarray(p, float).copy()
    pe[int(disturbance_bus)] += float(dP)
    pe -= float(dP) * np.asarray(gamma, float)
    # Numerical hygiene only; gamma is normalized so this is O(roundoff).
    pe -= pe.mean()
    return pe


def _face_phase1(A, e, s, half=np.pi/2):
    """Relative-interior LP for one face A[e]x=s*half.

    Maximize a common slack rho for all *other* inequalities.  The face equality
    itself is removed from the strict-slack constraints.
    """
    m, nvar = A.shape
    others = np.array([k for k in range(m) if k != e], int)
    Ao = A[others]
    # z=(x,rho)
    A_ub = np.block([
        [ Ao, np.ones((len(others), 1))],
        [-Ao, np.ones((len(others), 1))],
    ])
    b_ub = np.full(2*len(others), half)
    A_eq = np.zeros((1, nvar+1))
    A_eq[0, :nvar] = A[e]
    b_eq = np.array([s*half])
    c = np.zeros(nvar+1); c[-1] = -1.0
    bounds = [(None, None)]*nvar + [(0.0, half)]
    lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=bounds, method="highs")
    if not lp.success:
        return None
    return lp.x[:nvar], float(lp.x[-1])


def _barrier_face_min(A, K, pk, e, s, xri, *, half=np.pi/2,
                      mu0=2e-1, mu_min=2e-10, mu_factor=0.12,
                      newton_tol=2e-11, max_newton=80):
    """Minimize the convex potential on one face with equality eliminated.

    x = xri + Z y, with Z spanning null(A[e]).  A logarithmic barrier keeps all
    other angle constraints strictly feasible.  The final feasible point is
    evaluated in the *unbarriered* objective.
    """
    aeq = A[e:e+1]
    Z = null_space(aeq)  # nvar x (nvar-1)
    if Z.shape[1] == 0:
        x = xri.copy()
        val = float(-pk @ x - np.sum(K*np.cos(A@x)))
        return val, x
    others = np.array([k for k in range(A.shape[0]) if k != e], int)
    Ao = A[others]
    y = np.zeros(Z.shape[1])

    def f_x(x):
        return float(-pk @ x - np.sum(K*np.cos(A @ x)))

    mu = mu0
    while mu >= mu_min:
        for _ in range(max_newton):
            x = xri + Z @ y
            D = A @ x
            hp = half - D[others]
            hm = half + D[others]
            if hp.min() <= 0 or hm.min() <= 0:
                raise RuntimeError("barrier iterate left relative interior")
            gf = -pk + A.T @ (K*np.sin(D))
            Hf = A.T @ ((K*np.cos(D))[:, None] * A)
            gb = gf + mu * Ao.T @ (1.0/hp - 1.0/hm)
            Hb = Hf + mu * Ao.T @ ((1.0/hp**2 + 1.0/hm**2)[:, None] * Ao)
            gy = Z.T @ gb
            Hy = Z.T @ Hb @ Z
            if np.linalg.norm(gy, np.inf) < newton_tol * max(1.0, mu):
                break
            try:
                dy = np.linalg.solve(Hy, -gy)
            except np.linalg.LinAlgError:
                dy = np.linalg.lstsq(Hy, -gy, rcond=None)[0]

            # max feasible step in y, then Armijo on the barrier objective
            dD = Ao @ (Z @ dy)
            tmax = 1.0
            pos = dD > 0
            if np.any(pos):
                tmax = min(tmax, 0.99*np.min(hp[pos]/dD[pos]))
            neg = dD < 0
            if np.any(neg):
                tmax = min(tmax, 0.99*np.min(hm[neg]/(-dD[neg])))
            t = min(1.0, tmax)

            def barrier_obj(yy):
                xx = xri + Z @ yy
                DD = A @ xx
                hpp = half - DD[others]
                hmm = half + DD[others]
                if hpp.min() <= 0 or hmm.min() <= 0:
                    return np.inf
                return f_x(xx) - mu*(np.log(hpp).sum() + np.log(hmm).sum())

            b0 = barrier_obj(y)
            slope = float(gy @ dy)
            for _ in range(60):
                yn = y + t*dy
                if barrier_obj(yn) <= b0 + 1e-4*t*slope:
                    y = yn
                    break
                t *= 0.5
            else:
                break
        mu *= mu_factor

    x = xri + Z @ y
    return f_x(x), x


def critical_energy(Inc, K, p_eff, th_star, ref=0, verbose=False):
    """Numerically evaluate c* by minimizing on all 2L cohesive faces.

    The face decomposition is exact.  Each floating-point face minimum is found
    from a relative-interior Phase-I LP followed by an equality-eliminated
    logarithmic-barrier Newton solve.  This is substantially more robust than the
    old SLSQP warm-start implementation used in the early draft.
    """
    n, L = Inc.shape
    keep = np.array([i for i in range(n) if i != ref], int)
    A = Inc.T[:, keep]
    pk = p_eff[keep]
    Vstar = potential(th_star, Inc, K, p_eff)
    best = np.inf
    best_x = None
    best_face = None
    diagnostics = []

    for e in range(L):
        for s in (+1.0, -1.0):
            ph = _face_phase1(A, e, s)
            if ph is None:
                diagnostics.append((e, s, False, np.nan))
                continue
            xri, rho = ph
            try:
                val_abs, x = _barrier_face_min(A, K, pk, e, s, xri)
            except Exception:
                diagnostics.append((e, s, False, rho))
                continue
            if np.max(np.abs(A@x)) > np.pi/2 + 2e-7:
                diagnostics.append((e, s, False, rho))
                continue
            val = val_abs - Vstar
            diagnostics.append((e, s, True, rho))
            if val < best:
                best, best_x, best_face = float(val), x.copy(), (e, s)

    if not np.isfinite(best):
        raise RuntimeError("no cohesive boundary face was solved")
    if verbose:
        print(f"c*={best:.10g}, face={best_face}")
    return best, best_x, best_face, diagnostics


def initial_energy(Inc, K, p_eff, th_old, th_star):
    return potential(th_old, Inc, K, p_eff) - potential(th_star, Inc, K, p_eff)


def simulate_balanced_step(Inc, K, p_eff, damp, M, th_old,
                           *, T=20.0, dt=4e-3, stop_at_boundary=True):
    """RK4 simulation from (theta_old, omega=0) after the balanced step."""
    th = np.asarray(th_old, float).copy()
    om = np.zeros_like(th)
    damp = np.asarray(damp, float)
    M = np.asarray(M, float)
    half = np.pi/2
    dmax = float(np.max(np.abs(Inc.T@th)))
    lost = dmax >= half
    nsteps = int(np.ceil(T/dt))

    def rhs(tt, ww):
        return ww, (p_eff - damp*ww - Inc@(K*np.sin(Inc.T@tt))) / M

    for _ in range(nsteps):
        k1t, k1o = rhs(th, om)
        k2t, k2o = rhs(th+.5*dt*k1t, om+.5*dt*k1o)
        k3t, k3o = rhs(th+.5*dt*k2t, om+.5*dt*k2o)
        k4t, k4o = rhs(th+dt*k3t, om+dt*k3o)
        th += dt*(k1t+2*k2t+2*k3t+k4t)/6.0
        om += dt*(k1o+2*k2o+2*k3o+k4o)/6.0
        dm = float(np.max(np.abs(Inc.T@th)))
        dmax = max(dmax, dm)
        if not np.isfinite(dm) or dm >= half:
            lost = True
            if stop_at_boundary:
                break
    return {"cohesive": not lost, "dmax": dmax, "theta": th, "omega": om}


def static_threshold(Inc, Cyc, K, p, th_old, disturbance_bus, gamma,
                     campus, *, use_dcb=False, tol=0.01):
    """Largest balanced step with a strict-cohesion post-step equilibrium."""
    from syncnet import dcb_margin, particular_flow, solve_dual
    lo, hi = 0.0, float(campus)
    while hi-lo > tol:
        mid = .5*(lo+hi)
        pe = balanced_injection(p, mid, disturbance_bus, gamma)
        if use_dcb:
            ok = dcb_margin(Inc, K, pe)[0] < 1.0
        else:
            Fp, _ = particular_flow(Inc, K, pe)
            ok = solve_dual(Fp, Cyc, K)["ok"]
        if ok: lo = mid
        else: hi = mid
    return lo


def certified_threshold(Inc, K, p, th_old, disturbance_bus, gamma,
                        campus, *, tol=0.01):
    lo, hi = 0.0, float(campus)
    while hi-lo > tol:
        mid = .5*(lo+hi)
        pe = balanced_injection(p, mid, disturbance_bus, gamma)
        eq = newton_equilibrium(Inc, K, pe, th0=th_old)
        ok = False
        if eq["ok"] and eq["dmax"] < np.pi/2:
            cstar, *_ = critical_energy(Inc, K, pe, eq["theta"])
            W0 = initial_energy(Inc, K, pe, th_old, eq["theta"])
            ok = W0 < cstar
        if ok: lo = mid
        else: hi = mid
    return lo


def simulated_threshold(Inc, K, p, th_old, disturbance_bus, gamma, campus,
                        damp, M, *, tol=0.01, T=20.0, dt=4e-3):
    lo, hi = 0.0, float(campus)
    while hi-lo > tol:
        mid = .5*(lo+hi)
        pe = balanced_injection(p, mid, disturbance_bus, gamma)
        ok = simulate_balanced_step(Inc, K, pe, damp, M, th_old,
                                    T=T, dt=dt)["cohesive"]
        if ok: lo = mid
        else: hi = mid
    return lo
