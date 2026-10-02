"""Stage-1 stochastic nonlinear swing pilot for MATPOWER case39.

The model extends the CycleSpaceCertificate transient model by adding independent
white nodal injection fluctuations and generator-only feedback around a chosen
post-event balancing allocation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_continuous_are

@dataclass
class SwingProblem:
    Inc: np.ndarray
    Kedge: np.ndarray
    edges: list
    p_pre: np.ndarray
    p_post: np.ndarray
    M: np.ndarray
    damp: np.ndarray
    is_gen: np.ndarray
    theta_pre: np.ndarray
    theta_star: np.ndarray
    baseMVA: float
    disturbance: int
    dP: float
    gamma: np.ndarray
    ref: int = 0

    def __post_init__(self):
        self.n = len(self.p_pre)
        self.gen_idx = np.where(self.is_gen)[0]
        self.ng = len(self.gen_idx)
        self.keep = np.array([i for i in range(self.n) if i != self.ref], int)
        self.E = np.zeros((self.n, self.n-1))
        self.E[self.keep, np.arange(self.n-1)] = 1.0
        self.H = np.zeros((self.n-1, self.n))
        self.H[:, self.keep] = np.eye(self.n-1)
        self.H[:, self.ref] = -1.0
        self.Sg = np.zeros((self.n, self.ng))
        self.Sg[self.gen_idx, np.arange(self.ng)] = 1.0
        self.delta_star = self.Inc.T @ self.theta_star
        self.Lstar = self.Inc @ np.diag(self.Kedge*np.cos(self.delta_star)) @ self.Inc.T
        self.Ustar = self.potential(self.theta_star)

    def network_force(self, theta):
        """Inc [K sin(Inc^T theta)] for theta shape (...,n)."""
        return (np.sin(theta @ self.Inc) * self.Kedge) @ self.Inc.T

    def potential(self, theta):
        theta = np.asarray(theta)
        return -theta @ self.p_post - np.cos(theta @ self.Inc) @ self.Kedge

    def energy(self, theta, omega):
        pot = self.potential(theta) - self.Ustar
        kin = .5*np.sum((omega**2)*self.M, axis=-1)
        return pot + kin

    def barrier(self, theta, clip=1e-9):
        """Mean Bregman barrier of h(delta)=-log cos(delta) around delta_star."""
        d = theta @ self.Inc
        c = np.maximum(np.cos(d), clip)
        cstar = np.cos(self.delta_star)
        b = -np.log(c) + np.log(cstar) - np.tan(self.delta_star)*(d-self.delta_star)
        return np.mean(b, axis=-1)

    def reduced_state(self, theta, omega):
        q = (theta[:, self.keep] - theta[:, [self.ref]]) - \
            (self.theta_star[self.keep] - self.theta_star[self.ref])
        return np.concatenate([q, omega], axis=1)

    def lqr_gain(self, angle_weight=2.0, freq_weight=0.5, control_weight=0.2):
        Minv = np.diag(1.0/self.M)
        A = np.block([
            [np.zeros((self.n-1,self.n-1)), self.H],
            [-Minv@self.Lstar@self.E, -Minv@np.diag(self.damp)]
        ])
        B = np.vstack([np.zeros((self.n-1,self.ng)), Minv@self.Sg])
        Qq = self.E.T@self.Lstar@self.E
        Qq /= max(np.trace(Qq)/(self.n-1), 1e-12)
        Qw = np.diag(self.M)
        Qw /= max(np.trace(Qw)/self.n, 1e-12)
        Q = np.block([
            [angle_weight*Qq, np.zeros((self.n-1,self.n))],
            [np.zeros((self.n,self.n-1)), freq_weight*Qw]
        ])
        R = control_weight*np.eye(self.ng)
        P = solve_continuous_are(A,B,Q,R)
        gain = np.linalg.solve(R,B.T@P)
        return gain, dict(A=A,B=B,Q=Q,R=R,P=P)


def noise_profile(problem: SwingProblem, rms: float, kind="homogeneous"):
    """Independent nodal injection-noise amplitudes with prescribed RMS level."""
    rms = float(rms)
    if kind == "homogeneous":
        prof = np.ones(problem.n)
    elif kind == "activity_scaled":
        a = np.abs(problem.p_pre)
        scale = np.quantile(a[a>1e-10], .75)
        prof = 0.25 + np.minimum(a/max(scale,1e-12), 4.0)
        prof /= np.sqrt(np.mean(prof**2))
    else:
        raise ValueError(kind)
    return rms*prof


def controller(problem: SwingProblem, name: str, gain=None, droop_gain=0.5):
    """Return transient feedback v around the fixed post-event balance."""
    if name == "static":
        return lambda t,th,om: np.zeros((len(th),problem.ng))
    if name == "droop":
        return lambda t,th,om: -float(droop_gain)*om[:,problem.gen_idx]
    if name == "lqr":
        if gain is None:
            gain,_ = problem.lqr_gain()
        return lambda t,th,om: -(problem.reduced_state(th,om) @ gain.T)
    raise ValueError(name)


def simulate_ensemble(problem: SwingProblem, policy, noise, *, T=5.0, dt=5e-4,
                      n_paths=128, seed=1, control_weight=0.2,
                      energy_weight=1.0, barrier_weight=1.0, freq_weight=0.2,
                      terminal_weight=2.0, fail_penalty=50.0,
                      record_paths=8, record_stride=20):
    """Split-step Euler--Maruyama simulation with two candidate SOC objectives.

    Deterministic swing drift is advanced by semi-implicit Euler; additive
    injection noise is applied to omega. Costs are stopped at first cohesive
    boundary exit and receive a failure penalty.
    """
    rng = np.random.default_rng(seed)
    n,ng = problem.n,problem.ng
    th = np.repeat(problem.theta_pre[None,:], n_paths, axis=0).copy()
    om = np.zeros((n_paths,n))
    noise = np.asarray(noise,float)
    sqdt = np.sqrt(dt)
    steps = int(np.ceil(T/dt))
    active = np.ones(n_paths,dtype=bool)
    exit_time = np.full(n_paths,np.nan)
    dmax = np.max(np.abs(th@problem.Inc),axis=1)
    energy_int = np.zeros(n_paths); barrier_int=np.zeros(n_paths); control_int=np.zeros(n_paths)
    max_freq = np.zeros(n_paths)
    max_gen_freq = np.zeros(n_paths)
    max_coi_freq = np.zeros(n_paths)
    rec_ids=np.arange(min(record_paths,n_paths))
    times=[]; rec_dmax=[]; rec_freq=[]; rec_energy=[]
    Rdiag = float(control_weight)
    for k in range(steps):
        t=k*dt
        v=policy(t,th,om)
        nodctrl=v@problem.Sg.T
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-
               problem.network_force(th)+nodctrl)/problem.M[None,:]
        om = om + dt*drift + (noise/problem.M)[None,:]*sqdt*rng.standard_normal((n_paths,n))
        th = th + dt*om
        delta=th@problem.Inc
        dm=np.max(np.abs(delta),axis=1)
        dmax=np.maximum(dmax,dm)
        max_freq=np.maximum(max_freq,np.max(np.abs(om),axis=1))
        max_gen_freq=np.maximum(max_gen_freq,np.max(np.abs(om[:,problem.gen_idx]),axis=1))
        coi=(om@problem.M)/np.sum(problem.M)
        max_coi_freq=np.maximum(max_coi_freq,np.abs(coi))
        alive_before=active.copy()
        newly = alive_before & (dm >= np.pi/2)
        exit_time[newly]=t+dt
        active[newly]=False
        # integrate only until first exit
        if np.any(alive_before):
            E=np.maximum(problem.energy(th,om),0.0)
            kin=.5*np.sum((om**2)*problem.M,axis=1)
            b=problem.barrier(th)
            uc=.5*Rdiag*np.sum(v*v,axis=1)
            energy_int[alive_before] += (energy_weight*E[alive_before]/n + uc[alive_before]/ng)*dt
            barrier_int[alive_before] += (barrier_weight*b[alive_before] +
                                           freq_weight*kin[alive_before]/n + uc[alive_before]/ng)*dt
            control_int[alive_before] += uc[alive_before]*dt/ng
        if k % record_stride == 0:
            times.append(t+dt)
            rec_dmax.append(np.degrees(np.max(np.abs(delta[rec_ids]),axis=1)))
            rec_freq.append(np.max(np.abs(om[rec_ids]),axis=1))
            rec_energy.append(problem.energy(th[rec_ids],om[rec_ids])/n)
    terminal=np.maximum(problem.energy(th,om),0.0)/n
    failed=~active
    J_energy=energy_int + terminal_weight*terminal
    J_barrier=barrier_int + terminal_weight*terminal
    J_energy[failed]+=fail_penalty
    J_barrier[failed]+=fail_penalty
    out=dict(
        failure_probability=float(failed.mean()),
        survival_probability=float(active.mean()),
        dmax_deg_quantiles=np.degrees(np.quantile(dmax,[.5,.9,.99])).tolist(),
        max_abs_frequency_quantiles=np.quantile(max_freq,[.5,.9,.99]).tolist(),
        max_abs_generator_frequency_quantiles=np.quantile(max_gen_freq,[.5,.9,.99]).tolist(),
        max_abs_coi_frequency_quantiles=np.quantile(max_coi_freq,[.5,.9,.99]).tolist(),
        terminal_energy_per_node_mean=float(terminal.mean()),
        control_effort_mean=float(control_int.mean()),
        J_energy_mean=float(J_energy.mean()),
        J_energy_se=float(J_energy.std(ddof=1)/np.sqrt(n_paths)),
        J_barrier_mean=float(J_barrier.mean()),
        J_barrier_se=float(J_barrier.std(ddof=1)/np.sqrt(n_paths)),
        exit_time_mean=float(np.nanmean(exit_time)) if failed.any() else None,
        n_paths=int(n_paths), T=float(T), dt=float(dt),
        record=dict(times=np.asarray(times),dmax_deg=np.asarray(rec_dmax),
                    max_abs_frequency=np.asarray(rec_freq),energy=np.asarray(rec_energy)),
    )
    return out


def simulate_soc_cost(problem: SwingProblem, policy, noise, *, T=3.0, dt=0.0025,
                      n_paths=512, seed=1, R=0.2, cost=None):
    """Monte-Carlo stopped-domain SOC cost consistent with the PIC surrogate.

    The objective is
        E[ G(tau,X_tau) + int_0^tau (V(X_t)+.5 u^T R u) dt ],
    with first exit from |Inc^T theta|<pi/2. ``cost`` must provide
    ``running(problem,theta,omega)``, ``terminal(...)`` and ``failure``.
    """
    if cost is None:
        # local import avoids a hard module cycle at import time
        from pic_case39 import PICCost
        cost=PICCost()
    rng=np.random.default_rng(seed); n=problem.n
    th=np.repeat(problem.theta_pre[None,:],n_paths,axis=0).copy(); om=np.zeros((n_paths,n))
    noise=np.asarray(noise,float); steps=int(np.ceil(T/dt)); dt=T/steps; sq=np.sqrt(dt)
    if np.ndim(R)==0: Rdiag=np.full(problem.ng,float(R))
    else: Rdiag=np.asarray(R,float)
    active=np.ones(n_paths,bool); failed=np.zeros(n_paths,bool); run=np.zeros(n_paths); ctrl=np.zeros(n_paths)
    dmax=np.max(np.abs(th@problem.Inc),axis=1)
    for k in range(steps):
        if not np.any(active): break
        t=k*dt; idx=np.where(active)[0]; tha=th[idx]; oma=om[idx]
        v=policy(t,tha,oma)
        run[idx]+=cost.running(problem,tha,oma)*dt
        cu=.5*np.sum(v*v*Rdiag[None,:],axis=1)
        run[idx]+=cu*dt; ctrl[idx]+=cu*dt
        drift=(problem.p_post[None,:]-problem.damp[None,:]*oma-problem.network_force(tha)+(v@problem.Sg.T))/problem.M[None,:]
        omnew=oma+dt*drift+(noise/problem.M)[None,:]*sq*rng.standard_normal((len(idx),n))
        thnew=tha+dt*omnew; th[idx]=thnew; om[idx]=omnew
        dm=np.max(np.abs(thnew@problem.Inc),axis=1); dmax[idx]=np.maximum(dmax[idx],dm)
        ex=dm>=np.pi/2
        if np.any(ex):
            exidx=idx[ex]; failed[exidx]=True; active[exidx]=False
    terminal=np.empty(n_paths); terminal[failed]=cost.failure
    surv=~failed; terminal[surv]=cost.terminal(problem,th[surv],om[surv])
    J=run+terminal
    return dict(J_mean=float(J.mean()),J_se=float(J.std(ddof=1)/np.sqrt(n_paths)),
                failure_probability=float(failed.mean()),control_effort_mean=float(ctrl.mean()),
                dmax_deg_quantiles=np.degrees(np.quantile(dmax,[.5,.9,.99])).tolist(),
                n_paths=int(n_paths),dt=float(dt),T=float(T),raw_J=J)
