"""Physical reinforcement planning built on the control-completion oracle.

The mathematical completion Delta G = Q/lambda-G is a *virtual* relaxation for
an existing grid.  This module reinterprets positive pieces of Delta G as
candidate fast active-power flexibility (battery/inverter/flexible-load)
reinforcement.  It deliberately distinguishes:

1. operational SOC cost J;
2. exact PIC-completed value W+;
3. effective actuator Gramian G (control geometry);
4. planning proxies such as device count, tr(Delta G), and an effective MW
   authority proxy;
5. scenario flexibility (fixed versus reconfigurable authority).

The MW proxy is not a battery nameplate rating.  If a dimensionless actuator
command a_i is mapped to physical power P_i a_i and is charged at .5*r_ref*a_i^2,
then a diagonal Gramian increment Delta G_ii corresponds to
P_i = baseMVA*sqrt(r_ref*Delta G_ii).  Saturation, state of charge, energy rating,
and degradation are separate constraints not represented by this proxy.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize


def generator_authority_fixed(problem, r0=0.2):
    """Independent-generator authority g_i=1/r0 in nodal injection coordinates."""
    return np.ones(problem.ng, dtype=float)/float(r0)


def generator_authority_noise_proportional(problem, Gamma, r0=0.2):
    """Allocate the same total generator authority proportional to Gamma_g^2.

    This makes q_g/g_g constant across generators, eliminating the scalar
    weakest-channel bottleneck while preserving sum_i g_i = ng/r0.
    """
    Gamma=np.asarray(Gamma,float)
    qg=Gamma[problem.gen_idx]**2
    total=problem.ng/float(r0)
    return total*qg/qg.sum()


def authority_to_Rdiag(g):
    g=np.asarray(g,float)
    if np.any(g<=0): raise ValueError('authority must be positive')
    return 1.0/g


@dataclass
class DiagonalCompletion:
    lam: float
    authority_gen: np.ndarray
    Gdiag: np.ndarray
    Gplus_diag: np.ndarray
    Delta_diag: np.ndarray
    burden_trace: float
    virtual_rank: int


def diagonal_completion(problem, Gamma, authority_gen):
    """Minimal scalar PIC completion for diagonal independent generator authority."""
    Gamma=np.asarray(Gamma,float); g=np.asarray(authority_gen,float)
    if Gamma.shape!=(problem.n,) or g.shape!=(problem.ng,):
        raise ValueError('shape mismatch')
    q=Gamma**2
    lam=float(np.min(q[problem.gen_idx]/g))
    Gdiag=np.zeros(problem.n); Gdiag[problem.gen_idx]=g
    Gplus=q/lam
    Delta=np.maximum(Gplus-Gdiag,0.0)
    tol=1e-10*max(1.0,float(np.max(Gplus)))
    return DiagonalCompletion(lam,g,Gdiag,Gplus,Delta,float(Delta.sum()),int(np.sum(Delta>tol)))


def effective_power_authority_MW(completion: DiagonalCompletion, *, baseMVA=100.0, r_ref=0.2):
    """Effective aggregate MW authority proxy associated with Delta G."""
    return float(baseMVA*np.sum(np.sqrt(float(r_ref)*np.maximum(completion.Delta_diag,0.0))))


def authority_vector_power_MW(completion: DiagonalCompletion, *, baseMVA=100.0, r_ref=0.2):
    return baseMVA*np.sqrt(float(r_ref)*np.maximum(completion.Delta_diag,0.0))


def robust_fixed_authority(problem, scenarios, r0=0.2, maxiter=None):
    """Exact minimax fixed authority for scalar-PIC completion burden.

    For scenario s, B_s+A = (sum q_s) max_i g_i/q_{s,i}.  Introducing
    a common epigraph t gives linear upper bounds
    g_i <= t q_{s,i}/sum(q_s).  Hence the minimax solution is available
    in closed form from the pointwise minimum of these caps.
    """
    qs=[np.asarray(G,float)**2 for G in scenarios]
    qgs=[q[problem.gen_idx] for q in qs]
    sums=np.asarray([q.sum() for q in qs])
    total=problem.ng/float(r0)
    caps=np.min(np.vstack([qg/s for qg,s in zip(qgs,sums)]),axis=0)
    t=total/caps.sum()
    g=t*caps
    burdens=np.asarray([q.sum()*np.max(g/qg)-total for q,qg in zip(qs,qgs)])
    class Result:
        success=True
        message='closed-form epigraph solution'
    return g,burdens,Result()


def lqr_for_nodal_authority(problem, controlled_buses, Rdiag, *, angle_weight=2.0, freq_weight=0.5):
    """Continuous LQR around the post-event equilibrium for arbitrary nodal devices."""
    buses=np.asarray(controlled_buses,int); Rdiag=np.asarray(Rdiag,float)
    if len(buses)!=len(Rdiag): raise ValueError('Rdiag length')
    n=problem.n; m=len(buses); Minv=np.diag(1.0/problem.M)
    A=np.block([
        [np.zeros((n-1,n-1)),problem.H],
        [-Minv@problem.Lstar@problem.E,-Minv@np.diag(problem.damp)]
    ])
    Anod=np.zeros((n,m)); Anod[buses,np.arange(m)]=1.0
    B=np.vstack([np.zeros((n-1,m)),Minv@Anod])
    Qq=problem.E.T@problem.Lstar@problem.E
    Qq/=max(np.trace(Qq)/(n-1),1e-12)
    Qw=np.diag(problem.M); Qw/=max(np.trace(Qw)/n,1e-12)
    Qcost=np.block([[angle_weight*Qq,np.zeros((n-1,n))],
                    [np.zeros((n,n-1)),freq_weight*Qw]])
    R=np.diag(Rdiag)
    P=solve_continuous_are(A,B,Qcost,R)
    K=np.linalg.solve(R,B.T@P)
    return K,Anod,dict(A=A,B=B,Q=Qcost,R=R,P=P)


def simulate_nodal_lqr(problem, Gamma, controlled_buses, Rdiag, gain, *, alpha=.01, clip=8.0,
                       T=3.0, dt=.005, n_paths=400, seed=1, cost=None):
    """Nonlinear smooth-objective simulation for arbitrary nodal fast-flexibility devices."""
    if cost is None:
        from smooth_barrier import SmoothPICCost
        cost=SmoothPICCost()
    buses=np.asarray(controlled_buses,int); Rdiag=np.asarray(Rdiag,float)
    rng=np.random.default_rng(seed); n=problem.n
    th=np.repeat(problem.theta_pre[None,:],n_paths,axis=0).copy(); om=np.zeros((n_paths,n))
    steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt)
    run=np.zeros(n_paths); ctrl=np.zeros(n_paths); cross=np.zeros(n_paths,bool)
    dmax=np.max(np.abs(th@problem.Inc),axis=1)
    for k in range(steps):
        v=-float(alpha)*(problem.reduced_state(th,om)@gain.T)
        if clip is not None: v=np.clip(v,-float(clip),float(clip))
        run+=cost.running(problem,th,om)*dt
        cu=.5*np.sum((v*v)*Rdiag[None,:],axis=1); run+=cu*dt; ctrl+=cu*dt
        nod=np.zeros((n_paths,n)); nod[:,buses]=v
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+nod)/problem.M[None,:]
        om+=dt*drift+(np.asarray(Gamma)/problem.M)[None,:]*sq*rng.standard_normal((n_paths,n))
        th+=dt*om
        dm=np.max(np.abs(th@problem.Inc),axis=1);dmax=np.maximum(dmax,dm);cross|=(dm>=np.pi/2)
    J=run+cost.terminal(problem,th,om)
    return dict(J_mean=float(J.mean()),J_se=float(J.std(ddof=1)/np.sqrt(n_paths)),
                cross_probability=float(cross.mean()),control_effort_mean=float(ctrl.mean()),
                dmax_deg_quantiles=np.degrees(np.quantile(dmax,[.5,.9,.99])).tolist(),raw_J=J)


def completed_lqr_shadow_proxy(problem, Gamma, completion_geometry, gain, *, alpha=.01,
                               T=3.,dt=.005,n_paths=300,seed=7,clip=8.0):
    """RMS nodal command under the H4 LQR guide, used only as a placement heuristic.

    This is *not* the exact Gramian shadow-price theorem, which would require the
    gradient of the exact completed value.  It is a dynamics-aware proxy that
    incorporates network response and the severe-event trajectory.
    """
    rng=np.random.default_rng(seed);n=problem.n
    th=np.repeat(problem.theta_pre[None,:],n_paths,axis=0).copy();om=np.zeros((n_paths,n))
    steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt)
    acc=np.zeros(n)
    for k in range(steps):
        u=-float(alpha)*(problem.reduced_state(th,om)@gain.T)
        if clip is not None:u=np.clip(u,-float(clip),float(clip))
        acc+=np.mean(u*u,axis=0)*dt
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+u)/problem.M[None,:]
        om+=dt*drift+(np.asarray(Gamma)/problem.M)[None,:]*sq*rng.standard_normal((n_paths,n))
        th+=dt*om
    return np.sqrt(acc/max(T,1e-12))
