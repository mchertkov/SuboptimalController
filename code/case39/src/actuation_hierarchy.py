"""Actuation-rank hierarchy and general PIC geometry for case39.

The hierarchy isolates the transient actuator subspace while keeping the same
post-event equilibrium and fixed equal balancing allocation.
Physical generator control is u_G = Q v with Q^T Q = I, so the physical
quadratic effort .5*r*||u_G||^2 equals .5*r*||v||^2.  This makes H0--H3
comparisons meaningful.

H0: one MATPOWER slack-bus actuator (bus 31).
H1: one equal-distributed generator mode.
H2: rank-3 nested multimode subspace containing H0 and H1 plus one
    generator-network spectral/coherency mode.
H3: independent control at all 10 generators.

For a general nodal injection-mode matrix A = S_g Q and R=r I, the largest
linearly-solvable PIC temperature is

    lambda_max = 1 / lambda_max( Gamma^{-1} A R^{-1} A^T Gamma^{-1} )
               = 1 / || Gamma^{-1} A R^{-1/2} ||_2^2,

where Gamma is the diagonal vector of nodal injection-noise amplitudes.  The
physical and surrogate omega-space diffusion matrices are related by the
congruence M^{-1}(.)M^{-1}; therefore inertia cancels from the lambda test.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_continuous_are, eigh


@dataclass
class ActuationClass:
    name: str
    label: str
    Qgen: np.ndarray        # ng x r, orthonormal columns
    bus_ids: np.ndarray
    slack_bus: int = 31

    @property
    def rank(self):
        return int(self.Qgen.shape[1])

    def nodal_matrix(self, problem):
        return problem.Sg @ self.Qgen


@dataclass
class GeneralPICGeometry:
    actuation: ActuationClass
    control_weight: float
    Gamma: np.ndarray
    lam: float
    injection_diffusion_physical: np.ndarray
    injection_diffusion_surrogate: np.ndarray
    omega_deleted: np.ndarray
    omega_deleted_eigvals: np.ndarray
    omega_deleted_eigvecs: np.ndarray

    @property
    def deleted_rank(self):
        tol=max(1e-12,1e-10*np.max(self.omega_deleted_eigvals) if self.omega_deleted_eigvals.size else 1e-12)
        return int(np.count_nonzero(self.omega_deleted_eigvals>tol))

    @property
    def deleted_trace(self):
        return float(np.trace(self.omega_deleted))

    @property
    def deleted_fro(self):
        return float(np.linalg.norm(self.omega_deleted,'fro'))

    def sqrt_deleted(self):
        vals=np.maximum(self.omega_deleted_eigvals,0.0)
        return self.omega_deleted_eigvecs @ np.diag(np.sqrt(vals)) @ self.omega_deleted_eigvecs.T


def _orthonormalize_columns(cols, tol=1e-10):
    Q=[]
    for c in cols:
        v=np.asarray(c,float).copy()
        for q in Q:
            v-=q*np.dot(q,v)
        nv=np.linalg.norm(v)
        if nv>tol:
            Q.append(v/nv)
    return np.column_stack(Q) if Q else np.zeros((len(cols[0]),0))


def _generator_kron_laplacian(problem):
    """Kron-reduced post-event small-signal stiffness on generator buses."""
    G=problem.gen_idx
    Ld=np.array([i for i in range(problem.n) if i not in set(G)],dtype=int)
    L=problem.Lstar
    Lgg=L[np.ix_(G,G)]
    if len(Ld)==0:
        return Lgg
    Lgl=L[np.ix_(G,Ld)]; Llg=L[np.ix_(Ld,G)]; Lll=L[np.ix_(Ld,Ld)]
    # Lll is SPD when at least one generator is retained in a connected network.
    return Lgg-Lgl@np.linalg.solve(Lll,Llg)


def build_hierarchy(problem, bus_ids):
    """Construct H0--H3 with comparable physical generator effort metric."""
    bus_ids=np.asarray(bus_ids)
    gen_bus_ids=bus_ids[problem.gen_idx]
    ng=problem.ng
    # MATPOWER case39 slack/reference generator is bus 31.
    slack_loc=int(np.where(gen_bus_ids==31)[0][0])
    e0=np.zeros(ng); e0[slack_loc]=1.0
    qdist=np.ones(ng)/np.sqrt(ng)

    # A generator coherency/spatial mode from the Kron-reduced stiffness.
    Lg=_generator_kron_laplacian(problem)
    vals,vecs=eigh((Lg+Lg.T)/2)
    order=np.argsort(vals)
    # first mode is approximately constant; take the first nonconstant mode
    spec=None
    for j in order:
        v=vecs[:,j]
        v=v-qdist*np.dot(qdist,v)
        if np.linalg.norm(v)>1e-7:
            spec=v/np.linalg.norm(v); break
    if spec is None:
        # deterministic fallback
        spec=np.arange(ng,dtype=float)-(ng-1)/2
        spec-=qdist*np.dot(qdist,spec); spec/=np.linalg.norm(spec)

    Q0=e0[:,None]
    Q1=qdist[:,None]
    # H2 is deliberately nested over both H0 and H1, then adds a network mode.
    Q2=_orthonormalize_columns([qdist,e0,spec])
    if Q2.shape[1]<3:
        # add canonical residual directions if the spectral mode was dependent
        for j in range(ng):
            Q2=_orthonormalize_columns([Q2[:,k] for k in range(Q2.shape[1])]+[np.eye(ng)[:,j]])
            if Q2.shape[1]>=3: break
    Q3=np.eye(ng)
    return [
        ActuationClass('H0','single slack bus 31',Q0,gen_bus_ids),
        ActuationClass('H1','equal distributed rank-one',Q1,gen_bus_ids),
        ActuationClass('H2','rank-3 nested generator modes',Q2[:,:3],gen_bus_ids),
        ActuationClass('H3','independent generator control',Q3,gen_bus_ids),
    ]


def lqr_for_actuation(problem, actuation: ActuationClass, *, angle_weight=2.0,
                       freq_weight=0.5, control_weight=0.2):
    """Continuous LQR around the post-event equilibrium for a restricted subspace."""
    Minv=np.diag(1.0/problem.M)
    A=np.block([
        [np.zeros((problem.n-1,problem.n-1)), problem.H],
        [-Minv@problem.Lstar@problem.E, -Minv@np.diag(problem.damp)]
    ])
    Anod=actuation.nodal_matrix(problem)
    B=np.vstack([np.zeros((problem.n-1,actuation.rank)),Minv@Anod])
    Qq=problem.E.T@problem.Lstar@problem.E
    Qq/=max(np.trace(Qq)/(problem.n-1),1e-12)
    Qw=np.diag(problem.M); Qw/=max(np.trace(Qw)/problem.n,1e-12)
    Qcost=np.block([[angle_weight*Qq,np.zeros((problem.n-1,problem.n))],
                    [np.zeros((problem.n,problem.n-1)),freq_weight*Qw]])
    R=float(control_weight)*np.eye(actuation.rank)
    P=solve_continuous_are(A,B,Qcost,R)
    K=np.linalg.solve(R,B.T@P)
    return K,dict(A=A,B=B,Q=Qcost,R=R,P=P)


def mode_policy(problem, actuation: ActuationClass, gain, alpha=1.0, clip=None):
    """Policy in mode coordinates v; physical generator injection is Qgen v."""
    a=float(alpha)
    def pol(t,theta,omega):
        v=-a*(problem.reduced_state(theta,omega)@gain.T)
        if clip is not None:
            v=np.clip(v,-float(clip),float(clip))
        return v
    return pol


def physical_generator_control(actuation: ActuationClass, v):
    return np.asarray(v) @ actuation.Qgen.T


def general_pic_geometry(problem, actuation: ActuationClass, Gamma, control_weight=0.2):
    """Largest feasible linearly-solvable diffusion for a general actuation subspace."""
    Gamma=np.asarray(Gamma,float)
    if Gamma.shape!=(problem.n,) or np.any(Gamma<=0):
        raise ValueError('Gamma must be positive on all nodes for this implementation')
    r=float(control_weight)
    Anod=actuation.nodal_matrix(problem) # n x m
    # whitened B R^{-1/2}; inertia cancels by congruence
    C=(Anod/ Gamma[:,None])/np.sqrt(r)
    smax=np.linalg.svd(C,compute_uv=False)[0]
    lam=float(1.0/(smax*smax))
    Dinj=.5*np.diag(Gamma**2)
    Dtilde_inj=.5*lam/r*(Anod@Anod.T)
    resid_inj=(Dinj-Dtilde_inj + (Dinj-Dtilde_inj).T)/2
    ev=np.linalg.eigvalsh(resid_inj)
    if ev.min() < -1e-9:
        raise RuntimeError(f'injection diffusion residual not PSD: min={ev.min()}')
    Minv=np.diag(1.0/problem.M)
    Aom=Minv@resid_inj@Minv
    Aom=(Aom+Aom.T)/2
    vals,vecs=eigh(Aom)
    vals[np.abs(vals)<1e-12]=0.0
    vals=np.maximum(vals,0.0)
    return GeneralPICGeometry(
        actuation=actuation,control_weight=r,Gamma=Gamma,lam=lam,
        injection_diffusion_physical=Dinj,
        injection_diffusion_surrogate=Dtilde_inj,
        omega_deleted=Aom,omega_deleted_eigvals=vals,omega_deleted_eigvecs=vecs)


def simulate_soc_cost_modes(problem, actuation: ActuationClass, policy, noise, *,
                            T=3.0,dt=0.0025,n_paths=512,seed=1,control_weight=0.2,cost=None):
    """True-system stopped SOC cost with restricted transient generator actuation."""
    if cost is None:
        from pic_case39 import PICCost
        cost=PICCost()
    rng=np.random.default_rng(seed); n=problem.n
    th=np.repeat(problem.theta_pre[None,:],n_paths,axis=0).copy(); om=np.zeros((n_paths,n))
    noise=np.asarray(noise,float); steps=int(np.ceil(T/dt)); dt=T/steps; sq=np.sqrt(dt)
    active=np.ones(n_paths,bool); failed=np.zeros(n_paths,bool); run=np.zeros(n_paths); ctrl=np.zeros(n_paths)
    dmax=np.max(np.abs(th@problem.Inc),axis=1)
    Anod=actuation.nodal_matrix(problem)
    r=float(control_weight)
    for k in range(steps):
        if not np.any(active): break
        t=k*dt; idx=np.where(active)[0]; tha=th[idx]; oma=om[idx]
        v=policy(t,tha,oma)
        run[idx]+=cost.running(problem,tha,oma)*dt
        cu=.5*r*np.sum(v*v,axis=1)
        run[idx]+=cu*dt; ctrl[idx]+=cu*dt
        nod=v@Anod.T
        drift=(problem.p_post[None,:]-problem.damp[None,:]*oma-problem.network_force(tha)+nod)/problem.M[None,:]
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

def scan_scaled_lqr_modes(problem, actuation: ActuationClass, gain, alphas, noise, *,
                          T=3.0,dt=0.003,n_paths=250,seed=1,control_weight=0.2,cost=None,clip=8.0):
    """Vectorized common-random-number scan of scaled LQR policies."""
    if cost is None:
        from pic_case39 import PICCost
        cost=PICCost()
    alphas=np.asarray(alphas,float); na=len(alphas); n=problem.n
    rng=np.random.default_rng(seed); noise=np.asarray(noise,float)
    # dimensions alpha x path x node
    th=np.repeat(problem.theta_pre[None,None,:],na*n_paths,axis=0).reshape(na,n_paths,n).copy()
    om=np.zeros((na,n_paths,n)); active=np.ones((na,n_paths),bool); failed=np.zeros((na,n_paths),bool)
    run=np.zeros((na,n_paths)); ctrl=np.zeros((na,n_paths)); dmax=np.max(np.abs(th@problem.Inc),axis=2)
    Anod=actuation.nodal_matrix(problem); r=float(control_weight)
    steps=int(np.ceil(T/dt)); dt=T/steps; sq=np.sqrt(dt)
    for k in range(steps):
        # common physical nodal noise across alpha for matched CRN comparison
        eps=rng.standard_normal((n_paths,n))
        for ia,a in enumerate(alphas):
            idx=np.where(active[ia])[0]
            if len(idx)==0: continue
            tha=th[ia,idx]; oma=om[ia,idx]
            v=-float(a)*(problem.reduced_state(tha,oma)@gain.T)
            if clip is not None: v=np.clip(v,-float(clip),float(clip))
            run[ia,idx]+=cost.running(problem,tha,oma)*dt
            cu=.5*r*np.sum(v*v,axis=1); run[ia,idx]+=cu*dt; ctrl[ia,idx]+=cu*dt
            drift=(problem.p_post[None,:]-problem.damp[None,:]*oma-problem.network_force(tha)+(v@Anod.T))/problem.M[None,:]
            omnew=oma+dt*drift+(noise/problem.M)[None,:]*sq*eps[idx]
            thnew=tha+dt*omnew; th[ia,idx]=thnew; om[ia,idx]=omnew
            dm=np.max(np.abs(thnew@problem.Inc),axis=1); dmax[ia,idx]=np.maximum(dmax[ia,idx],dm)
            ex=dm>=np.pi/2
            if np.any(ex):
                exidx=idx[ex]; failed[ia,exidx]=True; active[ia,exidx]=False
    rows=[]
    for ia,a in enumerate(alphas):
        terminal=np.empty(n_paths); terminal[failed[ia]]=cost.failure
        surv=~failed[ia]
        terminal[surv]=cost.terminal(problem,th[ia,surv],om[ia,surv])
        J=run[ia]+terminal
        rows.append(dict(alpha=float(a),J=float(J.mean()),J_se=float(J.std(ddof=1)/np.sqrt(n_paths)),
                         failure=float(failed[ia].mean()),effort=float(ctrl[ia].mean()),
                         dmax50=float(np.degrees(np.quantile(dmax[ia],.5))),
                         dmax90=float(np.degrees(np.quantile(dmax[ia],.9))),
                         dmax99=float(np.degrees(np.quantile(dmax[ia],.99)))))
    return rows
