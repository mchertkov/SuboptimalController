"""Control inflation/completion for linearly-solvable stochastic control.

General matrix idea
-------------------
For physical diffusion covariance Q = sigma sigma^T and physical quadratic-control
Gramian G = B R^{-1} B^T, exact path-integral linear solvability requires

    Q = lambda G.

If lambda G <= Q, the *control-completed* Gramian

    G_plus(lambda) = Q / lambda

contains the physical control authority and is exactly matched to the physical noise.
Factoring G_plus-G = C C^T gives an augmented problem with fictitious control Cv,
unit quadratic cost .5||v||^2, and exact Feynman--Kac value.  Because the augmented
control set contains the physical one, its value is automatically a lower bound on
the original optimum.  No Hessian-curvature condition is required.

This module implements the power-system specialization in nodal injection coordinates
and an importance-sampled Feynman--Kac estimator using the *physical* nodal noise.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp

from smooth_barrier import SmoothPICCost


def _psd_factor(A, tol=1e-12):
    A=(np.asarray(A,float)+np.asarray(A,float).T)/2
    vals,vecs=eigh(A)
    scale=max(1.0,float(np.max(np.abs(vals))))
    if vals.min() < -tol*scale:
        raise ValueError(f'matrix is not PSD: min eigenvalue={vals.min():.3e}')
    keep=vals>tol*scale
    if not np.any(keep):
        return np.zeros((A.shape[0],0)),np.maximum(vals,0.0)
    return vecs[:,keep]@np.diag(np.sqrt(vals[keep])),np.maximum(vals,0.0)


def lambda_max_from_covariances(Q,G,tol=1e-10):
    """Largest lambda such that lambda*G <= Q, allowing singular Q.

    Raises ValueError if range(G) is not contained in range(Q).
    """
    Q=(np.asarray(Q,float)+np.asarray(Q,float).T)/2
    G=(np.asarray(G,float)+np.asarray(G,float).T)/2
    vals,U=eigh(Q)
    qscale=max(1.0,float(np.max(np.abs(vals))))
    pos=vals>tol*qscale
    P=U[:,pos]@U[:,pos].T if np.any(pos) else np.zeros_like(Q)
    # PSD G has range contained in Q iff its component outside P vanishes.
    off=(np.eye(Q.shape[0])-P)@G@(np.eye(Q.shape[0])-P)
    cross=(np.eye(Q.shape[0])-P)@G@P
    gscale=max(1.0,float(np.linalg.norm(G,2)))
    if np.linalg.norm(off,2)>tol*gscale or np.linalg.norm(cross,2)>tol*gscale:
        raise ValueError('range(G) is not contained in range(Q)')
    if np.linalg.norm(G,2)<=tol:
        return math.inf
    Qpinvsqrt=U[:,pos]@np.diag(1.0/np.sqrt(vals[pos]))@U[:,pos].T
    M=(Qpinvsqrt@G@Qpinvsqrt)
    mu=float(np.max(eigh((M+M.T)/2,eigvals_only=True)))
    if mu<=tol:
        return math.inf
    return 1.0/mu


@dataclass
class ControlInflationGeometry:
    actuation_name: str
    control_weight: float
    Gamma: np.ndarray
    lam: float
    injection_gain_physical: np.ndarray
    injection_gain_completed: np.ndarray
    injection_gain_virtual: np.ndarray
    virtual_factor: np.ndarray
    virtual_rank: int
    generator_rank: int

    @property
    def inflation_trace(self):
        return float(np.trace(self.injection_gain_virtual))

    @property
    def inflation_fro(self):
        return float(np.linalg.norm(self.injection_gain_virtual,'fro'))


def control_inflation_geometry(problem, actuation, Gamma, control_weight=0.2, lam=None):
    """Minimal scalar PIC control completion in nodal injection coordinates.

    Physical generator/mode injection is A v, cost .5*r*||v||^2, hence
    G_inj = A A^T / r.  Physical injection noise has covariance Gamma^2.
    For lambda <= lambda_max, the completed matched gain is Gamma^2/lambda.
    At lambda=lambda_max this is the least inflated member of the scalar PIC family.
    """
    Gamma=np.asarray(Gamma,float)
    if Gamma.shape!=(problem.n,) or np.any(Gamma<=0):
        raise ValueError('Gamma must be strictly positive at all nodes')
    r=float(control_weight)
    A=actuation.nodal_matrix(problem)
    G=A@A.T/r
    Q=np.diag(Gamma**2)
    lmax=lambda_max_from_covariances(Q,G)
    if lam is None:
        lam=lmax
    lam=float(lam)
    if lam<=0 or lam>lmax*(1+1e-9):
        raise ValueError(f'lambda={lam} outside admissible interval (0,{lmax}]')
    Gplus=Q/lam
    Delta=(Gplus-G + (Gplus-G).T)/2
    C,ev=_psd_factor(Delta)
    return ControlInflationGeometry(
        actuation_name=actuation.name,control_weight=r,Gamma=Gamma,lam=lam,
        injection_gain_physical=G,injection_gain_completed=Gplus,
        injection_gain_virtual=Delta,virtual_factor=C,virtual_rank=C.shape[1],
        generator_rank=actuation.rank,
    )


def _prepare(problem,state):
    arr=np.asarray(state,float)
    if arr.ndim==2 and arr.shape==(2,problem.n): arr=arr[None,:,:]
    if arr.ndim!=3 or arr.shape[1:]!=(2,problem.n):
        raise ValueError('state must have shape (2,n) or (q,2,n)')
    return arr[:,0,:].copy(),arr[:,1,:].copy()


def feynman_kac_control_inflation(problem, geometry: ControlInflationGeometry, state, *,
                                   t0=0.,T=3.,dt=.005,n_paths=1200,seed=1,cost=None,
                                   guide_nodal=None,brownian=None,return_logweights=False):
    """Feynman--Kac value of the exactly matched control-inflated problem.

    The passive process uses the *physical* injection noise Gamma at every node.
    ``guide_nodal`` is an optional callable (t,theta,omega)->nodal injection drift
    used only for importance sampling.  Exact Girsanov correction removes its bias.
    It must lie in the noise range; Gamma is strictly positive in this implementation.
    """
    if cost is None: cost=SmoothPICCost()
    th0,om0=_prepare(problem,state); nq=len(th0); n=problem.n
    horizon=float(T)-float(t0)
    if horizon<=0:
        term=cost.terminal(problem,th0,om0)
        return dict(W=term,ess=np.full(nq,n_paths),lam=geometry.lam,cross_fraction=np.zeros(nq))
    steps=int(np.ceil(horizon/dt)); dt=horizon/steps; sq=math.sqrt(dt)
    rng=np.random.default_rng(seed)
    if brownian is None:
        brownian=rng.standard_normal((steps,n_paths,n))
    else:
        brownian=np.asarray(brownian,float)
        if brownian.shape!=(steps,n_paths,n): raise ValueError('brownian shape')
    th=np.repeat(th0[:,None,:],n_paths,axis=1).reshape(nq*n_paths,n)
    om=np.repeat(om0[:,None,:],n_paths,axis=1).reshape(nq*n_paths,n)
    run=np.zeros(nq*n_paths); ll=np.zeros(nq*n_paths); cross=np.zeros(nq*n_paths,bool)
    Gamma=geometry.Gamma
    for k in range(steps):
        run+=cost.running(problem,th,om)*dt
        pid=np.arange(nq*n_paths)%n_paths
        dW=sq*brownian[k,pid,:]
        if guide_nodal is None:
            nod=np.zeros_like(th)
        else:
            nod=np.asarray(guide_nodal(t0+k*dt,th,om),float)
            if nod.shape!=th.shape: raise ValueError('guide_nodal must return (paths,n)')
            a=nod/Gamma[None,:]
            ll+=-np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*dt
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+nod)/problem.M[None,:]
        om=om+dt*drift+(Gamma/problem.M)[None,:]*dW
        th=th+dt*om
        cross|=(np.max(np.abs(th@problem.Inc),axis=1)>=np.pi/2)
    S=run+cost.terminal(problem,th,om)
    lw=(-S/geometry.lam+ll).reshape(nq,n_paths)
    lse=logsumexp(lw,axis=1)
    W=-geometry.lam*(lse-math.log(n_paths))
    lwn=lw-lse[:,None]
    ess=1.0/np.sum(np.exp(2*lwn),axis=1)
    out=dict(W=W,ess=ess,lam=geometry.lam,cross_fraction=cross.reshape(nq,n_paths).mean(axis=1),
             logweight_std=np.std(lw,axis=1),n_paths=n_paths,dt=dt,steps=steps)
    if return_logweights: out['logweights']=lw
    return out


def generator_lqr_guide_nodal(problem, actuation, gain, alpha=0.01, clip=8.0):
    """Map a restricted generator LQR feedback to a nodal drift guide."""
    A=actuation.nodal_matrix(problem)
    a=float(alpha)
    def guide(t,theta,omega):
        v=-a*(problem.reduced_state(theta,omega)@gain.T)
        if clip is not None: v=np.clip(v,-float(clip),float(clip))
        return v@A.T
    return guide


def completion_lqr_gain(problem, geometry: ControlInflationGeometry, *, angle_weight=2.0, freq_weight=0.5):
    """Linear-quadratic guide for the fully completed nodal control geometry.

    Uses nodal injection u in all n buses with R_plus satisfying
    R_plus^{-1} = Gamma^2/lambda = G_plus.
    """
    from scipy.linalg import solve_continuous_are
    n=problem.n
    Minv=np.diag(1.0/problem.M)
    A=np.block([
        [np.zeros((n-1,n-1)),problem.H],
        [-Minv@problem.Lstar@problem.E,-Minv@np.diag(problem.damp)]
    ])
    B=np.vstack([np.zeros((n-1,n)),Minv])
    Qq=problem.E.T@problem.Lstar@problem.E
    Qq/=max(np.trace(Qq)/(n-1),1e-12)
    Qw=np.diag(problem.M);Qw/=max(np.trace(Qw)/n,1e-12)
    Qcost=np.block([[angle_weight*Qq,np.zeros((n-1,n))],
                    [np.zeros((n,n-1)),freq_weight*Qw]])
    Rplus=np.diag(geometry.lam/(geometry.Gamma**2))
    P=solve_continuous_are(A,B,Qcost,Rplus)
    K=np.linalg.solve(Rplus,B.T@P)
    return K,dict(A=A,B=B,Q=Qcost,R=Rplus,P=P)


def completion_lqr_guide_nodal(problem, gain, alpha=1.0, clip=None):
    a=float(alpha)
    def guide(t,theta,omega):
        u=-a*(problem.reduced_state(theta,omega)@gain.T)
        if clip is not None: u=np.clip(u,-float(clip),float(clip))
        return u
    return guide


def completion_diagnostics(Q, G, lam):
    """Coordinate-aware and dimensionless measures of scalar PIC completion.

    Parameters
    ----------
    Q : ndarray
        Positive-definite physical diffusion covariance in the same coordinates as G.
    G : ndarray
        Physical quadratic-control Gramian.
    lam : float
        Scalar matching temperature, normally lambda_max.

    Returns
    -------
    dict
        ``inflation_trace`` = tr(Q/lam-G), the absolute amount of virtual
        Gramian added in the chosen physical coordinates.

        ``whitened_deficit_trace`` = tr(I-lam Q^{-1/2}GQ^{-1/2}), a
        dimensionless noise-whitened coverage deficit.  It is invariant under
        an overall rescaling Q -> c^2 Q accompanied by lam -> c^2 lam and
        reduces to the number of missing noise directions for homogeneous
        isotropic noise and an orthogonal control projector.

    These quantities diagnose different aspects of mismatch and are not, by
    themselves, universal bounds on the value-function gap.
    """
    Q=np.asarray(Q,float); G=np.asarray(G,float); lam=float(lam)
    vals,U=eigh((Q+Q.T)/2)
    if np.min(vals)<=0: raise ValueError('Q must be positive definite')
    Qis=U@np.diag(1/np.sqrt(vals))@U.T
    Delta=(Q/lam-G); Delta=(Delta+Delta.T)/2
    M=np.eye(Q.shape[0])-lam*(Qis@G@Qis); M=(M+M.T)/2
    return dict(inflation_trace=float(np.trace(Delta)),
                inflation_fro=float(np.linalg.norm(Delta,'fro')),
                whitened_deficit_trace=float(np.trace(M)),
                whitened_deficit_fro=float(np.linalg.norm(M,'fro')),
                whitened_deficit_eigs=eigh(M,eigvals_only=True))
