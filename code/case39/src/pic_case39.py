"""Importance-sampled path-integral surrogate for the stochastic case39 swing model.

The module implements the *deflated* linearly-solvable surrogate

    dX = f(X) dt + B u dt + sigma_tilde dW,
    sigma_tilde sigma_tilde^T = lambda B R^{-1} B^T,

on the cohesive domain |B_e^T theta| < pi/2.  The passive surrogate value is
computed from a stopped Feynman--Kac expectation.  A generator feedback may be
used only as an importance-sampling guide; an exact stopped Girsanov likelihood
ratio removes the bias.

This is a diagnostic/constructive numerical layer.  Finite state sampling does
*not* by itself establish the global subsolution inequality required for a
certificate.  ``curvature_trace_diagnostic`` tests the one unresolved nonlinear
term on selected occupied states using a Hutchinson finite-difference estimator
with common random numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.special import logsumexp

from case39_reduced import load_case39
from syncnet import incidence_and_cycles, newton_equilibrium
from stochastic_swing import SwingProblem, noise_profile


@dataclass
class PICCost:
    """Smooth/stopped objective for the linearly-solvable surrogate.

    All costs use the same physical state normalization as the Stage-1 pilot.
    ``running_energy`` multiplies nonlinear relative swing energy per bus;
    ``running_barrier`` multiplies the mean cohesive Bregman barrier;
    ``terminal_energy`` multiplies relative swing energy per bus at T;
    ``failure`` is the Dirichlet cost at first cohesive-boundary exit.
    """
    running_energy: float = 0.020
    running_barrier: float = 0.004
    terminal_energy: float = 0.050
    failure: float = 0.20

    def running(self, problem: SwingProblem, theta, omega):
        e = np.maximum(problem.energy(theta, omega), 0.0) / problem.n
        b = problem.barrier(theta, clip=1e-12)
        return self.running_energy * e + self.running_barrier * b

    def terminal(self, problem: SwingProblem, theta, omega):
        e = np.maximum(problem.energy(theta, omega), 0.0) / problem.n
        return self.terminal_energy * e


@dataclass
class PICGeometry:
    Rdiag: np.ndarray
    Gamma: np.ndarray
    lam: float
    deleted_diag: np.ndarray
    surrogate_injection_noise: np.ndarray

    @property
    def deleted_rank(self):
        return int(np.count_nonzero(self.deleted_diag > 1e-14))


def build_case39_problem(alpha=4.0, damping=0.05):
    """Build the severe bus-20 load-loss problem used by the companion paper."""
    case = load_case39(alpha=alpha)
    G,p,M,is_gen,base,bus_ids = (case[k] for k in
        ['G','p','M','is_gen','baseMVA','bus_ids'])
    Inc,Cyc,edges,nodes = incidence_and_cycles(G)
    Kedge = np.array([G[u][v]['K'] for u,v in edges])
    eq_pre = newton_equilibrium(Inc,Kedge,p)
    if not eq_pre['ok']:
        raise RuntimeError('pre-event equilibrium failed')
    disturbance = int(np.argmin(p))
    dP = float(-p[disturbance])
    gamma = np.zeros(len(p)); gamma[is_gen] = 1.0/is_gen.sum()
    p_post = p.copy(); p_post[disturbance] += dP; p_post -= dP*gamma; p_post -= p_post.mean()
    eq_post = newton_equilibrium(Inc,Kedge,p_post,th0=eq_pre['theta'])
    if not eq_post['ok']:
        raise RuntimeError('post-event equilibrium failed')
    problem = SwingProblem(
        Inc=Inc,Kedge=Kedge,edges=edges,p_pre=p,p_post=p_post,M=M,
        damp=np.full(len(p),float(damping)),is_gen=is_gen,
        theta_pre=eq_pre['theta'],theta_star=eq_post['theta'],baseMVA=base,
        disturbance=disturbance,dP=dP,gamma=gamma,
    )
    return problem, dict(case=case, cycles=Cyc, eq_pre=eq_pre, eq_post=eq_post,
                         bus_ids=bus_ids)


def pic_geometry(problem: SwingProblem, Gamma, control_weight=0.2):
    """Return lambda_max and the deleted diffusion diagonal in omega coordinates.

    ``Gamma`` is the vector of nodal *injection* noise amplitudes, before M^{-1}.
    With R=diag(r_g),

        lambda_max = min_g r_g Gamma_g^2,

    and D-D_tilde has omega diagonal

        .5/M_i^2 [Gamma_i^2 - 1_{i in G} lambda/r_i].
    """
    Gamma = np.asarray(Gamma,float)
    if np.ndim(control_weight)==0:
        Rdiag=np.full(problem.ng,float(control_weight))
    else:
        Rdiag=np.asarray(control_weight,float).copy()
        if Rdiag.shape!=(problem.ng,):
            raise ValueError('control_weight vector must have ng entries')
    gg=Gamma[problem.gen_idx]
    if np.any(gg<=0):
        raise ValueError('all actuated generators need nonzero noise for positive lambda')
    lam=float(np.min(Rdiag*gg**2))
    deleted=.5*(Gamma/problem.M)**2
    deleted[problem.gen_idx] -= .5*lam/(Rdiag*problem.M[problem.gen_idx]**2)
    # floating point safety only
    deleted[np.abs(deleted)<1e-13]=0.0
    if np.min(deleted)<-1e-10:
        raise RuntimeError('D-Dtilde is not PSD')
    deleted=np.maximum(deleted,0.0)
    # Matching surrogate has injection noise sqrt(lambda/r_g) at generators, zero elsewhere.
    sur=np.zeros(problem.n)
    sur[problem.gen_idx]=np.sqrt(lam/Rdiag)
    return PICGeometry(Rdiag=Rdiag,Gamma=Gamma,lam=lam,deleted_diag=deleted,
                       surrogate_injection_noise=sur)


def _prepare_initial(problem, states):
    """states may be (theta,omega) or a list/array with shape (q,2,n)."""
    if isinstance(states, tuple):
        th=np.asarray(states[0],float); om=np.asarray(states[1],float)
        if th.ndim==1: th=th[None,:]
        if om.ndim==1: om=om[None,:]
    else:
        arr=np.asarray(states,float)
        if arr.ndim==2 and arr.shape==(2,problem.n): arr=arr[None,:,:]
        if arr.ndim!=3 or arr.shape[1:]!=(2,problem.n):
            raise ValueError('states must have shape (q,2,n)')
        th=arr[:,0,:].copy(); om=arr[:,1,:].copy()
    if th.shape!=om.shape or th.shape[1]!=problem.n:
        raise ValueError('bad state shapes')
    return th.copy(),om.copy()


def _guide_values(problem, theta, omega, gain, alpha, clip):
    if gain is None or alpha==0.0:
        return np.zeros((theta.shape[0],problem.ng))
    u=-float(alpha)*(problem.reduced_state(theta,omega)@gain.T)
    if clip is not None:
        u=np.clip(u,-float(clip),float(clip))
    return u


def _logmeanexp(a,axis=-1):
    a=np.asarray(a)
    n=a.shape[axis]
    return logsumexp(a,axis=axis)-math.log(n)


def feynman_kac_importance(problem: SwingProblem, geometry: PICGeometry, states,
                            *, t0=0.0, T=3.0, dt=0.005, n_paths=1500, seed=1,
                            cost=None, guide_gain=None, guide_alpha=0.0,
                            guide_clip=8.0, brownian=None, return_logweights=False):
    """Evaluate the stopped desirability/value at one or more states.

    The same Brownian increments are broadcast over all query states, providing
    common random numbers for finite-difference derivatives.  Under a guide ``v``
    the stopped Girsanov factor is

        exp[-int a.dW - .5 int |a|^2 dt],
        a_g = sqrt(r_g/lambda) v_g.

    Returns one result per query state, including importance-sampling ESS.
    """
    if cost is None: cost=PICCost()
    th0,om0=_prepare_initial(problem,states)
    nq=th0.shape[0]; n=problem.n; ng=problem.ng
    horizon=max(float(T)-float(t0),0.0)
    if horizon<=0:
        term=cost.terminal(problem,th0,om0)
        return dict(W=term.copy(),psi=np.exp(-term/geometry.lam),ess=np.full(nq,n_paths),
                    exit_fraction=np.zeros(nq),lam=geometry.lam)
    steps=int(np.ceil(horizon/dt)); dt=horizon/steps; sqdt=np.sqrt(dt)
    rng=np.random.default_rng(seed)
    if brownian is None:
        # generator noise only; shape steps x paths x ng
        brownian=rng.standard_normal((steps,n_paths,ng),dtype=np.float64)
    else:
        brownian=np.asarray(brownian,float)
        if brownian.shape!=(steps,n_paths,ng):
            raise ValueError(f'brownian must have shape {(steps,n_paths,ng)}')
    # Flatten query x path for network operations.
    th=np.repeat(th0[:,None,:],n_paths,axis=1).reshape(nq*n_paths,n)
    om=np.repeat(om0[:,None,:],n_paths,axis=1).reshape(nq*n_paths,n)
    active=np.ones(nq*n_paths,dtype=bool)
    failed=np.zeros(nq*n_paths,dtype=bool)
    running=np.zeros(nq*n_paths)
    logLR=np.zeros(nq*n_paths)
    sur=geometry.surrogate_injection_noise
    Rdiag=geometry.Rdiag
    alpha=float(guide_alpha)
    for k in range(steps):
        if not np.any(active): break
        # Evaluate guide and running cost at left endpoint for active paths.
        idx=np.where(active)[0]
        tha=th[idx]; oma=om[idx]
        v=_guide_values(problem,tha,oma,guide_gain,alpha,guide_clip)
        running[idx]+=cost.running(problem,tha,oma)*dt
        # Brownian increments corresponding to flat query/path ids.
        path_id=idx % n_paths
        dW=sqdt*brownian[k,path_id,:]
        if alpha!=0.0 and guide_gain is not None:
            a=v*np.sqrt(Rdiag[None,:]/geometry.lam)
            logLR[idx] += -np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*dt
        nodctrl=v@problem.Sg.T
        drift=(problem.p_post[None,:]-problem.damp[None,:]*oma-
               problem.network_force(tha)+nodctrl)/problem.M[None,:]
        omnew=oma+dt*drift
        # matching surrogate noise at generators only
        omnew[:,problem.gen_idx] += (sur[problem.gen_idx]/problem.M[problem.gen_idx])[None,:]*dW
        thnew=tha+dt*omnew
        th[idx]=thnew; om[idx]=omnew
        dm=np.max(np.abs(thnew@problem.Inc),axis=1)
        ex=dm>=np.pi/2
        if np.any(ex):
            exidx=idx[ex]
            failed[exidx]=True
            active[exidx]=False
    terminal=np.zeros(nq*n_paths)
    terminal[failed]=cost.failure
    surv=~failed
    if np.any(surv):
        terminal[surv]=cost.terminal(problem,th[surv],om[surv])
    S=running+terminal
    logw=(-S/geometry.lam+logLR).reshape(nq,n_paths)
    lm=_logmeanexp(logw,axis=1)
    W=-geometry.lam*lm
    # Stable ESS from normalized weights.
    lw_norm=logw-logsumexp(logw,axis=1)[:,None]
    ess=1.0/np.sum(np.exp(2*lw_norm),axis=1)
    exit_fraction=failed.reshape(nq,n_paths).mean(axis=1)
    out=dict(W=W,psi=np.exp(lm),ess=ess,exit_fraction=exit_fraction,
             lam=geometry.lam,dt=dt,n_paths=n_paths,steps=steps,
             logweight_std=np.std(logw,axis=1),
             logweight_range=np.ptp(logw,axis=1))
    if return_logweights:
        out['logweights']=logw
    return out


def tune_guide(problem, geometry, state, gain, *, alphas=(0,.05,.1,.2,.3,.4),
               t0=0.0,T=3.0,dt=.005,n_paths=800,seed=11,cost=None,clip=8.0):
    """Choose guide strength by effective sample size at a representative state."""
    steps=int(np.ceil((T-t0)/dt)); dt2=(T-t0)/steps
    rng=np.random.default_rng(seed)
    brown=rng.standard_normal((steps,n_paths,problem.ng))
    rows=[]
    for a in alphas:
        z=feynman_kac_importance(problem,geometry,state,t0=t0,T=T,dt=dt2,
            n_paths=n_paths,seed=seed,cost=cost,guide_gain=gain,guide_alpha=float(a),
            guide_clip=clip,brownian=brown)
        rows.append(dict(alpha=float(a),W=float(z['W'][0]),ess=float(z['ess'][0]),
                         ess_fraction=float(z['ess'][0]/n_paths),
                         exit_fraction=float(z['exit_fraction'][0]),
                         logweight_std=float(z['logweight_std'][0])))
    best=max(rows,key=lambda r:r['ess'])
    return rows,best


def deterministic_state_path(problem, policy, times, *, dt=0.001):
    """Generate deterministic states at requested times after the load-loss event."""
    times=np.asarray(times,float)
    order=np.argsort(times); target=times[order]
    th=problem.theta_pre.copy(); om=np.zeros(problem.n)
    out=[None]*len(times); ktarget=0; t=0.0
    maxT=float(target[-1]) if len(target) else 0.0
    steps=int(np.ceil(maxT/dt))
    for k in range(steps+1):
        while ktarget<len(target) and t+1e-12>=target[ktarget]:
            out[order[ktarget]]=np.stack([th.copy(),om.copy()])
            ktarget+=1
        if k==steps: break
        v=policy(t,th[None,:],om[None,:])[0]
        nod=problem.Sg@v
        drift=(problem.p_post-problem.damp*om-problem.network_force(th[None,:])[0]+nod)/problem.M
        om=om+dt*drift; th=th+dt*om; t+=(dt)
    return np.stack(out)


def generator_gradient(problem, geometry, state, *, h=0.02, **fk_kwargs):
    """Central finite-difference PIC feedback in generator frequency directions."""
    base=np.asarray(state,float)
    states=[base]
    for i in problem.gen_idx:
        p=base.copy(); m=base.copy(); p[1,i]+=h; m[1,i]-=h
        states.extend([p,m])
    states=np.stack(states)
    out=feynman_kac_importance(problem,geometry,states,**fk_kwargs)
    W=out['W']; grad=np.array([(W[1+2*j]-W[2+2*j])/(2*h) for j in range(problem.ng)])
    # B^T grad W has 1/M_g multiplying omega-gradient.
    u=-(grad/problem.M[problem.gen_idx])/geometry.Rdiag
    return dict(W0=float(W[0]),omega_gradient=grad,u_pic=u,ess=out['ess'],raw=out)


def curvature_trace_diagnostic(problem, geometry, state, *, h=0.006,
                               n_directions=8, seed_directions=123, **fk_kwargs):
    """Estimate tr[(D-Dtilde) Hess W] at one state.

    For A=diag(deleted_diag) in omega coordinates and Rademacher z,
    q=A^{1/2}z gives E[q^T H q]=tr(AH).  Each quadratic form is estimated with
    a common-random central difference.  Reported uncertainty is across
    Hutchinson directions and is therefore a *diagnostic*, not a global proof.
    """
    A=geometry.deleted_diag
    if np.all(A==0):
        return dict(trace=0.0,se=0.0,directions=np.zeros(n_directions),h=h,
                    min_direction=0.0,positive_fraction=1.0,ess_min=np.nan)
    rng=np.random.default_rng(seed_directions)
    base=np.asarray(state,float)
    states=[base]
    for _ in range(n_directions):
        z=rng.choice(np.array([-1.0,1.0]),size=problem.n)
        q=np.sqrt(A)*z
        p=base.copy(); m=base.copy(); p[1,:]+=h*q; m[1,:]-=h*q
        states.extend([p,m])
    states=np.stack(states)
    out=feynman_kac_importance(problem,geometry,states,**fk_kwargs)
    W=out['W']; W0=W[0]
    vals=np.array([(W[1+2*j]+W[2+2*j]-2*W0)/(h*h) for j in range(n_directions)])
    se=float(vals.std(ddof=1)/np.sqrt(n_directions)) if n_directions>1 else np.nan
    return dict(trace=float(vals.mean()),se=se,directions=vals,h=float(h),
                min_direction=float(vals.min()),positive_fraction=float(np.mean(vals>=0)),
                ess_min=float(np.min(out['ess'])),ess_median=float(np.median(out['ess'])),
                W0=float(W0),exit_fraction=float(out['exit_fraction'][0]),raw=out)


def pic_feedback_first_increment(problem, geometry, state, *, t0=0.0, T=3.0,
                                 dt=0.005, n_paths=1500, seed=1, cost=None,
                                 guide_gain=None, guide_alpha=0.0, guide_clip=8.0):
    """Path-integral feedback from the first surrogate-noise increment.

    Under the importance guide v, passive Brownian increments satisfy
    dW^0 = dW^Q + a dt with a=sqrt(R/lambda)v.  Therefore

        u* = v(x,t) + sqrt(lambda R^{-1}) E_w[dW^Q_0]/dt,

    where ``w`` includes both the Feynman--Kac cost and the stopped Girsanov
    likelihood ratio.  This avoids finite differences of log desirability.
    """
    if cost is None: cost=PICCost()
    th0,om0=_prepare_initial(problem,state)
    if th0.shape[0]!=1:
        raise ValueError('first-increment feedback currently takes one query state')
    horizon=float(T)-float(t0)
    steps=int(np.ceil(horizon/dt)); dt2=horizon/steps; sq=np.sqrt(dt2)
    rng=np.random.default_rng(seed)
    brown=rng.standard_normal((steps,n_paths,problem.ng))
    out=feynman_kac_importance(problem,geometry,(th0,om0),t0=t0,T=T,dt=dt2,
        n_paths=n_paths,seed=seed,cost=cost,guide_gain=guide_gain,
        guide_alpha=guide_alpha,guide_clip=guide_clip,brownian=brown,
        return_logweights=True)
    lw=out['logweights'][0]
    w=np.exp(lw-logsumexp(lw))
    dW0=sq*brown[0]
    v0=_guide_values(problem,th0,om0,guide_gain,float(guide_alpha),guide_clip)[0]
    root=np.sqrt(geometry.lam/geometry.Rdiag)
    u=v0 + root*(w[:,None]*dW0).sum(axis=0)/dt2
    # A rough componentwise weighted standard error of the first-increment term.
    centered=dW0-(w[:,None]*dW0).sum(axis=0)
    neff=max(float(out['ess'][0]),1.0)
    se=root*np.sqrt(np.sum(w[:,None]*centered**2,axis=0)/neff)/dt2
    return dict(W0=float(out['W'][0]),u_pic=u,se=se,guide=v0,
                ess=float(out['ess'][0]),ess_fraction=float(out['ess'][0]/n_paths),
                exit_fraction=float(out['exit_fraction'][0]),dt=dt2)


def pic_feedback_first_increment_batched(problem, geometry, state, *, t0=0.0,T=3.0,
                                         dt=0.005,n_paths=10000,batch=1000,seed=1,
                                         cost=None,guide_gain=None,guide_alpha=0.0,
                                         guide_clip=8.0):
    """Memory-bounded first-increment feedback estimator for one query state."""
    if cost is None: cost=PICCost()
    th0,om0=_prepare_initial(problem,state)
    if th0.shape[0]!=1: raise ValueError('one state only')
    horizon=float(T)-float(t0); steps=int(np.ceil(horizon/dt)); dt2=horizon/steps; sq=np.sqrt(dt2)
    rng=np.random.default_rng(seed); n=problem.n; ng=problem.ng
    logs=[]; firsts=[]; fails=0
    v0=_guide_values(problem,th0,om0,guide_gain,float(guide_alpha),guide_clip)[0]
    Rdiag=geometry.Rdiag; sur=geometry.surrogate_injection_noise
    done=0
    while done<n_paths:
        b=min(batch,n_paths-done)
        th=np.repeat(th0,b,axis=0); om=np.repeat(om0,b,axis=0)
        active=np.ones(b,bool); failed=np.zeros(b,bool); running=np.zeros(b); logLR=np.zeros(b)
        dW_first=None
        for k in range(steps):
            dW=sq*rng.standard_normal((b,ng))
            if k==0: dW_first=dW.copy()
            if not np.any(active): continue
            idx=np.where(active)[0]; tha=th[idx]; oma=om[idx]
            v=_guide_values(problem,tha,oma,guide_gain,float(guide_alpha),guide_clip)
            running[idx]+=cost.running(problem,tha,oma)*dt2
            dWa=dW[idx]
            if guide_gain is not None and guide_alpha!=0.0:
                a=v*np.sqrt(Rdiag[None,:]/geometry.lam)
                logLR[idx]+=-np.sum(a*dWa,axis=1)-.5*np.sum(a*a,axis=1)*dt2
            drift=(problem.p_post[None,:]-problem.damp[None,:]*oma-problem.network_force(tha)+(v@problem.Sg.T))/problem.M[None,:]
            omnew=oma+dt2*drift
            omnew[:,problem.gen_idx]+=(sur[problem.gen_idx]/problem.M[problem.gen_idx])[None,:]*dWa
            thnew=tha+dt2*omnew; th[idx]=thnew; om[idx]=omnew
            ex=np.max(np.abs(thnew@problem.Inc),axis=1)>=np.pi/2
            if np.any(ex):
                exidx=idx[ex]; failed[exidx]=True; active[exidx]=False
        terminal=np.empty(b); terminal[failed]=cost.failure
        surv=~failed
        terminal[surv]=cost.terminal(problem,th[surv],om[surv])
        logs.append(-(running+terminal)/geometry.lam+logLR); firsts.append(dW_first); fails+=int(failed.sum()); done+=b
    lw=np.concatenate(logs); dW0=np.concatenate(firsts,axis=0)
    lse=logsumexp(lw); w=np.exp(lw-lse); W=-geometry.lam*(lse-math.log(n_paths))
    ess=1.0/np.sum(w*w); root=np.sqrt(geometry.lam/geometry.Rdiag)
    mean_dw=(w[:,None]*dW0).sum(axis=0)
    u=v0+root*mean_dw/dt2
    centered=dW0-mean_dw; se=root*np.sqrt(np.sum(w[:,None]*centered**2,axis=0)/max(ess,1.0))/dt2
    return dict(W0=float(W),u_pic=u,se=se,guide=v0,ess=float(ess),ess_fraction=float(ess/n_paths),
                exit_fraction=float(fails/n_paths),dt=dt2,n_paths=n_paths)
