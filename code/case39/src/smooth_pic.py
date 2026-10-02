"""Smooth (unstopped) PIC and true-system simulations for the current paper."""
from __future__ import annotations
import math
import numpy as np
from scipy.special import logsumexp
from smooth_barrier import SmoothPICCost


def _prepare(problem,state):
    arr=np.asarray(state,float)
    if arr.ndim==2 and arr.shape==(2,problem.n): arr=arr[None,:,:]
    if arr.ndim!=3 or arr.shape[1:]!=(2,problem.n): raise ValueError('state shape')
    return arr[:,0,:].copy(),arr[:,1,:].copy()


def _guide(problem,act,theta,omega,gain,alpha,clip):
    if gain is None or alpha==0: return np.zeros((len(theta),act.rank))
    v=-float(alpha)*(problem.reduced_state(theta,omega)@gain.T)
    if clip is not None: v=np.clip(v,-float(clip),float(clip))
    return v


def feynman_kac_modes_smooth(problem,geom,state,*,t0=0.,T=3.,dt=.005,n_paths=500,
                              seed=1,cost=None,guide_gain=None,guide_alpha=0.,guide_clip=8.,
                              brownian=None,return_logweights=False):
    """Unstopped smooth-barrier Feynman--Kac estimator with exact Girsanov guide correction."""
    if cost is None: cost=SmoothPICCost()
    th0,om0=_prepare(problem,state);nq=len(th0);m=geom.actuation.rank;n=problem.n
    horizon=float(T)-float(t0)
    if horizon<=0:
        term=cost.terminal(problem,th0,om0)
        return dict(W=term,ess=np.full(nq,n_paths),cross_fraction=np.zeros(nq),lam=geom.lam)
    steps=int(np.ceil(horizon/dt));dt=horizon/steps;sq=math.sqrt(dt)
    rng=np.random.default_rng(seed)
    if brownian is None:brownian=rng.standard_normal((steps,n_paths,m))
    else:
        brownian=np.asarray(brownian,float)
        if brownian.shape!=(steps,n_paths,m):raise ValueError('brownian shape')
    th=np.repeat(th0[:,None,:],n_paths,axis=1).reshape(nq*n_paths,n)
    om=np.repeat(om0[:,None,:],n_paths,axis=1).reshape(nq*n_paths,n)
    run=np.zeros(nq*n_paths);ll=np.zeros(nq*n_paths);cross=np.zeros(nq*n_paths,bool)
    Anod=geom.actuation.nodal_matrix(problem);root=math.sqrt(geom.lam/geom.control_weight)
    for k in range(steps):
        v=_guide(problem,geom.actuation,th,om,guide_gain,float(guide_alpha),guide_clip)
        run+=cost.running(problem,th,om)*dt
        pid=np.arange(nq*n_paths)%n_paths;dW=sq*brownian[k,pid,:]
        if guide_gain is not None and guide_alpha!=0:
            a=v*math.sqrt(geom.control_weight/geom.lam)
            ll+=-np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*dt
        nod=v@Anod.T
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+nod)/problem.M[None,:]
        om=om+dt*drift+((root*dW)@Anod.T)/problem.M[None,:]
        th=th+dt*om
        cross|=(np.max(np.abs(th@problem.Inc),axis=1)>=np.pi/2)
    S=run+cost.terminal(problem,th,om)
    lw=(-S/geom.lam+ll).reshape(nq,n_paths)
    lse=logsumexp(lw,axis=1);W=-geom.lam*(lse-math.log(n_paths))
    lwn=lw-lse[:,None];ess=1/np.sum(np.exp(2*lwn),axis=1)
    out=dict(W=W,ess=ess,cross_fraction=cross.reshape(nq,n_paths).mean(axis=1),lam=geom.lam,
             logweight_std=np.std(lw,axis=1),dt=dt,steps=steps,n_paths=n_paths)
    if return_logweights:out['logweights']=lw
    return out


def curvature_trace_modes_smooth(problem,geom,state,*,h=.008,n_directions=6,seed_directions=123,**fk):
    A=geom.omega_deleted
    if np.max(np.abs(A))<1e-14:
        return dict(trace=0.,se=0.,min_direction=0.,positive_fraction=1.,directions=np.zeros(n_directions))
    vals,vecs=np.linalg.eigh((A+A.T)/2);vals=np.maximum(vals,0);As=vecs@np.diag(np.sqrt(vals))@vecs.T
    rng=np.random.default_rng(seed_directions);base=np.asarray(state,float);states=[base]
    for _ in range(n_directions):
        z=rng.choice(np.array([-1.,1.]),size=problem.n);q=As@z
        pp=base.copy();mm=base.copy();pp[1]+=h*q;mm[1]-=h*q;states.extend([pp,mm])
    out=feynman_kac_modes_smooth(problem,geom,np.stack(states),**fk)
    W=out['W'];W0=W[0]
    ds=np.array([(W[1+2*j]+W[2+2*j]-2*W0)/(h*h) for j in range(n_directions)])
    se=float(ds.std(ddof=1)/math.sqrt(n_directions)) if n_directions>1 else float('nan')
    npaths=fk.get('n_paths',500)
    return dict(trace=float(ds.mean()),se=se,min_direction=float(ds.min()),
                positive_fraction=float(np.mean(ds>=0)),directions=ds,W0=float(W0),
                ess_fraction_min=float(np.min(out['ess'])/npaths),cross_fraction=float(out['cross_fraction'][0]))


def simulate_soc_cost_modes_smooth(problem,actuation,policy,noise,*,T=3.,dt=.005,n_paths=300,
                                    seed=1,control_weight=.2,cost=None):
    """True physical diffusion with smooth performance cost; crossing is recorded, not absorbing."""
    if cost is None:cost=SmoothPICCost()
    rng=np.random.default_rng(seed);n=problem.n
    th=np.repeat(problem.theta_pre[None,:],n_paths,axis=0).copy();om=np.zeros((n_paths,n))
    noise=np.asarray(noise,float);steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt)
    run=np.zeros(n_paths);ctrl=np.zeros(n_paths);cross=np.zeros(n_paths,bool);dmax=np.max(np.abs(th@problem.Inc),axis=1)
    Anod=actuation.nodal_matrix(problem);r=float(control_weight)
    for k in range(steps):
        v=policy(k*dt,th,om)
        run+=cost.running(problem,th,om)*dt
        cu=.5*r*np.sum(v*v,axis=1);run+=cu*dt;ctrl+=cu*dt
        nod=v@Anod.T
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+nod)/problem.M[None,:]
        om=om+dt*drift+(noise/problem.M)[None,:]*sq*rng.standard_normal((n_paths,n))
        th=th+dt*om
        dm=np.max(np.abs(th@problem.Inc),axis=1);dmax=np.maximum(dmax,dm);cross|=(dm>=np.pi/2)
    J=run+cost.terminal(problem,th,om)
    return dict(J_mean=float(J.mean()),J_se=float(J.std(ddof=1)/math.sqrt(n_paths)),
                cross_probability=float(cross.mean()),control_effort_mean=float(ctrl.mean()),
                dmax_deg_quantiles=np.degrees(np.quantile(dmax,[.5,.9,.99])).tolist(),raw_J=J)
