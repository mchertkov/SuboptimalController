from __future__ import annotations
import math
import numpy as np
from scipy.linalg import solve_continuous_are


def metric_lqr_gain(problem, Rdiag, angle_weight=2.0, freq_weight=0.5):
    """H3 generator-nodal LQR for diagonal physical control price Rdiag."""
    Rdiag=np.asarray(Rdiag,float)
    if Rdiag.shape!=(problem.ng,) or np.any(Rdiag<=0):
        raise ValueError('Rdiag must be positive length-ng vector')
    Minv=np.diag(1.0/problem.M); n=problem.n
    A=np.block([[np.zeros((n-1,n-1)),problem.H],
                [-Minv@problem.Lstar@problem.E,-Minv@np.diag(problem.damp)]])
    B=np.vstack([np.zeros((n-1,problem.ng)),Minv@problem.Sg])
    Qq=problem.E.T@problem.Lstar@problem.E
    Qq/=max(np.trace(Qq)/(n-1),1e-12)
    Qw=np.diag(problem.M);Qw/=max(np.trace(Qw)/n,1e-12)
    Q=np.block([[angle_weight*Qq,np.zeros((n-1,n))],
                [np.zeros((n,n-1)),freq_weight*Qw]])
    R=np.diag(Rdiag)
    P=solve_continuous_are(A,B,Q,R)
    K=np.linalg.solve(R,B.T@P)
    return K,dict(A=A,B=B,Q=Q,R=R,P=P)


def scan_metric_lqr_scales(problem,K,Rdiag,scales,noise,*,T=3.,dt=.0025,n_per_seed=16,seeds=(51501,51502),cost=None):
    """CRN nonlinear scalar-scale scan for H3 with diagonal R."""
    from case118_stage3a import SmoothCost
    if cost is None: cost=SmoothCost()
    Rdiag=np.asarray(Rdiag,float); scales=np.asarray(scales,float)
    ns=len(scales);n=problem.n;N=int(n_per_seed)*len(seeds)
    steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt)
    rngs=[np.random.default_rng(int(s)) for s in seeds]
    th=np.repeat(problem.theta_pre[None,None,:],ns*N,axis=0).reshape(ns,N,n).copy()
    om=np.zeros((ns,N,n));run=np.zeros((ns,N));ctrl=np.zeros((ns,N));cross=np.zeros((ns,N),bool)
    dmax=np.max(np.abs(problem.edge_delta(th.reshape(ns*N,n)).reshape(ns,N,-1)),axis=2)
    maxgen=np.zeros((ns,N))
    for kk in range(steps):
        flatth=th.reshape(ns*N,n);flatom=om.reshape(ns*N,n)
        x=problem.reduced_state(flatth,flatom).reshape(ns,N,-1)
        u=-scales[:,None,None]*(x@K.T)
        run+=cost.running(problem,flatth,flatom).reshape(ns,N)*dt
        cu=.5*np.sum(u*u*Rdiag[None,None,:],axis=2);run+=cu*dt;ctrl+=cu*dt
        nod=np.zeros((ns,N,n));nod[:,:,problem.gen_idx]=u
        force=problem.network_force(flatth).reshape(ns,N,n)
        drift=(problem.p_post[None,None,:]-problem.damp[None,None,:]*om-force+nod)/problem.M[None,None,:]
        basez=np.concatenate([rg.standard_normal((n_per_seed,n)) for rg in rngs],axis=0)
        om=om+dt*drift+(np.asarray(noise)/problem.M)[None,None,:]*sq*basez[None,:,:]
        th=th+dt*om
        dm=np.max(np.abs(problem.edge_delta(th.reshape(ns*N,n)).reshape(ns,N,-1)),axis=2)
        dmax=np.maximum(dmax,dm);cross|=dm>=np.pi/2
        maxgen=np.maximum(maxgen,np.max(np.abs(om[:,:,problem.gen_idx]),axis=2))
    J=run+cost.terminal(problem,th.reshape(ns*N,n),om.reshape(ns*N,n)).reshape(ns,N)
    rows=[]
    for i,s in enumerate(scales):
        rows.append(dict(scale=float(s),J_mean=float(J[i].mean()),J_se=float(J[i].std(ddof=1)/math.sqrt(N)),
            cross_probability=float(cross[i].mean()),control_effort_mean=float(ctrl[i].mean()),
            dmax_q50_deg=float(np.degrees(np.quantile(dmax[i],.5))),dmax_q90_deg=float(np.degrees(np.quantile(dmax[i],.9))),
            dmax_q99_deg=float(np.degrees(np.quantile(dmax[i],.99))),genomega_q99=float(np.quantile(maxgen[i],.99)),
            n_paths=N,seeds=';'.join(str(int(x)) for x in seeds)))
    return rows


def simulate_metric_policy(problem,K,Rdiag,alpha,noise,*,T=3.,dt=.0025,n_paths=192,seeds=(51601,51602),paths_per_seed=96,cost=None):
    """Metric-aware H3 nonlinear simulation using reproducible concatenated RNG streams."""
    from case118_stage3a import SmoothCost
    if cost is None: cost=SmoothCost()
    if n_paths != paths_per_seed*len(seeds): raise ValueError('n_paths mismatch')
    n=problem.n;steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt)
    rngs=[np.random.default_rng(int(s)) for s in seeds]
    th=np.repeat(problem.theta_pre[None,:],n_paths,axis=0).copy();om=np.zeros((n_paths,n))
    run=np.zeros(n_paths);ctrl=np.zeros(n_paths);cross=np.zeros(n_paths,bool)
    dmax=np.max(np.abs(problem.edge_delta(th)),axis=1);maxgen=np.zeros(n_paths)
    Rdiag=np.asarray(Rdiag,float)
    for kk in range(steps):
        x=problem.reduced_state(th,om);u=-float(alpha)*(x@K.T)
        run+=cost.running(problem,th,om)*dt
        cu=.5*np.sum(u*u*Rdiag[None,:],axis=1);run+=cu*dt;ctrl+=cu*dt
        nod=np.zeros((n_paths,n));nod[:,problem.gen_idx]=u
        force=problem.network_force(th)
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-force+nod)/problem.M[None,:]
        z=np.concatenate([rg.standard_normal((paths_per_seed,n)) for rg in rngs],axis=0)
        om=om+dt*drift+(np.asarray(noise)/problem.M)[None,:]*sq*z
        th=th+dt*om
        dm=np.max(np.abs(problem.edge_delta(th)),axis=1);dmax=np.maximum(dmax,dm);cross|=dm>=np.pi/2
        maxgen=np.maximum(maxgen,np.max(np.abs(om[:,problem.gen_idx]),axis=1))
    J=run+cost.terminal(problem,th,om)
    return dict(J_mean=float(J.mean()),J_se=float(J.std(ddof=1)/math.sqrt(n_paths)),
                cross_probability=float(cross.mean()),control_effort_mean=float(ctrl.mean()),
                dmax_q50_deg=float(np.degrees(np.quantile(dmax,.5))),dmax_q90_deg=float(np.degrees(np.quantile(dmax,.9))),
                dmax_q99_deg=float(np.degrees(np.quantile(dmax,.99))),genomega_q99=float(np.quantile(maxgen,.99)),
                raw_J=J,raw_cross=cross,raw_dmax_deg=np.degrees(dmax),raw_genomega=maxgen,
                n_paths=n_paths,dt=dt,T=T)


def simulate_static_common(problem,noise,*,T=3.,dt=.0025,n_paths=192,seeds=(51601,51602),paths_per_seed=96,cost=None):
    """Static balancing baseline under exactly the same concatenated Brownian streams."""
    from case118_stage3a import SmoothCost
    if cost is None: cost=SmoothCost()
    if n_paths != paths_per_seed*len(seeds): raise ValueError('n_paths mismatch')
    n=problem.n;steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt)
    rngs=[np.random.default_rng(int(s)) for s in seeds]
    th=np.repeat(problem.theta_pre[None,:],n_paths,axis=0).copy();om=np.zeros((n_paths,n))
    run=np.zeros(n_paths);cross=np.zeros(n_paths,bool)
    dmax=np.max(np.abs(problem.edge_delta(th)),axis=1);maxgen=np.zeros(n_paths)
    for kk in range(steps):
        run+=cost.running(problem,th,om)*dt
        force=problem.network_force(th)
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-force)/problem.M[None,:]
        z=np.concatenate([rg.standard_normal((paths_per_seed,n)) for rg in rngs],axis=0)
        om=om+dt*drift+(np.asarray(noise)/problem.M)[None,:]*sq*z
        th=th+dt*om
        dm=np.max(np.abs(problem.edge_delta(th)),axis=1);dmax=np.maximum(dmax,dm);cross|=dm>=np.pi/2
        maxgen=np.maximum(maxgen,np.max(np.abs(om[:,problem.gen_idx]),axis=1))
    J=run+cost.terminal(problem,th,om)
    return dict(J_mean=float(J.mean()),J_se=float(J.std(ddof=1)/math.sqrt(n_paths)),cross_probability=float(cross.mean()),
                control_effort_mean=0.0,dmax_q50_deg=float(np.degrees(np.quantile(dmax,.5))),
                dmax_q90_deg=float(np.degrees(np.quantile(dmax,.9))),dmax_q99_deg=float(np.degrees(np.quantile(dmax,.99))),
                genomega_q99=float(np.quantile(maxgen,.99)),raw_J=J,raw_cross=cross,raw_dmax_deg=np.degrees(dmax),raw_genomega=maxgen,
                n_paths=n_paths,dt=dt,T=T)
