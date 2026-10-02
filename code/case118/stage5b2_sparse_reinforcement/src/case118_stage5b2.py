from __future__ import annotations
import math
import numpy as np
from scipy.linalg import solve_continuous_are


def lqr_for_nodal_authority(problem, controlled_buses, authority, angle_weight=2.0, freq_weight=0.5):
    """LQR around post-event equilibrium for independent nodal control channels.

    authority_j = 1/R_j is the power-injection control Gramian diagonal.
    """
    buses=np.asarray(controlled_buses,int); g=np.asarray(authority,float)
    if buses.ndim!=1 or g.shape!=(len(buses),) or np.any(g<=0):
        raise ValueError('bad controlled-bus/authority arrays')
    n=problem.n; Minv=np.diag(1.0/problem.M)
    A=np.block([[np.zeros((n-1,n-1)),problem.H],
                [-Minv@problem.Lstar@problem.E,-Minv@np.diag(problem.damp)]])
    S=np.zeros((n,len(buses))); S[buses,np.arange(len(buses))]=1.0
    B=np.vstack([np.zeros((n-1,len(buses))),Minv@S])
    Qq=problem.E.T@problem.Lstar@problem.E
    Qq/=max(np.trace(Qq)/(n-1),1e-12)
    Qw=np.diag(problem.M); Qw/=max(np.trace(Qw)/n,1e-12)
    Q=np.block([[angle_weight*Qq,np.zeros((n-1,n))],
                [np.zeros((n,n-1)),freq_weight*Qw]])
    R=np.diag(1.0/g)
    P=solve_continuous_are(A,B,Q,R)
    K=g[:,None]*(B.T@P)  # R^{-1} B^T P
    return K, dict(A=A,B=B,Q=Q,R=R,P=P,S=S)


def h4_shadow_proxy(problem, Gamma, gen_authority, *, alpha=0.005, T=3.0, dt=.0025,
                    seeds=(52901,52902), paths_per_seed=64):
    """Dynamics-aware proxy for marginal value of load-bus authority.

    Build the least-inflated exact completion at the generator-matched lambda,
    simulate its scaled full-nodal LQR, and estimate
        s_i = 0.5 E int (R_i u_i)^2 dt / T.
    Since u_i = -g_i * d_i V in an exactly matched quadratic geometry, this is
    an LQR-guide approximation to 0.5 E[(d_i V)^2], the first-order shadow value
    per unit added authority.  It is a placement heuristic, not an exact theorem
    for the nonlinear PIC value.
    """
    Gamma=np.asarray(Gamma,float); gg=np.asarray(gen_authority,float)
    q=Gamma**2
    lam=float(np.min(q[problem.gen_idx]/gg))
    gplus=q/lam
    buses=np.arange(problem.n,dtype=int)
    K,_=lqr_for_nodal_authority(problem,buses,gplus)
    N=paths_per_seed*len(seeds); n=problem.n
    steps=int(np.ceil(T/dt)); dt=T/steps; sq=math.sqrt(dt)
    rngs=[np.random.default_rng(int(s)) for s in seeds]
    th=np.repeat(problem.theta_pre[None,:],N,axis=0).copy(); om=np.zeros((N,n))
    grad2_int=np.zeros(n); u2_int=np.zeros(n)
    Rdiag=1.0/gplus
    for _ in range(steps):
        x=problem.reduced_state(th,om)
        u=-float(alpha)*(x@K.T)
        grad=Rdiag[None,:]*u
        grad2_int+=np.mean(grad*grad,axis=0)*dt
        u2_int+=np.mean(u*u,axis=0)*dt
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+u)/problem.M[None,:]
        z=np.concatenate([rg.standard_normal((paths_per_seed,n)) for rg in rngs],axis=0)
        om=om+dt*drift+(Gamma/problem.M)[None,:]*sq*z
        th=th+dt*om
    shadow=.5*grad2_int/max(T,1e-12)
    rms=np.sqrt(u2_int/max(T,1e-12))
    return dict(lambda_max=lam,gplus=gplus,K=K,shadow=shadow,command_rms=rms,
                alpha=float(alpha),seeds=list(map(int,seeds)),paths=N,dt=dt,T=T)


def matched_sparse_authority(problem, gplus, selected_a, selected_b, k):
    """Construct equal total authority budgets for two placement sets.

    The full exact completion has target load authority gplus_i. For each k we
    choose the largest common budget that can be placed on either selected set
    without exceeding any exact-completion target: A_k=min(sum targets on A,
    sum targets on B). Each set receives A_k, distributed in proportion to its
    own exact-completion target. Therefore 0<=g_add<=gplus componentwise and
    k=all-loads recovers exact completion.
    """
    A=np.asarray(selected_a[:k],int); B=np.asarray(selected_b[:k],int)
    sa=float(np.sum(gplus[A])); sb=float(np.sum(gplus[B])); budget=min(sa,sb)
    def alloc(S,s):
        if s<=0: return np.zeros(len(S))
        return gplus[S]*(budget/s)
    return budget,A,alloc(A,sa),B,alloc(B,sb),sa,sb


def simulate_sparse_policy(problem, Gamma, gen_authority, load_buses, load_authority, K, *,
                           alpha=.005,T=3.,dt=.0025,seeds=(53201,53202),paths_per_seed=96,cost=None):
    from case118_stage3a import SmoothCost
    if cost is None: cost=SmoothCost()
    load_buses=np.asarray(load_buses,int); load_authority=np.asarray(load_authority,float)
    buses=np.r_[problem.gen_idx,load_buses]
    authority=np.r_[np.asarray(gen_authority,float),load_authority]
    Rdiag=1.0/authority
    N=paths_per_seed*len(seeds); n=problem.n; steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt)
    rngs=[np.random.default_rng(int(s)) for s in seeds]
    th=np.repeat(problem.theta_pre[None,:],N,axis=0).copy();om=np.zeros((N,n))
    run=np.zeros(N);ctrl=np.zeros(N);cross=np.zeros(N,bool)
    dmax=np.max(np.abs(problem.edge_delta(th)),axis=1); maxgen=np.zeros(N)
    load_u2=np.zeros(N); gen_u2=np.zeros(N)
    for _ in range(steps):
        x=problem.reduced_state(th,om); u=-float(alpha)*(x@K.T)
        run+=cost.running(problem,th,om)*dt
        cu=.5*np.sum(u*u*Rdiag[None,:],axis=1); run+=cu*dt; ctrl+=cu*dt
        gen_u2+=np.sum(u[:,:problem.ng]**2,axis=1)*dt
        if len(load_buses): load_u2+=np.sum(u[:,problem.ng:]**2,axis=1)*dt
        nod=np.zeros((N,n)); nod[:,buses]=u
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+nod)/problem.M[None,:]
        z=np.concatenate([rg.standard_normal((paths_per_seed,n)) for rg in rngs],axis=0)
        om=om+dt*drift+(np.asarray(Gamma)/problem.M)[None,:]*sq*z
        th=th+dt*om
        dm=np.max(np.abs(problem.edge_delta(th)),axis=1);dmax=np.maximum(dmax,dm);cross|=dm>=np.pi/2
        maxgen=np.maximum(maxgen,np.max(np.abs(om[:,problem.gen_idx]),axis=1))
    J=run+cost.terminal(problem,th,om)
    return dict(J_mean=float(J.mean()),J_se=float(J.std(ddof=1)/math.sqrt(N)),
                cross_probability=float(cross.mean()),control_effort_mean=float(ctrl.mean()),
                dmax_q50_deg=float(np.degrees(np.quantile(dmax,.5))),dmax_q90_deg=float(np.degrees(np.quantile(dmax,.9))),
                dmax_q99_deg=float(np.degrees(np.quantile(dmax,.99))),genomega_q99=float(np.quantile(maxgen,.99)),
                gen_command_l2sq_mean=float(gen_u2.mean()),load_command_l2sq_mean=float(load_u2.mean()),
                raw_J=J,raw_cross=cross,raw_dmax_deg=np.degrees(dmax),raw_genomega=maxgen,
                raw_ctrl=ctrl,n_paths=N,dt=dt,T=T,buses=buses,authority=authority)


def simulate_generator_baseline(problem,Gamma,gen_authority,K,*,alpha=.005,T=3.,dt=.0025,seeds=(53201,53202),paths_per_seed=96,cost=None):
    return simulate_sparse_policy(problem,Gamma,gen_authority,np.array([],int),np.array([],float),K,
                                  alpha=alpha,T=T,dt=dt,seeds=seeds,paths_per_seed=paths_per_seed,cost=cost)


def scan_sparse_scales(problem,Gamma,gen_authority,load_buses,load_authority,K,scales,*,T=3.,dt=.0025,seeds=(53101,53102),paths_per_seed=16,cost=None):
    from case118_stage3a import SmoothCost
    if cost is None: cost=SmoothCost()
    scales=np.asarray(scales,float);ns=len(scales);n=problem.n;N=paths_per_seed*len(seeds);steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt)
    lb=np.asarray(load_buses,int);la=np.asarray(load_authority,float);buses=np.r_[problem.gen_idx,lb];g=np.r_[np.asarray(gen_authority,float),la];R=1.0/g
    rngs=[np.random.default_rng(int(s)) for s in seeds]
    th=np.repeat(problem.theta_pre[None,None,:],ns*N,axis=0).reshape(ns,N,n).copy();om=np.zeros((ns,N,n));run=np.zeros((ns,N));ctrl=np.zeros((ns,N));cross=np.zeros((ns,N),bool)
    for _ in range(steps):
        flatth=th.reshape(ns*N,n);flatom=om.reshape(ns*N,n);x=problem.reduced_state(flatth,flatom).reshape(ns,N,-1)
        u=-scales[:,None,None]*(x@K.T)
        run+=cost.running(problem,flatth,flatom).reshape(ns,N)*dt
        cu=.5*np.sum(u*u*R[None,None,:],axis=2);run+=cu*dt;ctrl+=cu*dt
        nod=np.zeros((ns,N,n));
        for j,b in enumerate(buses): nod[:,:,b]=u[:,:,j]
        force=problem.network_force(flatth).reshape(ns,N,n);drift=(problem.p_post[None,None,:]-problem.damp[None,None,:]*om-force+nod)/problem.M[None,None,:]
        z=np.concatenate([rg.standard_normal((paths_per_seed,n)) for rg in rngs],axis=0)
        om=om+dt*drift+(np.asarray(Gamma)/problem.M)[None,None,:]*sq*z[None,:,:];th=th+dt*om
        dm=np.max(np.abs(problem.edge_delta(th.reshape(ns*N,n)).reshape(ns,N,-1)),axis=2);cross|=dm>=np.pi/2
    J=run+cost.terminal(problem,th.reshape(ns*N,n),om.reshape(ns*N,n)).reshape(ns,N)
    out=[]
    for i,a in enumerate(scales): out.append(dict(alpha=float(a),J_mean=float(J[i].mean()),J_se=float(J[i].std(ddof=1)/math.sqrt(N)),cross_probability=float(cross[i].mean()),control_effort_mean=float(ctrl[i].mean()),n_paths=N,seeds=';'.join(map(str,seeds))))
    return out
