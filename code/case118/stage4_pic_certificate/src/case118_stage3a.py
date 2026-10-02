from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
from scipy.linalg import solve_continuous_are, eigh

from case118_data import BASE_MVA, BUS, GEN, BRANCH


def _build_network():
    bus_ids=BUS[:,0].astype(int); n=len(bus_ids); idx={b:i for i,b in enumerate(bus_ids)}
    Vm=BUS[:,2].astype(float); Pd=BUS[:,1].astype(float)/BASE_MVA
    Pg=np.zeros(n); is_gen=np.zeros(n,bool)
    for b,p in GEN:
        i=idx[int(b)]; Pg[i]+=p/BASE_MVA; is_gen[i]=True
    # Preserve every energized MATPOWER branch, including parallel lines.
    # This keeps L=186 and raw cycle rank L-n+1=69, exactly as frozen in Stage 0.
    edges=[]; kvals=[]
    for fb,tb,x,tap in BRANCH:
        i,j=idx[int(fb)],idx[int(tb)]
        tau=abs(float(tap)) if abs(float(tap))>1e-14 else 1.0
        kij=Vm[i]*Vm[j]/(abs(float(x))*tau)
        # canonical orientation only changes the sign of the corresponding angle coordinate.
        edges.append((min(i,j),max(i,j))); kvals.append(kij)
    Inc=np.zeros((n,len(edges))); K=np.asarray(kvals,float)
    for e,(i,j) in enumerate(edges):
        Inc[i,e]=1.; Inc[j,e]=-1.
    p0=Pg-Pd; p0-=p0.mean()
    return dict(bus_ids=bus_ids,Vm=Vm,Pd=Pd,Pg=Pg,is_gen=is_gen,Inc=Inc,Kedge=K,edges=edges,p0=p0)


def newton_equilibrium(Inc,K,p,th0=None,ref=0,tol=1e-12,maxit=300):
    n=Inc.shape[0]; th=np.zeros(n) if th0 is None else np.asarray(th0,float).copy(); th-=th[ref]
    keep=np.array([i for i in range(n) if i!=ref])
    for it in range(maxit):
        d=Inc.T@th; r=Inc@(K*np.sin(d))-p; rn=float(np.max(np.abs(r[keep])))
        H=Inc@np.diag(K*np.cos(d))@Inc.T
        if rn<tol:
            ev=np.linalg.eigvalsh((H+H.T)/2)
            return dict(ok=True,theta=th,Delta=d,residual=rn,lam2=float(np.sort(ev)[1]),dmax=float(np.max(np.abs(d))),iters=it)
        try: stepk=np.linalg.solve(H[np.ix_(keep,keep)],-r[keep])
        except np.linalg.LinAlgError: return dict(ok=False,reason='singular',theta=th,residual=rn)
        step=np.zeros(n);step[keep]=stepk;t=1.;accepted=False
        for _ in range(80):
            cand=th+t*step; rc=Inc@(K*np.sin(Inc.T@cand))-p
            if np.max(np.abs(rc[keep]))<rn: accepted=True;break
            t*=.5
        if not accepted: return dict(ok=False,reason='line search',theta=th,residual=rn)
        th=cand
    return dict(ok=False,reason='maxit',theta=th)

@dataclass
class SwingProblem:
    Inc: np.ndarray; Kedge: np.ndarray; edges: list; bus_ids: np.ndarray
    p_pre: np.ndarray; p_post: np.ndarray; M: np.ndarray; damp: np.ndarray; is_gen: np.ndarray
    theta_pre: np.ndarray; theta_star: np.ndarray; disturbance: int; dP: float; gamma: np.ndarray
    ref: int=0
    def __post_init__(self):
        self.n=len(self.bus_ids);self.gen_idx=np.where(self.is_gen)[0];self.ng=len(self.gen_idx)
        self.keep=np.array([i for i in range(self.n) if i!=self.ref])
        self.E=np.zeros((self.n,self.n-1)); self.E[self.keep,np.arange(self.n-1)]=1.
        self.H=np.zeros((self.n-1,self.n));self.H[:,self.keep]=np.eye(self.n-1);self.H[:,self.ref]=-1.
        self.Sg=np.zeros((self.n,self.ng));self.Sg[self.gen_idx,np.arange(self.ng)]=1.
        self.edge_i=np.array([e[0] for e in self.edges],dtype=int); self.edge_j=np.array([e[1] for e in self.edges],dtype=int)
        self.delta_star=self.theta_star[self.edge_i]-self.theta_star[self.edge_j]
        self.Lstar=self.Inc@np.diag(self.Kedge*np.cos(self.delta_star))@self.Inc.T
        self.Ustar=self.potential(self.theta_star)
    def edge_delta(self,theta):
        theta=np.asarray(theta); return theta[...,self.edge_i]-theta[...,self.edge_j]
    def network_force(self,theta):
        theta=np.asarray(theta); d=self.edge_delta(theta); f=np.sin(d)*self.Kedge
        out=np.zeros_like(theta,dtype=float)
        if theta.ndim==1:
            np.add.at(out,self.edge_i,f); np.add.at(out,self.edge_j,-f)
        else:
            rows=np.arange(theta.shape[0])[:,None]
            np.add.at(out,(rows,self.edge_i[None,:]),f); np.add.at(out,(rows,self.edge_j[None,:]),-f)
        return out
    def potential(self,theta):
        theta=np.asarray(theta);return -theta@self.p_post-np.cos(self.edge_delta(theta))@self.Kedge
    def energy(self,theta,omega): return self.potential(theta)-self.Ustar+.5*np.sum(np.asarray(omega)**2*self.M,axis=-1)
    def reduced_state(self,theta,omega):
        theta=np.asarray(theta);omega=np.asarray(omega)
        q=(theta[:,self.keep]-theta[:,[self.ref]])-(self.theta_star[self.keep]-self.theta_star[self.ref])
        return np.concatenate([q,omega],axis=1)


def build_problem(alpha=4.0,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05):
    d=_build_network(); p=float(alpha)*d['p0'];
    eq_pre=newton_equilibrium(d['Inc'],d['Kedge'],p)
    if not eq_pre['ok']: raise RuntimeError(f"pre equilibrium failed {eq_pre}")
    disturbance=int(np.where(d['bus_ids']==int(event_bus))[0][0]);dP=float(-p[disturbance])
    gamma=np.zeros(len(p));gamma[d['is_gen']]=1./d['is_gen'].sum()
    p_post=p.copy();p_post[disturbance]+=dP;p_post-=dP*gamma;p_post-=p_post.mean()
    eq_post=newton_equilibrium(d['Inc'],d['Kedge'],p_post,th0=eq_pre['theta'])
    if not eq_post['ok']: raise RuntimeError(f"post equilibrium failed {eq_post}")
    M=np.where(d['is_gen'],float(inertia_gen),float(inertia_load))
    problem=SwingProblem(Inc=d['Inc'],Kedge=d['Kedge'],edges=d['edges'],bus_ids=d['bus_ids'],p_pre=p,p_post=p_post,
        M=M,damp=np.full(len(p),float(damping)),is_gen=d['is_gen'],theta_pre=eq_pre['theta'],theta_star=eq_post['theta'],
        disturbance=disturbance,dP=dP,gamma=gamma)
    return problem,dict(eq_pre=eq_pre,eq_post=eq_post,data=d)


def noise_profile(problem,rms,kind):
    if kind=='homogeneous': prof=np.ones(problem.n)
    elif kind=='activity_scaled':
        a=np.abs(problem.p_pre);q75=np.quantile(a,.75);prof=.25+np.minimum(a/max(q75,1e-12),4.);prof/=np.sqrt(np.mean(prof**2))
    else: raise ValueError(kind)
    return float(rms)*prof

# Smooth C4 cohesion barrier

def _h_derivs(g):
    c=math.cos(g);t=math.tan(g);sec2=1/(c*c)
    return -math.log(c),t,sec2,2*sec2*t,4*sec2*t*t+2*sec2*sec2

def h_ext(x,guard=math.radians(82.)):
    a=np.abs(np.asarray(x,float));h0,h1,h2,h3,h4=_h_derivs(guard);inside=a<=guard;out=np.empty_like(a)
    out[inside]=-np.log(np.cos(a[inside]));z=a[~inside]-guard
    out[~inside]=h0+h1*z+.5*h2*z*z+h3*z**3/6+h4*z**4/24
    return out

def h_ext_prime(x,guard=math.radians(82.)):
    a=np.abs(np.asarray(x,float));sg=np.sign(np.asarray(x,float));_,h1,h2,h3,h4=_h_derivs(guard);inside=a<=guard;out=np.empty_like(a)
    out[inside]=np.tan(a[inside]);z=a[~inside]-guard;out[~inside]=h1+h2*z+.5*h3*z*z+h4*z**3/6
    return sg*out

def barrier_bregman(problem,theta,guard_deg=82.):
    d=problem.edge_delta(theta);ds=problem.delta_star;g=math.radians(guard_deg)
    return np.mean(h_ext(d,g)-h_ext(ds,g)-h_ext_prime(ds,g)*(d-ds),axis=-1)

@dataclass
class SmoothCost:
    c_E:float=.08;c_B:float=.02;c_T:float=.20;guard_deg:float=82.
    def running(self,p,th,om):return self.c_E*np.maximum(p.energy(th,om),0)/p.n+self.c_B*barrier_bregman(p,th,self.guard_deg)
    def terminal(self,p,th,om):return self.c_T*np.maximum(p.energy(th,om),0)/p.n

@dataclass
class Actuation:
    name:str;label:str;Qgen:np.ndarray;gen_bus_ids:np.ndarray
    @property
    def rank(self):return int(self.Qgen.shape[1])
    def nodal(self,p):return p.Sg@self.Qgen


def _orth(cols,tol=1e-10):
    Q=[]
    for c in cols:
        v=np.asarray(c,float).copy()
        for q in Q:v-=q*np.dot(q,v)
        nv=np.linalg.norm(v)
        if nv>tol:Q.append(v/nv)
    return np.column_stack(Q)

def generator_kron_laplacian(problem):
    G=problem.gen_idx; mask=np.ones(problem.n,bool);mask[G]=False;Lidx=np.where(mask)[0];L=problem.Lstar
    Lgg=L[np.ix_(G,G)]
    if len(Lidx)==0:return Lgg
    return Lgg-L[np.ix_(G,Lidx)]@np.linalg.solve(L[np.ix_(Lidx,Lidx)],L[np.ix_(Lidx,G)])

def build_architectures(problem,h2_ranks=(3,5,10),slack_bus=69):
    genbus=problem.bus_ids[problem.gen_idx];ng=problem.ng
    sloc=int(np.where(genbus==slack_bus)[0][0]);e=np.zeros(ng);e[sloc]=1.;q=np.ones(ng)/np.sqrt(ng)
    Lg=generator_kron_laplacian(problem);vals,vecs=eigh((Lg+Lg.T)/2)
    # remove the constant null direction; order remaining eigenmodes from soft to stiff
    non=[]
    for j in np.argsort(vals):
        v=vecs[:,j]-q*np.dot(q,vecs[:,j]);nv=np.linalg.norm(v)
        if nv>1e-7:non.append(v/nv)
    out={'H0':Actuation('H0',f'single MATPOWER slack generator bus {slack_bus}',e[:,None],genbus),
         'H1':Actuation('H1','equal distributed generator rank-one mode',q[:,None],genbus),
         'H3':Actuation('H3',f'independent control at all {ng} generators',np.eye(ng),genbus)}
    for r in h2_ranks:
        Q=_orth([q]+non[:int(r)-1])
        if Q.shape[1]!=r:raise RuntimeError(f'H2 rank construction failed {r} -> {Q.shape}')
        out[f'H2_r{r}']=Actuation(f'H2_r{r}',f'rank-{r} uniform + {r-1} soft Kron generator modes',Q,genbus)
    return out,dict(kron_eigenvalues=vals,constant_mode=q)


def lqr_gain(problem,act,angle_weight=2.,freq_weight=.5,control_weight=.2):
    Minv=np.diag(1/problem.M);A=np.block([[np.zeros((problem.n-1,problem.n-1)),problem.H],[-Minv@problem.Lstar@problem.E,-Minv@np.diag(problem.damp)]])
    B=np.vstack([np.zeros((problem.n-1,act.rank)),Minv@act.nodal(problem)])
    Qq=problem.E.T@problem.Lstar@problem.E;Qq/=max(np.trace(Qq)/(problem.n-1),1e-12)
    Qw=np.diag(problem.M);Qw/=max(np.trace(Qw)/problem.n,1e-12)
    Q=np.block([[angle_weight*Qq,np.zeros((problem.n-1,problem.n))],[np.zeros((problem.n,problem.n-1)),freq_weight*Qw]])
    R=control_weight*np.eye(act.rank);P=solve_continuous_are(A,B,Q,R);K=np.linalg.solve(R,B.T@P)
    return K

def policy_static(act): return lambda t,th,om: np.zeros((len(th),act.rank))
def policy_droop(problem,act,k):
    # Project generator-frequency droop into the architecture's orthonormal mode subspace.
    return lambda t,th,om: -float(k)*(om[:,problem.gen_idx]@act.Qgen)
def policy_lqr(problem,act,K,alpha): return lambda t,th,om: -float(alpha)*(problem.reduced_state(th,om)@K.T)


def simulate_smooth(problem,act,policy,noise,*,T=3.,dt=.0025,n_paths=64,seed=1,control_weight=.2,cost=None,brownian=None):
    if cost is None:cost=SmoothCost()
    n=problem.n;steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt)
    if brownian is None:brownian=np.random.default_rng(seed).standard_normal((steps,n_paths,n))
    else:
        brownian=np.asarray(brownian,float)
        if brownian.shape!=(steps,n_paths,n):raise ValueError((brownian.shape,(steps,n_paths,n)))
    th=np.repeat(problem.theta_pre[None,:],n_paths,axis=0).copy();om=np.zeros((n_paths,n));run=np.zeros(n_paths);ctrl=np.zeros(n_paths)
    cross=np.zeros(n_paths,bool);dmax=np.max(np.abs(problem.edge_delta(th)),axis=1);maxgen=np.zeros(n_paths);An=act.nodal(problem);r=float(control_weight)
    for kk in range(steps):
        v=policy(kk*dt,th,om);run+=cost.running(problem,th,om)*dt;cu=.5*r*np.sum(v*v,axis=1);run+=cu*dt;ctrl+=cu*dt
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+(v@An.T))/problem.M[None,:]
        om=om+dt*drift+(np.asarray(noise)/problem.M)[None,:]*sq*brownian[kk]
        th=th+dt*om;dm=np.max(np.abs(problem.edge_delta(th)),axis=1);dmax=np.maximum(dmax,dm);cross|=dm>=np.pi/2
        maxgen=np.maximum(maxgen,np.max(np.abs(om[:,problem.gen_idx]),axis=1))
    J=run+cost.terminal(problem,th,om)
    return dict(J_mean=float(J.mean()),J_se=float(J.std(ddof=1)/math.sqrt(n_paths)),cross_probability=float(cross.mean()),
                control_effort_mean=float(ctrl.mean()),dmax_deg_quantiles=np.degrees(np.quantile(dmax,[.5,.9,.99])).tolist(),
                genomega_quantiles=np.quantile(maxgen,[.5,.9,.99]).tolist(),raw_J=J,raw_cross=cross,n_paths=n_paths,dt=dt,T=T)

def scan_scales(problem,act,controller_kind,scales,noise,*,gain=None,T=3.,dt=.0025,n_per_seed=12,seeds=(31001,31002),control_weight=.2,cost=None):
    """CRN scalar scan for droop or scaled-LQR, vectorized over candidate scales.

    Independent RNG streams are concatenated into one small training ensemble. The same
    standard-normal draws are replayed for every candidate scale, so pairwise comparisons
    have low Monte-Carlo noise without contaminating future Stage-3B seeds.
    """
    if cost is None:cost=SmoothCost()
    scales=np.asarray(scales,float);ns=len(scales);n=problem.n;N=int(n_per_seed)*len(seeds)
    steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt);rngs=[np.random.default_rng(int(s)) for s in seeds]
    th=np.repeat(problem.theta_pre[None,None,:],ns*N,axis=0).reshape(ns,N,n).copy();om=np.zeros((ns,N,n))
    run=np.zeros((ns,N));ctrl=np.zeros((ns,N));cross=np.zeros((ns,N),bool);dmax=np.max(np.abs(problem.edge_delta(th.reshape(ns*N,n)).reshape(ns,N,-1)),axis=2)
    maxgen=np.zeros((ns,N));An=act.nodal(problem);r=float(control_weight)
    for kk in range(steps):
        flatth=th.reshape(ns*N,n);flatom=om.reshape(ns*N,n)
        if controller_kind=='droop':
            proj=(om[:,:,problem.gen_idx]@act.Qgen) # ns,N,r
            v=-scales[:,None,None]*proj
        elif controller_kind=='lqr':
            if gain is None:raise ValueError('gain required')
            x=problem.reduced_state(flatth,flatom).reshape(ns,N,-1)
            v=-scales[:,None,None]*(x@gain.T)
        else:raise ValueError(controller_kind)
        run+=cost.running(problem,flatth,flatom).reshape(ns,N)*dt
        cu=.5*r*np.sum(v*v,axis=2);run+=cu*dt;ctrl+=cu*dt
        nod=(v.reshape(ns*N,act.rank)@An.T).reshape(ns,N,n)
        force=problem.network_force(flatth).reshape(ns,N,n)
        drift=(problem.p_post[None,None,:]-problem.damp[None,None,:]*om-force+nod)/problem.M[None,None,:]
        basez=np.concatenate([rg.standard_normal((n_per_seed,n)) for rg in rngs],axis=0)
        om=om+dt*drift+(np.asarray(noise)/problem.M)[None,None,:]*sq*basez[None,:,:]
        th=th+dt*om;dm=np.max(np.abs(problem.edge_delta(th.reshape(ns*N,n)).reshape(ns,N,-1)),axis=2);dmax=np.maximum(dmax,dm);cross|=dm>=np.pi/2
        maxgen=np.maximum(maxgen,np.max(np.abs(om[:,:,problem.gen_idx]),axis=2))
    J=run+cost.terminal(problem,th.reshape(ns*N,n),om.reshape(ns*N,n)).reshape(ns,N)
    rows=[]
    for i,s in enumerate(scales):
        rows.append(dict(scale=float(s),J_mean=float(J[i].mean()),J_se=float(J[i].std(ddof=1)/math.sqrt(N)),
            cross_probability=float(cross[i].mean()),control_effort_mean=float(ctrl[i].mean()),
            dmax_q50_deg=float(np.degrees(np.quantile(dmax[i],.5))),dmax_q90_deg=float(np.degrees(np.quantile(dmax[i],.9))),
            dmax_q99_deg=float(np.degrees(np.quantile(dmax[i],.99))),genomega_q99=float(np.quantile(maxgen[i],.99)),
            n_paths=N,seeds=[int(x) for x in seeds]))
    return rows
