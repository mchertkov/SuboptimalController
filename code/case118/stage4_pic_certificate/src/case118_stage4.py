from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import eigh, solve_continuous_are
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar

from case118_stage3a import Actuation, SmoothCost


@dataclass
class GeneralPICGeometry:
    actuation: Actuation
    control_weight: float
    Gamma: np.ndarray
    lam: float
    injection_diffusion_physical: np.ndarray
    injection_diffusion_surrogate: np.ndarray
    omega_deleted: np.ndarray
    omega_deleted_eigvals: np.ndarray
    omega_deleted_eigvecs: np.ndarray
    injection_gain_physical: np.ndarray

    @property
    def deleted_rank(self):
        vals=self.omega_deleted_eigvals
        tol=max(1e-12,1e-10*np.max(vals) if vals.size else 1e-12)
        return int(np.count_nonzero(vals>tol))

    @property
    def deleted_trace(self): return float(np.trace(self.omega_deleted))
    @property
    def deleted_fro(self): return float(np.linalg.norm(self.omega_deleted,'fro'))


@dataclass
class ControlInflationGeometry:
    actuation: Actuation
    control_weight: float
    Gamma: np.ndarray
    lam: float
    injection_gain_physical: np.ndarray
    injection_gain_completed: np.ndarray
    injection_gain_virtual: np.ndarray
    virtual_rank: int

    @property
    def inflation_trace(self): return float(np.trace(self.injection_gain_virtual))
    @property
    def inflation_fro(self): return float(np.linalg.norm(self.injection_gain_virtual,'fro'))


def load_architectures(problem, matrix_path):
    z=np.load(matrix_path)
    genbus=np.asarray(z['generator_bus_ids'],int)
    if not np.array_equal(genbus,problem.bus_ids[problem.gen_idx]):
        raise RuntimeError('generator bus IDs mismatch')
    out={
        'H0':Actuation('H0','single physical MATPOWER slack generator',np.asarray(z['Qgen_H0'],float),genbus),
        'H1':Actuation('H1','equal distributed rank-one generator mode',np.asarray(z['Qgen_H1'],float),genbus),
        'H2':Actuation('H2','frozen rank-3 Kron generator mode',np.asarray(z['Qgen_H2_r3'],float),genbus),
        'H3':Actuation('H3','independent actuation at all generators',np.asarray(z['Qgen_H3'],float),genbus),
    }
    gains={k:np.asarray(z[{'H0':'LQR_K_H0','H1':'LQR_K_H1','H2':'LQR_K_H2_r3','H3':'LQR_K_H3'}[k]],float) for k in out}
    return out,gains


def lambda_max_from_covariances(Q,G,tol=1e-10):
    Q=(np.asarray(Q,float)+np.asarray(Q,float).T)/2
    G=(np.asarray(G,float)+np.asarray(G,float).T)/2
    vals,U=eigh(Q); scale=max(1.0,float(np.max(np.abs(vals)))); pos=vals>tol*scale
    P=U[:,pos]@U[:,pos].T if np.any(pos) else np.zeros_like(Q)
    I=np.eye(Q.shape[0]); gscale=max(1.0,float(np.linalg.norm(G,2)))
    if np.linalg.norm((I-P)@G@(I-P),2)>tol*gscale or np.linalg.norm((I-P)@G@P,2)>tol*gscale:
        raise ValueError('range(G) is not contained in range(Q)')
    if np.linalg.norm(G,2)<=tol: return math.inf
    Qis=U[:,pos]@np.diag(1/np.sqrt(vals[pos]))@U[:,pos].T
    M=Qis@G@Qis
    mu=float(np.max(eigh((M+M.T)/2,eigvals_only=True)))
    return math.inf if mu<=tol else 1.0/mu


def general_pic_geometry(problem, actuation, Gamma, control_weight=.2):
    Gamma=np.asarray(Gamma,float); r=float(control_weight)
    A=actuation.nodal(problem)
    C=(A/Gamma[:,None])/math.sqrt(r)
    smax=float(np.linalg.svd(C,compute_uv=False)[0])
    lam=1.0/(smax*smax)
    Dinj=.5*np.diag(Gamma**2)
    Ginj=A@A.T/r
    Dtilde=.5*lam*Ginj
    resid=(Dinj-Dtilde + (Dinj-Dtilde).T)/2
    ev=np.linalg.eigvalsh(resid)
    if ev.min() < -1e-9: raise RuntimeError(f'negative deleted injection diffusion {ev.min()}')
    Minv=np.diag(1/problem.M)
    Aom=Minv@resid@Minv; Aom=(Aom+Aom.T)/2
    vals,vecs=eigh(Aom); vals[np.abs(vals)<1e-12]=0.; vals=np.maximum(vals,0.)
    return GeneralPICGeometry(actuation,r,Gamma,lam,Dinj,Dtilde,Aom,vals,vecs,Ginj)


def control_inflation_geometry(problem, actuation, Gamma, control_weight=.2, lam=None):
    Gamma=np.asarray(Gamma,float); r=float(control_weight); A=actuation.nodal(problem)
    G=A@A.T/r; Q=np.diag(Gamma**2)
    lmax=lambda_max_from_covariances(Q,G)
    lam=lmax if lam is None else float(lam)
    if lam<=0 or lam>lmax*(1+1e-9): raise ValueError('invalid lambda')
    Gp=Q/lam; Delta=(Gp-G + (Gp-G).T)/2
    vals=eigh(Delta,eigvals_only=True); tol=max(1e-12,1e-10*np.max(np.abs(vals)))
    if vals.min() < -tol: raise RuntimeError(f'completion not PSD {vals.min()}')
    vr=int(np.count_nonzero(vals>tol))
    return ControlInflationGeometry(actuation,r,Gamma,lam,G,Gp,Delta,vr)


def completion_diagnostics(geom):
    Q=np.diag(geom.Gamma**2); G=geom.injection_gain_physical; lam=geom.lam
    qi=1/geom.Gamma
    M=np.eye(len(qi))-lam*((qi[:,None]*G)*qi[None,:]); M=(M+M.T)/2
    return dict(inflation_trace=geom.inflation_trace,inflation_fro=geom.inflation_fro,
                whitened_deficit_trace=float(np.trace(M)),whitened_deficit_fro=float(np.linalg.norm(M,'fro')),
                whitened_deficit_min_eig=float(np.min(eigh(M,eigvals_only=True))),
                whitened_deficit_max_eig=float(np.max(eigh(M,eigvals_only=True))))


def _prepare(problem,state):
    a=np.asarray(state,float)
    if a.ndim==2 and a.shape==(2,problem.n): a=a[None,:,:]
    if a.ndim!=3 or a.shape[1:]!=(2,problem.n): raise ValueError('state shape')
    return a[:,0,:].copy(),a[:,1,:].copy()


def feynman_kac_modes(problem,geom,state,*,t0=0.,T=3.,dt=.005,n_paths=512,seed=1,cost=None,
                      guide_gain=None,guide_alpha=0.,guide_clip=8.,brownian=None,return_logweights=False):
    if cost is None: cost=SmoothCost()
    th0,om0=_prepare(problem,state); nq=len(th0); n=problem.n; m=geom.actuation.rank
    horizon=float(T)-float(t0)
    if horizon<=0:
        term=cost.terminal(problem,th0,om0); return dict(W=term,ess=np.full(nq,n_paths),cross_fraction=np.zeros(nq),lam=geom.lam)
    steps=int(np.ceil(horizon/dt)); dt=horizon/steps; sq=math.sqrt(dt)
    rng=np.random.default_rng(seed)
    if brownian is None: brownian=rng.standard_normal((steps,n_paths,m))
    th=np.repeat(th0[:,None,:],n_paths,axis=1).reshape(nq*n_paths,n)
    om=np.repeat(om0[:,None,:],n_paths,axis=1).reshape(nq*n_paths,n)
    run=np.zeros(nq*n_paths); ll=np.zeros(nq*n_paths); cross=np.zeros(nq*n_paths,bool)
    An=geom.actuation.nodal(problem); root=math.sqrt(geom.lam/geom.control_weight)
    for k in range(steps):
        if guide_gain is None or guide_alpha==0:
            v=np.zeros((nq*n_paths,m))
        else:
            v=-float(guide_alpha)*(problem.reduced_state(th,om)@guide_gain.T)
            if guide_clip is not None: v=np.clip(v,-float(guide_clip),float(guide_clip))
        run += cost.running(problem,th,om)*dt
        pid=np.arange(nq*n_paths)%n_paths; dW=sq*brownian[k,pid,:]
        if guide_gain is not None and guide_alpha!=0:
            a=v*math.sqrt(geom.control_weight/geom.lam)
            ll += -np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*dt
        nod=v@An.T
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+nod)/problem.M[None,:]
        om=om+dt*drift+((root*dW)@An.T)/problem.M[None,:]
        th=th+dt*om
        cross |= np.max(np.abs(problem.edge_delta(th)),axis=1)>=np.pi/2
    S=run+cost.terminal(problem,th,om)
    lw=(-S/geom.lam+ll).reshape(nq,n_paths)
    lse=logsumexp(lw,axis=1); W=-geom.lam*(lse-math.log(n_paths))
    lwn=lw-lse[:,None]; ess=1/np.sum(np.exp(2*lwn),axis=1)
    out=dict(W=W,ess=ess,cross_fraction=cross.reshape(nq,n_paths).mean(axis=1),lam=geom.lam,
             logweight_std=np.std(lw,axis=1),dt=dt,steps=steps,n_paths=n_paths)
    if return_logweights: out['logweights']=lw
    return out


def completion_lqr_gain(problem,geom,angle_weight=2.,freq_weight=.5):
    n=problem.n; Minv=np.diag(1/problem.M)
    A=np.block([[np.zeros((n-1,n-1)),problem.H],[-Minv@problem.Lstar@problem.E,-Minv@np.diag(problem.damp)]])
    B=np.vstack([np.zeros((n-1,n)),Minv])
    Qq=problem.E.T@problem.Lstar@problem.E; Qq/=max(np.trace(Qq)/(n-1),1e-12)
    Qw=np.diag(problem.M); Qw/=max(np.trace(Qw)/n,1e-12)
    Q=np.block([[angle_weight*Qq,np.zeros((n-1,n))],[np.zeros((n,n-1)),freq_weight*Qw]])
    R=np.diag(geom.lam/(geom.Gamma**2))
    P=solve_continuous_are(A,B,Q,R); K=np.linalg.solve(R,B.T@P)
    return K


def completion_guide(problem,K,alpha=0.01,clip=8.):
    def guide(t,th,om):
        u=-float(alpha)*(problem.reduced_state(th,om)@K.T)
        return np.clip(u,-clip,clip) if clip is not None else u
    return guide


def shared_plus_actions(problem,Gamma,guide,*,T=3.,dt=.005,n_paths=1024,seed=1,cost=None):
    if cost is None: cost=SmoothCost()
    n=problem.n; steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt);rng=np.random.default_rng(seed)
    th=np.repeat(problem.theta_pre[None,:],n_paths,axis=0).copy();om=np.zeros((n_paths,n));run=np.zeros(n_paths);ll=np.zeros(n_paths);cross=np.zeros(n_paths,bool)
    for k in range(steps):
        run += cost.running(problem,th,om)*dt
        dW=sq*rng.standard_normal((n_paths,n)); nod=np.asarray(guide(k*dt,th,om),float) if guide is not None else np.zeros_like(th)
        a=nod/Gamma[None,:]
        ll += -np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*dt
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+nod)/problem.M[None,:]
        om=om+dt*drift+(Gamma/problem.M)[None,:]*dW; th=th+dt*om
        cross |= np.max(np.abs(problem.edge_delta(th)),axis=1)>=np.pi/2
    S=run+cost.terminal(problem,th,om)
    return dict(S=S,ll=ll,cross_fraction=float(cross.mean()),dt=dt,n_paths=n_paths)


def eval_plus_actions(actions,lam):
    lw=-actions['S']/float(lam)+actions['ll']; lse=logsumexp(lw)
    W=-float(lam)*(lse-math.log(len(lw))); wn=np.exp(lw-lse);ess=1/np.sum(wn*wn)
    return dict(W=float(W),ess=float(ess),ess_fraction=float(ess/len(lw)),logweight_std=float(np.std(lw)))


def tune_completion_guide(problem,Gamma,geom_ref,alphas,*,seed=40101,n_paths=256,T=3.,dt=.005):
    K=completion_lqr_gain(problem,geom_ref); rows=[]
    for a in alphas:
        guide=completion_guide(problem,K,float(a),clip=8.) if float(a)!=0 else None
        act=shared_plus_actions(problem,Gamma,guide,T=T,dt=dt,n_paths=n_paths,seed=seed,cost=SmoothCost())
        z=eval_plus_actions(act,geom_ref.lam)
        rows.append(dict(alpha=float(a),W=z['W'],ess_fraction=z['ess_fraction'],logweight_std=z['logweight_std'],cross_fraction=act['cross_fraction']))
    best=max(rows,key=lambda x:x['ess_fraction'])
    return rows,best,K


def curvature_trace_modes(problem,geom,state,*,h=.008,n_directions=6,seed_directions=123,**fk):
    A=geom.omega_deleted
    if np.max(np.abs(A))<1e-14: return dict(trace=0.,se=0.,min_direction=0.,positive_fraction=1.,directions=[])
    vals,vecs=eigh((A+A.T)/2); vals=np.maximum(vals,0); As=vecs@np.diag(np.sqrt(vals))@vecs.T
    rng=np.random.default_rng(seed_directions);base=np.asarray(state,float);states=[base]
    for _ in range(n_directions):
        z=rng.choice(np.array([-1.,1.]),size=problem.n);q=As@z
        pp=base.copy();mm=base.copy();pp[1]+=h*q;mm[1]-=h*q;states.extend([pp,mm])
    out=feynman_kac_modes(problem,geom,np.stack(states),**fk)
    W=out['W'];W0=W[0];ds=np.array([(W[1+2*j]+W[2+2*j]-2*W0)/(h*h) for j in range(n_directions)])
    se=float(ds.std(ddof=1)/math.sqrt(n_directions)) if n_directions>1 else float('nan')
    npaths=fk.get('n_paths',512)
    return dict(trace=float(ds.mean()),se=se,min_direction=float(ds.min()),positive_fraction=float(np.mean(ds>=0)),directions=ds.tolist(),
                W0=float(W0),ess_fraction_min=float(np.min(out['ess'])/npaths),cross_fraction=float(out['cross_fraction'][0]))


def deterministic_state_path(problem,policy,times,dt=.001):
    times=np.asarray(times,float);order=np.argsort(times);target=times[order];out=[None]*len(times)
    th=problem.theta_pre.copy();om=np.zeros(problem.n);it=0;t=0.;maxT=float(target[-1]) if len(target) else 0.;steps=int(np.ceil(maxT/dt))
    for k in range(steps+1):
        while it<len(target) and t+1e-12>=target[it]: out[order[it]]=np.stack([th.copy(),om.copy()]);it+=1
        if k==steps:break
        v=policy(t,th[None,:],om[None,:])[0]
        nod=v
        drift=(problem.p_post-problem.damp*om-problem.network_force(th)+nod)/problem.M
        om=om+dt*drift;th=th+dt*om;t+=dt
    return np.stack(out)


def _h_derivs(g):
    c=math.cos(g);t=math.tan(g);sec2=1/(c*c)
    return -math.log(c),t,sec2,2*sec2*t,4*sec2*t*t+2*sec2*sec2

def _h2ext(x,guard):
    a=abs(float(x));_,_,h2,h3,h4=_h_derivs(guard)
    if a<=guard:return 1/(math.cos(a)**2)
    z=a-guard;return h2+h3*z+.5*h4*z*z


def _terminal_matrices(problem,Ginj,running_energy=.08,terminal_energy=.20):
    n=problem.n;d0=terminal_energy/n;M=np.diag(problem.M);Minv=np.diag(1/problem.M);D=np.diag(problem.damp)
    P=np.asarray(Ginj,float);C=(running_energy/n)*M-2*d0*D-d0*d0*P;Ac=Minv@(D+d0*P);G=Minv@P@Minv;Q=.5*(Ac.T@C+C@Ac)
    return d0,M,Minv,D,P,C,Ac,G,Q


def terminal_rho_coefficients(problem,geom,running_energy=.08,running_barrier=.02,terminal_energy=.20,delta_bar_deg=84.,guard_deg=82.):
    d0,M,Minv,D,P,C,Ac,G,Q=_terminal_matrices(problem,geom.injection_gain_physical,running_energy,terminal_energy)
    A=geom.omega_deleted
    rho0=float(d0*np.trace(A@M));rho1=float(np.trace(A@(2*d0*D-(running_energy/problem.n)*M+d0*d0*P)));rho2=float(-np.trace(A@(Ac.T@C+C@Ac)))
    B=problem.Inc;k=problem.Kedge;me=B.shape[1];const=float(np.trace(A@(2*C@G@C-2*(Ac.T@Q+Q@Ac))))
    db=math.radians(delta_bar_deg);guard=math.radians(guard_deg);mins=[];maxs=[]
    for e in range(me):
        be=B[:,e];ae=float(be@A@be);ce=float(be@A@C@Minv@be);coef=float(k[e]*((running_energy/problem.n)*ae-ce));bcoef=float((running_barrier/me)*ae)
        def fun(x):return coef*math.cos(x)+bcoef*_h2ext(x,guard)
        cand=[fun(0.),fun(min(guard,db)),fun(db)]
        for lo,hi in [(0.,min(guard,db)),(min(guard,db),db)]:
            if hi-lo>1e-12:
                rr=minimize_scalar(fun,bounds=(lo,hi),method='bounded');cand.append(float(rr.fun));rr=minimize_scalar(lambda x:-fun(x),bounds=(lo,hi),method='bounded');cand.append(float(-rr.fun))
        mins.append(min(cand));maxs.append(max(cand))
    raw_lo=const-2*sum(maxs);raw_hi=const-2*sum(mins);pad=.01*max(abs(raw_lo),abs(raw_hi),1.)+1e-9
    disc=rho1*rho1-2*rho0*rho2;smin=rho1/rho2 if rho2>0 else float('nan');qmin=rho0-rho1*rho1/(2*rho2) if rho2>0 else float('nan')
    return dict(rho_T=rho0,rho_t_T=rho1,rho_tt_T=rho2,rho_ttt_T_raw_interval=[float(raw_lo),float(raw_hi)],rho_ttt_T_inflated_interval=[float(raw_lo-pad),float(raw_hi+pad)],quadratic_discriminant=float(disc),quadratic_s_min=float(smin),quadratic_min=float(qmin))


def terminal_quadratic_positive_layer(coeffs,max_layer=.05):
    # rho(T-s) ~= rho0-rho1*s+.5*rho2*s^2. Return interval to first zero if any.
    a=.5*coeffs['rho_tt_T'];b=-coeffs['rho_t_T'];c=coeffs['rho_T']
    if c<=0:return 0.0
    roots=[]
    if abs(a)<1e-16:
        if b<0: roots=[-c/b]
    else:
        d=b*b-4*a*c
        if d>=0:
            sq=math.sqrt(d); roots=[r for r in [(-b-sq)/(2*a),(-b+sq)/(2*a)] if r>0]
    return float(min([max_layer]+roots)) if roots else float(max_layer)

def terminal_taylor_residual_bound(problem,geom,coeffs,running_energy=.08,running_barrier=.02,terminal_energy=.20,delta_bar_deg=84.,omega_bar=6.5,guard_deg=82.):
    """Conservative pointwise positive terminal-layer test on the compact tube.

    Generalizes the case39 implementation to an arbitrary frozen actuation Gramian.
    It bounds the residual of the explicit second-order terminal HJB Taylor candidate;
    no Monte Carlo enters this bound.  It is a compact-tube certificate, not a global
    interior certificate for the whole smooth problem.
    """
    n=problem.n;B=problem.Inc;k=problem.Kedge;me=B.shape[1]
    db=math.radians(delta_bar_deg);wb=float(omega_bar);guard=math.radians(guard_deg)
    d0,M,Minv,D,P,C,Ac,G,Q=_terminal_matrices(problem,geom.injection_gain_physical,running_energy,terminal_energy)
    Wnorm=math.sqrt(n)*wb
    hppmax=max(abs(_h2ext(-db,guard)),abs(_h2ext(db,guard)))
    d_sin=[];d_hp=[]
    _,hps,_,_,_=_h_derivs(guard)
    def hp_ext(x):
        a=abs(float(x));sg=1 if x>=0 else -1
        _,h1,h2,h3,h4=_h_derivs(guard)
        if a<=guard:return math.tan(float(x))
        z=a-guard;return sg*(h1+h2*z+.5*h3*z*z+(h4/6)*z**3)
    for ds in problem.delta_star:
        d_sin.append(max(abs(math.sin(-db)-math.sin(float(ds))),abs(math.sin(db)-math.sin(float(ds)))))
        hpstar=math.tan(float(ds));d_hp.append(max(abs(hp_ext(-db)-hpstar),abs(hp_ext(db)-hpstar)))
    d_sin=np.asarray(d_sin);d_hp=np.asarray(d_hp)
    a0max=float(np.linalg.norm(Minv@B@np.diag(np.sqrt(k)),2)*np.linalg.norm(np.sqrt(k)*d_sin))
    edge_grad=(running_energy/n)*k*d_sin+(running_barrier/me)*d_hp
    rmax=float(np.linalg.norm(B,2)*np.linalg.norm(edge_grad))
    L0=B@np.diag(k)@B.T;Lmax=float(np.linalg.norm(L0,2))
    Hthmax=float((running_energy/n)*Lmax+(running_barrier/me)*hppmax*np.linalg.norm(B@B.T,2))
    Jlmax=float(Hthmax+np.linalg.norm(C@Minv,2)*Lmax)
    lmax=float(rmax+np.linalg.norm(C,2)*a0max)
    gradw2=float(lmax+2*np.linalg.norm(Q,2)*Wnorm)
    gom=float(a0max+np.linalg.norm(Ac,2)*Wnorm)
    gradH0w=float(np.linalg.norm(C,2)*Wnorm)
    Amax=float(np.linalg.norm(G,2)*gradH0w**2)
    Dtilde=.5*geom.lam*G
    DHmax=float(Wnorm*Jlmax*Wnorm+gom*gradw2+2*abs(np.trace(Dtilde@Q)))
    w3max=float(Amax+DHmax);crossmax=float(np.linalg.norm(G,2)*gradH0w*gradw2);gradw2G=float(np.linalg.norm(G,2)*gradw2**2)
    def rhoquad(s):return coeffs['rho_T']-coeffs['rho_t_T']*s+.5*coeffs['rho_tt_T']*s*s
    def lower(s):
        s=float(s);neg=.5*s*s*w3max+.5*s**3*crossmax+.125*s**4*gradw2G;return rhoquad(s)-neg
    lo=0.;hi=.1
    if lower(hi)>=0:safe=hi
    else:
        for _ in range(100):
            mid=.5*(lo+hi)
            if lower(mid)>=0:lo=mid
            else:hi=mid
        safe=lo
    return dict(delta_bar_deg=float(delta_bar_deg),omega_bar=float(omega_bar),w3_abs_max=w3max,cross_abs_max=crossmax,gradw2_G_gradw2_max=gradw2G,analytic_safe_layer_s=float(safe),residual_lower_at_safe=float(lower(safe)),residual_lower_samples={str(x):float(lower(x)) for x in [1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2]})

def completion_actual_cost_lqr_gain(problem,geom,cost=None):
    """CARE guide using the local quadratic expansion of the *actual* smooth Stage-3 cost."""
    if cost is None: cost=SmoothCost()
    n=problem.n; Minv=np.diag(1/problem.M)
    A=np.block([[np.zeros((n-1,n-1)),problem.H],[-Minv@problem.Lstar@problem.E,-Minv@np.diag(problem.damp)]])
    Bc=np.vstack([np.zeros((n-1,n)),Minv])
    # Energy Hessian in reduced angle coordinates and frequency coordinates.
    Qq=(cost.c_E/n)*(problem.E.T@problem.Lstar@problem.E)
    # Bregman barrier Hessian at post-event equilibrium.
    hedges=1.0/np.cos(problem.delta_star)**2
    Hb=problem.Inc@np.diag(hedges)@problem.Inc.T
    Qq += (cost.c_B/problem.Inc.shape[1])*(problem.E.T@Hb@problem.E)
    Qw=(cost.c_E/n)*np.diag(problem.M)
    Q=np.block([[Qq,np.zeros((n-1,n))],[np.zeros((n,n-1)),Qw]])
    R=np.diag(geom.lam/(geom.Gamma**2))
    P=solve_continuous_are(A,Bc,Q,R);K=np.linalg.solve(R,Bc.T@P)
    return K,dict(A=A,B=Bc,Q=Q,R=R,P=P)
