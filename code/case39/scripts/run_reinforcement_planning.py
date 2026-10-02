#!/usr/bin/env python3
from pathlib import Path
import sys,json,csv,math
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pic_case39 import build_case39_problem
from stochastic_swing import noise_profile
from actuation_hierarchy import build_hierarchy
from control_inflation import control_inflation_geometry,completion_lqr_gain,completion_lqr_guide_nodal
from smooth_barrier import SmoothPICCost
from reinforcement_planning import (
 generator_authority_fixed,generator_authority_noise_proportional,authority_to_Rdiag,
 diagonal_completion,effective_power_authority_MW,authority_vector_power_MW,robust_fixed_authority,
 lqr_for_nodal_authority,simulate_nodal_lqr,completed_lqr_shadow_proxy)

T=3.;DT=.005;R0=.2;BASE=100.;NP_PLUS=800
FRACS=[1.0,.8,.6,.4,.25]
KLIST=[5,10,20,29]


def shared_plus_actions(problem,Gamma,guide,seed,n_paths=NP_PLUS):
    cost=SmoothPICCost();n=problem.n;steps=int(np.ceil(T/DT));dt=T/steps;sq=np.sqrt(dt)
    rng=np.random.default_rng(seed)
    th=np.repeat(problem.theta_pre[None,:],n_paths,axis=0).copy();om=np.zeros((n_paths,n))
    run=np.zeros(n_paths);ll=np.zeros(n_paths);cross=np.zeros(n_paths,bool)
    for k in range(steps):
        run+=cost.running(problem,th,om)*dt
        dW=sq*rng.standard_normal((n_paths,n))
        nod=np.asarray(guide(k*dt,th,om),float)
        a=nod/Gamma[None,:]
        ll+=-np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*dt
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+nod)/problem.M[None,:]
        om+=dt*drift+(Gamma/problem.M)[None,:]*dW
        th+=dt*om
        cross|=(np.max(np.abs(th@problem.Inc),axis=1)>=np.pi/2)
    S=run+cost.terminal(problem,th,om)
    return S,ll,float(cross.mean())


def eval_plus(S,ll,lam):
    lw=-S/float(lam)+ll;lse=logsumexp(lw);W=-float(lam)*(lse-math.log(len(lw)))
    wn=np.exp(lw-lse);ess=1/np.sum(wn*wn)
    return dict(W=float(W),ess_fraction=float(ess/len(lw)),logweight_std=float(np.std(lw)))


def best_policy(problem,Gamma,g,seed):
    buses=problem.gen_idx;Rdiag=authority_to_Rdiag(g)
    K,_,_=lqr_for_nodal_authority(problem,buses,Rdiag)
    best=None
    for ia,a in enumerate([.01]):
        z=simulate_nodal_lqr(problem,Gamma,buses,Rdiag,K,alpha=a,T=T,dt=DT,n_paths=180,seed=seed,cost=SmoothPICCost())
        row=dict(alpha=a,**{k:v for k,v in z.items() if k!='raw_J'})
        if best is None or row['J_mean']<best['J_mean']:best=row
    return best


def main():
    problem,meta=build_case39_problem(); H3=build_hierarchy(problem,meta['bus_ids'])[-1]
    out=ROOT/'results';out.mkdir(exist_ok=True)
    allres={'metadata':dict(n=problem.n,ng=problem.ng,T=T,dt=DT,r0=R0,baseMVA=BASE),
            'frontier':{},'metric_design':{},'scenario_flexibility':{},'sparse':{}}
    shared={}
    for ik,kind in enumerate(['homogeneous','activity_scaled']):
        Gamma=noise_profile(problem,.1,kind)
        geom0=control_inflation_geometry(problem,H3,Gamma,R0)
        Kplus,_=completion_lqr_gain(problem,geom0)
        guide=completion_lqr_guide_nodal(problem,Kplus,alpha=(.01 if kind=='homogeneous' else .022),clip=8.)
        S,ll,cross=shared_plus_actions(problem,Gamma,guide,seed=9100+ik)
        shared[kind]=(Gamma,S,ll)
        # exact-PIC reinforcement frontier
        rows=[]
        for frac in FRACS:
            lam=frac*geom0.lam
            comp=diagonal_completion(problem,Gamma,generator_authority_fixed(problem,R0))
            # override completion at the selected lambda
            q=Gamma**2;Gdiag=np.zeros(problem.n);Gdiag[problem.gen_idx]=problem.ng*0+1/R0
            D=np.maximum(q/lam-Gdiag,0.)
            burden=float(D.sum()); Pmw=float(BASE*np.sum(np.sqrt(R0*D)))
            ev=eval_plus(S,ll,lam)
            rows.append(dict(lambda_fraction=frac,lambda_value=lam,completion_trace=burden,
                             effective_power_authority_MW=Pmw,physical_generator_authority=float(problem.ng/R0),
                             crossing_fraction_under_IS_guide=cross,**ev))
        allres['frontier'][kind]=rows
        # generator metric/authority design with same total physical authority
        g_fixed=generator_authority_fixed(problem,R0)
        g_equal=generator_authority_noise_proportional(problem,Gamma,R0)
        designs={'fixed_R':g_fixed,'noise_proportional_equalized_R':g_equal}
        dr={}
        for j,(name,g) in enumerate(designs.items()):
            c=diagonal_completion(problem,Gamma,g);ev=eval_plus(S,ll,c.lam)
            pol=best_policy(problem,Gamma,g,seed=12000+ik*100+j)
            dr[name]=dict(lambda_value=c.lam,completion_trace=c.burden_trace,
                          effective_power_authority_MW=effective_power_authority_MW(c,baseMVA=BASE,r_ref=R0),
                          generator_authority=g.tolist(),Rdiag=authority_to_Rdiag(g).tolist(),**ev,policy=pol)
        allres['metric_design'][kind]=dr

    # scenario flexibility: one fixed authority vector must serve both noise shapes
    Gh,Sh,llh=shared['homogeneous']; Ga,Sa,lla=shared['activity_scaled']
    g_unif=generator_authority_fixed(problem,R0)
    g_act=generator_authority_noise_proportional(problem,Ga,R0)
    g_rob,b_rob,opt=robust_fixed_authority(problem,[Gh,Ga],R0)
    designs={'uniform':g_unif,'robust_fixed':g_rob,'activity_specific':g_act}
    for kind,(Gamma,S,ll) in {'homogeneous':(Gh,Sh,llh),'activity_scaled':(Ga,Sa,lla)}.items():
        rr={}
        for j,(name,g) in enumerate(designs.items()):
            c=diagonal_completion(problem,Gamma,g);ev=eval_plus(S,ll,c.lam)
            pol=best_policy(problem,Gamma,g,seed=13000+(0 if kind=='homogeneous' else 100)+j)
            rr[name]=dict(lambda_value=c.lam,completion_trace=c.burden_trace,
                          effective_power_authority_MW=effective_power_authority_MW(c,baseMVA=BASE,r_ref=R0),
                          generator_authority=g.tolist(),**ev,policy=pol)
        allres['scenario_flexibility'][kind]=rr
    allres['scenario_flexibility']['robust_optimizer']=dict(success=bool(opt.success),message=str(opt.message),
                                                            authority=g_rob.tolist(),burdens=b_rob.tolist())

    # sparse physical reinforcement: H3 generators + k load-bus devices.  Compare
    # noise-only placement with an H4-LQR command-utilization (shadow) proxy.
    for ik,kind in enumerate(['homogeneous','activity_scaled']):
        Gamma,S,ll=shared[kind]
        geom0=control_inflation_geometry(problem,H3,Gamma,R0);Kplus,_=completion_lqr_gain(problem,geom0)
        shadow=completed_lqr_shadow_proxy(problem,Gamma,geom0,Kplus,alpha=(.01 if kind=='homogeneous' else .022),
                                          T=T,dt=DT,n_paths=120,seed=15000+ik)
        load=np.array([i for i in range(problem.n) if i not in set(problem.gen_idx)],dtype=int)
        noise_order=load[np.argsort(Gamma[load]**2)[::-1]]
        shadow_order=load[np.argsort(shadow[load])[::-1]]
        kres={}
        for method,order in [('noise',noise_order),('shadow_proxy',shadow_order)]:
            rows=[]
            for k in KLIST:
                added=order[:k];buses=np.r_[problem.gen_idx,added]
                Rdiag=np.ones(len(buses))*R0
                K,_,_=lqr_for_nodal_authority(problem,buses,Rdiag)
                # fixed mild scaling; common seed across methods for each k/noise
                z=simulate_nodal_lqr(problem,Gamma,buses,Rdiag,K,alpha=.01,T=T,dt=DT,n_paths=180,
                                     seed=16000+ik*100+k,cost=SmoothPICCost())
                rows.append(dict(k=int(k),added_bus_ids=(meta['bus_ids'][added]).astype(int).tolist(),
                                 J_mean=z['J_mean'],J_se=z['J_se'],cross_probability=z['cross_probability'],
                                 control_effort_mean=z['control_effort_mean']))
            kres[method]=rows
        kres['shadow_scores']=dict(bus_ids=meta['bus_ids'].astype(int).tolist(),scores=shadow.tolist())
        allres['sparse'][kind]=kres

    (out/'reinforcement_planning.json').write_text(json.dumps(allres,indent=2))
    # convenient flattened tables
    with open(out/'pic_planning_frontier.csv','w',newline='') as f:
        fields=['noise','lambda_fraction','lambda_value','completion_trace','effective_power_authority_MW','W','ess_fraction']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for kind,rows in allres['frontier'].items():
            for r in rows:w.writerow({k:(kind if k=='noise' else r[k]) for k in fields})
    print('wrote',out/'reinforcement_planning.json')

if __name__=='__main__':main()
