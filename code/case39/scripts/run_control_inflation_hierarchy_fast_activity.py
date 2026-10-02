#!/usr/bin/env python3
from pathlib import Path
import sys, csv, json, math
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pic_case39 import build_case39_problem
from stochastic_swing import noise_profile
from actuation_hierarchy import build_hierarchy, general_pic_geometry, lqr_for_actuation, mode_policy
from smooth_pic import feynman_kac_modes_smooth, simulate_soc_cost_modes_smooth
from smooth_barrier import SmoothPICCost
from control_inflation import control_inflation_geometry, completion_lqr_gain, completion_lqr_guide_nodal

SIGMAS=[0.05,0.10,0.20,0.30]
KINDS=['activity_scaled']
ALPHA={'H0':.0025,'H1':.0025,'H2':.01,'H3':.01}
R=.2; T=3.; DT=.005; NP_MINUS=100; NP_J=140; NP_PLUS=400

def shared_plus_actions(problem,Gamma,guide,seed=991):
    cost=SmoothPICCost(); n=problem.n; steps=int(np.ceil(T/DT));dt=T/steps;sq=np.sqrt(dt)
    rng=np.random.default_rng(seed)
    th=np.repeat(problem.theta_pre[None,:],NP_PLUS,axis=0).copy(); om=np.zeros((NP_PLUS,n))
    run=np.zeros(NP_PLUS); ll=np.zeros(NP_PLUS); cross=np.zeros(NP_PLUS,bool)
    for k in range(steps):
        run += cost.running(problem,th,om)*dt
        dW=sq*rng.standard_normal((NP_PLUS,n))
        nod=np.asarray(guide(k*dt,th,om),float)
        a=nod/Gamma[None,:]
        ll += -np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*dt
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+nod)/problem.M[None,:]
        om += dt*drift + (Gamma/problem.M)[None,:]*dW
        th += dt*om
        cross |= np.max(np.abs(th@problem.Inc),axis=1)>=np.pi/2
    S=run+cost.terminal(problem,th,om)
    return S,ll,float(cross.mean())

def eval_plus(S,ll,lam):
    lw=-S/lam+ll
    lse=logsumexp(lw)
    W=-lam*(lse-math.log(len(lw)))
    wn=np.exp(lw-lse); ess=1.0/np.sum(wn*wn)
    return float(W),float(ess/len(lw)),float(np.std(lw))

def main():
    p,meta=build_case39_problem(); Hs=build_hierarchy(p,meta['bus_ids']); cost=SmoothPICCost()
    gains={H.name:lqr_for_actuation(p,H,control_weight=R)[0] for H in Hs}
    rows=[]
    for kind in KINDS:
      print('KIND',kind,flush=True)
      # plus-guide metric independent of RMS strength for fixed noise shape
      G01=noise_profile(p,.1,kind); H3=Hs[-1]; gH4_01=control_inflation_geometry(p,H3,G01,R)
      Kplus,_=completion_lqr_gain(p,gH4_01)
      aplus=.01 if kind=='homogeneous' else .022
      guide=completion_lqr_guide_nodal(p,Kplus,alpha=aplus,clip=8.0)
      for sigma in SIGMAS:
        print(' sigma',sigma,flush=True)
        Gamma=noise_profile(p,sigma,kind)
        # one physical-noise path ensemble serves all completion temperatures
        S,ll,cross_plus=shared_plus_actions(p,Gamma,guide,seed=991+int(1000*sigma)+(0 if kind=='homogeneous' else 10000))
        # common H4 = H3 minimal completion
        gH4=control_inflation_geometry(p,H3,Gamma,R)
        Wh4,essh4,_=eval_plus(S,ll,gH4.lam)
        for H in Hs:
          geom_minus=general_pic_geometry(p,H,Gamma,R)
          st=np.stack([p.theta_pre,np.zeros(p.n)])
          zm=feynman_kac_modes_smooth(p,geom_minus,st,T=T,dt=DT,n_paths=NP_MINUS,
              seed=1701+int(1000*sigma)+H.rank+(0 if kind=='homogeneous' else 10000),cost=cost,
              guide_gain=gains[H.name],guide_alpha=ALPHA[H.name],guide_clip=8.0)
          pol=mode_policy(p,H,gains[H.name],alpha=ALPHA[H.name],clip=8.0)
          zj=simulate_soc_cost_modes_smooth(p,H,pol,Gamma,T=T,dt=DT,n_paths=NP_J,
              seed=2701+int(1000*sigma)+(0 if kind=='homogeneous' else 10000),control_weight=R,cost=cost)
          gp=control_inflation_geometry(p,H,Gamma,R)
          Wp,essp,lws=eval_plus(S,ll,gp.lam)
          trplus=float(np.trace(gp.injection_gain_completed)); froplus=float(np.linalg.norm(gp.injection_gain_completed,'fro'))
          rows.append(dict(noise=kind,sigma_rms=sigma,H=H.name,rank=H.rank,alpha=ALPHA[H.name],
              J=zj['J_mean'],J_se=zj['J_se'],cross_probability=zj['cross_probability'],control_effort=zj['control_effort_mean'],
              Wminus=float(zm['W'][0]),Wminus_ess=float(zm['ess'][0]/NP_MINUS),deleted_rank=geom_minus.deleted_rank,
              Wplus=Wp,Wplus_ess=essp,H4_oracle=Wh4,H4_ess=essh4,
              lambda_max=gp.lam,virtual_rank=gp.virtual_rank,
              inflation_trace=gp.inflation_trace,inflation_fraction_trace=gp.inflation_trace/trplus,
              inflation_fro=gp.inflation_fro,inflation_fraction_fro=gp.inflation_fro/froplus,
              gap_minus=zj['J_mean']-float(zm['W'][0]),gap_plus=zj['J_mean']-Wp,gap_H4=zj['J_mean']-Wh4,
              bound_improvement=Wp-float(zm['W'][0]),plus_shared_cross=cross_plus,plus_logweight_std=lws))
          print('  ',H.name,'J',rows[-1]['J'],'W-',rows[-1]['Wminus'],'W+',Wp,'lam',gp.lam,'ess+',essp,flush=True)
    out=ROOT/'results';
    (out/'control_inflation_hierarchy_activity.json').write_text(json.dumps(rows,indent=2))
    with open(out/'control_inflation_hierarchy_activity.csv','w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('WROTE',out/'control_inflation_hierarchy_activity.csv')
if __name__=='__main__':main()
