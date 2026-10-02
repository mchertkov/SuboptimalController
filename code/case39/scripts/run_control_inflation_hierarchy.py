#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, math, csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pic_case39 import build_case39_problem
from stochastic_swing import noise_profile
from actuation_hierarchy import build_hierarchy, general_pic_geometry, lqr_for_actuation, mode_policy
from smooth_pic import feynman_kac_modes_smooth, simulate_soc_cost_modes_smooth
from smooth_barrier import SmoothPICCost
from control_inflation import (control_inflation_geometry, completion_lqr_gain,
    completion_lqr_guide_nodal, feynman_kac_control_inflation)

SIGMAS=[0.05,0.10,0.20,0.30]
KINDS=['homogeneous','activity_scaled']
ALPHA_DEPLOY={'H0':0.0025,'H1':0.0025,'H2':0.01,'H3':0.01}
SEEDS_W=[1103,2207]
NP_W=400
NP_J=500
DT=0.005
T=3.0
R=0.2

# Importance-guide scale for completed problem.  This is selected from the nominal
# H3 tuning and scaled by sqrt(0.002/lambda_shape) to compensate for the scalar
# temperature bottleneck under heterogeneous noise. It is kept fixed across RMS
# strength because the noise shape and completed control metric are unchanged.
def plus_guide_alpha(lam_at_sigma_01):
    return float(np.clip(0.01*np.sqrt(0.002/max(lam_at_sigma_01,1e-12)),0.004,0.025))

def _build(kind,sigma,Hname):
    p,meta=build_case39_problem(); hs={h.name:h for h in build_hierarchy(p,meta['bus_ids'])}; H=hs[Hname]
    Gamma=noise_profile(p,sigma,kind)
    return p,meta,H,Gamma

def task_J(args):
    kind,sigma,Hname,seed=args
    p,meta,H,Gamma=_build(kind,sigma,Hname)
    K,_=lqr_for_actuation(p,H,control_weight=R)
    pol=mode_policy(p,H,K,alpha=ALPHA_DEPLOY[Hname],clip=8.0)
    z=simulate_soc_cost_modes_smooth(p,H,pol,Gamma,T=T,dt=DT,n_paths=NP_J,seed=seed,
                                     control_weight=R,cost=SmoothPICCost())
    return dict(type='J',kind=kind,sigma=sigma,H=Hname,rank=H.rank,
                alpha=ALPHA_DEPLOY[Hname],J=z['J_mean'],J_se=z['J_se'],
                cross=z['cross_probability'],effort=z['control_effort_mean'])

def task_Wminus(args):
    kind,sigma,Hname,seed=args
    p,meta,H,Gamma=_build(kind,sigma,Hname)
    geom=general_pic_geometry(p,H,Gamma,control_weight=R)
    K,_=lqr_for_actuation(p,H,control_weight=R)
    st=np.stack([p.theta_pre,np.zeros(p.n)])
    z=feynman_kac_modes_smooth(p,geom,st,T=T,dt=DT,n_paths=NP_W,seed=seed,cost=SmoothPICCost(),
        guide_gain=K,guide_alpha=ALPHA_DEPLOY[Hname],guide_clip=8.0)
    return dict(type='Wminus',kind=kind,sigma=sigma,H=Hname,rank=H.rank,
                lam=geom.lam,Wminus=float(z['W'][0]),ess=float(z['ess'][0]/NP_W),
                cross_sur=float(z['cross_fraction'][0]),deleted_rank=geom.deleted_rank,
                deleted_trace=geom.deleted_trace)

def task_Wplus(args):
    kind,sigma,Hname,seed=args
    p,meta,H,Gamma=_build(kind,sigma,Hname)
    geom=control_inflation_geometry(p,H,Gamma,control_weight=R)
    # nominal shape lambda for scale selection
    G01=noise_profile(p,0.1,kind)
    geom01=control_inflation_geometry(p,H,G01,control_weight=R)
    a=plus_guide_alpha(geom01.lam)
    K,_=completion_lqr_gain(p,geom)
    guide=completion_lqr_guide_nodal(p,K,alpha=a,clip=8.0)
    st=np.stack([p.theta_pre,np.zeros(p.n)])
    z=feynman_kac_control_inflation(p,geom,st,T=T,dt=DT,n_paths=NP_W,seed=seed,
        cost=SmoothPICCost(),guide_nodal=guide)
    trplus=float(np.trace(geom.injection_gain_completed)); trv=geom.inflation_trace
    froplus=float(np.linalg.norm(geom.injection_gain_completed,'fro'))
    return dict(type='Wplus',kind=kind,sigma=sigma,H=Hname,rank=H.rank,
                lam=geom.lam,Wplus=float(z['W'][0]),ess=float(z['ess'][0]/NP_W),
                cross_phys=float(z['cross_fraction'][0]),guide_alpha=a,
                virtual_rank=geom.virtual_rank,inflation_trace=trv,
                inflation_fraction_trace=trv/trplus,
                inflation_fro=geom.inflation_fro,
                inflation_fraction_fro=geom.inflation_fro/froplus)

def main():
    tasks=[]
    # J one seed; W two independent reps. For homogeneous W+ is identical for all H
    # because all orthonormal architectures have the same lambda_max. Compute H3 once
    # per sigma/rep and broadcast later.
    for kind in KINDS:
      for sigma in SIGMAS:
       for H in ['H0','H1','H2','H3']:
        tasks.append(('J',(kind,sigma,H,3331)))
        for seed in SEEDS_W: tasks.append(('Wminus',(kind,sigma,H,seed)))
        if kind=='activity_scaled':
         for seed in SEEDS_W: tasks.append(('Wplus',(kind,sigma,H,seed)))
       if kind=='homogeneous':
        for seed in SEEDS_W: tasks.append(('Wplus',(kind,sigma,'H3',seed)))
    print('tasks',len(tasks),flush=True)
    funcs={'J':task_J,'Wminus':task_Wminus,'Wplus':task_Wplus}
    raw=[]
    with ProcessPoolExecutor(max_workers=6) as ex:
      futs=[ex.submit(funcs[t],a) for t,a in tasks]
      for i,f in enumerate(as_completed(futs),1):
        raw.append(f.result())
        if i%10==0: print('done',i,'/',len(futs),flush=True)
    # Broadcast homogeneous Wplus and geometry to H0-H2, but recompute geometry metrics
    # algebraically for each H at sigma=.1 then they are RMS-scale invariant.
    p,meta=build_case39_problem(); hs={h.name:h for h in build_hierarchy(p,meta['bus_ids'])}
    for sigma in SIGMAS:
      src=[r for r in raw if r['type']=='Wplus' and r['kind']=='homogeneous' and r['sigma']==sigma and r['H']=='H3']
      for Hname in ['H0','H1','H2']:
        Gamma=noise_profile(p,sigma,'homogeneous'); g=control_inflation_geometry(p,hs[Hname],Gamma,R)
        trplus=float(np.trace(g.injection_gain_completed)); froplus=float(np.linalg.norm(g.injection_gain_completed,'fro'))
        for s in src:
          d=dict(s);d['H']=Hname;d['rank']=hs[Hname].rank;d['lam']=g.lam
          d['virtual_rank']=g.virtual_rank;d['inflation_trace']=g.inflation_trace
          d['inflation_fraction_trace']=g.inflation_trace/trplus
          d['inflation_fro']=g.inflation_fro;d['inflation_fraction_fro']=g.inflation_fro/froplus
          raw.append(d)
    # aggregate
    rows=[]
    for kind in KINDS:
      for sigma in SIGMAS:
       for Hname in ['H0','H1','H2','H3']:
        j=[r for r in raw if r['type']=='J' and r['kind']==kind and r['sigma']==sigma and r['H']==Hname][0]
        wm=[r for r in raw if r['type']=='Wminus' and r['kind']==kind and r['sigma']==sigma and r['H']==Hname]
        wp=[r for r in raw if r['type']=='Wplus' and r['kind']==kind and r['sigma']==sigma and r['H']==Hname]
        def ms(key,rr):
          v=np.array([x[key] for x in rr],float);return float(v.mean()),float(v.std(ddof=1)) if len(v)>1 else 0.
        Wm,Wm_sd=ms('Wminus',wm);Wp,Wp_sd=ms('Wplus',wp)
        # H4 common oracle is H3-completed value at same kind/sigma
        wp_h4=[r for r in raw if r['type']=='Wplus' and r['kind']==kind and r['sigma']==sigma and r['H']=='H3']
        Wh4,Wh4_sd=ms('Wplus',wp_h4)
        g=wp[0]
        rows.append(dict(noise=kind,sigma_rms=sigma,H=Hname,rank=j['rank'],alpha=j['alpha'],
          J=j['J'],J_se=j['J_se'],cross_probability=j['cross'],control_effort=j['effort'],
          Wminus=Wm,Wminus_rep_sd=Wm_sd,Wminus_ess=float(np.mean([x['ess'] for x in wm])),
          Wplus=Wp,Wplus_rep_sd=Wp_sd,Wplus_ess=float(np.mean([x['ess'] for x in wp])),
          H4_oracle=Wh4,H4_rep_sd=Wh4_sd,lambda_max=g['lam'],virtual_rank=g['virtual_rank'],
          inflation_trace=g['inflation_trace'],inflation_fraction_trace=g['inflation_fraction_trace'],
          inflation_fro=g['inflation_fro'],inflation_fraction_fro=g['inflation_fraction_fro'],
          gap_minus=j['J']-Wm,gap_plus=j['J']-Wp,gap_H4=j['J']-Wh4,
          bound_improvement=Wp-Wm))
    outdir=ROOT/'results'; outdir.mkdir(exist_ok=True)
    (outdir/'control_inflation_hierarchy_raw.json').write_text(json.dumps(raw,indent=2))
    (outdir/'control_inflation_hierarchy.json').write_text(json.dumps(rows,indent=2))
    with open(outdir/'control_inflation_hierarchy.csv','w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print('wrote',outdir/'control_inflation_hierarchy.csv')

if __name__=='__main__':main()
