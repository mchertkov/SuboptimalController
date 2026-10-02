from pathlib import Path
import sys,json,csv,math,os
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import load_architectures,general_pic_geometry,control_inflation_geometry,completion_lqr_gain,completion_guide,shared_plus_actions,eval_plus_actions,feynman_kac_modes

R=.2;T=3.;DT=.005
ARCHS=['H0','H1','H2','H3'];KINDS=['homogeneous','activity_scaled'];SIGMAS=[.10,.15,.20]
ALPHA_MINUS={'H0':.00125,'H1':.00125,'H2':.0025,'H3':.005}
ALPHA_PLUS={'homogeneous':.005,'activity_scaled':.02}
SEEDS_MINUS=[42101,42201];SEEDS_PLUS=[43101,43201]
NP_MINUS=512
NP_PLUS={'homogeneous':1024,'activity_scaled':2048}

def build():
 p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,gains=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');return p,Hs,gains

def task_minus(args):
 noise,s,H,seed=args;p,Hs,gains=build();G=noise_profile(p,s,noise);gm=general_pic_geometry(p,Hs[H],G,R);st=np.stack([p.theta_pre,np.zeros(p.n)])
 z=feynman_kac_modes(p,gm,st,T=T,dt=DT,n_paths=NP_MINUS,seed=seed,cost=SmoothCost(),guide_gain=gains[H],guide_alpha=ALPHA_MINUS[H],guide_clip=8.)
 return dict(type='minus',noise=noise,sigma_rms=s,architecture=H,rank=Hs[H].rank,seed=seed,n_paths=NP_MINUS,lambda_max=gm.lam,W=float(z['W'][0]),ess_fraction=float(z['ess'][0]/NP_MINUS),logweight_std=float(z['logweight_std'][0]),surrogate_cross=float(z['cross_fraction'][0]),deleted_rank=gm.deleted_rank,deleted_trace=gm.deleted_trace)

def task_plus(args):
 noise,s,seed=args;p,Hs,gains=build();G=noise_profile(p,s,noise)
 # H3 completion metric has sigma-invariant Rplus for fixed noise shape, so build guide at this sigma.
 gref=control_inflation_geometry(p,Hs['H3'],G,R);K=completion_lqr_gain(p,gref);guide=completion_guide(p,K,ALPHA_PLUS[noise],clip=8.)
 n=NP_PLUS[noise];a=shared_plus_actions(p,G,guide,T=T,dt=DT,n_paths=n,seed=seed,cost=SmoothCost())
 rows=[]
 for H in ARCHS:
  gp=control_inflation_geometry(p,Hs[H],G,R);z=eval_plus_actions(a,gp.lam)
  trplus=float(np.trace(gp.injection_gain_completed));froplus=float(np.linalg.norm(gp.injection_gain_completed,'fro'))
  rows.append(dict(type='plus',noise=noise,sigma_rms=s,architecture=H,rank=Hs[H].rank,seed=seed,n_paths=n,lambda_max=gp.lam,W=z['W'],ess_fraction=z['ess_fraction'],logweight_std=z['logweight_std'],physical_cross_under_guide=a['cross_fraction'],guide_alpha=ALPHA_PLUS[noise],virtual_rank=gp.virtual_rank,inflation_trace=gp.inflation_trace,inflation_fraction_trace=gp.inflation_trace/trplus,inflation_fro=gp.inflation_fro,inflation_fraction_fro=gp.inflation_fro/froplus))
 return rows

def main():
 tasks=[]
 for noise in KINDS:
  for s in SIGMAS:
   for H in ARCHS:
    for seed in SEEDS_MINUS: tasks.append(('minus',(noise,s,H,seed)))
   for seed in SEEDS_PLUS: tasks.append(('plus',(noise,s,seed)))
 print('tasks',len(tasks),flush=True)
 raw=[]
 with ProcessPoolExecutor(max_workers=4) as ex:
  futs=[]
  for typ,arg in tasks:futs.append(ex.submit(task_minus if typ=='minus' else task_plus,arg))
  for i,f in enumerate(as_completed(futs),1):
   x=f.result();raw.extend(x if isinstance(x,list) else [x]);print('done',i,'/',len(futs),flush=True)
 (ROOT/'results/stage4_pic_raw.json').write_text(json.dumps(raw,indent=2))
 with open(ROOT/'results/stage4_pic_raw.csv','w',newline='') as f:
  keys=sorted(set().union(*(r.keys() for r in raw)));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(raw)
 # aggregate plus/minus reps and join Stage3B frozen LQR upper values
 import pandas as pd
 upper=pd.read_csv(ROOT/'data/stage3b_all_results.csv');upper=upper[upper['controller']=='lqr'].copy()
 rows=[]
 for noise in KINDS:
  for s in SIGMAS:
   for H in ARCHS:
    wm=[r for r in raw if r['type']=='minus' and r['noise']==noise and r['sigma_rms']==s and r['architecture']==H]
    wp=[r for r in raw if r['type']=='plus' and r['noise']==noise and r['sigma_rms']==s and r['architecture']==H]
    def agg(rr):
     a=np.array([x['W'] for x in rr]);return float(a.mean()),float(a.std(ddof=1)) if len(a)>1 else 0.,float(np.mean([x['ess_fraction'] for x in rr])),float(np.min([x['ess_fraction'] for x in rr]))
    Wm,Wmsd,Wme,Wmemin=agg(wm);Wp,Wpsd,Wpe,Wpemin=agg(wp)
    u=upper[(upper.noise==noise)&(np.isclose(upper.sigma_rms,s))&(upper.policy==f'{H}_lqr')].iloc[0]
    gp=wp[0];gm=wm[0]
    rows.append(dict(noise=noise,sigma_rms=s,architecture=H,rank=gp['rank'],J=float(u.J_mean),J_se=float(u.J_se),cross_probability=float(u.cross_probability),control_effort=float(u.control_effort_mean),Wminus=Wm,Wminus_rep_sd=Wmsd,Wminus_ess_mean=Wme,Wminus_ess_min=Wmemin,Wplus=Wp,Wplus_rep_sd=Wpsd,Wplus_ess_mean=Wpe,Wplus_ess_min=Wpemin,lambda_max=gp['lambda_max'],deleted_rank=gm['deleted_rank'],virtual_rank=gp['virtual_rank'],inflation_trace=gp['inflation_trace'],inflation_fraction_trace=gp['inflation_fraction_trace'],gap_minus=float(u.J_mean)-Wm,gap_plus=float(u.J_mean)-Wp,Wplus_minus_Wminus=Wp-Wm))
 with open(ROOT/'results/stage4_pic_summary.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (ROOT/'results/stage4_pic_summary.json').write_text(json.dumps(rows,indent=2))
 print('SUMMARY')
 for r in rows: print(r['noise'],r['sigma_rms'],r['architecture'],'J',r['J'],'W-',r['Wminus'],'W+',r['Wplus'],'ess-',r['Wminus_ess_mean'],'ess+',r['Wplus_ess_mean'],flush=True)
if __name__=='__main__':main()
