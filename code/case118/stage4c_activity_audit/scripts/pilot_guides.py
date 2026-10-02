from pathlib import Path
import sys,csv,json
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import load_architectures,control_inflation_geometry,completion_lqr_gain,completion_actual_cost_lqr_gain,completion_guide,shared_plus_actions,eval_plus_actions
SIG=.15;DT=.0025;N=128;SEEDS=[49101,49102]
CANDS=[('generic',a) for a in [.0125,.0175,.0225,.0275]]+[('actual',a) for a in [.75,1.,1.25,1.5]]
def one(t):
 fam,a,seed=t
 p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz')
 G=noise_profile(p,SIG,'activity_scaled');gp=control_inflation_geometry(p,Hs['H3'],G,.2)
 K=completion_lqr_gain(p,gp) if fam=='generic' else completion_actual_cost_lqr_gain(p,gp,SmoothCost())[0]
 guide=completion_guide(p,K,a,clip=8.)
 act=shared_plus_actions(p,G,guide,T=3.,dt=DT,n_paths=N,seed=seed,cost=SmoothCost());z=eval_plus_actions(act,gp.lam)
 return dict(family=fam,alpha=a,seed=seed,W=z['W'],ess_fraction=z['ess_fraction'],ess=z['ess'],logweight_std=z['logweight_std'],cross_fraction=act['cross_fraction'])
def main():
 rows=[]
 tasks=[(fam,a,s) for fam,a in CANDS for s in SEEDS]
 with ProcessPoolExecutor(max_workers=4) as ex:
  fs=[ex.submit(one,t) for t in tasks]
  for i,f in enumerate(as_completed(fs),1):
   r=f.result();rows.append(r);print(i,len(tasks),r,flush=True)
 rows.sort(key=lambda r:(r['family'],r['alpha'],r['seed']))
 with open(ROOT/'results/pilot_guides.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 import numpy as np
 agg=[]
 for fam,a in CANDS:
  rr=[r for r in rows if r['family']==fam and r['alpha']==a]
  agg.append(dict(family=fam,alpha=a,W_mean=float(np.mean([r['W'] for r in rr])),W_range=float(np.ptp([r['W'] for r in rr])),ess_fraction_mean=float(np.mean([r['ess_fraction'] for r in rr])),ess_fraction_min=float(np.min([r['ess_fraction'] for r in rr])),logweight_std_mean=float(np.mean([r['logweight_std'] for r in rr]))))
 agg.sort(key=lambda r:-r['ess_fraction_mean'])
 with open(ROOT/'results/pilot_guides_agg.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=agg[0].keys());w.writeheader();w.writerows(agg)
 print('AGG');[print(x) for x in agg]
if __name__=='__main__':main()
