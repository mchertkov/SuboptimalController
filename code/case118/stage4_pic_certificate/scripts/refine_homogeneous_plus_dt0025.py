from pathlib import Path
import sys,csv,json,math
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import load_architectures,control_inflation_geometry,completion_lqr_gain,completion_guide,shared_plus_actions,eval_plus_actions
R=.2;DT=.0025;T=3.;SIGMAS=[.10,.15,.20];SEEDS=[47101,47201];N=1024;ALPHA=.005

def task(arg):
 s,seed=arg;p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');G=noise_profile(p,s,'homogeneous');gp=control_inflation_geometry(p,hs['H3'],G,R);K=completion_lqr_gain(p,gp);guide=completion_guide(p,K,ALPHA,clip=8.);a=shared_plus_actions(p,G,guide,T=T,dt=DT,n_paths=N,seed=seed,cost=SmoothCost());z=eval_plus_actions(a,gp.lam);return dict(noise='homogeneous',sigma_rms=s,seed=seed,n_paths=N,dt=DT,lambda_max=gp.lam,Wplus=z['W'],ess_fraction=z['ess_fraction'],logweight_std=z['logweight_std'],guide_alpha=ALPHA)

def main():
 rows=[]
 with ProcessPoolExecutor(max_workers=3) as ex:
  fut=[ex.submit(task,j) for j in [(s,se) for s in SIGMAS for se in SEEDS]]
  for i,f in enumerate(as_completed(fut),1):r=f.result();rows.append(r);print('done',i,'/6',r,flush=True)
 rows=sorted(rows,key=lambda r:(r['sigma_rms'],r['seed']));agg=[]
 for s in SIGMAS:
  rr=[r for r in rows if r['sigma_rms']==s];v=np.array([r['Wplus'] for r in rr]);agg.append(dict(noise='homogeneous',sigma_rms=s,Wplus=float(v.mean()),Wplus_rep_sd=float(v.std(ddof=1)),ess_fraction_mean=float(np.mean([r['ess_fraction'] for r in rr])),ess_fraction_min=float(np.min([r['ess_fraction'] for r in rr])),n_paths_per_rep=N,n_reps=2,dt=DT,guide_alpha=ALPHA))
 with open(ROOT/'results/stage4_homogeneous_Wplus_dt0025_raw.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with open(ROOT/'results/stage4_homogeneous_Wplus_dt0025.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
 (ROOT/'results/stage4_homogeneous_Wplus_dt0025.json').write_text(json.dumps(agg,indent=2))
if __name__=='__main__':main()
