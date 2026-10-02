from pathlib import Path
import sys,json,csv,math
from types import SimpleNamespace
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import completion_actual_cost_lqr_gain,completion_guide,shared_plus_actions,eval_plus_actions
SIG=.15;R=.2;DT=.0025;T=3.;N=512;SEEDS=[51301,51401];ALPHA=1.0

def setup(p):
 Gh=noise_profile(p,SIG,'homogeneous');Ga=noise_profile(p,SIG,'activity_scaled');A=p.ng/R;qs=[Gh**2,Ga**2];caps=np.min(np.vstack([q[p.gen_idx]/q.sum() for q in qs]),axis=0);g=A*caps/caps.sum();return Gh,Ga,g

def task(arg):
 noise,seed=arg;p,_=build_problem();Gh,Ga,g=setup(p);G=Gh if noise=='homogeneous' else Ga;q=G**2;lam=float(np.min(q[p.gen_idx]/g));fake=SimpleNamespace(lam=lam,Gamma=G);K,_=completion_actual_cost_lqr_gain(p,fake,SmoothCost());guide=completion_guide(p,K,ALPHA,clip=8.);a=shared_plus_actions(p,G,guide,T=T,dt=DT,n_paths=N,seed=seed,cost=SmoothCost());z=eval_plus_actions(a,lam);return dict(design='robust_fixed',noise=noise,sigma_rms=SIG,seed=seed,n_paths=N,dt=DT,lambda_max=lam,guide_alpha=ALPHA,Wplus=z['W'],Z=math.exp(-z['W']/lam),ess_fraction=z['ess_fraction'],logweight_std=z['logweight_std'])

def main():
 rows=[]
 with ProcessPoolExecutor(max_workers=2) as ex:
  fut=[ex.submit(task,j) for j in [(n,s) for n in ['homogeneous','activity_scaled'] for s in SEEDS]]
  for f in as_completed(fut): r=f.result();rows.append(r);print('DONE',r,flush=True)
 rows=sorted(rows,key=lambda r:(r['noise'],r['seed']));agg=[]
 for noise in ['homogeneous','activity_scaled']:
  rr=[r for r in rows if r['noise']==noise];Z=float(np.mean([r['Z'] for r in rr]));lam=rr[0]['lambda_max'];v=np.array([r['Wplus'] for r in rr]);agg.append(dict(design='robust_fixed',noise=noise,sigma_rms=SIG,lambda_max=lam,Wplus=float(-lam*np.log(Z)),Wplus_rep_mean=float(v.mean()),Wplus_rep_sd=float(v.std(ddof=1)),ess_fraction_mean=float(np.mean([r['ess_fraction'] for r in rr])),ess_fraction_min=float(np.min([r['ess_fraction'] for r in rr])),guide_alpha=ALPHA,n_reps=2,n_paths_per_rep=N,dt=DT))
 out=ROOT/'results'
 with open(out/'stage5a2_robust_reps.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with open(out/'stage5a2_robust_values.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
 (out/'stage5a2_robust_values.json').write_text(json.dumps(agg,indent=2));print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
