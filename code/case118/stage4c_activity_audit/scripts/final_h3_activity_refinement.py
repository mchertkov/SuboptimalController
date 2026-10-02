from pathlib import Path
import sys,json,csv,math
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import *
DT=.0025;N=512;SEEDS=[49701,49702,49703,49704]
fit=json.loads((ROOT/'results/ce_fit_actual.json').read_text());ALPHAS=np.asarray(fit['alpha_smoothed'],float)

def one(seed):
 p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');G=noise_profile(p,.15,'activity_scaled');gp=control_inflation_geometry(p,Hs['H3'],G,.2);K=completion_actual_cost_lqr_gain(p,gp,SmoothCost())[0]
 def guide(t,th,om): b=min(5,int(t/3.*6));return -ALPHAS[b]*(p.reduced_state(th,om)@K.T)
 a=shared_plus_actions(p,G,guide,T=3.,dt=DT,n_paths=N,seed=seed,cost=SmoothCost());z=eval_plus_actions(a,gp.lam)
 return dict(seed=seed,n_paths=N,dt=DT,Wplus=z['W'],ess=z['ess'],ess_fraction=z['ess_fraction'],logweight_std=z['logweight_std'],cross_fraction=a['cross_fraction'],lambda_max=gp.lam)

def main():
 rows=[];raw=ROOT/'results/final_refinement_reps.csv'
 if raw.exists():raw.unlink()
 with ProcessPoolExecutor(max_workers=4) as ex:
  fs=[ex.submit(one,s) for s in SEEDS]
  for f in as_completed(fs):
   r=f.result();rows.append(r);print(r,flush=True)
   with open(raw,'a',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=r.keys());
    if fh.tell()==0:w.writeheader()
    w.writerow(r)
 rows.sort(key=lambda r:r['seed']);v=np.array([r['Wplus'] for r in rows]);ess=np.array([r['ess_fraction'] for r in rows])
 out=dict(stage='4C',target='case118 H3 activity-scaled sigma_rms=0.15 completed PIC value',guide='actual-cost CARE with 6-bin cross-entropy-adapted scalar schedule',guide_alpha_bins=ALPHAS.tolist(),bin_edges=fit['bin_edges'],dt=DT,n_paths_per_rep=N,n_reps=len(rows),total_paths=N*len(rows),seeds=SEEDS,lambda_max=rows[0]['lambda_max'],Wplus_mean=float(v.mean()),Wplus_rep_sd=float(v.std(ddof=1)),Wplus_rep_se=float(v.std(ddof=1)/math.sqrt(len(v))),Wplus_min=float(v.min()),Wplus_max=float(v.max()),ess_fraction_mean=float(ess.mean()),ess_fraction_min=float(ess.min()),ess_fraction_max=float(ess.max()),ess_absolute_mean=float(np.mean([r['ess'] for r in rows])),per_rep=rows)
 (ROOT/'results/STAGE4C_FINAL.json').write_text(json.dumps(out,indent=2));
 with open(ROOT/'results/STAGE4C_FINAL.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=[k for k,vv in out.items() if not isinstance(vv,(list,dict))]);w.writeheader();w.writerow({k:v for k,v in out.items() if not isinstance(v,(list,dict))})
 print('FINAL',json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
