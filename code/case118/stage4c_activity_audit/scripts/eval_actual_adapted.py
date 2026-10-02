from pathlib import Path
import sys,json,csv,numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import *
DT=.0025;N=256;SEEDS=[49601,49602,49603,49604]
fit=json.loads((ROOT/'results/ce_fit_actual.json').read_text())
CANDS={'actual_const_1p25':[1.25]*6,'actual_ce_smoothed':fit['alpha_smoothed'],'actual_ce_raw':fit['alpha_raw']}
def one(arg):
 name,alphas,seed=arg
 p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');G=noise_profile(p,.15,'activity_scaled');gp=control_inflation_geometry(p,Hs['H3'],G,.2);K=completion_actual_cost_lqr_gain(p,gp,SmoothCost())[0];aa=np.asarray(alphas,float)
 def guide(t,th,om): b=min(5,int(t/3.*6)); return -aa[b]*(p.reduced_state(th,om)@K.T)
 a=shared_plus_actions(p,G,guide,T=3.,dt=DT,n_paths=N,seed=seed,cost=SmoothCost());z=eval_plus_actions(a,gp.lam)
 return dict(candidate=name,seed=seed,W=z['W'],ess_fraction=z['ess_fraction'],ess=z['ess'],logweight_std=z['logweight_std'],cross_fraction=a['cross_fraction'])
def main():
 tasks=[(n,a,s) for n,a in CANDS.items() for s in SEEDS];rows=[]
 with ProcessPoolExecutor(max_workers=4) as ex:
  fs=[ex.submit(one,t) for t in tasks]
  for f in as_completed(fs):r=f.result();rows.append(r);print(r,flush=True)
 rows.sort(key=lambda r:(r['candidate'],r['seed']));
 with open(ROOT/'results/actual_adapted_eval.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 agg=[]
 for n in CANDS:
  rr=[r for r in rows if r['candidate']==n];v=np.array([r['W'] for r in rr]);agg.append(dict(candidate=n,W_mean=float(v.mean()),W_rep_sd=float(v.std(ddof=1)),W_rep_se=float(v.std(ddof=1)/np.sqrt(len(v))),ess_fraction_mean=float(np.mean([r['ess_fraction'] for r in rr])),ess_fraction_min=float(np.min([r['ess_fraction'] for r in rr])),logweight_std_mean=float(np.mean([r['logweight_std'] for r in rr]))))
 agg.sort(key=lambda r:r['W_rep_sd']);print('AGG');[print(x) for x in agg]
 with open(ROOT/'results/actual_adapted_eval_agg.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=agg[0]);w.writeheader();w.writerows(agg)
if __name__=='__main__':main()
