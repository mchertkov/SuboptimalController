from pathlib import Path
import sys,json,csv,math
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import load_architectures,control_inflation_geometry,completion_actual_cost_lqr_gain,completion_guide,shared_plus_actions,eval_plus_actions
R=.2;T=3.;DT=.0025;S=.15;SEEDS=[48101,48201]
ARCHS=['H0','H1','H2','H3'];ALPHA={'H0':.75,'H1':1.0,'H2':1.0,'H3':1.25};NP={'H0':1024,'H1':512,'H2':512,'H3':1536}
def task(arg):
 h,seed=arg;p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');G=noise_profile(p,S,'activity_scaled');gp=control_inflation_geometry(p,hs[h],G,R);K,_=completion_actual_cost_lqr_gain(p,gp,SmoothCost());guide=completion_guide(p,K,ALPHA[h],clip=8.);a=shared_plus_actions(p,G,guide,T=T,dt=DT,n_paths=NP[h],seed=seed,cost=SmoothCost());z=eval_plus_actions(a,gp.lam);ef=z['ess_fraction'];wse=float(gp.lam*math.sqrt(max(1/ef-1,0)/NP[h]));return dict(noise='activity_scaled',sigma_rms=S,architecture=h,rank=hs[h].rank,seed=seed,n_paths=NP[h],dt=DT,lambda_max=gp.lam,Wplus=z['W'],Wplus_delta_se=wse,ess_fraction=ef,guide_alpha=ALPHA[h])
def main():
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as ex:
  fut=[ex.submit(task,j) for j in [(h,s) for h in ARCHS for s in SEEDS]]
  for i,f in enumerate(as_completed(fut),1):r=f.result();rows.append(r);print('done',i,'/8',r,flush=True)
 rows=sorted(rows,key=lambda r:(r['architecture'],r['seed']));agg=[]
 for h in ARCHS:
  rr=[r for r in rows if r['architecture']==h];v=np.array([r['Wplus'] for r in rr]);agg.append(dict(noise='activity_scaled',sigma_rms=S,architecture=h,rank=rr[0]['rank'],lambda_max=rr[0]['lambda_max'],Wplus=float(v.mean()),Wplus_rep_sd=float(v.std(ddof=1)),Wplus_delta_se_mean=float(np.mean([r['Wplus_delta_se'] for r in rr])),ess_fraction_mean=float(np.mean([r['ess_fraction'] for r in rr])),ess_fraction_min=float(np.min([r['ess_fraction'] for r in rr])),guide_alpha=ALPHA[h],n_paths_per_rep=NP[h],n_reps=2,dt=DT))
 with open(ROOT/'results/stage4_activity_Wplus_sigma015_dt0025_raw.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with open(ROOT/'results/stage4_activity_Wplus_sigma015_dt0025.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
 (ROOT/'results/stage4_activity_Wplus_sigma015_dt0025.json').write_text(json.dumps(agg,indent=2))
if __name__=='__main__':main()
