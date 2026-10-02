from pathlib import Path
import sys,json,time,argparse
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import completion_actual_cost_lqr_gain,completion_guide,shared_plus_actions,eval_plus_actions

def allocations(p,Gh,Ga,R=.2):
 A=p.ng/R
 qga=Ga[p.gen_idx]**2
 act=A*qga/qga.sum()
 qs=[Gh**2,Ga**2];caps=np.min(np.vstack([q[p.gen_idx]/q.sum() for q in qs]),axis=0);rob=A*caps/caps.sum()
 return {'activity_specific':act,'robust_fixed':rob}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--design',choices=['robust_fixed','activity_specific'],required=True);ap.add_argument('--noise',choices=['homogeneous','activity_scaled'],required=True);ap.add_argument('--seed',type=int,default=51101);ap.add_argument('--n-paths',type=int,default=128);args=ap.parse_args()
 p,_=build_problem();Gh=noise_profile(p,.15,'homogeneous');Ga=noise_profile(p,.15,'activity_scaled');Gamma=Gh if args.noise=='homogeneous' else Ga;g=allocations(p,Gh,Ga)[args.design];q=Gamma**2;lam=float(np.min(q[p.gen_idx]/g));fake=SimpleNamespace(lam=lam,Gamma=Gamma);K,_=completion_actual_cost_lqr_gain(p,fake,SmoothCost())
 rows=[]
 for a in [0.5,0.75,1.0,1.25,1.5]:
  t=time.time();guide=completion_guide(p,K,a,clip=8.);actions=shared_plus_actions(p,Gamma,guide,T=3.,dt=.0025,n_paths=args.n_paths,seed=args.seed,cost=SmoothCost());z=eval_plus_actions(actions,lam);rows.append(dict(alpha=a,Wplus=float(z['W']),ess_fraction=float(z['ess_fraction']),logweight_std=float(z['logweight_std']),elapsed_s=time.time()-t));print('ALPHA',a,rows[-1],flush=True)
 med=float(np.median([r['Wplus'] for r in rows]));top=sorted(rows,key=lambda r:r['ess_fraction'],reverse=True)[:2];best=min(top,key=lambda r:abs(r['Wplus']-med));out={'design':args.design,'noise':args.noise,'seed':args.seed,'n_paths':args.n_paths,'lambda_max':lam,'rows':rows,'median_W':med,'selected_alpha':best['alpha'],'selection_rule':'among top-2 ESS candidates choose Wplus closest to median scan W'}
 d=ROOT/'results'/'stage5a2_tuning';d.mkdir(parents=True,exist_ok=True);path=d/f"tune_{args.design}_{args.noise}.json";path.write_text(json.dumps(out,indent=2)+'\n');print('SELECTED',best['alpha'],'SAVED',path,flush=True)
if __name__=='__main__':main()
