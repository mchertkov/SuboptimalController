from pathlib import Path
import sys,json,math,time,argparse
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import completion_actual_cost_lqr_gain,completion_guide,shared_plus_actions,eval_plus_actions

def allocations(p,Gh,Ga,R=.2):
 A=p.ng/R
 qga=Ga[p.gen_idx]**2
 act=A*qga/qga.sum()
 qs=[Gh**2,Ga**2]
 caps=np.min(np.vstack([q[p.gen_idx]/q.sum() for q in qs]),axis=0)
 robust=A*caps/caps.sum()
 return {'activity_specific':act,'robust_fixed':robust}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--design',choices=['robust_fixed','activity_specific'],required=True);ap.add_argument('--noise',choices=['homogeneous','activity_scaled'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--alpha',type=float,required=True);ap.add_argument('--n-paths',type=int,default=512);args=ap.parse_args();t0=time.time()
 p,_=build_problem();Gh=noise_profile(p,.15,'homogeneous');Ga=noise_profile(p,.15,'activity_scaled');Gamma=Gh if args.noise=='homogeneous' else Ga;g=allocations(p,Gh,Ga)[args.design];q=Gamma**2;lam=float(np.min(q[p.gen_idx]/g));fake=SimpleNamespace(lam=lam,Gamma=Gamma);K,_=completion_actual_cost_lqr_gain(p,fake,SmoothCost());guide=completion_guide(p,K,args.alpha,clip=8.);actions=shared_plus_actions(p,Gamma,guide,T=3.,dt=.0025,n_paths=args.n_paths,seed=args.seed,cost=SmoothCost());z=eval_plus_actions(actions,lam)
 row=dict(design=args.design,noise=args.noise,sigma_rms=.15,seed=args.seed,n_paths=args.n_paths,dt=.0025,lambda_max=lam,guide_alpha=args.alpha,Wplus=float(z['W']),Z=float(math.exp(-z['W']/lam)),ess_fraction=float(z['ess_fraction']),logweight_std=float(z['logweight_std']),elapsed_s=float(time.time()-t0))
 d=ROOT/'results'/'stage5a2_chunks';d.mkdir(parents=True,exist_ok=True);path=d/f"{args.design}_{args.noise}_seed{args.seed}.json";path.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2),flush=True);print('SAVED',path,flush=True)
if __name__=='__main__':main()
