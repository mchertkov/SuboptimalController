from pathlib import Path
import sys,json,math,time,argparse
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import completion_actual_cost_lqr_gain,completion_guide,shared_plus_actions,eval_plus_actions

def setup(p,sig=0.15,R=0.2):
    Gh=noise_profile(p,sig,'homogeneous')
    Ga=noise_profile(p,sig,'activity_scaled')
    A=p.ng/R
    qs=[Gh**2,Ga**2]
    caps=np.min(np.vstack([q[p.gen_idx]/q.sum() for q in qs]),axis=0)
    g=A*caps/caps.sum()
    return Gh,Ga,g

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--noise',choices=['homogeneous','activity_scaled'],required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--n-paths',type=int,default=512)
    ap.add_argument('--dt',type=float,default=0.0025)
    ap.add_argument('--alpha',type=float,default=1.0)
    args=ap.parse_args()
    tstart=time.time()
    p,_=build_problem(); Gh,Ga,g=setup(p)
    Gamma=Gh if args.noise=='homogeneous' else Ga
    q=Gamma**2
    lam=float(np.min(q[p.gen_idx]/g))
    fake=SimpleNamespace(lam=lam,Gamma=Gamma)
    K,_=completion_actual_cost_lqr_gain(p,fake,SmoothCost())
    guide=completion_guide(p,K,args.alpha,clip=8.)
    actions=shared_plus_actions(p,Gamma,guide,T=3.,dt=args.dt,n_paths=args.n_paths,seed=args.seed,cost=SmoothCost())
    z=eval_plus_actions(actions,lam)
    row=dict(design='robust_fixed',noise=args.noise,sigma_rms=0.15,seed=args.seed,n_paths=args.n_paths,dt=args.dt,
             lambda_max=lam,guide_alpha=args.alpha,Wplus=float(z['W']),Z=float(math.exp(-z['W']/lam)),
             ess_fraction=float(z['ess_fraction']),logweight_std=float(z['logweight_std']),elapsed_s=float(time.time()-tstart))
    out=ROOT/'results'/'stage5a2_chunks';out.mkdir(parents=True,exist_ok=True)
    path=out/f"robust_{args.noise}_seed{args.seed}.json"
    path.write_text(json.dumps(row,indent=2)+"\n")
    print(json.dumps(row,indent=2),flush=True)
    print('SAVED',path,flush=True)
if __name__=='__main__': main()
