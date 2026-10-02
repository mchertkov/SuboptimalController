#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage5a3 import simulate_metric_policy
ap=argparse.ArgumentParser();ap.add_argument('--rep',type=int,required=True);args=ap.parse_args()
rep=int(args.rep); seed1=51701+2*rep;seed2=seed1+1; pps=192;N=384;t0=time.time()
p,_=build_problem();G=noise_profile(p,.15,'activity_scaled');z=np.load(ROOT/'data/stage5a3_activity_specific_lqr.npz')
r=simulate_metric_policy(p,z['K'],z['Rdiag'],float(z['selected_alpha']),G,T=3.,dt=.0025,n_paths=N,seeds=(seed1,seed2),paths_per_seed=pps,cost=SmoothCost())
J=np.asarray(r['raw_J']);out=dict(rep=rep,seeds=[seed1,seed2],n_paths=N,J_mean=float(J.mean()),J_sd=float(J.std(ddof=1)),J_sum=float(J.sum()),J_sumsq=float(np.sum(J*J)),cross_count=int(np.sum(r['raw_cross'])),control_effort_mean=r['control_effort_mean'],dmax_q99_deg=r['dmax_q99_deg'],genomega_q99=r['genomega_q99'],runtime_s=time.time()-t0)
outdir=ROOT/'results/stage5a3_eval/activity_specific_act_audit';outdir.mkdir(parents=True,exist_ok=True);(outdir/f'rep_{rep:02d}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
