#!/usr/bin/env python3
from pathlib import Path
import argparse,json,math,sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage5b2 import simulate_sparse_policy,simulate_generator_baseline
ap=argparse.ArgumentParser();ap.add_argument('--noise',choices=['homogeneous','activity_scaled'],required=True);ap.add_argument('--method',choices=['baseline','noise','shadow'],required=True);ap.add_argument('--k',type=int,default=0);args=ap.parse_args()
noise=args.noise;method=args.method;k=int(args.k);sigma=.15;seeds=(53201,53202);pps=96;alpha=.005;t0=time.time()
p,_=build_problem();Gamma=noise_profile(p,sigma,noise);freeze=json.loads((ROOT/'data/STAGE5A_GEOMETRY_FREEZE.json').read_text());design='uniform' if noise=='homogeneous' else 'activity_specific';gen_auth=np.asarray(freeze['designs'][design]['authority'],float);z=np.load(ROOT/'data'/f'stage5b2_gains_{noise}.npz')
if method=='baseline':
    res=simulate_generator_baseline(p,Gamma,gen_auth,z['K_baseline'],alpha=alpha,seeds=seeds,paths_per_seed=pps,cost=SmoothCost());added=peff=bf=0.;buses='';auth=''
else:
    if k not in (5,10,20,40,64):raise ValueError(k)
    df=pd.read_csv(ROOT/'results'/f'stage5b2_design_{noise}.csv');r=df[(df.method==method)&(df.k==k)].iloc[0]
    ids=np.array([int(x) for x in str(r.load_bus_ids).split(';') if x],int);idx={int(b):i for i,b in enumerate(p.bus_ids)};lb=np.array([idx[int(b)] for b in ids],int);ga=np.array([float(x) for x in str(r.load_authorities).split(';') if x],float)
    K=z[str(r.K_key)];res=simulate_sparse_policy(p,Gamma,gen_auth,lb,ga,K,alpha=alpha,seeds=seeds,paths_per_seed=pps,cost=SmoothCost());added=float(r.added_authority_sum);peff=float(r.effective_power_proxy_MW);bf=float(r.budget_fraction_of_full);buses=str(r.load_bus_ids);auth=str(r.load_authorities)
N=res['n_paths'];count=int(res['raw_cross'].sum());zv=1.959963984540054;ph=count/N;den=1+zv*zv/N;c=(ph+zv*zv/(2*N))/den;h=zv*math.sqrt(ph*(1-ph)/N+zv*zv/(4*N*N))/den
row=dict(noise=noise,baseline_metric=design,method=method,k=k,alpha=alpha,J=float(res['J_mean']),J_se=float(res['J_se']),crossing=float(res['cross_probability']),cross_ci_lo=max(0,c-h),cross_ci_hi=min(1,c+h),control_effort=float(res['control_effort_mean']),dmax_q50_deg=float(res['dmax_q50_deg']),dmax_q90_deg=float(res['dmax_q90_deg']),dmax_q99_deg=float(res['dmax_q99_deg']),genomega_q99=float(res['genomega_q99']),gen_command_l2sq_mean=float(res['gen_command_l2sq_mean']),load_command_l2sq_mean=float(res['load_command_l2sq_mean']),added_authority_sum=added,effective_power_proxy_MW=peff,budget_fraction_of_full=bf,load_bus_ids=buses,load_authorities=auth,n_paths=N,seeds=';'.join(map(str,seeds)),runtime_s=time.time()-t0)
out=ROOT/'results/chunks';out.mkdir(parents=True,exist_ok=True);stem=f'{noise}_{method}_k{k}';(out/f'{stem}.json').write_text(json.dumps(row,indent=2));np.savez_compressed(out/f'{stem}.npz',J=res['raw_J'],cross=res['raw_cross'].astype(np.int8),dmax_deg=res['raw_dmax_deg'],genomega=res['raw_genomega'],ctrl=res['raw_ctrl'])
print(json.dumps(row,indent=2))
