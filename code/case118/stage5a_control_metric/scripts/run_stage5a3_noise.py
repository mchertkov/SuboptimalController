#!/usr/bin/env python3
from pathlib import Path
import argparse,json,math,sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage5a3 import simulate_metric_policy,simulate_static_common

ap=argparse.ArgumentParser();ap.add_argument('--noise',choices=['homogeneous','activity_scaled'],required=True);args=ap.parse_args()
noise_kind=args.noise; sigma=.15; seeds=(51601,51602); pps=96; N=pps*len(seeds); t0=time.time()
p,_=build_problem(); G=noise_profile(p,sigma,noise_kind)
freeze=json.loads((ROOT/'results/STAGE5A_GEOMETRY_FREEZE.json').read_text())
base=np.load(ROOT/'data/stage3a_controller_matrices.npz')
policies={}
policies['uniform']=dict(K=base['LQR_K_H3'],Rdiag=np.ones(p.ng)*.2,alpha=.005)
for design in ['robust_fixed','activity_specific']:
    z=np.load(ROOT/'data'/f'stage5a3_{design}_lqr.npz')
    policies[design]=dict(K=z['K'],Rdiag=z['Rdiag'],alpha=float(z['selected_alpha']))

# exact completed values are analytic lower values; these are the frozen MC point estimates from Stage4/4C/5A2.
jplus={
 ('homogeneous','uniform'):(0.075024198177028,5.992258932968614e-05),
 ('homogeneous','robust_fixed'):(0.07357254905123932,5.638583481921175e-05),
 ('homogeneous','activity_specific'):(0.06836612843647452,0.00019804242477192488),
 ('activity_scaled','uniform'):(0.024956690901558123,1.6615740016435685e-05),
 ('activity_scaled','robust_fixed'):(0.03877712402305069,1.8451381587343483e-05),
 ('activity_scaled','activity_specific'):(0.04057418273984063,2.2211836855163706e-05),
}

def wilson(k,n,z=1.959963984540054):
    ph=k/n;den=1+z*z/n;c=(ph+z*z/(2*n))/den;h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return max(0,c-h),min(1,c+h)

static=simulate_static_common(p,G,T=3.,dt=.0025,n_paths=N,seeds=seeds,paths_per_seed=pps,cost=SmoothCost())
rows=[];raw={'static_J':static['raw_J'],'static_cross':static['raw_cross'].astype(np.int8)}
k=int(static['raw_cross'].sum());lo,hi=wilson(k,N)
rows.append(dict(noise=noise_kind,design='static',alpha=0.0,J=static['J_mean'],J_se=static['J_se'],cross_probability=static['cross_probability'],cross_ci_lo=lo,cross_ci_hi=hi,
                 control_effort=0.0,dmax_q50_deg=static['dmax_q50_deg'],dmax_q90_deg=static['dmax_q90_deg'],dmax_q99_deg=static['dmax_q99_deg'],genomega_q99=static['genomega_q99'],
                 Jplus=np.nan,gap_plus=np.nan,gap_fraction=np.nan,gap_mc_se_approx=np.nan,delta_J_vs_static=0.0,delta_J_vs_static_se=0.0,n_paths=N,seeds=';'.join(map(str,seeds))))
for design,d in policies.items():
    z=simulate_metric_policy(p,d['K'],d['Rdiag'],d['alpha'],G,T=3.,dt=.0025,n_paths=N,seeds=seeds,paths_per_seed=pps,cost=SmoothCost())
    jp,jpsd=jplus[(noise_kind,design)]; gap=z['J_mean']-jp
    diff=z['raw_J']-static['raw_J'];k=int(z['raw_cross'].sum());lo,hi=wilson(k,N)
    rows.append(dict(noise=noise_kind,design=design,alpha=d['alpha'],J=z['J_mean'],J_se=z['J_se'],cross_probability=z['cross_probability'],cross_ci_lo=lo,cross_ci_hi=hi,
                     control_effort=z['control_effort_mean'],dmax_q50_deg=z['dmax_q50_deg'],dmax_q90_deg=z['dmax_q90_deg'],dmax_q99_deg=z['dmax_q99_deg'],genomega_q99=z['genomega_q99'],
                     Jplus=jp,gap_plus=gap,gap_fraction=gap/z['J_mean'],gap_mc_se_approx=math.sqrt(z['J_se']**2+jpsd**2),
                     delta_J_vs_static=float(diff.mean()),delta_J_vs_static_se=float(diff.std(ddof=1)/math.sqrt(N)),n_paths=N,seeds=';'.join(map(str,seeds))))
    raw[f'{design}_J']=z['raw_J'];raw[f'{design}_cross']=z['raw_cross'].astype(np.int8);raw[f'{design}_dmax_deg']=z['raw_dmax_deg'];raw[f'{design}_genomega']=z['raw_genomega']

df=pd.DataFrame(rows);outdir=ROOT/'results/stage5a3_eval';outdir.mkdir(parents=True,exist_ok=True)
df.to_csv(outdir/f'{noise_kind}.csv',index=False);np.savez_compressed(ROOT/'data'/f'stage5a3_eval_{noise_kind}.npz',**raw)
meta={'stage':'5A3_metric_aware_controller_evaluation','noise':noise_kind,'sigma_rms':sigma,'seeds':list(seeds),'paths_per_seed':pps,'n_paths':N,'dt':.0025,'T':3.0,'runtime_s':time.time()-t0,'common_random_numbers':True,'saturation':'none'}
(outdir/f'{noise_kind}_meta.json').write_text(json.dumps(meta,indent=2))
print(df[['design','alpha','J','J_se','Jplus','gap_plus','gap_fraction','cross_probability','control_effort','delta_J_vs_static','delta_J_vs_static_se']].to_string(index=False))
print(json.dumps(meta,indent=2))
