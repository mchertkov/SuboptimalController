#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage5a3 import metric_lqr_gain,scan_metric_lqr_scales

ap=argparse.ArgumentParser();ap.add_argument('--design',choices=['robust_fixed','activity_specific'],required=True);args=ap.parse_args()
scales=[0.000625,0.00125,0.0025,0.005,0.01,0.02]
seeds=(51501,51502);n_per_seed=16; sigma=.15;t0=time.time()
p,_=build_problem();freeze=json.loads((ROOT/'results/STAGE5A_GEOMETRY_FREEZE.json').read_text())
g=np.array(freeze['designs'][args.design]['authority'],float);Rdiag=1/g
K,meta=metric_lqr_gain(p,Rdiag)
rows=[]
for noise_kind in ['homogeneous','activity_scaled']:
    G=noise_profile(p,sigma,noise_kind)
    z=scan_metric_lqr_scales(p,K,Rdiag,scales,G,T=3.,dt=.0025,n_per_seed=n_per_seed,seeds=seeds,cost=SmoothCost())
    for r in z: rows.append(dict(design=args.design,noise=noise_kind,**r))
df=pd.DataFrame(rows)
piv=df.pivot_table(index='scale',columns='noise',values='J_mean')
piv['J_avg']=.5*(piv['homogeneous']+piv['activity_scaled'])
best_scale=float(piv['J_avg'].idxmin())
# Require an interior selected point; report if grid edge was selected.
edge=best_scale in [min(scales),max(scales)]
out=dict(stage='5A3_metric_aware_lqr_tuning',design=args.design,sigma_rms=sigma,
         training_seeds=list(seeds),paths_per_seed_per_noise=n_per_seed,n_paths_per_noise=n_per_seed*len(seeds),
         scales=scales,selection_objective='equal-weight average smooth J across homogeneous and activity-scaled noise',
         selected_alpha=best_scale,selected_at_grid_edge=edge,
         selected_J_hom=float(piv.loc[best_scale,'homogeneous']),selected_J_act=float(piv.loc[best_scale,'activity_scaled']),
         selected_J_avg=float(piv.loc[best_scale,'J_avg']),runtime_s=time.time()-t0,
         Rdiag_min=float(Rdiag.min()),Rdiag_max=float(Rdiag.max()))
outdir=ROOT/'results/stage5a3_tuning';outdir.mkdir(parents=True,exist_ok=True)
df.to_csv(outdir/f'{args.design}_scan.csv',index=False)
(outdir/f'{args.design}_selection.json').write_text(json.dumps(out,indent=2))
np.savez_compressed(ROOT/'data'/f'stage5a3_{args.design}_lqr.npz',K=K,Rdiag=Rdiag,g=g,selected_alpha=np.array(best_scale))
print(df[['noise','scale','J_mean','J_se','cross_probability','control_effort_mean']].to_string(index=False))
print(json.dumps(out,indent=2))
