from pathlib import Path
import argparse,json,sys,time,os
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3b import POLICY_NAMES,simulate_policy,paired_vs_static
ap=argparse.ArgumentParser();ap.add_argument('--sigma',type=float,required=True);ap.add_argument('--noise',choices=['homogeneous','activity_scaled'],required=True);ap.add_argument('--workers',type=int,default=9);args=ap.parse_args()
t0=time.time();kwargs=dict(matrix_path=str(ROOT/'data/stage3a_controller_matrices.npz'),sigma=args.sigma,noise_kind=args.noise,seeds=(31301,31302),paths_per_seed=96,T=3.,dt=.0025,control_weight=.2)
rows_by={};raws={}
with ProcessPoolExecutor(max_workers=args.workers) as ex:
    futs={ex.submit(simulate_policy,policy_name=p,**kwargs):p for p in POLICY_NAMES}
    for f in as_completed(futs):
        p=futs[f];row,raw=f.result();rows_by[p]=row;raws[p]=raw;print('done',p,row['J_mean'],row['cross_probability'],flush=True)
rows=[rows_by[p] for p in POLICY_NAMES];paired=paired_vs_static(rows,raws)
tag=f"{args.noise}_s{args.sigma:.2f}".replace('.','p')
pd.DataFrame(rows).to_csv(ROOT/'results'/f'stage3b_{tag}.csv',index=False);pd.DataFrame(paired).to_csv(ROOT/'results'/f'stage3b_paired_{tag}.csv',index=False)
np.savez_compressed(ROOT/'data'/f'stage3b_raw_{tag}.npz',**{f'{p}_{k}':v for p,r in raws.items() for k,v in r.items()})
meta={'sigma_rms':args.sigma,'noise':args.noise,'runtime_s':time.time()-t0,'seeds':[31301,31302],'paths_per_seed':96,'n_paths':192,'workers':args.workers}
(ROOT/'results'/f'stage3b_{tag}_meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2));print(pd.DataFrame(rows)[['policy','J_mean','J_se','cross_probability','control_effort_mean','dmax_q99_deg','genomega_q99']].to_string(index=False))
