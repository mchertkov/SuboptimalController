#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage5b2 import scan_sparse_scales
ap=argparse.ArgumentParser();ap.add_argument('--noise',required=True);ap.add_argument('--method',required=True);ap.add_argument('--k',type=int,required=True);args=ap.parse_args();noise=args.noise;method=args.method;k=args.k
p,_=build_problem();G=noise_profile(p,.15,noise);freeze=json.loads((ROOT/'data/STAGE5A_GEOMETRY_FREEZE.json').read_text());design='uniform' if noise=='homogeneous' else 'activity_specific';gg=np.asarray(freeze['designs'][design]['authority'],float);z=np.load(ROOT/'data'/f'stage5b2_gains_{noise}.npz');df=pd.read_csv(ROOT/'results'/f'stage5b2_design_{noise}.csv');r=df[(df.method==method)&(df.k==k)].iloc[0];idx={int(b):i for i,b in enumerate(p.bus_ids)};lb=np.array([idx[int(b)] for b in str(r.load_bus_ids).split(';') if b],int);la=np.array([float(x) for x in str(r.load_authorities).split(';') if x],float);K=z[str(r.K_key)]
rows=scan_sparse_scales(p,G,gg,lb,la,K,[.0025,.005,.01,.02],seeds=(53101,53102),paths_per_seed=16,cost=SmoothCost());out=ROOT/'results/alpha_scan';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/f'{noise}_{method}_k{k}.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
