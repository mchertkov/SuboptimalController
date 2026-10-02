#!/usr/bin/env python3
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
ok=True
for noise in ['homogeneous','activity_scaled']:
    d=pd.read_csv(ROOT/'results'/f'stage5b2_design_{noise}.csv')
    gains=np.load(ROOT/'data'/f'stage5b2_gains_{noise}.npz')
    gplus=gains['gplus'];busids=np.asarray(__import__('sys').path and [],dtype=int)
    # map bus id from case118 module
    sys.path.insert(0,str(ROOT/'src'))
    from case118_stage3a import build_problem
    p,_=build_problem(); idx={int(b):i for i,b in enumerate(p.bus_ids)}
    for k in [5,10,20,40,64]:
        a=d[(d.k==k)&(d.method=='noise')].iloc[0];b=d[(d.k==k)&(d.method=='shadow')].iloc[0]
        assert abs(a.added_authority_sum-b.added_authority_sum)<1e-9
        for r in [a,b]:
            ids=[int(x) for x in str(r.load_bus_ids).split(';') if x]
            ga=[float(x) for x in str(r.load_authorities).split(';') if x]
            assert len(ids)==k and len(ga)==k
            for bid,g in zip(ids,ga): assert g <= gplus[idx[bid]]*(1+2e-9)+1e-12
    a=d[(d.k==64)&(d.method=='noise')].iloc[0];b=d[(d.k==64)&(d.method=='shadow')].iloc[0]
    assert abs(a.added_authority_sum-b.added_authority_sum)<1e-10
    ev=pd.read_csv(ROOT/'results'/f'stage5b2_eval_{noise}.csv') if (ROOT/'results'/f'stage5b2_eval_{noise}.csv').exists() else None
# main aggregate checks
f=pd.read_csv(ROOT/'results/stage5b2_sparse_frontier.csv')
for noise in ['homogeneous','activity_scaled']:
    s=f[f.noise==noise]
    assert len(s)==11
    n=s[(s.k==64)&(s.method=='noise')].iloc[0];q=s[(s.k==64)&(s.method=='shadow')].iloc[0]
    assert abs(n.J-q.J)<1e-15 and abs(n.added_authority_sum-q.added_authority_sum)<1e-12
# alpha spot checks
for name in ['homogeneous_noise_k64','activity_scaled_noise_k64','homogeneous_shadow_k20','activity_scaled_shadow_k20']:
    a=pd.read_csv(ROOT/'results/alpha_scan'/f'{name}.csv')
    best=float(a.loc[a.J_mean.idxmin(),'alpha']); assert abs(best-.005)<1e-12,(name,best)
# activity dynamics-aware placement must beat noise-only in the evaluated matched-budget CRN comparison k<64
cmp=pd.read_csv(ROOT/'results/stage5b2_shadow_vs_noise.csv')
act=cmp[(cmp.noise=='activity_scaled')&(cmp.k<64)]
assert np.all(act.delta_J_shadow_minus_noise.values<0)
print('Stage 5B2 verification PASSED')
