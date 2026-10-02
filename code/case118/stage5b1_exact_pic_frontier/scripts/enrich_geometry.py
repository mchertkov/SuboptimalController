from pathlib import Path
import sys, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(ROOT/'src'))
from run_stage5b1 import setup, COMBOS, FRACTIONS, R0
rows=[]
for d,n in COMBOS:
 for f in FRACTIONS:
  p,G,g,lmax,lam,delta,proxy=setup(d,n,f); gm=np.zeros(p.n,bool);gm[p.gen_idx]=True
  tol=1e-10*max(1.,float((G**2/lam).max()))
  row=dict(design=d,noise=n,sigma_rms=.15,fraction=f,lambda_max=lmax,lambda_pic=lam,total_generator_authority=float(g.sum()),completion_trace=float(delta.sum()),completion_trace_gen=float(delta[gm].sum()),completion_trace_load=float(delta[~gm].sum()),effective_power_authority_MW=proxy,effective_power_gen_MW=float(100*np.sum(np.sqrt(R0*delta[gm]))),effective_power_load_MW=float(100*np.sum(np.sqrt(R0*delta[~gm]))),virtual_rank=int(np.count_nonzero(delta>tol)),virtual_rank_gen=int(np.count_nonzero(delta[gm]>tol)),virtual_rank_load=int(np.count_nonzero(delta[~gm]>tol)))
  rows.append(row)
df=pd.DataFrame(rows)
for (d,n),idx in df.groupby(['design','noise']).groups.items():
 ids=list(idx);b=df.loc[ids][df.loc[ids,'fraction']==1].iloc[0]
 df.loc[ids,'incremental_trace_above_least']=df.loc[ids,'completion_trace']-b.completion_trace
 df.loc[ids,'incremental_effective_power_proxy_MW']=df.loc[ids,'effective_power_authority_MW']-b.effective_power_authority_MW
df.to_csv(ROOT/'results/stage5b1_geometry.csv',index=False)
print(df.to_string(index=False))
