from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/'results/stage5b1_frontier_final.csv')
assert len(df)==20
for (d,n),s in df.groupby(['design','noise']):
 s=s.sort_values('fraction',ascending=False)
 assert np.all(np.diff(s.lambda_pic.values)<0)
 assert np.all(np.diff(s.completion_trace.values)>0)
 assert np.all(np.diff(s.effective_power_authority_MW.values)>0)
 assert np.all(np.diff(s.Wplus.values)<0), (d,n,s.Wplus.values)
# scenario-matched least-completion must add only load-side authority (up to roundoff)
g=pd.read_csv(ROOT/'results/stage5b1_geometry.csv')
for d,n in [('uniform','homogeneous'),('activity_specific','activity_scaled')]:
 r=g[(g.design==d)&(g.noise==n)&(g.fraction==1.0)].iloc[0]
 assert abs(r.completion_trace_gen)<1e-8
 assert int(r.virtual_rank_load)==64
print('Stage5B1 verification passed: monotone exact-PIC frontiers and matched-baseline geometry checks.')
