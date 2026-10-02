from pathlib import Path
import json, csv, math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
geom=pd.read_csv(ROOT/'results/stage5b1_geometry.csv')
# Frozen baseline values from Stage 5A2/4C
base=pd.read_csv('/mnt/data/CASE118_STAGE5A3_METRIC_AWARE_CONTROL_09_24_26/results/stage5a2_metric_comparison.csv')
combos=[('uniform','homogeneous'),('activity_specific','activity_scaled'),('robust_fixed','homogeneous'),('robust_fixed','activity_scaled')]
rows=[]
for d,n in combos:
    gsub=geom[(geom.design==d)&(geom.noise==n)].sort_values('fraction',ascending=False)
    for _,g in gsub.iterrows():
        f=float(g.fraction); lam=float(g.lambda_pic)
        if abs(f-1.0)<1e-12:
            b=base[(base.design==d)&(base.noise==n)].iloc[0]
            rows.append({**g.to_dict(),'Wplus':float(b.Wplus),'Wplus_rep_sd':None if pd.isna(b.Wplus_rep_sd) else float(b.Wplus_rep_sd),'Wplus_rep_se':None,'ess_fraction_mean':None if pd.isna(b.ess_fraction_mean) else float(b.ess_fraction_mean),'ess_fraction_min':None,'n_total_paths':None,'n_reps':None,'source':'frozen Stage5A2/Stage4 baseline'})
            continue
        files=list((ROOT/'results/chunks').glob(f'{d}_{n}_f{f:g}_s*_n*.json'))
        rr=[]
        for p in files:
            z=json.loads(p.read_text())
            # production rule: use alpha=1 only; other alpha scans were guide-tuning diagnostics
            if abs(float(z['guide_alpha'])-1.0)>1e-12: continue
            # exclude seed 52112 used in guide tuning
            if int(z['seed'])==52112: continue
            rr.append(z)
        if not rr: raise RuntimeError((d,n,f))
        ns=np.array([r['n_paths'] for r in rr],float); Ws=np.array([r['Wplus'] for r in rr],float)
        # pool unbiased normalizer estimates Z=E exp(-S/lambda) across independent batches by path count
        Z=np.exp(-Ws/lam); Zp=float(np.sum(ns*Z)/np.sum(ns)); Wp=float(-lam*math.log(Zp))
        sd=float(Ws.std(ddof=1)) if len(Ws)>1 else float('nan'); se=sd/math.sqrt(len(Ws)) if len(Ws)>1 else float('nan')
        ess=np.array([r['ess_fraction'] for r in rr],float)
        rows.append({**g.to_dict(),'Wplus':Wp,'Wplus_rep_sd':sd,'Wplus_rep_se':se,'ess_fraction_mean':float(np.average(ess,weights=ns)),'ess_fraction_min':float(ess.min()),'n_total_paths':int(ns.sum()),'n_reps':len(rr),'source':'Stage5B1 independent importance-sampling batches pooled at normalizer level'})
out=pd.DataFrame(rows)
# Derived improvement relative to each frontier's least-inflated exact-PIC point
for (d,n),idx in out.groupby(['design','noise']).groups.items():
    ids=list(idx); b=out.loc[ids][out.loc[ids,'fraction']==1.0].iloc[0]
    out.loc[ids,'Wplus_reduction_from_least']=float(b.Wplus)-out.loc[ids,'Wplus']
    out.loc[ids,'Wplus_reduction_fraction']=out.loc[ids,'Wplus_reduction_from_least']/float(b.Wplus)
    out.loc[ids,'added_proxy_MW_from_least']=out.loc[ids,'effective_power_authority_MW']-float(b.effective_power_authority_MW)
out.to_csv(ROOT/'results/stage5b1_frontier_final.csv',index=False)
# compact JSON
(ROOT/'results/STAGE5B1_FRONTIER_FINAL.json').write_text(json.dumps(out.replace({np.nan:None}).to_dict(orient='records'),indent=2))
print(out[['design','noise','fraction','lambda_pic','completion_trace','effective_power_authority_MW','Wplus','Wplus_rep_sd','ess_fraction_mean','n_total_paths','Wplus_reduction_fraction']].to_string(index=False))
