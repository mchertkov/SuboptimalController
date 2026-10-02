#!/usr/bin/env python3
from pathlib import Path
import json,math,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
rows=[]; paired=[]; direct=[]
for noise in ['homogeneous','activity_scaled']:
    ch=ROOT/'results/chunks'
    base=json.loads((ch/f'{noise}_baseline_k0.json').read_text()); zb=np.load(ch/f'{noise}_baseline_k0.npz'); J0=zb['J']; c0=zb['cross'].astype(bool)
    jplus=0.075024198177028 if noise=='homogeneous' else 0.04057418273984063
    base.update(Jplus_common_oracle=jplus,gap_to_common_Jplus=base['J']-jplus,gap_fraction=(base['J']-jplus)/base['J'],delta_J_vs_k0=0.,delta_J_vs_k0_se=0.,relative_delta_J_pct=0.,delta_cross_vs_k0=0.)
    rows.append(base)
    cache={}
    for method in ['noise','shadow']:
      for k in [5,10,20,40,64]:
        # full completion identical; reuse noise run for shadow k64
        stem=f'{noise}_{method}_k{k}'
        if method=='shadow' and k==64:
            src=f'{noise}_noise_k64'; row=json.loads((ch/f'{src}.json').read_text()).copy(); row['method']='shadow'
            z=np.load(ch/f'{src}.npz')
        else:
            row=json.loads((ch/f'{stem}.json').read_text()); z=np.load(ch/f'{stem}.npz')
        J=z['J'];cr=z['cross'].astype(bool);d=J-J0;dc=cr.astype(float)-c0.astype(float)
        row.update(Jplus_common_oracle=jplus,gap_to_common_Jplus=row['J']-jplus,gap_fraction=(row['J']-jplus)/row['J'],
                   delta_J_vs_k0=float(d.mean()),delta_J_vs_k0_se=float(d.std(ddof=1)/math.sqrt(len(d))),relative_delta_J_pct=float(100*d.mean()/J0.mean()),delta_cross_vs_k0=float(dc.mean()))
        rows.append(row); cache[(method,k)]=(row,J,cr)
        paired.append(dict(noise=noise,method=method,k=k,delta_J_mean=float(d.mean()),delta_J_se=float(d.std(ddof=1)/math.sqrt(len(d))),
                           delta_J_ci95_lo=float(d.mean()-1.95996398454*d.std(ddof=1)/math.sqrt(len(d))),delta_J_ci95_hi=float(d.mean()+1.95996398454*d.std(ddof=1)/math.sqrt(len(d))),
                           relative_delta_J_pct=float(100*d.mean()/J0.mean()),delta_cross_probability=float(dc.mean()),discordant_baseline_only=int(np.sum(c0 & ~cr)),discordant_policy_only=int(np.sum(~c0 & cr)),n_paths=len(d)))
    for k in [5,10,20,40,64]:
        rn,Jn,cn=cache[('noise',k)];rs,Js,cs=cache[('shadow',k)];d=Js-Jn;dc=cs.astype(float)-cn.astype(float);se=float(d.std(ddof=1)/math.sqrt(len(d)))
        direct.append(dict(noise=noise,k=k,delta_J_shadow_minus_noise=float(d.mean()),delta_J_se=se,ci95_lo=float(d.mean()-1.95996398454*se),ci95_hi=float(d.mean()+1.95996398454*se),
                           delta_cross_shadow_minus_noise=float(dc.mean()),shadow_better_fraction=float(np.mean(Js<Jn)),n_paths=len(d)))

df=pd.DataFrame(rows).sort_values(['noise','k','method']);df.to_csv(ROOT/'results/stage5b2_sparse_frontier.csv',index=False)
pd.DataFrame(paired).to_csv(ROOT/'results/stage5b2_paired_vs_baseline.csv',index=False)
pd.DataFrame(direct).to_csv(ROOT/'results/stage5b2_shadow_vs_noise.csv',index=False)
summary=[]
for noise in ['homogeneous','activity_scaled']:
    sub=df[df.noise==noise]
    for k in [0,5,10,20,40,64]:
        ss=sub[sub.k==k]
        for _,r in ss.iterrows(): summary.append({c:r[c] for c in ['noise','method','k','J','J_se','crossing','control_effort','added_authority_sum','effective_power_proxy_MW','gap_to_common_Jplus','gap_fraction','delta_J_vs_k0','relative_delta_J_pct']})
(ROOT/'results/STAGE5B2_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(df[['noise','method','k','J','J_se','delta_J_vs_k0','relative_delta_J_pct','crossing','control_effort','added_authority_sum','gap_fraction']].to_string(index=False))
print('\nSHADOW-NOISE\n',pd.DataFrame(direct).to_string(index=False))
