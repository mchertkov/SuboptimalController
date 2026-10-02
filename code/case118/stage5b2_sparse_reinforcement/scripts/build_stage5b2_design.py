#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,time,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile
from case118_stage5b2 import h4_shadow_proxy,matched_sparse_authority,lqr_for_nodal_authority

ap=argparse.ArgumentParser(); ap.add_argument('--noise',choices=['homogeneous','activity_scaled'],required=True); args=ap.parse_args()
noise=args.noise; sigma=.15; R0=.2; KLIST=[5,10,20,40,64]
p,_=build_problem(); Gamma=noise_profile(p,sigma,noise)
freeze=json.loads((ROOT/'data/STAGE5A_GEOMETRY_FREEZE.json').read_text())
design='uniform' if noise=='homogeneous' else 'activity_specific'
gen_auth=np.asarray(freeze['designs'][design]['authority'],float)
print('building H4 shadow',noise,design,flush=True); t0=time.time()
sh=h4_shadow_proxy(p,Gamma,gen_auth,alpha=.005,seeds=(52901,52902),paths_per_seed=48)
load=np.array([i for i in range(p.n) if not p.is_gen[i]],int)
# Noise-only ranking. For homogeneous q is degenerate; stable bus-ID tie-break makes that explicit/reproducible.
q=Gamma**2
noise_order=np.array(sorted(load,key=lambda i:(-q[i],int(p.bus_ids[i]))),int)
shadow_order=np.array(sorted(load,key=lambda i:(-sh['shadow'][i],int(p.bus_ids[i]))),int)
rows=[]; mats={}
# baseline generator-only LQR
K0,_=lqr_for_nodal_authority(p,p.gen_idx,gen_auth); mats['K_baseline']=K0
for k in KLIST:
    budget,A,gA,B,gB,sumA,sumB=matched_sparse_authority(p,sh['gplus'],noise_order,shadow_order,k)
    for method,S,g,rawcap in [('noise',A,gA,sumA),('shadow',B,gB,sumB)]:
        # k=64 is the same full-load set/authority irrespective of order; still store both labels for table clarity.
        buses=np.r_[p.gen_idx,S]; auth=np.r_[gen_auth,g]
        key=f'K_{method}_k{k}'
        if k==64 and method=='shadow' and 'K_noise_k64' in mats:
            K=mats['K_noise_k64']; keyref='K_noise_k64'
        else:
            print('CARE',noise,method,k,'m=',len(buses),flush=True)
            K,_=lqr_for_nodal_authority(p,buses,auth); mats[key]=K; keyref=key
        peff=100.0*np.sum(np.sqrt(R0*np.maximum(g,0)))
        rows.append(dict(noise=noise,baseline_metric=design,method=method,k=k,added_authority_sum=float(np.sum(g)),
                         selected_target_capacity=float(rawcap),common_budget=float(budget),budget_fraction_of_full=float(budget/np.sum(sh['gplus'][load])),
                         effective_power_proxy_MW=float(peff),lambda_max=float(sh['lambda_max']),
                         load_bus_ids=';'.join(map(str,p.bus_ids[S].astype(int))),load_authorities=';'.join(f'{x:.12g}' for x in g),
                         K_key=keyref))
# ranking table
rankrows=[]
pos_noise={int(i):r+1 for r,i in enumerate(noise_order)};pos_shadow={int(i):r+1 for r,i in enumerate(shadow_order)}
for i in load:
    rankrows.append(dict(bus_id=int(p.bus_ids[i]),bus_index=int(i),noise_variance=float(q[i]),exact_completion_authority=float(sh['gplus'][i]),
                         h4_command_rms=float(sh['command_rms'][i]),h4_shadow_proxy=float(sh['shadow'][i]),
                         noise_rank=pos_noise[int(i)],shadow_rank=pos_shadow[int(i)]))
out=ROOT/'results'; out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out/f'stage5b2_design_{noise}.csv',index=False)
pd.DataFrame(rankrows).sort_values('bus_id').to_csv(out/f'stage5b2_load_ranking_{noise}.csv',index=False)
np.savez_compressed(ROOT/'data'/f'stage5b2_gains_{noise}.npz',generator_bus_ids=p.bus_ids[p.gen_idx],load_bus_ids=p.bus_ids[load],gen_authority=gen_auth,Gamma=Gamma,gplus=sh['gplus'],noise_order=noise_order,shadow_order=shadow_order,shadow=sh['shadow'],command_rms=sh['command_rms'],**mats)
meta=dict(stage='5B2_sparse_design',noise=noise,baseline_metric=design,sigma_rms=sigma,shadow_alpha=.005,shadow_seeds=sh['seeds'],shadow_paths=sh['paths'],dt=sh['dt'],T=sh['T'],lambda_max=sh['lambda_max'],full_load_completion_authority=float(np.sum(sh['gplus'][load])),runtime_s=time.time()-t0,
          budget_rule='A_k=min(sum exact-completion target on noise set, sum target on shadow set); each selected set receives A_k proportional to its own target, so g_add<=gplus and k=64 is exact completion')
(out/f'stage5b2_design_{noise}.json').write_text(json.dumps(meta,indent=2))
print(pd.DataFrame(rows)[['method','k','added_authority_sum','budget_fraction_of_full','effective_power_proxy_MW']].to_string(index=False))
print(json.dumps(meta,indent=2))
