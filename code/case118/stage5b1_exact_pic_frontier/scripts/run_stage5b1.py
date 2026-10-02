from pathlib import Path
import sys, math, json, csv, argparse, time
from types import SimpleNamespace
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem, noise_profile, SmoothCost
from case118_stage4 import completion_actual_cost_lqr_gain, completion_guide, shared_plus_actions, eval_plus_actions

SIG=.15; DT=.0025; T=3.; R0=.2; FRACTIONS=[1.0,.8,.6,.4,.25]
COMBOS=[('uniform','homogeneous'),('activity_specific','activity_scaled'),('robust_fixed','homogeneous'),('robust_fixed','activity_scaled')]

def allocations(p,Gh,Ga):
    A=p.ng/R0
    uniform=np.ones(p.ng)/R0
    qga=Ga[p.gen_idx]**2; activity=A*qga/qga.sum()
    qs=[Gh**2,Ga**2]; caps=np.min(np.vstack([q[p.gen_idx]/q.sum() for q in qs]),axis=0); robust=A*caps/caps.sum()
    return {'uniform':uniform,'activity_specific':activity,'robust_fixed':robust}

def setup(design,noise,fraction):
    p,_=build_problem(); Gh=noise_profile(p,SIG,'homogeneous'); Ga=noise_profile(p,SIG,'activity_scaled'); G=Gh if noise=='homogeneous' else Ga
    g=allocations(p,Gh,Ga)[design]; q=G**2; lammax=float(np.min(q[p.gen_idx]/g)); lam=float(fraction*lammax)
    Gdiag=np.zeros(p.n); Gdiag[p.gen_idx]=g; delta=np.maximum(q/lam-Gdiag,0.)
    proxy=float(100*np.sum(np.sqrt(R0*delta)))
    return p,G,g,lammax,lam,delta,proxy

def one(task):
    design,noise,fraction,seed,npaths,alpha=task; t0=time.time(); p,G,g,lmax,lam,delta,proxy=setup(design,noise,fraction)
    fake=SimpleNamespace(lam=lam,Gamma=G); K,_=completion_actual_cost_lqr_gain(p,fake,SmoothCost()); guide=completion_guide(p,K,alpha,clip=8.)
    a=shared_plus_actions(p,G,guide,T=T,dt=DT,n_paths=npaths,seed=seed,cost=SmoothCost()); z=eval_plus_actions(a,lam)
    row=dict(design=design,noise=noise,sigma_rms=SIG,fraction=float(fraction),lambda_max=lmax,lambda_pic=lam,completion_trace=float(delta.sum()),virtual_rank=int(np.count_nonzero(delta>1e-10*max(1.,float((G**2/lam).max())))),effective_power_authority_MW=proxy,seed=int(seed),n_paths=int(npaths),dt=DT,guide_alpha=float(alpha),Wplus=float(z['W']),ess_fraction=float(z['ess_fraction']),ess=float(z['ess']),logweight_std=float(z['logweight_std']),cross_fraction=float(a['cross_fraction']),elapsed_s=float(time.time()-t0))
    out=ROOT/'results'/'chunks'; out.mkdir(parents=True,exist_ok=True); (out/f'{design}_{noise}_f{fraction:g}_s{seed}_n{npaths}.json').write_text(json.dumps(row,indent=2)+'\n')
    return row

def geometry():
    rows=[]
    for design,noise in COMBOS:
        base=None
        for f in FRACTIONS:
            p,G,g,lmax,lam,delta,proxy=setup(design,noise,f)
            if base is None: base=(float(delta.sum()),proxy)
            rows.append(dict(design=design,noise=noise,sigma_rms=SIG,fraction=f,lambda_max=lmax,lambda_pic=lam,total_generator_authority=float(g.sum()),completion_trace=float(delta.sum()),incremental_trace_above_least=float(delta.sum()-base[0]),effective_power_authority_MW=proxy,incremental_effective_power_proxy_MW=proxy-base[1],virtual_rank=int(np.count_nonzero(delta>1e-10*max(1.,float((G**2/lam).max()))))))
    with open(ROOT/'results/stage5b1_geometry.csv','w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (ROOT/'results/STAGE5B1_GEOMETRY.json').write_text(json.dumps(rows,indent=2))
    return rows

def run_tasks(tasks,workers=4):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(one,t) for t in tasks]
        for f in as_completed(fs):
            r=f.result(); rows.append(r); print(json.dumps(r),flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['geometry','pilot','rep'],required=True); ap.add_argument('--seed',type=int,default=52101); ap.add_argument('--n-paths',type=int,default=128); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--alpha',type=float,default=1.0); args=ap.parse_args()
    geometry()
    if args.mode=='geometry': return
    tasks=[]
    for d,n in COMBOS:
        for f in FRACTIONS[1:]: # baseline f=1 already accurately available from 5A2/4; do not recompute here
            tasks.append((d,n,f,args.seed,args.n_paths,args.alpha))
    run_tasks(tasks,args.workers)
if __name__=='__main__': main()
