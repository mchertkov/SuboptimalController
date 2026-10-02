from pathlib import Path
import sys, json, csv, math
from types import SimpleNamespace
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem, noise_profile, SmoothCost
from case118_stage4 import completion_actual_cost_lqr_gain, completion_guide, shared_plus_actions, eval_plus_actions

SIGMA=.15; R0=.2; DT=.0025; T=3.0
TUNE_ALPHAS=[0.5,0.75,1.0,1.25,1.5]
TUNE_SEED=51101; TUNE_N=128
FINAL_SEEDS=[51201,51202]; FINAL_N=512


def allocations(problem,Gh,Ga):
    A=problem.ng/R0
    unif=np.ones(problem.ng)/R0
    qga=Ga[problem.gen_idx]**2
    act=A*qga/qga.sum()
    qs=[Gh**2,Ga**2]
    caps=np.min(np.vstack([q[problem.gen_idx]/q.sum() for q in qs]),axis=0)
    robust=A*caps/caps.sum()
    return {'uniform':unif,'activity_specific':act,'robust_fixed':robust}

def geometry(problem,Gamma,g):
    q=Gamma**2; qg=q[problem.gen_idx]
    lam=float(np.min(qg/g))
    Gdiag=np.zeros(problem.n); Gdiag[problem.gen_idx]=g
    Gplus=q/lam; Delta=np.maximum(Gplus-Gdiag,0.)
    tol=1e-10*max(1.,float(np.max(Gplus)))
    M=np.eye(problem.n)-lam*((1/Gamma[:,None])*np.diag(Gdiag))*(1/Gamma[None,:])
    return dict(lambda_max=lam,completion_trace=float(Delta.sum()),virtual_rank=int(np.sum(Delta>tol)),
                whitened_deficit_trace=float(np.trace(M)),authority_sum=float(g.sum()),
                authority_min=float(g.min()),authority_max=float(g.max()),
                Rdiag_min=float((1/g).min()),Rdiag_max=float((1/g).max()))

def tune_and_eval(design,noise):
    p,_=build_problem(); Gh=noise_profile(p,SIGMA,'homogeneous'); Ga=noise_profile(p,SIGMA,'activity_scaled')
    G={'homogeneous':Gh,'activity_scaled':Ga}[noise]; g=allocations(p,Gh,Ga)[design]; geo=geometry(p,G,g)
    fake=SimpleNamespace(lam=geo['lambda_max'],Gamma=G)
    K,_=completion_actual_cost_lqr_gain(p,fake,SmoothCost())
    tune=[]
    for a in TUNE_ALPHAS:
        guide=completion_guide(p,K,a,clip=8.)
        actions=shared_plus_actions(p,G,guide,T=T,dt=DT,n_paths=TUNE_N,seed=TUNE_SEED,cost=SmoothCost())
        z=eval_plus_actions(actions,geo['lambda_max'])
        tune.append(dict(alpha=a,Wplus=z['W'],ess_fraction=z['ess_fraction'],logweight_std=z['logweight_std']))
    # ESS is used only as proposal-overlap diagnostic; among top-2 ESS candidates choose the one whose W is closest to the median scan W.
    med=float(np.median([r['Wplus'] for r in tune])); ordered=sorted(tune,key=lambda r:r['ess_fraction'],reverse=True)[:2]
    best=min(ordered,key=lambda r:abs(r['Wplus']-med)); alpha=float(best['alpha'])
    reps=[]
    for seed in FINAL_SEEDS:
        guide=completion_guide(p,K,alpha,clip=8.)
        actions=shared_plus_actions(p,G,guide,T=T,dt=DT,n_paths=FINAL_N,seed=seed,cost=SmoothCost())
        z=eval_plus_actions(actions,geo['lambda_max'])
        # desirability normalizer estimate, for equal-size pooling across reps
        Z=math.exp(-z['W']/geo['lambda_max'])
        reps.append(dict(seed=seed,alpha=alpha,Wplus=z['W'],Z=Z,ess_fraction=z['ess_fraction'],logweight_std=z['logweight_std']))
    Zpool=float(np.mean([r['Z'] for r in reps])); Wpool=float(-geo['lambda_max']*math.log(Zpool))
    vals=np.array([r['Wplus'] for r in reps])
    return dict(design=design,noise=noise,geometry=geo,tune=tune,selected_alpha=alpha,reps=reps,
                Wplus_pooled=Wpool,Wplus_rep_mean=float(vals.mean()),Wplus_rep_sd=float(vals.std(ddof=1)),
                ess_fraction_mean=float(np.mean([r['ess_fraction'] for r in reps])),ess_fraction_min=float(np.min([r['ess_fraction'] for r in reps])))

def main():
    p,_=build_problem(); Gh=noise_profile(p,SIGMA,'homogeneous');Ga=noise_profile(p,SIGMA,'activity_scaled');alloc=allocations(p,Gh,Ga)
    georows=[]
    for d,g in alloc.items():
        for noise,G in [('homogeneous',Gh),('activity_scaled',Ga)]:
            row=dict(design=d,noise=noise,sigma_rms=SIGMA,**geometry(p,G,g));georows.append(row)
    # only new designs need MC; uniform values come from frozen Stage4/4C and are inserted downstream.
    jobs=[(d,n) for d in ['activity_specific','robust_fixed'] for n in ['homogeneous','activity_scaled']]
    vals=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs={ex.submit(tune_and_eval,*j):j for j in jobs}
        for f in as_completed(futs):
            r=f.result();vals.append(r); print('DONE',r['design'],r['noise'],'W+',r['Wplus_pooled'],'ESS',r['ess_fraction_mean'],'alpha',r['selected_alpha'],flush=True)
    vals=sorted(vals,key=lambda z:(z['design'],z['noise']))
    out=ROOT/'results';out.mkdir(exist_ok=True)
    with open(out/'stage5a_geometry.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(georows[0]));w.writeheader();w.writerows(georows)
    (out/'stage5a_mc_new_designs.json').write_text(json.dumps(vals,indent=2))
    tunerows=[]; reprows=[]; finalrows=[]
    for z in vals:
        for r in z['tune']: tunerows.append(dict(design=z['design'],noise=z['noise'],**r))
        for r in z['reps']: reprows.append(dict(design=z['design'],noise=z['noise'],lambda_max=z['geometry']['lambda_max'],**r))
        finalrows.append(dict(design=z['design'],noise=z['noise'],sigma_rms=SIGMA,lambda_max=z['geometry']['lambda_max'],completion_trace=z['geometry']['completion_trace'],virtual_rank=z['geometry']['virtual_rank'],selected_alpha=z['selected_alpha'],Wplus=z['Wplus_pooled'],Wplus_rep_sd=z['Wplus_rep_sd'],ess_fraction_mean=z['ess_fraction_mean'],ess_fraction_min=z['ess_fraction_min'],n_reps=len(z['reps']),n_paths_per_rep=FINAL_N,dt=DT))
    for name,rows in [('stage5a_guide_scan.csv',tunerows),('stage5a_mc_reps.csv',reprows),('stage5a_new_design_values.csv',finalrows)]:
        with open(out/name,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # allocations
    with open(out/'stage5a_generator_authority.csv','w',newline='') as f:
        fields=['bus_id']+list(alloc.keys());w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for j,idx in enumerate(p.gen_idx): w.writerow(dict(bus_id=int(p.bus_ids[idx]),**{k:float(v[j]) for k,v in alloc.items()}))
    print('wrote results')
if __name__=='__main__':main()
