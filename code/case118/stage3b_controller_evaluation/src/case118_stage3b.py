from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from case118_stage3a import build_problem, Actuation, noise_profile, SmoothCost

ARCH_KEYS={'H0':('Qgen_H0','LQR_K_H0'),'H1':('Qgen_H1','LQR_K_H1'),'H2':('Qgen_H2_r3','LQR_K_H2_r3'),'H3':('Qgen_H3','LQR_K_H3')}
DROOP_K={'H0':0.0125,'H1':0.0050,'H2':0.0125,'H3':0.0125}
LQR_ALPHA={'H0':0.00125,'H1':0.00125,'H2':0.00250,'H3':0.00500}
POLICY_NAMES=['static']+[f'{a}_{k}' for a in ('H0','H1','H2','H3') for k in ('droop','lqr')]


def wilson_interval(k,n,z=1.959963984540054):
    p=k/n;den=1+z*z/n;ctr=(p+z*z/(2*n))/den;rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,ctr-rad),min(1.,ctr+rad)


def _load_policy(problem,matrix_path,name):
    z=np.load(matrix_path);genbus=np.asarray(z['generator_bus_ids'],int)
    if not np.array_equal(genbus,problem.bus_ids[problem.gen_idx]):raise RuntimeError('generator bus IDs mismatch')
    if name=='static':
        act=Actuation('H3','static carrier',np.asarray(z['Qgen_H3'],float),genbus)
        return 'static','static',act,None,0.0
    arch,kind=name.split('_',1);qkey,kkey=ARCH_KEYS[arch];act=Actuation(arch,arch,np.asarray(z[qkey],float),genbus)
    if kind=='droop':return arch,kind,act,None,DROOP_K[arch]
    return arch,kind,act,np.asarray(z[kkey],float),LQR_ALPHA[arch]


def simulate_policy(*,matrix_path,policy_name,sigma,noise_kind,seeds=(31301,31302),paths_per_seed=96,T=3.,dt=.0025,control_weight=.2):
    problem,_=build_problem(alpha=4.0,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05)
    arch,kind,act,K,scale=_load_policy(problem,matrix_path,policy_name)
    cost=SmoothCost(.08,.02,.20,82.);noise=noise_profile(problem,float(sigma),noise_kind)
    n=problem.n;N=len(seeds)*paths_per_seed;steps=int(np.ceil(T/dt));dt=T/steps;sq=math.sqrt(dt);rngs=[np.random.default_rng(int(s)) for s in seeds]
    th=np.repeat(problem.theta_pre[None,:],N,axis=0).copy();om=np.zeros((N,n));run=np.zeros(N);ctrl=np.zeros(N);cross=np.zeros(N,bool)
    dmax=np.max(np.abs(problem.edge_delta(th)),axis=1);maxgen=np.zeros(N);An=act.nodal(problem);r=float(control_weight)
    for kk in range(steps):
        if kind=='static':v=np.zeros((N,act.rank))
        elif kind=='droop':v=-scale*(om[:,problem.gen_idx]@act.Qgen)
        else:v=-scale*(problem.reduced_state(th,om)@K.T)
        run+=cost.running(problem,th,om)*dt;cu=.5*r*np.sum(v*v,axis=1);run+=cu*dt;ctrl+=cu*dt
        drift=(problem.p_post[None,:]-problem.damp[None,:]*om-problem.network_force(th)+(v@An.T))/problem.M[None,:]
        basez=np.concatenate([rg.standard_normal((paths_per_seed,n)) for rg in rngs],axis=0)
        om=om+dt*drift+(noise/problem.M)[None,:]*sq*basez
        th=th+dt*om;dm=np.max(np.abs(problem.edge_delta(th)),axis=1);dmax=np.maximum(dmax,dm);cross|=dm>=np.pi/2
        maxgen=np.maximum(maxgen,np.max(np.abs(om[:,problem.gen_idx]),axis=1))
    J=run+cost.terminal(problem,th,om);k=int(cross.sum());lo,hi=wilson_interval(k,N)
    row=dict(policy=policy_name,architecture=arch,controller=kind,scale=float(scale),sigma_rms=float(sigma),noise=noise_kind,
        J_mean=float(J.mean()),J_se=float(J.std(ddof=1)/math.sqrt(N)),cross_probability=float(k/N),cross_ci95_lo=float(lo),cross_ci95_hi=float(hi),cross_count=k,n_paths=N,
        control_effort_mean=float(ctrl.mean()),dmax_q50_deg=float(np.degrees(np.quantile(dmax,.5))),dmax_q90_deg=float(np.degrees(np.quantile(dmax,.9))),dmax_q99_deg=float(np.degrees(np.quantile(dmax,.99))),
        genomega_q50=float(np.quantile(maxgen,.5)),genomega_q90=float(np.quantile(maxgen,.9)),genomega_q99=float(np.quantile(maxgen,.99)),
        seeds=';'.join(map(str,seeds)),paths_per_seed=int(paths_per_seed),dt=float(dt),T=float(T))
    return row,dict(J=J,cross=cross,dmax=dmax,maxgen=maxgen,ctrl=ctrl)


def paired_vs_static(rows,raws):
    ref=raws['static']['J'];cref=raws['static']['cross'];refmean=ref.mean();out=[]
    for row in rows:
        if row['policy']=='static':continue
        p=row['policy'];d=raws[p]['J']-ref;dc=raws[p]['cross'].astype(float)-cref.astype(float);se=d.std(ddof=1)/math.sqrt(len(d))
        out.append(dict(policy=p,architecture=row['architecture'],controller=row['controller'],scale=row['scale'],sigma_rms=row['sigma_rms'],noise=row['noise'],
            delta_J_mean=float(d.mean()),delta_J_se=float(se),delta_J_ci95_lo=float(d.mean()-1.959963984540054*se),delta_J_ci95_hi=float(d.mean()+1.959963984540054*se),
            relative_delta_J_pct=float(100*d.mean()/refmean),delta_cross_probability=float(dc.mean()),discordant_static_only=int(np.sum(cref & ~raws[p]['cross'])),discordant_policy_only=int(np.sum(~cref & raws[p]['cross'])),n_paths=len(d)))
    return out
