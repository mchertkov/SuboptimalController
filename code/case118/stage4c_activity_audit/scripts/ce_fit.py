from pathlib import Path
import sys,math,json,csv,os
import numpy as np
from scipy.special import logsumexp
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import load_architectures,control_inflation_geometry,completion_lqr_gain
SIG=.15;DT=.0025;T=3.;N=256;SEEDS=[49201,49202,49203,49204];ALPHA0=.015;NBIN=6

def simulate(seed):
 p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz')
 Gamma=noise_profile(p,SIG,'activity_scaled');gp=control_inflation_geometry(p,Hs['H3'],Gamma,.2);K=completion_lqr_gain(p,gp);cost=SmoothCost()
 n=p.n;steps=int(np.ceil(T/DT));dt=T/steps;sq=math.sqrt(dt);rng=np.random.default_rng(seed)
 th=np.repeat(p.theta_pre[None,:],N,axis=0).copy();om=np.zeros((N,n));run=np.zeros(N);ll=np.zeros(N)
 Ab=np.zeros((N,NBIN));Bb=np.zeros((N,NBIN));maxu=0.
 for k in range(steps):
  run += cost.running(p,th,om)*dt
  uunit=-(p.reduced_state(th,om)@K.T) # nodal power command at alpha=1
  nod=ALPHA0*uunit; maxu=max(maxu,float(np.max(np.abs(nod))))
  aunit=uunit/Gamma[None,:]; a=ALPHA0*aunit
  dW=sq*rng.standard_normal((N,n)); b=min(NBIN-1,int((k*dt)/T*NBIN))
  Ab[:,b]+=np.sum(aunit*dW,axis=1); Bb[:,b]+=np.sum(aunit*aunit,axis=1)*dt
  ll += -np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*dt
  drift=(p.p_post[None,:]-p.damp[None,:]*om-p.network_force(th)+nod)/p.M[None,:]
  om=om+dt*drift+(Gamma/p.M)[None,:]*dW;th=th+dt*om
 S=run+cost.terminal(p,th,om);lw=-S/gp.lam+ll
 out=ROOT/'results'/f'ce_seed_{seed}.npz';np.savez_compressed(out,lw=lw,A=Ab,B=Bb,S=S,ll=ll,maxu=np.array(maxu),lam=np.array(gp.lam))
 return dict(seed=seed,W=float(-gp.lam*(logsumexp(lw)-math.log(N))),ess_fraction=float(1/np.sum(np.exp(2*(lw-logsumexp(lw))))/N),maxu=maxu)

def main():
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as ex:
  fs=[ex.submit(simulate,s) for s in SEEDS]
  for f in as_completed(fs): r=f.result();rows.append(r);print(r,flush=True)
 # combine all samples under same proposal
 L=[];A=[];B=[]
 for s in SEEDS:
  z=np.load(ROOT/'results'/f'ce_seed_{s}.npz');L.append(z['lw']);A.append(z['A']);B.append(z['B'])
 L=np.concatenate(L);A=np.concatenate(A);B=np.concatenate(B);lse=logsumexp(L);w=np.exp(L-lse);lam=float(np.load(ROOT/'results'/f'ce_seed_{SEEDS[0]}.npz')['lam'])
 ess=1/np.sum(w*w); W=-lam*(lse-math.log(len(L)))
 num=np.sum(w[:,None]*A,axis=0);den=np.sum(w[:,None]*B,axis=0);delta=num/np.maximum(den,1e-30);raw=ALPHA0+delta
 # shrink + smooth for robustness
 shr=ALPHA0+0.5*delta
 sm=shr.copy()
 for i in range(NBIN):
  vals=shr[max(0,i-1):min(NBIN,i+2)];sm[i]=np.mean(vals)
 sm=np.clip(sm,0.0025,0.05)
 result=dict(alpha0=ALPHA0,nbins=NBIN,bin_edges=np.linspace(0,T,NBIN+1).tolist(),delta_raw=delta.tolist(),alpha_raw=raw.tolist(),alpha_shrunk=shr.tolist(),alpha_smoothed=sm.tolist(),combined_W=float(W),combined_ess=float(ess),combined_ess_fraction=float(ess/len(L)),n_total=len(L),seeds=SEEDS,per_seed=sorted(rows,key=lambda x:x['seed']))
 (ROOT/'results/ce_fit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
