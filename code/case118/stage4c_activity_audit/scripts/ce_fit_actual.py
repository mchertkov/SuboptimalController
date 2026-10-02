from pathlib import Path
import sys,math,json
import numpy as np
from scipy.special import logsumexp
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import *
SIG=.15;DT=.0025;T=3.;N=256;SEEDS=[49501,49502,49503,49504];ALPHA0=1.25;NBIN=6

def simulate(seed):
 p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');G=noise_profile(p,SIG,'activity_scaled');gp=control_inflation_geometry(p,Hs['H3'],G,.2);K=completion_actual_cost_lqr_gain(p,gp,SmoothCost())[0];cost=SmoothCost()
 n=p.n;steps=int(T/DT);sq=math.sqrt(DT);rng=np.random.default_rng(seed);th=np.repeat(p.theta_pre[None,:],N,axis=0).copy();om=np.zeros((N,n));run=np.zeros(N);ll=np.zeros(N);Ab=np.zeros((N,NBIN));Bb=np.zeros((N,NBIN));maxu=0.
 for k in range(steps):
  run += cost.running(p,th,om)*DT;uunit=-(p.reduced_state(th,om)@K.T);nod=ALPHA0*uunit;maxu=max(maxu,float(np.max(np.abs(nod))));aunit=uunit/G[None,:];a=ALPHA0*aunit;dW=sq*rng.standard_normal((N,n));b=min(NBIN-1,int(k/steps*NBIN));Ab[:,b]+=np.sum(aunit*dW,axis=1);Bb[:,b]+=np.sum(aunit*aunit,axis=1)*DT;ll += -np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*DT;drift=(p.p_post[None,:]-p.damp[None,:]*om-p.network_force(th)+nod)/p.M[None,:];om+=DT*drift+(G/p.M)[None,:]*dW;th+=DT*om
 S=run+cost.terminal(p,th,om);lw=-S/gp.lam+ll;np.savez_compressed(ROOT/'results'/f'cea_seed_{seed}.npz',lw=lw,A=Ab,B=Bb,S=S,ll=ll,lam=np.array(gp.lam));lse=logsumexp(lw);wn=np.exp(lw-lse);return dict(seed=seed,W=float(-gp.lam*(lse-math.log(N))),ess_fraction=float(1/np.sum(wn*wn)/N),maxu=maxu)
def main():
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as ex:
  fs=[ex.submit(simulate,s) for s in SEEDS]
  for f in as_completed(fs):r=f.result();rows.append(r);print(r,flush=True)
 L=[];A=[];B=[]
 for s in SEEDS:
  z=np.load(ROOT/'results'/f'cea_seed_{s}.npz');L.append(z['lw']);A.append(z['A']);B.append(z['B'])
 L=np.concatenate(L);A=np.concatenate(A);B=np.concatenate(B);lse=logsumexp(L);w=np.exp(L-lse);lam=float(np.load(ROOT/'results'/f'cea_seed_{SEEDS[0]}.npz')['lam']);num=np.sum(w[:,None]*A,0);den=np.sum(w[:,None]*B,0);delta=num/np.maximum(den,1e-30);raw=ALPHA0+delta;shr=ALPHA0+.5*delta;sm=shr.copy();
 for i in range(NBIN):sm[i]=np.mean(shr[max(0,i-1):min(NBIN,i+2)])
 sm=np.clip(sm,.1,3.)
 out=dict(alpha0=ALPHA0,bin_edges=np.linspace(0,T,NBIN+1).tolist(),delta_raw=delta.tolist(),alpha_raw=raw.tolist(),alpha_shrunk=shr.tolist(),alpha_smoothed=sm.tolist(),combined_W=float(-lam*(lse-math.log(len(L)))),combined_ess_fraction=float(1/np.sum(w*w)/len(L)),per_seed=sorted(rows,key=lambda x:x['seed']))
 (ROOT/'results/ce_fit_actual.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
