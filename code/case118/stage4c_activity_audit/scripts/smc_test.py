from pathlib import Path
import sys,math
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import *

def systematic(w,rng):
 n=len(w); u0=rng.random()/n; c=np.cumsum(w); return np.searchsorted(c,u0+np.arange(n)/n)

def run(family='actual',alpha=1.25,N=256,seed=49401,threshold=.5):
 p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');Gamma=noise_profile(p,.15,'activity_scaled');gp=control_inflation_geometry(p,Hs['H3'],Gamma,.2);cost=SmoothCost()
 K=completion_lqr_gain(p,gp) if family=='generic' else completion_actual_cost_lqr_gain(p,gp,cost)[0]
 dt=.0025;T=3.;steps=int(T/dt);sq=math.sqrt(dt);rng=np.random.default_rng(seed);n=p.n
 th=np.repeat(p.theta_pre[None,:],N,axis=0).copy();om=np.zeros((N,n));lw=np.zeros(N);logZ=0.;resamps=0;essmins=[]
 for k in range(steps):
  V=cost.running(p,th,om)
  u=-alpha*(p.reduced_state(th,om)@K.T);u=np.clip(u,-8.,8.);a=u/Gamma[None,:]
  dW=sq*rng.standard_normal((N,n))
  lw += -V*dt/gp.lam - np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*dt
  ls=logsumexp(lw);wn=np.exp(lw-ls);ess=1/np.sum(wn*wn);essmins.append(ess/N)
  if ess < threshold*N:
   logZ += ls-math.log(N);idx=systematic(wn,rng);th=th[idx].copy();om=om[idx].copy();lw.fill(0.);resamps+=1
  drift=(p.p_post[None,:]-p.damp[None,:]*om-p.network_force(th)+u if False else None)
  # IMPORTANT: if resampled, u/dW correspond to pre-resample particles; dynamics must happen BEFORE resampling.
  # This line is unreachable; algorithm is implemented correctly below in a separate loop.
 raise RuntimeError('placeholder')

# Correct implementation: propagate then resample the propagated particles using current incremental weights.
def run_correct(family='actual',alpha=1.25,N=256,seed=49401,threshold=.5):
 p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');Gamma=noise_profile(p,.15,'activity_scaled');gp=control_inflation_geometry(p,Hs['H3'],Gamma,.2);cost=SmoothCost()
 K=completion_lqr_gain(p,gp) if family=='generic' else completion_actual_cost_lqr_gain(p,gp,cost)[0]
 dt=.0025;T=3.;steps=int(T/dt);sq=math.sqrt(dt);rng=np.random.default_rng(seed);n=p.n
 th=np.repeat(p.theta_pre[None,:],N,axis=0).copy();om=np.zeros((N,n));lw=np.zeros(N);logZ=0.;resamps=0;miness=1.;maxu=0.
 for k in range(steps):
  V=cost.running(p,th,om)
  u=-alpha*(p.reduced_state(th,om)@K.T);u=np.clip(u,-8.,8.);maxu=max(maxu,float(np.max(np.abs(u))));a=u/Gamma[None,:]
  dW=sq*rng.standard_normal((N,n))
  lw += -V*dt/gp.lam - np.sum(a*dW,axis=1)-.5*np.sum(a*a,axis=1)*dt
  drift=(p.p_post[None,:]-p.damp[None,:]*om-p.network_force(th)+u)/p.M[None,:]
  om=om+dt*drift+(Gamma/p.M)[None,:]*dW;th=th+dt*om
  ls=logsumexp(lw);wn=np.exp(lw-ls);ess=1/np.sum(wn*wn);miness=min(miness,ess/N)
  if ess < threshold*N:
   logZ += ls-math.log(N);idx=systematic(wn,rng);th=th[idx].copy();om=om[idx].copy();lw.fill(0.);resamps+=1
 phi=cost.terminal(p,th,om);lw += -phi/gp.lam
 ls=logsumexp(lw);wn=np.exp(lw-ls);ess=1/np.sum(wn*wn);miness=min(miness,ess/N);logZ += ls-math.log(N)
 return dict(family=family,alpha=alpha,N=N,seed=seed,W=float(-gp.lam*logZ),resamples=resamps,min_ess_fraction=float(miness),final_ess_fraction=float(ess/N),maxu=maxu)

if __name__=='__main__':
 for fam,a in [('actual',1.25),('generic',.015)]: print(run_correct(fam,a),flush=True)
