from pathlib import Path
import sys,json,csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import load_architectures,general_pic_geometry,curvature_trace_modes

R=.2;DT=.005;T=3.;SIGMA=.15;N=128;NDIR=4
ALPHA={'H0':.00125,'H1':.00125,'H2':.0025,'H3':.005}
TIMES=[0.,.75,1.5,2.25,2.8]

def setup():
 p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,gains=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');return p,Hs,gains

def deterministic_h3_states():
 p,Hs,gains=setup();H=Hs['H3'];K=gains['H3'];An=H.nodal(p)
 th=p.theta_pre.copy();om=np.zeros(p.n);out=[];idx=0;dt=.001;steps=int(max(TIMES)/dt+1e-9)
 for k in range(steps+1):
  t=k*dt
  while idx<len(TIMES) and t+1e-12>=TIMES[idx]:out.append(np.stack([th.copy(),om.copy()]));idx+=1
  if k==steps:break
  x=p.reduced_state(th[None,:],om[None,:]);v=-ALPHA['H3']*(x@K.T);nod=(v@An.T)[0]
  drift=(p.p_post-p.damp*om-p.network_force(th)+nod)/p.M
  om=om+dt*drift;th=th+dt*om
 return np.stack(out)
STATES=deterministic_h3_states()

def task(args):
 noise,Hname,it=args;p,Hs,gains=setup();G=noise_profile(p,SIGMA,noise);gm=general_pic_geometry(p,Hs[Hname],G,R);t=TIMES[it];state=STATES[it]
 z=curvature_trace_modes(p,gm,state,h=.008,n_directions=NDIR,seed_directions=51001+100*it+(0 if noise=='homogeneous' else 1000)+Hs[Hname].rank,
   t0=t,T=T,dt=DT,n_paths=N,seed=52001+100*it+(0 if noise=='homogeneous' else 1000)+Hs[Hname].rank,cost=SmoothCost(),guide_gain=gains[Hname],guide_alpha=ALPHA[Hname],guide_clip=8.)
 return dict(noise=noise,sigma_rms=SIGMA,architecture=Hname,rank=Hs[Hname].rank,t=t,remaining=T-t,lambda_max=gm.lam,deleted_rank=gm.deleted_rank,trace=z['trace'],trace_se=z['se'],trace_z=z['trace']/z['se'] if z['se']>0 else 0.,min_direction=z['min_direction'],positive_fraction=z['positive_fraction'],ess_fraction_min=z['ess_fraction_min'],surrogate_cross=z['cross_fraction'],W0=z['W0'],max_edge_deg=float(np.degrees(np.max(np.abs(p.edge_delta(state[0]))))))

def main():
 # Full initial-state H0-H3 screen + transient-path H2/H3 screen.
 jobs=[]
 for noise in ['homogeneous','activity_scaled']:
  for H in ['H0','H1','H2','H3']:jobs.append((noise,H,0))
  for H in ['H2','H3']:
   for it in range(1,len(TIMES)):jobs.append((noise,H,it))
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as ex:
  fut=[ex.submit(task,j) for j in jobs]
  for i,f in enumerate(as_completed(fut),1):
   r=f.result();rows.append(r);print('done',i,'/',len(fut),r['noise'],r['architecture'],r['t'],'trace',r['trace'],'se',r['trace_se'],'ess',r['ess_fraction_min'],flush=True)
 rows=sorted(rows,key=lambda r:(r['noise'],r['architecture'],r['t']))
 with open(ROOT/'results/stage4_curvature_screen.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (ROOT/'results/stage4_curvature_screen.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
