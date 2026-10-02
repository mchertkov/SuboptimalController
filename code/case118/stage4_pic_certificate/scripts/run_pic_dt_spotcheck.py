from pathlib import Path
import sys,csv,json,math
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import load_architectures,general_pic_geometry,control_inflation_geometry,feynman_kac_modes,completion_lqr_gain,completion_actual_cost_lqr_gain,completion_guide,shared_plus_actions,eval_plus_actions
R=.2;S=.15;T=3.;ALPHA_MINUS=.005

def setup():
 p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);hs,gains=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');return p,hs,gains

def run(arg):
 kind,noise,dt=arg;p,hs,gains=setup();H=hs['H3'];G=noise_profile(p,S,noise);cost=SmoothCost()
 if kind=='minus':
  gm=general_pic_geometry(p,H,G,R);st=np.stack([p.theta_pre,np.zeros(p.n)]);z=feynman_kac_modes(p,gm,st,T=T,dt=dt,n_paths=256,seed=46101+(0 if noise=='homogeneous' else 1000),cost=cost,guide_gain=gains['H3'],guide_alpha=ALPHA_MINUS,guide_clip=8.)
  return dict(kind=kind,noise=noise,dt=dt,W=float(z['W'][0]),ess_fraction=float(z['ess'][0]/256),n_paths=256)
 gp=control_inflation_geometry(p,H,G,R)
 if noise=='homogeneous':K=completion_lqr_gain(p,gp);alpha=.005;n=512;seed=46201
 else:K,_=completion_actual_cost_lqr_gain(p,gp,cost);alpha=1.25;n=768;seed=46301
 guide=completion_guide(p,K,alpha,clip=8.);a=shared_plus_actions(p,G,guide,T=T,dt=dt,n_paths=n,seed=seed,cost=cost);z=eval_plus_actions(a,gp.lam)
 return dict(kind=kind,noise=noise,dt=dt,W=z['W'],ess_fraction=z['ess_fraction'],n_paths=n,guide_alpha=alpha)

def main():
 jobs=[(k,n,d) for k in ['minus','plus'] for n in ['homogeneous','activity_scaled'] for d in [.005,.0025]];rows=[]
 with ProcessPoolExecutor(max_workers=4) as ex:
  fut=[ex.submit(run,j) for j in jobs]
  for i,f in enumerate(as_completed(fut),1):r=f.result();rows.append(r);print('done',i,len(fut),r,flush=True)
 rows=sorted(rows,key=lambda r:(r['kind'],r['noise'],-r['dt']))
 # relative/fine differences per pair
 for k in ['minus','plus']:
  for n in ['homogeneous','activity_scaled']:
   rr=[x for x in rows if x['kind']==k and x['noise']==n];co=[x for x in rr if x['dt']==.005][0];fi=[x for x in rr if x['dt']==.0025][0];co['fine_minus_coarse']=fi['W']-co['W'];fi['fine_minus_coarse']=fi['W']-co['W']
 with open(ROOT/'results/stage4_pic_dt_spotcheck.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
 (ROOT/'results/stage4_pic_dt_spotcheck.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
