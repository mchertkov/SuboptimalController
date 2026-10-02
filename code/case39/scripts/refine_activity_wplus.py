from pathlib import Path
import sys,csv,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pic_case39 import build_case39_problem
from stochastic_swing import noise_profile
from actuation_hierarchy import build_hierarchy
from control_inflation import control_inflation_geometry,completion_lqr_gain,completion_lqr_guide_nodal,feynman_kac_control_inflation
from smooth_barrier import SmoothPICCost
p,m=build_case39_problem();hs=build_hierarchy(p,m['bus_ids']); st=np.stack([p.theta_pre,np.zeros(p.n)])
alphas={'H0':.01,'H1':.015,'H2':.006,'H3':.01}
rows=[]
for s in [.05,.1,.2,.3]:
 G=noise_profile(p,s,'activity_scaled')
 print('sigma',s,flush=True)
 for H in hs:
  g=control_inflation_geometry(p,H,G,.2);K,_=completion_lqr_gain(p,g);a=alphas[H.name]
  guide=completion_lqr_guide_nodal(p,K,alpha=a,clip=8)
  z=feynman_kac_control_inflation(p,g,st,T=3,dt=.005,n_paths=700,seed=4501+int(s*1000)+H.rank,cost=SmoothPICCost(),guide_nodal=guide)
  d=dict(sigma_rms=s,H=H.name,lambda_max=g.lam,Wplus=float(z['W'][0]),ess=float(z['ess'][0]/700),guide_alpha=a,cross=float(z['cross_fraction'][0]))
  rows.append(d);print(H.name,d,flush=True)
with open(ROOT/'results'/'control_inflation_activity_wplus_refined.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'results'/'control_inflation_activity_wplus_refined.json').write_text(json.dumps(rows,indent=2))
