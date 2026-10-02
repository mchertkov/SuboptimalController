from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile
from case118_stage4 import load_architectures,control_inflation_geometry,tune_completion_guide
p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz')
rows=[];chosen={}
for j,noise in enumerate(['homogeneous','activity_scaled']):
 G=noise_profile(p,.15,noise); geom=control_inflation_geometry(p,Hs['H3'],G,.2)
 alphas=[0.,.0025,.005,.01,.02,.03,.04] if noise=='homogeneous' else [0.,.005,.01,.02,.03,.04,.06]
 rr,best,K=tune_completion_guide(p,G,geom,alphas,seed=40101+j*100,n_paths=64,T=3.,dt=.005)
 for x in rr:x.update(noise=noise,lambda_H3=geom.lam);rows.append(x)
 chosen[noise]=best['alpha'];print(noise,'best',best)
with open(ROOT/'results/stage4_plus_guide_scan.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'results/stage4_plus_guide_choice.json').write_text(json.dumps(chosen,indent=2))
