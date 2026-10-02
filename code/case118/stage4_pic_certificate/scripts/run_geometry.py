from pathlib import Path
import sys,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile
from case118_stage4 import load_architectures,general_pic_geometry,control_inflation_geometry,completion_diagnostics,terminal_rho_coefficients,terminal_quadratic_positive_layer
p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05)
Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz')
rows=[];term=[]
for noise in ['homogeneous','activity_scaled']:
  for s in [.10,.15,.20]:
    G=noise_profile(p,s,noise)
    for h,H in Hs.items():
      gm=general_pic_geometry(p,H,G,.2);gp=control_inflation_geometry(p,H,G,.2);diag=completion_diagnostics(gp)
      rows.append(dict(noise=noise,sigma_rms=s,architecture=h,rank=H.rank,lambda_max=gp.lam,deleted_rank=gm.deleted_rank,deleted_trace_omega=gm.deleted_trace,deleted_fro_omega=gm.deleted_fro,virtual_rank=gp.virtual_rank,**{k:v for k,v in diag.items() if k not in ('whitened_deficit_min_eig','whitened_deficit_max_eig')},whitened_min_eig=diag['whitened_deficit_min_eig'],whitened_max_eig=diag['whitened_deficit_max_eig']))
      if abs(s-.15)<1e-12:
        c=terminal_rho_coefficients(p,gm);c.update(noise=noise,sigma_rms=s,architecture=h,rank=H.rank,lambda_max=gm.lam,quadratic_positive_layer_s=terminal_quadratic_positive_layer(c,.05));term.append(c)
for fn,data in [('stage4_geometry.csv',rows),('stage4_terminal_layer.csv',term)]:
  with open(ROOT/'results'/fn,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(ROOT/'results/stage4_geometry.json').write_text(json.dumps(rows,indent=2))
(ROOT/'results/stage4_terminal_layer.json').write_text(json.dumps(term,indent=2))
print('GEOMETRY')
for r in rows:
 if r['sigma_rms']==.15: print(r['noise'],r['architecture'],'lam',r['lambda_max'],'defrank',r['deleted_rank'],'vrank',r['virtual_rank'],'I',r['inflation_trace'],'Dw',r['whitened_deficit_trace'])
print('TERMINAL')
for r in term: print(r['noise'],r['architecture'],'rho0',r['rho_T'],'rho1',r['rho_t_T'],'rho2',r['rho_tt_T'],'layer',r['quadratic_positive_layer_s'])
