from pathlib import Path
import sys,csv,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile
from case118_stage4 import load_architectures,general_pic_geometry,terminal_rho_coefficients,terminal_taylor_residual_bound
p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz')
rows=[]
for noise in ['homogeneous','activity_scaled']:
 G=noise_profile(p,.15,noise)
 for h,H in Hs.items():
  gm=general_pic_geometry(p,H,G,.2);c=terminal_rho_coefficients(p,gm,delta_bar_deg=84.,guard_deg=82.);b=terminal_taylor_residual_bound(p,gm,c,delta_bar_deg=84.,omega_bar=6.5,guard_deg=82.)
  row=dict(noise=noise,sigma_rms=.15,architecture=h,rank=H.rank,lambda_max=gm.lam,deleted_rank=gm.deleted_rank,**{k:v for k,v in c.items() if not isinstance(v,(list,dict))},analytic_safe_layer_s=b['analytic_safe_layer_s'],analytic_safe_layer_ms=1000*b['analytic_safe_layer_s'],w3_abs_max=b['w3_abs_max'])
  rows.append(row);print(noise,h,'safe ms',row['analytic_safe_layer_ms'],'rhoT',row['rho_T'])
with open(ROOT/'results/stage4_terminal_certificate.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'results/stage4_terminal_certificate.json').write_text(json.dumps(rows,indent=2))
