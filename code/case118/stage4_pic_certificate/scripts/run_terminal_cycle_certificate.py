from pathlib import Path
import sys,csv,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile
from case118_stage4 import load_architectures,general_pic_geometry
from certification_bounds_case118 import cycle_energy_edge_enclosure,terminal_cycle_energy_residual_enclosure
p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz')
enc=cycle_energy_edge_enclosure(p,.34*p.n,84.0); rows=[]; full=[]
for noise in ['homogeneous','activity_scaled']:
 G=noise_profile(p,.15,noise)
 for h,H in Hs.items():
  gm=general_pic_geometry(p,H,G,.2);z=terminal_cycle_energy_residual_enclosure(p,gm,enclosure=enc,energy_per_bus_bar=.34,omega_inf_bar=6.5,delta_bar_deg=84.,max_layer=.02)
  row=dict(noise=noise,sigma_rms=.15,architecture=h,rank=H.rank,lambda_max=gm.lam,deleted_rank=gm.deleted_rank,rho_T=z['rho_T'],rho_t_T=z['rho_t_T'],rho_tt_T=z['rho_tt_T'],analytic_safe_layer_s=z['analytic_safe_layer_s'],analytic_safe_layer_ms=1000*z['analytic_safe_layer_s'],edge_enclosure_iterations=z['edge_enclosure_iterations'],edge_max_abs_deg=z['edge_max_abs_deg'])
  rows.append(row);full.append(dict(**row,details=z));print(noise,h,'safe_ms',row['analytic_safe_layer_ms'],flush=True)
with open(ROOT/'results/stage4_terminal_cycle_certificate.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'results/stage4_terminal_cycle_certificate.json').write_text(json.dumps(full,indent=2))
