from pathlib import Path
import json,pandas as pd
R=Path(__file__).resolve().parent
D=json.loads((R/'results/STAGE4D_DECISION.json').read_text())
assert D['status']=='PROTOTYPE_COMPLETE_DO_NOT_EXPAND_NAIVE_GLOBAL_COVER'
assert D['rigorous_status']['new_interior_cells_certified']==0
s=pd.read_csv(R/'results/stage4d_lipschitz_stress_test.csv')
assert len(s)==10
assert (s['rho_center_minus_2se']>0).all()
assert s[s.noise=='homogeneous'].implied_edge_radius_deg.max()<1e-9
assert s[s.noise=='activity_scaled'].implied_edge_radius_deg.max()<1e-11
v=pd.read_csv(R/'results/stage4d_short_step_variations.csv')
assert v[v.step_s==.002].J1.max()<1.001
print('Stage4D verification passed')
