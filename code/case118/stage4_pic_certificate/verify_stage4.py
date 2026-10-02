from pathlib import Path
import pandas as pd, json, numpy as np
R=Path(__file__).resolve().parent
s=pd.read_csv(R/'results/stage4_pic_certificate_summary_final.csv')
c=pd.read_csv(R/'results/stage4_curvature_screen.csv')
t=pd.read_csv(R/'results/stage4_terminal_cycle_certificate.csv')
g=pd.read_csv(R/'results/stage4_geometry.csv')
assert len(s)==24 and len(g)==24
assert set(s.architecture)=={'H0','H1','H2','H3'}
assert set(s.noise)=={'homogeneous','activity_scaled'}
assert np.all(s.lambda_max>0)
assert np.all(s.Wminus_ess_mean>0.90)
assert np.all(c.trace>0) and np.all(c.min_direction>0)
assert np.all(t.analytic_safe_layer_ms>0)
# frozen geometry regressions from the handoff
x=g[(g.noise=='homogeneous')&(np.isclose(g.sigma_rms,.10))&(g.architecture=='H3')].iloc[0]
assert abs(x.lambda_max-.002)<1e-12 and abs(x.inflation_trace-320)<1e-6 and int(x.virtual_rank)==64
x=g[(g.noise=='activity_scaled')&(np.isclose(g.sigma_rms,.10))&(g.architecture=='H3')].iloc[0]
assert abs(x.inflation_trace-14832.272593)<1e-3 and int(x.virtual_rank)==117
status=json.loads((R/'results/STAGE4_CERTIFICATE_STATUS.json').read_text())
assert status['manuscript_modified'] is False and status['Wminus']['domain_wide_interior_enclosure'] is False
print('Stage4 verification: PASS')
