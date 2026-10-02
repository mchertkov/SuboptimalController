from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'STAGE3B_EVALUATION_CONFIG.json').read_text())
assert cfg['evaluation']['seeds']==[31301,31302]
assert cfg['evaluation']['training_seed_overlap'] is False
D=pd.read_csv(ROOT/'results/stage3b_all_results.csv')
P=pd.read_csv(ROOT/'results/stage3b_all_paired_vs_static.csv')
assert len(D)==54 and len(P)==48
assert set(D['policy'])=={'static','H0_droop','H0_lqr','H1_droop','H1_lqr','H2_droop','H2_lqr','H3_droop','H3_lqr'}
assert set(np.round(D['sigma_rms'],2))=={0.10,0.15,0.20}
assert set(D['noise'])=={'homogeneous','activity_scaled'}
assert np.all(D['n_paths']==192)
# Verify frozen Stage-3A matrix hash recorded by Stage 3A.
sha=hashlib.sha256((ROOT/'data/stage3a_controller_matrices.npz').read_bytes()).hexdigest()
assert sha=='f5a4d417f7bbe32742d71abc2f71b653f53d9e0c8735c6c1dfd860d0034257d0',sha
# The new static cost regression should remain within 2 combined SE of Stage 2 on every point.
R=pd.read_csv(ROOT/'results/stage3b_static_regression_vs_stage2.csv')
assert np.max(np.abs(R['z_combined']))<2.0
# No manuscript source is included or edited here.
assert not list(ROOT.rglob('*.tex'))
print('Stage 3B verification passed.')
print('rows:',len(D),'paired rows:',len(P),'max |Stage2-vs-Stage3B static z|:',float(np.max(np.abs(R['z_combined']))))
