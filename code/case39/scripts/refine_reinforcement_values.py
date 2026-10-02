#!/usr/bin/env python3
"""Higher-statistics refresh of selected W+ values used in the planning figures.

The broad planning sweep shares one importance guide for efficiency.  That is useful for
trend discovery but can be statistically poor when the scalar PIC temperature changes by
an order of magnitude.  This script recomputes the four planning points used for the
control-metric/scenario-flexibility discussion with a completion-specific LQR guide and
exact Girsanov correction, then patches ``reinforcement_planning.json``.

It is intentionally separate from the fast broad sweep so the paper can distinguish a
screening calculation from the refined values used in quantitative comparisons.
"""
from pathlib import Path
import sys, json
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pic_case39 import build_case39_problem
from stochastic_swing import noise_profile
from reinforcement_planning import generator_authority_fixed, generator_authority_noise_proportional, robust_fixed_authority, diagonal_completion
from control_inflation import completion_lqr_gain, completion_lqr_guide_nodal, feynman_kac_control_inflation
from smooth_barrier import SmoothPICCost

T=3.0; DT=.005; R0=.2; NPATHS=700

def eval_case(problem,Gamma,lam,alpha,seed):
    geom=SimpleNamespace(Gamma=np.asarray(Gamma,float),lam=float(lam))
    K,_=completion_lqr_gain(problem,geom)
    guide=completion_lqr_guide_nodal(problem,K,alpha=float(alpha),clip=8.)
    state=np.stack([problem.theta_pre,np.zeros(problem.n)])
    z=feynman_kac_control_inflation(problem,geom,state,T=T,dt=DT,n_paths=NPATHS,seed=int(seed),cost=SmoothPICCost(),guide_nodal=guide)
    return dict(W=float(z['W'][0]),ess_fraction=float(z['ess'][0]/NPATHS),logweight_std=float(z['logweight_std'][0]),n_paths=NPATHS,guide_alpha=float(alpha),seed=int(seed))

def main():
    pth=ROOT/'results'/'reinforcement_planning.json'
    data=json.loads(pth.read_text())
    problem,_=build_case39_problem()
    Gh=noise_profile(problem,.1,'homogeneous')
    Ga=noise_profile(problem,.1,'activity_scaled')
    g_unif=generator_authority_fixed(problem,R0)
    g_act=generator_authority_noise_proportional(problem,Ga,R0)
    g_rob,_,_=robust_fixed_authority(problem,[Gh,Ga],R0)

    jobs=[
      ('homogeneous','robust_fixed',Gh,g_rob,.012,500),
      ('activity_scaled','robust_fixed',Ga,g_rob,.018,501),
      ('activity_scaled','activity_specific',Ga,g_act,.020,850),
      ('homogeneous','activity_specific',Gh,g_act,.012,851),
    ]
    rec={}
    for noise,name,Gamma,g,alpha,seed in jobs:
        c=diagonal_completion(problem,Gamma,g)
        z=eval_case(problem,Gamma,c.lam,alpha,seed)
        row=data['scenario_flexibility'][noise][name]
        row.update(z)
        # Keep metric-design activity equalized consistent with same physical authority.
        if noise=='activity_scaled' and name=='activity_specific':
            data['metric_design']['activity_scaled']['noise_proportional_equalized_R'].update(z)
        rec[f'{noise}:{name}']=dict(lambda_value=c.lam,**z)
        print(noise,name,rec[f'{noise}:{name}'])
    data['refined_values']=rec
    pth.write_text(json.dumps(data,indent=2))
    print('patched',pth)

if __name__=='__main__': main()
