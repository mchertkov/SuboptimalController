from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import *
p,_=build_problem(alpha=4.,event_bus=90,inertia_gen=.30,inertia_load=.02,damping=.05);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz')
G=noise_profile(p,.15,'activity_scaled');cost=SmoothCost()
for Hname in ['H0','H1','H2','H3']:
 gp=control_inflation_geometry(p,Hs[Hname],G,.2);K,_=completion_actual_cost_lqr_gain(p,gp,cost)
 print('\n',Hname,'lam',gp.lam,'Knorm',__import__('numpy').linalg.norm(K),flush=True)
 for a in [0.,.25,.5,.75,1.,1.25,1.5]:
  guide=None if a==0 else completion_guide(p,K,a,clip=8.)
  act=shared_plus_actions(p,G,guide,T=3.,dt=.005,n_paths=128,seed=44001+Hs[Hname].rank,cost=cost);z=eval_plus_actions(act,gp.lam)
  print(a,'W',z['W'],'ess',z['ess_fraction'],'lwstd',z['logweight_std'],'cross',act['cross_fraction'],flush=True)
