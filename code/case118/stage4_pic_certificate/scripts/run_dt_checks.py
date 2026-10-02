from pathlib import Path
import sys,json,csv,math
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile,SmoothCost
from case118_stage4 import *

def task(arg):
 noise,s,Hname,dt,n,seed,alpha,guidekind=arg
 p,_=build_problem(alpha=4,event_bus=90);Hs,_=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz');G=noise_profile(p,s,noise);gp=control_inflation_geometry(p,Hs[Hname],G,.2)
 if guidekind=='actual':K,_=completion_actual_cost_lqr_gain(p,gp,SmoothCost())
 else:K=completion_lqr_gain(p,gp)
 guide=completion_guide(p,K,alpha,8.)
 a=shared_plus_actions(p,G,guide,T=3,dt=dt,n_paths=n,seed=seed,cost=SmoothCost());z=eval_plus_actions(a,gp.lam);ef=z['ess_fraction'];se=gp.lam*math.sqrt(max(1/ef-1,0)/n)
 return dict(noise=noise,sigma_rms=s,architecture=Hname,dt=dt,n_paths=n,seed=seed,lambda_max=gp.lam,Wplus=z['W'],delta_se=se,ess_fraction=ef,guide=guidekind,guide_alpha=alpha)

def main():
 jobs=[]
 for s in [.1,.15,.2]:jobs.append(('activity_scaled',s,'H0',.0025,512,46101+int(s*1000),.75,'actual'))
 jobs.append(('activity_scaled',.15,'H3',.0025,768,46501,1.25,'actual'))
 jobs.append(('homogeneous',.15,'H3',.0025,768,46601,.005,'legacy'))
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as ex:
  fut=[ex.submit(task,j) for j in jobs]
  for f in as_completed(fut):r=f.result();rows.append(r);print(r,flush=True)
 with open(ROOT/'results/stage4_dt_checks.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (ROOT/'results/stage4_dt_checks.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
