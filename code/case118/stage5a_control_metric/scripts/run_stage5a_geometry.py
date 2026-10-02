from pathlib import Path
import sys,csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile
SIGMA=.15;R0=.2
p,_=build_problem(); Gh=noise_profile(p,SIGMA,'homogeneous'); Ga=noise_profile(p,SIGMA,'activity_scaled')
A=p.ng/R0
uniform=np.ones(p.ng)/R0
qga=Ga[p.gen_idx]**2
activity=A*qga/qga.sum()
qs=[Gh**2,Ga**2]
caps=np.min(np.vstack([q[p.gen_idx]/q.sum() for q in qs]),axis=0)
robust=A*caps/caps.sum()
alloc={'uniform':uniform,'activity_specific':activity,'robust_fixed':robust}

def geom(G,g):
 q=G**2;qg=q[p.gen_idx];lam=float(np.min(qg/g));Gdiag=np.zeros(p.n);Gdiag[p.gen_idx]=g;Gp=q/lam;D=np.maximum(Gp-Gdiag,0.);tol=1e-10*max(1.,float(Gp.max()))
 M=np.eye(p.n)-lam*((1/G[:,None])*np.diag(Gdiag))*(1/G[None,:])
 return dict(lambda_max=lam,completion_trace=float(D.sum()),virtual_rank=int(np.sum(D>tol)),whitened_deficit_trace=float(np.trace(M)),authority_sum=float(g.sum()),authority_min=float(g.min()),authority_max=float(g.max()),Rdiag_min=float((1/g).min()),Rdiag_max=float((1/g).max()))
rows=[]
for d,g in alloc.items():
 for noise,G in [('homogeneous',Gh),('activity_scaled',Ga)]: rows.append(dict(design=d,noise=noise,sigma_rms=SIGMA,**geom(G,g)))
out=ROOT/'results';out.mkdir(exist_ok=True)
with open(out/'stage5a_geometry.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with open(out/'stage5a_generator_authority.csv','w',newline='') as f:
 fields=['bus_id']+list(alloc);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for j,idx in enumerate(p.gen_idx): w.writerow({'bus_id':int(p.bus_ids[idx]),**{k:float(v[j]) for k,v in alloc.items()}})
summary={'stage':'5A1_control_metric_geometry','sigma_rms':SIGMA,'R0':R0,'total_generator_authority':A,'n_generators':p.ng,'designs':{d:{'authority':g.tolist(),'authority_min':float(g.min()),'authority_max':float(g.max())} for d,g in alloc.items()},'geometry':rows,'status':'geometry_frozen; dynamic/PIC Monte Carlo intentionally deferred to short Stage5A2 chunks after timeout of combined run','manuscript_modified':False}
(out/'STAGE5A_GEOMETRY_FREEZE.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(rows,indent=2))
