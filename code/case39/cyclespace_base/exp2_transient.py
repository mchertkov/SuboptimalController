"""Balanced-step transient experiment reported in the revised manuscript."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

from grids import load_case39
from syncnet import incidence_and_cycles, newton_equilibrium
from transient import certified_threshold, simulated_threshold, static_threshold
from paper_results import CONCENTRATIONS, TRANSIENT_MW

ALPHA = 4.0
DAMP = 0.05
T_SIM = 20.0
DT_SIM = 4e-3
TOL_PU = 0.01   # 1 MW on a 100 MVA base


def main(outdir="results", do_refinement_checks=False):
    outdir=Path(outdir); outdir.mkdir(exist_ok=True,parents=True)
    case=load_case39()
    G,p0,M,is_gen,base,bus_ids=(case[k] for k in ["G","p","M","is_gen","baseMVA","bus_ids"])
    Inc,Cyc,edges,nodes=incidence_and_cycles(G)
    K=np.array([G[u][v]["K"] for u,v in edges])
    p=ALPHA*p0
    eq0=newton_equilibrium(Inc,K,p)
    if not eq0["ok"]: raise RuntimeError("pre-step equilibrium did not converge")
    th_old=eq0["theta"]
    d=int(np.argmin(p))       # MATPOWER bus 20
    slack=int(np.argmax(p))   # MATPOWER bus 38
    campus=-p[d]
    damp=np.full(len(p),DAMP)

    print(f"disturbance bus={bus_ids[d]}, concentrated balancing bus={bus_ids[slack]}")
    print(f"campus={campus*base:.3f} MW, pre-step max angle={np.degrees(eq0['dmax']):.3f} deg")

    def gamma_of(conc):
        g=np.zeros(len(p))
        g[slack]+=conc
        g[is_gen]+=(1.0-conc)/int(is_gen.sum())
        return g/g.sum()

    rows=[]
    for conc in CONCENTRATIONS:
        gamma=gamma_of(float(conc))
        t0=time.time()
        cert=certified_threshold(Inc,K,p,th_old,d,gamma,campus,tol=TOL_PU)
        sim=simulated_threshold(Inc,K,p,th_old,d,gamma,campus,damp,M,
                                tol=TOL_PU,T=T_SIM,dt=DT_SIM)
        sex=static_threshold(Inc,Cyc,K,p,th_old,d,gamma,campus,use_dcb=False,tol=TOL_PU)
        sdc=static_threshold(Inc,Cyc,K,p,th_old,d,gamma,campus,use_dcb=True,tol=TOL_PU)
        rows.append([conc,cert,sim,sex,sdc])
        print(f"c={conc:4.2f}: cert={cert*base:7.1f}, sim={sim*base:7.1f}, "
              f"static={sex*base:7.1f}, DCB={sdc*base:7.1f} MW [{time.time()-t0:.1f}s]")
    rows=np.asarray(rows,float)

    checks={}
    if do_refinement_checks:
        for conc in (0.0,1.0):
            gamma=gamma_of(conc)
            a=simulated_threshold(Inc,K,p,th_old,d,gamma,campus,damp,M,
                                  tol=TOL_PU,T=20.0,dt=2e-3)
            b=simulated_threshold(Inc,K,p,th_old,d,gamma,campus,damp,M,
                                  tol=TOL_PU,T=30.0,dt=4e-3)
            checks[str(conc)]={"dt_0.002_T20_MW":a*base,"dt_0.004_T30_MW":b*base}

    payload={
      "alpha":ALPHA,"damping":DAMP,"T":T_SIM,"dt":DT_SIM,
      "disturbance_bus":int(bus_ids[d]),"concentrated_bus":int(bus_ids[slack]),
      "campus_MW":float(campus*base),"prestep_dmax_deg":float(np.degrees(eq0["dmax"])),
      "columns":["concentration","certified_pu","simulated_pu","static_exact_pu","static_dcb_pu"],
      "rows":rows.tolist(),"rows_MW":(rows[:,1:]*base).tolist(),
      "paper_expected_MW":{k:v.tolist() for k,v in TRANSIENT_MW.items()},
      "refinement_checks":checks,
    }
    (outdir/"transient_results.json").write_text(json.dumps(payload,indent=2))
    np.savez(outdir/"transient_results.npz",rows=rows,base=base,campus=campus)
    return payload

if __name__=="__main__":
    main()
