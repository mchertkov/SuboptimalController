"""Static/validation experiments reported in the revised paper."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import networkx as nx

from grids import load_case39, load_case118
from syncnet import *
from paper_results import STATIC


def graph_arrays(case):
    G = case["G"]
    Inc, Cyc, edges, nodes = incidence_and_cycles(G)
    K = np.array([G[u][v]["K"] for u, v in edges])
    return Inc, Cyc, edges, K


def validation():
    rng = np.random.default_rng(0)
    G = nx.gnm_random_graph(12, 22, seed=3)
    Inc, Cyc, edges, nodes = incidence_and_cycles(G)
    K = 1.0 + rng.random(Inc.shape[1])
    p = rng.normal(size=Inc.shape[0]); p -= p.mean()
    Fp, _ = particular_flow(Inc, K, p)
    c = Cyc.shape[1]
    psi = 0.15*rng.normal(size=c)
    ph = phase1_cycle_box(Fp, Cyc, K)
    if not np.max(np.abs((Fp+Cyc@psi)/K)) < 0.8:
        psi = ph["psi"]
    h = 1e-6
    E = np.eye(c)
    gn = np.array([(Phi(psi+h*E[k],Fp,Cyc,K)-Phi(psi-h*E[k],Fp,Cyc,K))/(2*h)
                   for k in range(c)])
    Hn = np.column_stack([(gradPhi(psi+h*E[k],Fp,Cyc,K)-gradPhi(psi-h*E[k],Fp,Cyc,K))/(2*h)
                          for k in range(c)])
    ga = gradPhi(psi,Fp,Cyc,K); Ha = hessPhi(psi,Fp,Cyc,K)
    out = solve_dual(Fp,Cyc,K)
    th,res = reconstruct_theta(Inc,out["Delta"])
    return {"grad_fd_err": float(np.max(np.abs(gn-ga))),
            "hess_fd_err": float(np.max(np.abs(Hn-Ha))),
            "reconstruction_residual": res}


def ring_counts():
    ns = np.arange(3,26)
    cnt=[]
    for n in ns:
        G=nx.cycle_graph(int(n))
        Inc,Cyc,_,_=incidence_and_cycles(G)
        K=np.ones(Inc.shape[1]); Fp=np.zeros(Inc.shape[1])
        q=(n-1)//4
        cnt.append(sum(solve_dual(Fp,Cyc,K,m=[w])["ok"] for w in range(-q-1,q+2)))
    pred=2*((ns-1)//4)+1
    return ns,np.array(cnt),pred


def static_case(case, label):
    Inc,Cyc,edges,K=graph_arrays(case)
    p0=case["p"]
    sdc,_,_=dcb_margin(Inc,K,p0)
    a_dcb=1.0/sdc
    a_coh=cohesive_threshold(Inc,Cyc,K,p0,tol=1e-6)
    # Start just inside the cohesive boundary and continue the locally stable branch.
    a0=0.995*a_coh
    eq0=newton_equilibrium(Inc,K,a0*p0)
    a_cont,eqc=continued_stable_limit(Inc,K,p0,a0,start_theta=eq0["theta"],
                                      step0=0.15,tol=1e-6)
    bind=int(np.argmax(np.abs((particular_flow(Inc,K,a_dcb*p0)[0]/K))))
    # exact binding edge at cohesive threshold
    Fp,_=particular_flow(Inc,K,a_coh*p0)
    od=solve_dual(Fp,Cyc,K)
    if od.get("ok",False): bind=int(np.argmax(np.abs(od["s"])))
    u,v=edges[bind]
    return {"label":label,"n":Inc.shape[0],"L":Inc.shape[1],"c":Cyc.shape[1],
            "dcb":a_dcb,"cohesive":a_coh,"continued":a_cont,
            "continued_lam2":float(eqc.get("lam2",np.nan)),
            "continued_dmax_deg":float(np.degrees(eqc.get("dmax",np.nan))),
            "binding_internal_edge":[int(u),int(v)],
            "binding_matpower_edge":[int(case["bus_ids"][u]),int(case["bus_ids"][v])]}


def main(outdir="results"):
    outdir=Path(outdir); outdir.mkdir(exist_ok=True,parents=True)
    print("Validation:")
    val=validation(); print(json.dumps(val,indent=2))
    ns,cnt,pred=ring_counts()
    print("Ring counts correct:",bool(np.all(cnt==pred)))

    c39=load_case39(); c118=load_case118()
    r39=static_case(c39,"case39")
    r118=static_case(c118,"case118")
    print(json.dumps(r39,indent=2)); print(json.dumps(r118,indent=2))
    payload={"validation":val,"ring_n":ns.tolist(),"ring_count":cnt.tolist(),
             "ring_prediction":pred.tolist(),"case39":r39,"case118":r118,
             "paper_expected":STATIC}
    (outdir/"static_results.json").write_text(json.dumps(payload,indent=2))
    np.savez(outdir/"static_results.npz",ring_n=ns,ring_count=cnt,ring_prediction=pred,
             case39=np.array([r39["dcb"],r39["cohesive"],r39["continued"]]),
             case118=np.array([r118["dcb"],r118["cohesive"],r118["continued"]]))
    return payload

if __name__=="__main__":
    main()
