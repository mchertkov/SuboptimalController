"""Generate the two compact figures used in the final marked-up paper."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from paper_results import STATIC, CONCENTRATIONS, TRANSIENT_MW


def make_figures(outdir="figures", static_json=None, transient_json=None):
    outdir=Path(outdir); outdir.mkdir(exist_ok=True,parents=True)
    static=STATIC.copy()
    if static_json and Path(static_json).exists():
        j=json.loads(Path(static_json).read_text())
        static={k:{q:j[k][q] for q in ["dcb","cohesive","continued"]} for k in ["case118","case39"]}
    labels=["MATPOWER 118","MATPOWER 39"]
    vals=np.array([[static["case118"][q],static["case39"][q]] for q in ["dcb","cohesive","continued"]])
    x=np.arange(2); w=.22
    fig,ax=plt.subplots(figsize=(6.7,3.6))
    ax.bar(x-w,vals[0],width=w,label="DCB sufficient")
    ax.bar(x,vals[1],width=w,label="strict cohesion")
    ax.bar(x+w,vals[2],width=w,label="continued branch")
    ax.set_xticks(x,labels); ax.set_ylabel(r"loading multiplier $\alpha$"); ax.set_ylim(0,6.2)
    ax.legend(frameon=False); ax.set_title("Static loading thresholds"); fig.tight_layout()
    fig.savefig(outdir/"fig1_theory.pdf"); fig.savefig(outdir/"fig1_theory.png",dpi=180); plt.close(fig)

    tr={k:v.copy() for k,v in TRANSIENT_MW.items()}
    if transient_json and Path(transient_json).exists():
        j=json.loads(Path(transient_json).read_text())
        a=np.asarray(j["rows_MW"])
        tr={"certified":a[:,0],"simulated":a[:,1],"static_exact":a[:,2],"static_dcb":a[:,3]}
    fig,ax=plt.subplots(figsize=(6.7,3.8))
    ax.plot(CONCENTRATIONS,tr["static_exact"],marker="^",label="static strict cohesion / DCB")
    ax.plot(CONCENTRATIONS,tr["simulated"],marker="o",label="finite-horizon simulated cohesion")
    ax.plot(CONCENTRATIONS,tr["certified"],marker="s",label="energy-certified")
    ax.set_xlabel("concentration of balancing action"); ax.set_ylabel("threshold [MW]")
    ax.set_xticks(CONCENTRATIONS); ax.set_ylim(0,3000); ax.legend(frameon=False)
    ax.set_title(r"Synthetic MATPOWER case39 balanced step at $\alpha=4$")
    fig.tight_layout(); fig.savefig(outdir/"fig2_transient.pdf"); fig.savefig(outdir/"fig2_transient.png",dpi=180); plt.close(fig)

if __name__=="__main__":
    make_figures()
