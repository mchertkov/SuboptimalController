#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'results/stage5b2_sparse_frontier.csv');out=ROOT/'figures';out.mkdir(exist_ok=True)
for noise in ['homogeneous','activity_scaled']:
    s=df[(df.noise==noise)&(df.k>0)]
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for method in ['noise','shadow']:
        q=s[s.method==method].sort_values('budget_fraction_of_full')
        ax.plot(100*q.budget_fraction_of_full,-100*q.relative_delta_J_pct/100,marker='o',label=('noise-only' if method=='noise' else 'H4-LQR shadow'))
    ax.set_xlabel('added load authority (% of full least completion)')
    ax.set_ylabel('deployed cost reduction vs generator-only (%)')
    ax.set_title(('Homogeneous' if noise=='homogeneous' else 'Activity-scaled')+' noise: sparse reinforcement')
    ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();
    fig.savefig(out/f'stage5b2_cost_reduction_{noise}.pdf');fig.savefig(out/f'stage5b2_cost_reduction_{noise}.png',dpi=180);plt.close(fig)

# crossing charts, one per noise
for noise in ['homogeneous','activity_scaled']:
    s=df[df.noise==noise]
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    b=s[s.k==0].iloc[0]
    ax.scatter([0],[b.crossing],label='generator-only')
    for method in ['noise','shadow']:
        q=s[(s.method==method)&(s.k>0)].sort_values('k')
        ax.plot(q.k,q.crossing,marker='o',label=('noise-only' if method=='noise' else 'H4-LQR shadow'))
    ax.set_xlabel('number of load-side devices k');ax.set_ylabel('empirical 90° crossing fraction')
    ax.set_title(('Homogeneous' if noise=='homogeneous' else 'Activity-scaled')+' noise: first-exit diagnostic')
    ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/f'stage5b2_crossing_{noise}.pdf');fig.savefig(out/f'stage5b2_crossing_{noise}.png',dpi=180);plt.close(fig)
