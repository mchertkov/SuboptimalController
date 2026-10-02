#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
d=json.load(open(ROOT/'results'/'reinforcement_planning.json'))
F=ROOT/'figures';F.mkdir(exist_ok=True)

def save(name):
    plt.tight_layout();plt.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.savefig(F/(name+'.png'),dpi=180,bbox_inches='tight');plt.close()

# 1. PIC reinforcement frontier
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for kind,label in [('homogeneous','homogeneous'),('activity_scaled','activity scaled')]:
    rows=d['frontier'][kind]
    x=[r['effective_power_authority_MW'] for r in rows];y=[r['W'] for r in rows]
    ax[0].plot(x,y,'o-',label=label);ax[1].plot([r['completion_trace'] for r in rows],y,'o-',label=label)
ax[0].set_xlabel('effective added power-authority proxy (MW)');ax[1].set_xlabel(r'completion authority $\mathrm{tr}\,\Delta G$')
for a in ax:a.set_ylabel(r'completed PIC value $\mathcal{J}^+$');a.grid(alpha=.25);a.legend()
save('case39_pic_planning_frontier')

# 2. control metric design
fig,ax=plt.subplots(1,2,figsize=(10.8,4.2))
for j,kind in enumerate(['homogeneous','activity_scaled']):
    r=d['metric_design'][kind]; names=['fixed_R','noise_proportional_equalized_R']; labels=['fixed R','noise-equalized R']
    vals=[r[n]['completion_trace'] for n in names]; bars=ax[j].bar(labels,vals)
    ax[j].set_title(kind.replace('_',' '));ax[j].set_ylabel(r'completion burden $\mathrm{tr}\,\Delta G$');ax[j].grid(axis='y',alpha=.25)
    for b,n in zip(bars,names):ax[j].text(b.get_x()+b.get_width()/2,b.get_height(),rf'$\mathcal{{J}}^+={r[n]["W"]:.3f}$',ha='center',va='bottom',fontsize=9)
save('case39_control_metric_design')

# 3. same total authority allocation: burden + W+
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
names=['fixed_R','noise_proportional_equalized_R'];labels=['uniform\nauthority','noise-proportional\nauthority']
x=np.arange(2);width=.36
for i,kind in enumerate(['homogeneous','activity_scaled']):
    r=d['metric_design'][kind]
    vals=[r[n]['completion_trace'] for n in names]
    ax[0].bar(x+(i-.5)*width,vals,width,label=kind.replace('_',' '))
    vals2=[r[n]['W'] for n in names]
    ax[1].bar(x+(i-.5)*width,vals2,width,label=kind.replace('_',' '))
ax[0].set_xticks(x,labels);ax[1].set_xticks(x,labels);ax[0].set_ylabel(r'completion burden $\mathrm{tr}\,\Delta G$');ax[1].set_ylabel(r'completed value $\mathcal{J}^+$')
ax[0].set_title('same total generator authority');ax[1].set_title('same physical authority budget')
for a in ax:a.grid(axis='y',alpha=.25);a.legend()
save('case39_authority_allocation')

# 4. scenario flexibility
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for j,kind in enumerate(['homogeneous','activity_scaled']):
    r=d['scenario_flexibility'][kind];names=['uniform','robust_fixed','activity_specific'];labels=['uniform','robust fixed','activity-specific']
    vals=[r[n]['completion_trace'] for n in names];bars=ax[j].bar(labels,vals)
    ax[j].set_title(kind.replace('_',' '));ax[j].set_ylabel(r'completion burden $\mathrm{tr}\,\Delta G$');ax[j].tick_params(axis='x',rotation=15);ax[j].grid(axis='y',alpha=.25)
    for b,n in zip(bars,names):
        txt=rf'$\mathcal{{J}}^+={r[n]["W"]:.3f}$'+'\n'+rf'$J={r[n]["policy"]["J_mean"]:.3f}$'
        ax[j].text(b.get_x()+b.get_width()/2,b.get_height(),txt,ha='center',va='bottom',fontsize=8)
fig.suptitle('fixed versus scenario-reconfigurable generator authority')
save('case39_scenario_flexibility')

# 5. sparse physical reinforcement
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for j,kind in enumerate(['activity_scaled','homogeneous']):
    r=d['sparse'][kind]
    for method,label in [('noise','noise'),('shadow_proxy','dynamic shadow proxy')]:
        rows=r[method];x=[z['k'] for z in rows];y=[z['J_mean'] for z in rows];e=[z['J_se'] for z in rows]
        ax[j].errorbar(x,y,yerr=e,marker='o',label=label,capsize=2)
    ax[j].set_title(kind.replace('_',' '));ax[j].set_xlabel('number of added nodal devices');ax[j].set_ylabel('nonlinear deployed-policy cost');ax[j].grid(alpha=.25);ax[j].legend()
save('case39_sparse_reinforcement')
