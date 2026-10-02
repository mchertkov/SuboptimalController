from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(ROOT/'results'/'control_inflation_hierarchy_combined.csv')))
for r in rows:
 for k in ['sigma_rms','J','Wminus','Wplus','H4_oracle','lambda_max','inflation_trace','inflation_fraction_trace','gap_minus','gap_plus','gap_H4','bound_improvement','Wplus_ess']:
  r[k]=float(r[k])
 r['rank']=int(r['rank']);r['virtual_rank']=int(r['virtual_rank'])
Hs=['H0','H1','H2','H3'];x=np.arange(4)
# Figure 1: geometry + nominal values
fig,axs=plt.subplots(2,2,figsize=(10,7.2))
for kind,label,mark in [('homogeneous','homogeneous','o'),('activity_scaled','activity-scaled','s')]:
 rr=[r for r in rows if r['noise']==kind and abs(r['sigma_rms']-.1)<1e-12]
 rr=sorted(rr,key=lambda r:Hs.index(r['H']))
 axs[0,0].plot(x,[r['inflation_trace'] for r in rr],marker=mark,label=label)
 axs[0,1].plot(x,[r['lambda_max'] for r in rr],marker=mark,label=label)
axs[0,0].set_ylabel(r'$\mathrm{tr}(G_+-G)$');axs[0,0].set_title('absolute completion burden')
axs[0,1].set_ylabel(r'$\lambda_{\max}$');axs[0,1].set_yscale('log');axs[0,1].set_title('scalar matching temperature')
for j,(kind,title) in enumerate([('homogeneous',r'homogeneous, $\sigma_{\mathrm{rms}}=0.1$'),('activity_scaled',r'activity-scaled, $\sigma_{\mathrm{rms}}=0.1$')]):
 ax=axs[1,j];rr=sorted([r for r in rows if r['noise']==kind and abs(r['sigma_rms']-.1)<1e-12],key=lambda r:Hs.index(r['H']))
 ax.plot(x,[r['J'] for r in rr],'o-',label=r'$J_{H_k}$')
 ax.plot(x,[r['Wminus'] for r in rr],'s--',label=r'$\mathcal{J}^-_{H_k}$')
 ax.plot(x,[r['Wplus'] for r in rr],'^-',label=r'$\mathcal{J}^+_{H_k}$')
 ax.axhline(rr[-1]['H4_oracle'],linestyle=':',label='H4 oracle')
 ax.set_ylabel('cost / value');ax.set_title(title)
 ax.legend(fontsize=8)
for ax in axs.flat:
 ax.set_xticks(x,Hs);ax.grid(alpha=.25)
axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
fig.tight_layout();
for ext in ['png','pdf']:
 fig.savefig(ROOT/'figures'/f'case39_control_inflation_hierarchy.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
# Figure 2: across noise strengths
fig,axs=plt.subplots(2,2,figsize=(10,7.2),sharex=True)
for rowi,(kind,title) in enumerate([('homogeneous','homogeneous noise'),('activity_scaled','activity-scaled noise')]):
 for H in Hs:
  rr=sorted([r for r in rows if r['noise']==kind and r['H']==H],key=lambda r:r['sigma_rms'])
  s=[r['sigma_rms'] for r in rr]
  axs[rowi,0].plot(s,[r['gap_plus'] for r in rr],marker='o',label=H)
  axs[rowi,1].plot(s,[r['bound_improvement'] for r in rr],marker='o',label=H)
 axs[rowi,0].set_title(title+': $J-\mathcal{J}^+$')
 axs[rowi,1].set_title(title+': $\mathcal{J}^+-\mathcal{J}^-$')
 axs[rowi,0].set_ylabel('completed certificate gap')
 axs[rowi,1].set_ylabel('completion vs deflation')
for ax in axs[-1]: ax.set_xlabel(r'$\sigma_{\mathrm{rms}}$')
for ax in axs.flat: ax.grid(alpha=.25)
axs[0,0].legend(fontsize=8,ncol=2);axs[0,1].legend(fontsize=8,ncol=2)
fig.tight_layout()
for ext in ['png','pdf']:
 fig.savefig(ROOT/'figures'/f'case39_control_inflation_noise.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
print('made figures')
