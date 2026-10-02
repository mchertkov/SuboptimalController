from pathlib import Path
import json, csv
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RES=ROOT/'results'; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)

def save(name):
    plt.tight_layout(); plt.savefig(FIG/f'{name}.pdf',bbox_inches='tight'); plt.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight'); plt.close()

def read_csv(name):
    with open(RES/name,newline='') as f:return list(csv.DictReader(f))

# deterministic transient
rows=read_csv('deterministic_transient.csv')
t=np.array([float(r['time']) for r in rows]); a=np.array([float(r['max_edge_angle_deg']) for r in rows]); w=np.array([float(r['max_abs_frequency']) for r in rows])
fig,ax=plt.subplots(1,2,figsize=(9,3.4)); ax[0].plot(t,a); ax[0].axhline(90,ls='--',lw=1); ax[0].set(xlabel='time (s)',ylabel='max energized-line angle (deg)',title='deterministic severe load-loss transient'); ax[1].plot(t,w);ax[1].set(xlabel='time (s)',ylabel='max |frequency state|',title='frequency excursion'); save('case39_deterministic_transient')

# noise sweep
rows=read_csv('noise_sweep.csv'); pols=['static','droop','H3 scaled LQR']; noises=['homogeneous','activity scaled']
fig,ax=plt.subplots(2,2,figsize=(9,6))
for i,nz in enumerate(noises):
    for p in pols:
        rr=[r for r in rows if r['noise']==nz and r['policy']==p]; x=[float(r['sigma_rms']) for r in rr]; c=[float(r['cost']) for r in rr]; q=[float(r['exit_probability']) for r in rr]
        ax[i,0].plot(x,c,'o-',label=p); ax[i,1].plot(x,q,'o-',label=p)
    ax[i,0].set_title(f'{nz} noise'); ax[i,1].set_title(f'{nz} noise'); ax[i,0].set_ylabel('smooth SOC cost'); ax[i,1].set_ylabel('P[cross 90 deg by T]');
    ax[i,0].set_xlabel('nodal injection-noise RMS'); ax[i,1].set_xlabel('nodal injection-noise RMS')
ax[0,0].legend(fontsize=8); save('case39_noise_sweep')

# hierarchy
rows=read_csv('h0_h3.csv'); fig,ax=plt.subplots(1,2,figsize=(9,3.5))
for j,nz in enumerate(noises):
    rr=[r for r in rows if r['noise']==nz]; H=[r['H'] for r in rr]; J=np.array([float(r['J']) for r in rr]); W=np.array([float(r['W']) for r in rr]); rk=np.array([int(r['deleted_rank']) for r in rr])
    x=np.arange(4); ax[j].plot(x,J,'o-',label=r'deployed cost $J$'); ax[j].plot(x,W,'s-',label=r'deflated value $\mathcal{J}^-$'); ax[j].fill_between(x,W,J,alpha=.12,label='conditional gap'); ax[j].set_xticks(x,H); ax[j].set_ylabel('cost / value'); ax[j].set_title(nz)
    a2=ax[j].twinx(); a2.plot(x,rk,'x--',alpha=.6); a2.set_ylabel('deleted diffusion rank')
ax[0].legend(fontsize=8); save('case39_h0_h3_hierarchy')

# implementation
impl=json.loads((RES/'implementation.json').read_text()); fig,ax=plt.subplots(2,2,figsize=(9,6)); items=[('sample_hold','sample/hold interval (s)'),('lag','lag time constant (s)'),('saturation','saturation cap / baseline q99 command'),('dropout','generator dropout probability')]
for a,(key,xlab) in zip(ax.ravel(),items):
    a.plot(impl[key]['x'],impl[key]['y'],'o-'); a.axhline(0,ls='--',lw=1); a.set(xlabel=xlab,ylabel='cost increase vs baseline',title=key.replace('_',' '))
save('case39_implementation')

# locality
loc=json.loads((RES/'locality.json').read_text()); plt.figure(figsize=(6,3.7)); plt.semilogy(loc['kappa'],np.maximum(loc['deltaJ'],1e-7),'o-'); plt.xlabel('measurement radius kappa (graph hops)'); plt.ylabel('cost increase vs full H3'); plt.title('case39 graph-local truncation'); save('case39_locality')

# PIC projection
p=json.loads((RES/'pic_feedback_projection.json').read_text()); plt.figure(figsize=(6.2,3.7)); plt.plot(p['time'],p['homogeneous'],'o-',label='homogeneous'); plt.plot(p['time'],p['activity_scaled'],'o-',label='activity scaled'); plt.axhline(p['frozen_alpha'],ls='--',label='frozen H3 scale 0.01'); plt.xlabel('time'); plt.ylabel('PIC projection onto full LQR command'); plt.title('Sparse PIC feedback estimates'); plt.legend(); save('case39_pic_feedback_projection')

# curvature
c=json.loads((RES/'curvature_screen.json').read_text()); plt.figure(figsize=(7,3.8)); plt.plot(c['state'],c['homogeneous'],'o',ms=4,label='homogeneous'); plt.plot(c['state'],c['activity_scaled'],'s',ms=4,label='activity scaled'); plt.axhline(0,lw=1); plt.xlabel('representative compact-domain state'); plt.ylabel('deleted-noise curvature rho'); plt.legend(); plt.title('case39 compact-domain curvature screen'); save('compact_curvature_screen')

# radial
r=json.loads((RES/'radial_comparison.json').read_text()); fig,ax=plt.subplots(1,2,figsize=(9,3.5)); ax[0].plot(r['angle_deg'],r['stopped_value_hom'],'o-',label='stopped failure cost'); ax[0].plot(r['angle_deg'],r['smooth_value_hom'],'o-',label='smooth barrier'); ax[0].set(xlabel='initial max edge angle (deg)',ylabel='PIC value',title='value'); ax[0].legend(fontsize=8); ax[1].plot(r['angle_deg'],r['stopped_ess_hom'],'o-',label='stopped ESS/N'); ax[1].plot(r['angle_deg'],r['smooth_ess_hom'],'o-',label='smooth ESS/N'); ax[1].set(xlabel='initial max edge angle (deg)',ylabel='effective sample fraction',title='conditioning'); ax[1].legend(fontsize=8); save('radial_formulation_comparison')

# terminal layer
term=json.loads((RES/'terminal_layer.json').read_text()); plt.figure(figsize=(6,3.7)); names=['homogeneous','activity scaled']; vals=[term['homogeneous_ms'],term['activity_scaled_ms']]; bars=plt.bar(names,vals); plt.ylabel('rigorous terminal-layer width (ms)'); plt.title('cycle-consistent energy-coupled terminal enclosure');
for b,v in zip(bars,vals): plt.text(b.get_x()+b.get_width()/2,v+0.02,f'{v:.3f}',ha='center')
save('terminal_cycle_energy_enclosure')

# cellwise
d=json.loads((RES/'cellwise_bounds.json').read_text()); plt.figure(figsize=(6.5,4)); plt.semilogy(d['radius_deg'],d['J'],'o-',label='first variation J'); plt.semilogy(d['radius_deg'],d['H'],'s-',label='second variation H'); plt.semilogy(d['radius_deg'],d['K3'],'^-',label='third variation K'); plt.xlabel('local angular cell radius (deg)'); plt.ylabel('analytic majorant after 2.5 s'); plt.legend(); plt.title('cellwise mechanical sensitivity bounds'); save('cellwise_sensitivity_radius')

# balance concentration
b=json.loads((RES/'balance_concentration.json').read_text()); fig,ax=plt.subplots(1,2,figsize=(8.7,3.4)); ax[0].plot(b['c'],b['deterministic_max_angle_deg'],'o-'); ax[0].axhline(90,ls='--',lw=1); ax[0].set(xlabel='balancing concentration c',ylabel='deterministic max angle (deg)',title='deterministic transient'); ax[1].plot(b['c'],b['exit_probability'],'o-'); ax[1].set(xlabel='balancing concentration c',ylabel='nominal stochastic exit probability',title='stochastic cohesion loss'); save('case39_balance_concentration')

# convergence
co=json.loads((RES/'numerical_convergence.json').read_text()); fig,ax=plt.subplots(1,2,figsize=(8.5,3.4)); ax[0].plot(co['dt'],co['max_angle_deg'],'o-'); ax[0].invert_xaxis(); ax[0].set(xlabel='time step dt (s)',ylabel='max angle (deg)',title='deterministic regression'); ax[1].plot(co['dt'],co['cost'],'o-'); ax[1].invert_xaxis(); ax[1].set(xlabel='time step dt (s)',ylabel='SOC cost',title='stochastic cost regression'); save('case39_numerical_convergence')

print('wrote figures to',FIG)

# control inflation/completion versus noise deflation for H3
ci=json.loads((RES/'control_inflation_h3.json').read_text())
hrows=read_csv('h0_h3.csv')
h3={r['noise']:r for r in hrows if r['H']=='H3'}
labels=['homogeneous','activity scaled']; keys=['homogeneous','activity_scaled']; x=np.arange(2); w=.24
Wd=[float(h3[l]['W']) for l in labels]; Wi=[ci[k]['W_inflation_mean'] for k in keys]; J=[float(h3[l]['J']) for l in labels]; Err=[ci[k]['W_inflation_rep_sd'] for k in keys]
plt.figure(figsize=(6.8,4.0)); plt.bar(x-w,Wd,width=w,label='noise-deflation PIC (conditional)'); plt.bar(x,Wi,width=w,yerr=Err,capsize=3,label='control-inflation PIC (automatic subsolution)'); plt.bar(x+w,J,width=w,label='deployed H3 cost (upper)'); plt.xticks(x,labels); plt.ylabel('cost / value at initial state'); plt.title('H3: dual PIC relaxations at nominal noise'); plt.legend(fontsize=8); save('case39_control_inflation')
