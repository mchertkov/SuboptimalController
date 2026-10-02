from pathlib import Path
import pandas as pd, numpy as np, json, math, hashlib
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
base=pd.read_csv(RES/'stage4_pic_summary.csv')
ref=pd.read_csv(RES/'stage4_activity_Wplus_refined.csv')
# replace activity Wplus diagnostics with refined architecture-specific runs
for _,r in ref.iterrows():
 m=(base.noise==r.noise)&np.isclose(base.sigma_rms,r.sigma_rms)&(base.architecture==r.architecture)
 base.loc[m,'Wplus']=r.Wplus;base.loc[m,'Wplus_rep_sd']=r.Wplus_rep_sd;base.loc[m,'Wplus_ess_mean']=r.ess_fraction_mean;base.loc[m,'Wplus_ess_min']=r.ess_fraction_min
 base.loc[m,'Wplus_delta_se']=r.Wplus_delta_se_mean;base.loc[m,'Wplus_guide_alpha']=r.guide_alpha;base.loc[m,'Wplus_npaths_per_rep']=r.n_paths_per_rep
base['gap_minus']=base.J-base.Wminus;base['gap_plus']=base.J-base.Wplus
base['gap_plus_se_approx']=np.sqrt(base.J_se**2+base.Wplus_rep_sd.fillna(0.)**2+base.get('Wplus_delta_se',pd.Series(0,index=base.index)).fillna(0.)**2)
# H4 is H3 completion within each noise,sigma
h4=base[base.architecture=='H3'][['noise','sigma_rms','Wplus']].rename(columns={'Wplus':'H4_common_value'})
base=base.merge(h4,on=['noise','sigma_rms'],how='left');base['gap_to_H4']=base.J-base.H4_common_value
base['Wminus_status']='candidate: positive representative curvature screen + analytic terminal layer; no full-domain interior enclosure'
base['Wplus_status']='exact completed value is analytic HJB lower certificate; number shown is a Monte Carlo estimate of that value'
base.to_csv(RES/'stage4_pic_certificate_summary.csv',index=False)
(RES/'stage4_pic_certificate_summary.json').write_text(base.to_json(orient='records',indent=2))
# nominal compact table
nom=base[np.isclose(base.sigma_rms,.15)].copy();nom.to_csv(RES/'stage4_nominal_sigma015.csv',index=False)
# figures per noise
for noise in ['homogeneous','activity_scaled']:
 d=nom[nom.noise==noise].copy();x=np.arange(len(d));w=.25
 fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(x-w,d.Wminus,width=w,label='J- noise-deflated PIC');ax.bar(x,d.Wplus,width=w,label='J+ completed PIC');ax.bar(x+w,d.J,width=w,label='deployed LQR J');ax.set_xticks(x,d.architecture);ax.set_ylabel('smooth cost');ax.set_title(f'case118 PIC hierarchy, sigma=0.15, {noise.replace("_"," ")}');ax.legend();fig.tight_layout();fig.savefig(FIG/f'stage4_pic_hierarchy_{noise}.png',dpi=180);fig.savefig(FIG/f'stage4_pic_hierarchy_{noise}.pdf');plt.close(fig)
# curvature time plot H2/H3
cur=pd.read_csv(RES/'stage4_curvature_screen.csv')
for noise in ['homogeneous','activity_scaled']:
 d=cur[(cur.noise==noise)&cur.architecture.isin(['H2','H3'])]
 fig,ax=plt.subplots(figsize=(7.5,4.2))
 for h,g in d.groupby('architecture'):
  g=g.sort_values('t');ax.errorbar(g.t,g.trace,yerr=2*g.trace_se,marker='o',label=h)
 ax.axhline(0,lw=1);ax.set_xlabel('time t (s)');ax.set_ylabel('deleted-noise curvature trace');ax.set_title(f'case118 curvature screen, sigma=0.15, {noise.replace("_"," ")}');ax.legend();fig.tight_layout();fig.savefig(FIG/f'stage4_curvature_{noise}.png',dpi=180);fig.savefig(FIG/f'stage4_curvature_{noise}.pdf');plt.close(fig)
# concise machine decision
term=pd.read_csv(RES/'stage4_terminal_cycle_certificate.csv')
dec={
 'stage':'case118_stage4_pic_certificate',
 'status':'NUMERICS_COMPLETE_WPLUS_CERTIFIED_WMINUS_CANDIDATE_PENDING_GLOBAL_INTERIOR_ENCLOSURE',
 'manuscript_modified':False,
 'primary_sigma':0.15,
 'Wplus':{'status':'analytic HJB subsolution / exact lower value; reported values are MC estimates','activity_values_use_refined_actual_cost_CARE_guides':True},
 'Wminus':{'status':'candidate lower value only','representative_curvature_all_positive':bool((cur.trace>0).all() and (cur.min_direction>0).all()),'screen_min_trace':float(cur.trace.min()),'screen_min_direction':float(cur.min_direction.min()),'global_domain_enclosure_completed':False},
 'terminal_layer':{'cycle_energy_compact_tube':True,'min_safe_ms':float(term.analytic_safe_layer_ms.min()),'max_safe_ms':float(term.analytic_safe_layer_ms.max()),'energy_per_bus_cap':0.34,'omega_inf_bar':6.5,'delta_bar_deg':84.0},
 'pic_dt_s':0.005,
 'upper_cost_dt_s':0.0025,
 'next_theory_task':'domain-wide/adaptive-cell interior curvature enclosure if promotion of Wminus from candidate to rigorous lower certificate is required'
}
(RES/'STAGE4_CERTIFICATE_STATUS.json').write_text(json.dumps(dec,indent=2))
