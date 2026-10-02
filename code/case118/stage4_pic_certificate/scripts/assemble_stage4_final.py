from pathlib import Path
import pandas as pd,numpy as np,json,math,hashlib,os
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
base=pd.read_csv(RES/'stage4_pic_summary.csv')
# 1) Replace all activity-scaled coarse W+ with refined importance-guided values at dt=.005.
act=pd.read_csv(RES/'stage4_activity_Wplus_refined.csv')
base['Wplus_dt']=.005;base['Wplus_delta_se']=np.nan;base['Wplus_guide_alpha']=np.nan;base['Wplus_npaths_per_rep']=np.nan
for _,r in act.iterrows():
 m=(base.noise==r.noise)&np.isclose(base.sigma_rms,r.sigma_rms)&(base.architecture==r.architecture)
 for col,val in [('Wplus',r.Wplus),('Wplus_rep_sd',r.Wplus_rep_sd),('Wplus_ess_mean',r.ess_fraction_mean),('Wplus_ess_min',r.ess_fraction_min)]:base.loc[m,col]=val
 base.loc[m,'Wplus_delta_se']=r.Wplus_delta_se_mean;base.loc[m,'Wplus_guide_alpha']=r.guide_alpha;base.loc[m,'Wplus_npaths_per_rep']=r.n_paths_per_rep
# 2) Homogeneous W+ is architecture-independent; replace all sigmas by dt=.0025 refinement.
hom=pd.read_csv(RES/'stage4_homogeneous_Wplus_dt0025.csv')
for _,r in hom.iterrows():
 m=(base.noise=='homogeneous')&np.isclose(base.sigma_rms,r.sigma_rms)
 for col,val in [('Wplus',r.Wplus),('Wplus_rep_sd',r.Wplus_rep_sd),('Wplus_ess_mean',r.ess_fraction_mean),('Wplus_ess_min',r.ess_fraction_min)]:base.loc[m,col]=val
 base.loc[m,'Wplus_dt']=r['dt'];base.loc[m,'Wplus_guide_alpha']=r.guide_alpha;base.loc[m,'Wplus_npaths_per_rep']=r.n_paths_per_rep
# 3) Primary sigma=.15 activity completion values also refined at dt=.0025.
act15=pd.read_csv(RES/'stage4_activity_Wplus_sigma015_dt0025.csv')
for _,r in act15.iterrows():
 m=(base.noise=='activity_scaled')&np.isclose(base.sigma_rms,.15)&(base.architecture==r.architecture)
 for col,val in [('Wplus',r.Wplus),('Wplus_rep_sd',r.Wplus_rep_sd),('Wplus_ess_mean',r.ess_fraction_mean),('Wplus_ess_min',r.ess_fraction_min)]:base.loc[m,col]=val
 base.loc[m,'Wplus_dt']=r['dt'];base.loc[m,'Wplus_delta_se']=r.Wplus_delta_se_mean;base.loc[m,'Wplus_guide_alpha']=r.guide_alpha;base.loc[m,'Wplus_npaths_per_rep']=r.n_paths_per_rep
base['Wminus_dt']=.005;base['gap_minus']=base.J-base.Wminus;base['gap_plus']=base.J-base.Wplus
# approximate MC uncertainty only; not a rigorous confidence interval
base['gap_plus_mc_se_approx']=np.sqrt(base.J_se**2+base.Wplus_rep_sd.fillna(0.)**2+base.Wplus_delta_se.fillna(0.)**2)
h4=base[base.architecture=='H3'][['noise','sigma_rms','Wplus']].rename(columns={'Wplus':'H4_common_value'})
base=base.drop(columns=['H4_common_value'],errors='ignore').merge(h4,on=['noise','sigma_rms'],how='left');base['gap_to_H4']=base.J-base.H4_common_value
base['Wplus_status']='analytic HJB lower certificate for exact completed value; numerical entry is MC estimate'
base['Wminus_status']='candidate only: representative curvature screen positive and terminal layer analytic, but no full-domain interior enclosure'
base.to_csv(RES/'stage4_pic_certificate_summary_final.csv',index=False);(RES/'stage4_pic_certificate_summary_final.json').write_text(base.to_json(orient='records',indent=2))
nom=base[np.isclose(base.sigma_rms,.15)].copy();nom.to_csv(RES/'stage4_nominal_sigma015_final.csv',index=False)
# Figures at primary sigma.
for noise in ['homogeneous','activity_scaled']:
 d=nom[nom.noise==noise].copy();x=np.arange(len(d));w=.25
 fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(x-w,d.Wminus,width=w,label='noise-deflated PIC candidate');ax.bar(x,d.Wplus,width=w,label='completed PIC lower value');ax.bar(x+w,d.J,width=w,label='deployed LQR cost');ax.set_xticks(x,d.architecture);ax.set_ylabel('smooth cost');ax.set_title(f'case118 PIC hierarchy, sigma=0.15, {noise.replace("_"," ")}');ax.legend();fig.tight_layout();fig.savefig(FIG/f'stage4_pic_hierarchy_{noise}.png',dpi=180);fig.savefig(FIG/f'stage4_pic_hierarchy_{noise}.pdf');plt.close(fig)
cur=pd.read_csv(RES/'stage4_curvature_screen.csv')
for noise in ['homogeneous','activity_scaled']:
 d=cur[(cur.noise==noise)&cur.architecture.isin(['H2','H3'])]
 fig,ax=plt.subplots(figsize=(7.5,4.2))
 for h,g in d.groupby('architecture'):
  g=g.sort_values('t');ax.errorbar(g.t,g.trace,yerr=2*g.trace_se,marker='o',label=h)
 ax.axhline(0,lw=1);ax.set_xlabel('time t (s)');ax.set_ylabel('deleted-noise curvature trace');ax.set_title(f'case118 curvature screen, sigma=0.15, {noise.replace("_"," ")}');ax.legend();fig.tight_layout();fig.savefig(FIG/f'stage4_curvature_{noise}.png',dpi=180);fig.savefig(FIG/f'stage4_curvature_{noise}.pdf');plt.close(fig)
term=pd.read_csv(RES/'stage4_terminal_cycle_certificate.csv');dtc=pd.read_csv(RES/'stage4_pic_dt_spotcheck.csv')
status={
 'stage':'case118_stage4_pic_certificate','date':'2026-09-24','status':'COMPLETE_FOR_NUMERICAL_PIC_AND_WPLUS_CERTIFICATE; WMINUS_REMAINS_CANDIDATE_PENDING_DOMAIN_WIDE_INTERIOR_ENCLOSURE','manuscript_modified':False,'primary_sigma_rms':.15,
 'numerical_discretization':{'upper_cost_dt_s':.0025,'Wminus_dt_s':.005,'Wplus_homogeneous_dt_s':.0025,'Wplus_activity_primary_dt_s':.0025,'Wplus_activity_sigma_0p10_0p20_dt_s':.005,'dt_spotcheck_file':'results/stage4_pic_dt_spotcheck.csv'},
 'Wplus':{'mathematical_status':'automatic HJB subsolution / rigorous exact lower value','numerical_status':'importance-sampled Monte Carlo estimate; not itself a rigorous finite-sample confidence lower bound','H4_definition':'H3 completed value'},
 'Wminus':{'mathematical_status':'candidate lower value until curvature sign is enclosed on entire compact domain','representative_screen_all_positive':bool((cur.trace>0).all() and (cur.min_direction>0).all()),'min_trace':float(cur.trace.min()),'min_sampled_direction':float(cur.min_direction.min()),'domain_wide_interior_enclosure':False},
 'terminal_cycle_energy':{'status':'analytic compact-tube terminal layer','energy_per_bus_bar':.34,'omega_inf_bar':6.5,'delta_bar_deg':84.,'min_safe_layer_ms':float(term.analytic_safe_layer_ms.min()),'max_safe_layer_ms':float(term.analytic_safe_layer_ms.max())},
 'next_if_full_Wminus_rigor_needed':'adaptive cell/interval enclosure of the interior directional curvature over the compact stopped domain'
}
(RES/'STAGE4_CERTIFICATE_STATUS.json').write_text(json.dumps(status,indent=2))
