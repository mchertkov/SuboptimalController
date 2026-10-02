from pathlib import Path
import sys,csv,json,numpy as np
from scipy.stats import spearmanr, pearsonr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pic_case39 import build_case39_problem
from stochastic_swing import noise_profile
from actuation_hierarchy import build_hierarchy
from control_inflation import control_inflation_geometry, completion_diagnostics
# Homogeneous exploratory values from the systematic run (common physical-noise completion).
hom={
 .05:{'H0':(.06710116077660976,.05748545288789803),'H1':(.06721329032538668,.0576861908502615),'H2':(.0640366317039431,.05461323463183966),'H3':(.06304510727334559,.05413138265774125),'Wp':.0590502436062935,'ess':.10098338619144245},
 .10:{'H0':(.09572870193412829,.057697678479027674),'H1':(.09584403873034926,.05778127160029497),'H2':(.0927013693311533,.055107398542212395),'H3':(.0915581849442513,.056188966353778215),'Wp':.08464734576582213,'ess':.2510557070683586},
 .20:{'H0':(.20985006250901994,.058549736041662914),'H1':(.20996896312860325,.058207108354840775),'H2':(.20676934870890704,.05714747072373983),'H3':(.20517053186325693,.06426223024467811),'Wp':.18739078322510958,'ess':.3614666903919934},
 .30:{'H0':(.3953698824903767,.06040251476046517),'H1':(.39549837109557057,.0589285780039433),'H2':(.3924725541399571,.06051298108513104),'H3':(.3901186694073035,.0774158454183972),'Wp':.3588035503334406,'ess':.3615593510466418}}
# activity J,Wminus from cached exploratory hierarchy run
actrows=list(csv.DictReader(open(ROOT/'results'/'control_inflation_hierarchy_activity.csv')))
act={(float(r['sigma_rms']),r['H']):(float(r['J']),float(r['Wminus']),float(r['J_se']),float(r['cross_probability'])) for r in actrows}
# refined architecture-specific Wplus values
actWp={
 (.05,'H0'):(.059460247400713005,.5745011920669638),(.05,'H1'):(.053672794058581885,.02932683488847731),(.05,'H2'):(.05115489673720562,.004963841669291071),(.05,'H3'):(.039440883520281925,.0014292062530572801),
 (.10,'H0'):(.07846160214115662,.8140477276155571),(.10,'H1'):(.0720727414561491,.3341834172590871),(.10,'H2'):(.06881530803690536,.038937082809509756),(.10,'H3'):(.05318268128005419,.0023160476754712275),
 (.20,'H0'):(.155391054157089,.8257318054024566),(.20,'H1'):(.14494319120127552,.1639228512304148),(.20,'H2'):(.1379097004005237,.13488532181390517),(.20,'H3'):(.1131509279151211,.01028174912904058),
 (.30,'H0'):(.28169885607443546,.8201898568240961),(.30,'H1'):(.26468092387980335,.20032202005599425),(.30,'H2'):(.2542761002601527,.09635191210466634),(.30,'H3'):(.21153007600295393,.020666079089310646)}
p,meta=build_case39_problem(); hs={h.name:h for h in build_hierarchy(p,meta['bus_ids'])}; rows=[]
for kind in ['homogeneous','activity_scaled']:
 for s in [.05,.1,.2,.3]:
  Gamma=noise_profile(p,s,kind); h4=actWp[(s,'H3')][0] if kind=='activity_scaled' else hom[s]['Wp']
  for hn in ['H0','H1','H2','H3']:
   gp=control_inflation_geometry(p,hs[hn],Gamma,.2); trp=float(np.trace(gp.injection_gain_completed)); frp=float(np.linalg.norm(gp.injection_gain_completed,'fro')); diag=completion_diagnostics(np.diag(Gamma**2),gp.injection_gain_physical,gp.lam)
   if kind=='homogeneous':J,Wm=hom[s][hn];Jse=np.nan;cross=np.nan;Wp=hom[s]['Wp'];essp=hom[s]['ess']
   else:J,Wm,Jse,cross=act[(s,hn)];Wp,essp=actWp[(s,hn)]
   rows.append(dict(noise=kind,sigma_rms=s,H=hn,rank=hs[hn].rank,J=J,J_se=Jse,cross_probability=cross,
     Wminus=Wm,Wplus=Wp,H4_oracle=h4,lambda_max=gp.lam,virtual_rank=gp.virtual_rank,
     inflation_trace=gp.inflation_trace,inflation_fraction_trace=gp.inflation_trace/trp,
     inflation_fro=gp.inflation_fro,inflation_fraction_fro=gp.inflation_fro/frp,whitened_deficit_trace=diag['whitened_deficit_trace'],whitened_deficit_fro=diag['whitened_deficit_fro'],Wplus_ess=essp,
     gap_minus=J-Wm,gap_plus=J-Wp,gap_H4=J-h4,bound_improvement=Wp-Wm))
with open(ROOT/'results'/'control_inflation_hierarchy_combined.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'results'/'control_inflation_hierarchy_combined.json').write_text(json.dumps(rows,indent=2))
# Correlations architecture burden vs gaps, calculated at each sigma and pooled (burden repeats across sigma by shape).
summary={}
for kind in ['homogeneous','activity_scaled']:
 rr=[r for r in rows if r['noise']==kind]; summary[kind]={}
 for s in [.05,.1,.2,.3]:
  ss=[r for r in rr if r['sigma_rms']==s]
  x=np.array([r['inflation_fraction_trace'] for r in ss]); y=np.array([r['gap_plus'] for r in ss]); z=np.array([r['gap_H4'] for r in ss])
  summary[kind][str(s)]={'burden':x.tolist(),'gap_plus':y.tolist(),'gap_H4':z.tolist(),
     'spearman_burden_gap_plus':float(spearmanr(x,y).statistic),'pearson_burden_gap_plus':float(pearsonr(x,y).statistic),
     'spearman_burden_gap_H4':float(spearmanr(x,z).statistic),'pearson_burden_gap_H4':float(pearsonr(x,z).statistic)}
(ROOT/'results'/'control_inflation_correlation_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
