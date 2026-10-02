from pathlib import Path
import sys,math,json,csv
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from case118_stage3a import build_problem,noise_profile
from case118_stage4 import load_architectures,general_pic_geometry
from cellwise_stage4d import local_cell_constants,flow_majorants,global_compact_cost_derivative_bounds,action_derivative_majorants,value_third_bound

TIMES=[0.,.75,1.5,2.25,2.8]; RADII=[.02,.05,.1,.25]; SHORT=[.001,.002,.005,.01,.02]
p,_=build_problem(); hs,gains=load_architectures(p,ROOT/'data/stage3a_controller_matrices.npz')
H=hs['H3']; K=gains['H3']; An=H.nodal(p); alpha=.005; dt=.001
th=p.theta_pre.copy();om=np.zeros(p.n);states={0.:(th.copy(),om.copy())};targets={int(round(t/dt)):t for t in TIMES[1:]}
for kk in range(1,int(round(max(TIMES)/dt))+1):
    x=p.reduced_state(th[None,:],om[None,:]);v=-alpha*(x@K.T);nod=(v@An.T)[0]
    drift=(p.p_post-p.damp*om-p.network_force(th)+nod)/p.M;om=om+dt*drift;th=th+dt*om
    if kk in targets:states[targets[kk]]=(th.copy(),om.copy())
rows=[]
for t in TIMES:
    for r in RADII:
        c=local_cell_constants(p,states[t][0],r,t=t)
        rows.append(dict(t=t,radius_deg=r,dmax_deg=c.dmax_deg,mu=c.mu,c2=c.c2,c3=c.c3,edge_dual=c.edge_dual))
pd.DataFrame(rows).to_csv(ROOT/'results/stage4d_local_constants.csv',index=False)
rows2=[]
for t in TIMES:
    c=local_cell_constants(p,states[t][0],.02,t=t)
    for h in SHORT:
        j,hh,k=flow_majorants(c.mu,c.c2,c.c3,h)
        rows2.append(dict(t=t,radius_deg=.02,step_s=h,J1=j,H2=hh,K3=k))
pd.DataFrame(rows2).to_csv(ROOT/'results/stage4d_short_step_variations.csv',index=False)
# Deliberately optimistic feasibility test: use the very small 0.02-degree local drift constants for the ENTIRE remaining horizon.
# Global compact-domain cost derivative bounds remain rigorous on the stopped tube. If this optimistic construction already yields
# microscopic cells, the generic sup-norm D^3W/Lipschitz route is not useful for a full cover.
cost=global_compact_cost_derivative_bounds(p)
cur=pd.read_csv(ROOT/'data/stage4_curvature_screen.csv')
stress=[]
for t in TIMES:
    c=local_cell_constants(p,states[t][0],.02,t=t); horizon=3.-t
    C1,C2,C3=action_derivative_majorants(c.mu,c.c2,c.c3,cost,horizon)
    for noise in ['homogeneous','activity_scaled']:
        gm=general_pic_geometry(p,H,noise_profile(p,.15,noise),.2)
        W3=value_third_bound(C1,C2,C3,gm.lam)
        trace_hat=float(np.trace(gm.omega_deleted@np.diag(p.M))); Lrho=trace_hat*W3
        rr=cur[(cur.noise==noise)&(cur.architecture=='H3')&(np.isclose(cur.t,t))].iloc[0]
        margin=float(rr.trace-2*rr.trace_se)
        r_mech=margin/Lrho
        edge_deg=math.degrees(c.edge_dual*r_mech)
        target_mech=math.radians(.1)/c.edge_dual
        ratio=Lrho*target_mech/margin
        hopf_term=8*C1**3/(gm.lam**2)
        stress.append(dict(noise=noise,architecture='H3',sigma_rms=.15,t=t,remaining_s=horizon,lambda_max=gm.lam,
                           rho_center=float(rr.trace),rho_se=float(rr.trace_se),rho_center_minus_2se=margin,
                           local_radius_deg_used=.02,mu=c.mu,c2=c.c2,c3=c.c3,C1=C1,C2=C2,C3=C3,
                           W3_bound=W3,hopf_C1_term=hopf_term,trace_hat=trace_hat,Lrho_energy=Lrho,
                           admissible_mechanical_radius_from_numerical_anchor=r_mech,
                           implied_edge_radius_deg=edge_deg,bound_loss_ratio_for_0p1deg_ball=ratio))
pd.DataFrame(stress).to_csv(ROOT/'results/stage4d_lipschitz_stress_test.csv',index=False)
term=pd.read_csv(ROOT/'data/stage4_terminal_cycle_certificate.csv')
# summary/decision
sdf=pd.DataFrame(stress); ldf=pd.DataFrame(rows); vdf=pd.DataFrame(rows2)
decision={
 'stage':'case118_stage4D_adaptive_cell_prototype','status':'PROTOTYPE_COMPLETE_DO_NOT_EXPAND_NAIVE_GLOBAL_COVER',
 'manuscript_modified':False,
 'reference':'frozen H3 scaled-LQR deterministic transient; sigma=0.15 curvature anchors from Stage 4',
 'local_mechanical_result':{
   'mu_radius_0p02_range':[float(ldf[ldf.radius_deg==.02].mu.min()),float(ldf[ldf.radius_deg==.02].mu.max())],
   'c2_radius_0p02_range':[float(ldf[ldf.radius_deg==.02].c2.min()),float(ldf[ldf.radius_deg==.02].c2.max())],
   'c3_radius_0p02_range':[float(ldf[ldf.radius_deg==.02].c3.min()),float(ldf[ldf.radius_deg==.02].c3.max())],
   'max_J1_over_2ms':float(vdf[vdf.step_s==.002].J1.max()),
   'max_H2_over_2ms':float(vdf[vdf.step_s==.002].H2.max()),
   'max_K3_over_2ms':float(vdf[vdf.step_s==.002].K3.max())},
 'lipschitz_stress_test':{
   'max_implied_edge_radius_deg_homogeneous':float(sdf[sdf.noise=='homogeneous'].implied_edge_radius_deg.max()),
   'max_implied_edge_radius_deg_activity_scaled':float(sdf[sdf.noise=='activity_scaled'].implied_edge_radius_deg.max()),
   'min_loss_ratio_for_0p1deg_homogeneous':float(sdf[sdf.noise=='homogeneous'].bound_loss_ratio_for_0p1deg_ball.min()),
   'min_loss_ratio_for_0p1deg_activity_scaled':float(sdf[sdf.noise=='activity_scaled'].bound_loss_ratio_for_0p1deg_ball.min())},
 'rigorous_status':{
   'new_interior_cells_certified':0,
   'terminal_layer_remains_rigorous':True,
   'reason':'The local flow-variation ODEs are numerically benign on millisecond cells, but the generic sup-norm Feynman-Kac D^3W bound, dominated by inverse powers of the scalar PIC temperature, makes the spatial rho Lipschitz enclosure many orders too conservative. Curvature center values are Monte Carlo anchors and cannot by themselves certify an interior continuum cell.'},
 'recommendation':'Stop the naive full-domain adaptive-cell expansion. Retain J- as a candidate lower value with strong positive numerical evidence and analytic terminal layer; use J+ as the rigorous lower certificate. Move next to case118 control-metric/reinforcement experiments unless a sharper structure-exploiting curvature theorem is developed.'}
(ROOT/'results/STAGE4D_DECISION.json').write_text(json.dumps(decision,indent=2))
print(json.dumps(decision,indent=2))
