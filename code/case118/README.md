# Case118 campaign

Seven stages, run in this order. Each stage keeps its own `src/`, `scripts/`, frozen `data/`
inputs and `results_reference/` outputs; common modules are deliberately duplicated so that the
provenance of each frozen stage is explicit.

1. `stage3b_controller_evaluation/` — deployed H0–H3 policies (static, droop, scaled LQR).
2. `stage4_pic_certificate/` — matching geometry, deflated and completed Feynman–Kac values,
   curvature screen, terminal certificate.
3. `stage4c_activity_audit/` — variance-reduction audit of the hard activity-scaled H3 completed value.
4. `stage4d_curvature_prototype/` — adaptive-cell prototype for the interior curvature certificate
   (Appendix A; shows why a generic enclosure is out of reach).
5. `stage5a_control_metric/` — pricing of generator authority at fixed total authority.
6. `stage5b1_exact_pic_frontier/` — linearly solvable reinforcement frontiers.
7. `stage5b2_sparse_reinforcement/` — sparse load-side reinforcement, shadow vs. noise ranking.

`provenance/` holds the stage reports written when each stage was frozen. `case118_data.py`
contains the IEEE 118-bus data used, so no MATPOWER/PYPOWER installation is required.

The objective used in every stage is `SmoothCost` in `src/case118_stage3a.py`:
$V=c_E\,\max(\mathcal E,0)/n+c_B\,\overline{\mathcal B}$, $\phi=c_T\,\max(\mathcal E,0)/n$ with
$(c_E,c_B,c_T)=(0.08,0.02,0.20)$, edge-averaged Bregman barrier and an 82° guard — matching
equation (6.2) of the paper.
