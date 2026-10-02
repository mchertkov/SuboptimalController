"""Production recomputation map for the case39 campaign.

This file is intentionally a concise orchestration guide rather than a hidden
one-click black box.  The paper comparisons should be rerun with fixed seeds and
saved raw data before overwriting cached summaries.
"""
STEPS = [
    "cyclespace_base/exp2_transient.py: reproduce deterministic bus-20 load-loss baseline",
    "src/stochastic_swing.py: Monte Carlo true-system noise and controller sweeps",
    "src/actuation_hierarchy.py: construct H0--H3, LQR families, and matching geometry",
    "src/smooth_pic.py + src/pic_case39.py: importance-guided Feynman--Kac/PIC values",
    "src/variational_bounds.py + src/certification_bounds.py: terminal and cellwise bounds",
    "scripts/make_figures.py: regenerate all manuscript figures from saved summaries",
]
if __name__ == '__main__':
    print("Case39 production recomputation order:\n")
    for i,s in enumerate(STEPS,1): print(f"{i}. {s}")
