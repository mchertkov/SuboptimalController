"""Numerical values reported in the final marked-up manuscript."""
import numpy as np

STATIC = {
    "case118": {"dcb": 4.363697, "cohesive": 5.068736, "continued": 5.268367},
    "case39":  {"dcb": 5.544474, "cohesive": 5.544472, "continued": 5.544472},
}
CONCENTRATIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
TRANSIENT_MW = {
    "certified": np.array([2035., 1751., 1442., 1186., 991.]),
    "simulated": np.array([2724., 2724., 2585., 2442., 2313.]),
    "static_exact": np.array([2724., 2724., 2724., 2724., 2724.]),
    "static_dcb": np.array([2724., 2724., 2724., 2724., 2724.]),
}
CAMPUS_MW = 2724.5
ALPHA_TRANSIENT = 4.0
DISTURBANCE_BUS = 20   # MATPOWER numbering
CONCENTRATED_BUS = 38  # MATPOWER numbering
PRESTEP_DMAX_DEG = 46.17
