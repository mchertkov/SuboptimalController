# CASE118 Stage 5A control-metric checkpoint

This package contains the first bounded step of the case118 control-metric/reinforcement campaign. It freezes the fixed-total-authority generator metric designs and their exact scalar-PIC completion geometry. Heavy Monte Carlo is intentionally split into subsequent short Stage-5A2 runs after the combined run exceeded the execution window.

Use `scripts/run_stage5a_geometry.py` to reproduce the tables. `scripts/run_stage5a.py` is retained as the attempted combined driver for provenance but should not be used as the next execution path; run the new short design-by-design jobs instead.
