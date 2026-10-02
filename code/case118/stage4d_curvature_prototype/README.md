# CASE118 Stage 4D checkpoint

This package is the bounded Stage-4D adaptive-cell/interval-curvature feasibility prototype. It does not alter the manuscript.

Run from the package root:

```bash
python scripts/run_stage4d_prototype.py
```

The central result is negative in a useful sense: local mechanical variation bounds are numerically mild on millisecond cells, but the generic Feynman--Kac third-derivative spatial-Lipschitz closure is far too conservative to certify a nontrivial interior continuum cell. See `STAGE4D_REPORT.md` and `results/STAGE4D_DECISION.json`.
