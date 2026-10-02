"""Adapter from the finalized cycle-space MATPOWER reduction to this project."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from grids import load_case39 as _load_base

def load_case39(alpha: float = 4.0, data_dir=None):
    if data_dir is None:
        data_dir=Path(__file__).resolve().parents[1]/'cyclespace_base'/'data'
    c=_load_base(data_dir)
    c=dict(c)
    c['p0']=np.asarray(c['p'],float).copy()
    c['p']=float(alpha)*c['p0']
    c['alpha']=float(alpha)
    return c
