"""Regenerate every case39 manuscript figure from cached machine-readable results."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('make_figures.py')), run_name='__main__')
