"""Utilities for loading the MATPOWER 8.1 case39/case118 files without MATPOWER.

The paper uses only static MATPOWER data to build a deliberately simplified
lossless, fixed-voltage swing model.  This module parses the native ``.m`` case
files directly; no pandapower/PYPOWER dependency is required.

If ``data/case39.m`` or ``data/case118.m`` is missing, :func:`ensure_case`
tries to download the file from the MATPOWER 8.1 release commit.  For fully
offline use, put the two standard MATPOWER files in the local ``data`` folder.
"""
from __future__ import annotations

from pathlib import Path
import re
import urllib.request
import numpy as np

MATPOWER_COMMIT = "1a828c7"  # MATPOWER 8.1 release commit (July 2025)
RAW_BASES = [
    f"https://raw.githubusercontent.com/MATPOWER/matpower/{MATPOWER_COMMIT}/data",
    "https://raw.githubusercontent.com/MATPOWER/matpower/8.1/data",
    "https://raw.githubusercontent.com/MATPOWER/matpower/master/data",
]


def ensure_case(name: str, data_dir: str | Path = "data") -> Path:
    """Return a local MATPOWER case file, downloading it if necessary."""
    if not name.endswith(".m"):
        name += ".m"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / name
    if path.exists():
        return path

    errors = []
    for base in RAW_BASES:
        url = f"{base}/{name}"
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as exc:  # pragma: no cover - network dependent
            errors.append(f"{url}: {exc}")
    msg = "\n".join(errors)
    raise FileNotFoundError(
        f"Could not obtain {name}. Put the standard MATPOWER 8.1 file at {path}.\n{msg}"
    )


def _extract_scalar(text: str, name: str) -> float:
    pat = re.compile(rf"{re.escape(name)}\s*=\s*([+\-0-9.eE]+)\s*;")
    m = pat.search(text)
    if not m:
        raise ValueError(f"Could not find scalar {name}")
    return float(m.group(1))


def _extract_matrix(text: str, name: str) -> np.ndarray:
    pat = re.compile(rf"{re.escape(name)}\s*=\s*\[(.*?)\];", re.S)
    m = pat.search(text)
    if not m:
        raise ValueError(f"Could not find matrix {name}")
    block = m.group(1)
    rows = []
    for row in block.split(";"):
        row = row.split("%", 1)[0].strip()
        if not row:
            continue
        vals = [float(tok) for tok in row.replace("\t", " ").split()]
        rows.append(vals)
    if not rows:
        return np.empty((0, 0), float)
    width = max(map(len, rows))
    if any(len(r) != width for r in rows):
        raise ValueError(f"Ragged matrix while parsing {name}")
    return np.asarray(rows, float)


def load_matpower_case(path: str | Path) -> dict:
    """Parse the parts of a MATPOWER v2 case needed by this project."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {
        "baseMVA": _extract_scalar(text, "mpc.baseMVA"),
        "bus": _extract_matrix(text, "mpc.bus"),
        "gen": _extract_matrix(text, "mpc.gen"),
        "branch": _extract_matrix(text, "mpc.branch"),
        "source": str(path),
    }


if __name__ == "__main__":
    for case in ("case39", "case118"):
        p = ensure_case(case)
        mpc = load_matpower_case(p)
        print(case, mpc["bus"].shape, mpc["gen"].shape, mpc["branch"].shape)
