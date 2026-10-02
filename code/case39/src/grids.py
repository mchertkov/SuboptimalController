"""Build the lossless fixed-voltage reductions used in the paper."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import networkx as nx

from matpower_loader import ensure_case, load_matpower_case


def build_lossless_case(mpc: dict, inertia_gen=0.30, inertia_load=0.02):
    """Convert MATPOWER static data to the simplified lossless model.

    Construction used in the revised paper:
      * active in-service branches only;
      * fixed voltage magnitudes ``Vm`` from the MATPOWER solved case;
      * coupling K_ij = Vm_i Vm_j / (|x_ij| tap_ij), with tap=1 when MATPOWER
        stores 0; parallel branches are added;
      * nodal injections p=(Pg-Pd)/baseMVA and then projected onto sum(p)=0,
        because resistive losses are absent from the reduced model;
      * simple synthetic inertias: generator buses vs. load-only buses.
    """
    bus = mpc["bus"]
    gen = mpc["gen"]
    branch = mpc["branch"]
    base = float(mpc["baseMVA"])

    bus_ids = bus[:, 0].astype(int)
    index = {bid: i for i, bid in enumerate(bus_ids)}
    n = len(bus_ids)
    Vm = bus[:, 7].astype(float)
    Pd = bus[:, 2].astype(float) / base

    Pg = np.zeros(n)
    is_gen = np.zeros(n, dtype=bool)
    for row in gen:
        status = int(round(row[7])) if len(row) > 7 else 1
        if status <= 0:
            continue
        i = index[int(round(row[0]))]
        Pg[i] += row[1] / base
        is_gen[i] = True

    K_by_edge: dict[tuple[int, int], float] = {}
    for row in branch:
        status = int(round(row[10])) if len(row) > 10 else 1
        if status <= 0:
            continue
        fb, tb = int(round(row[0])), int(round(row[1]))
        if fb == tb:
            continue
        x = float(row[3])
        if abs(x) < 1e-14:
            continue
        tap = float(row[8]) if len(row) > 8 else 0.0
        tap = tap if abs(tap) > 1e-14 else 1.0
        i, j = index[fb], index[tb]
        kij = Vm[i] * Vm[j] / (abs(x) * abs(tap))
        key = (min(i, j), max(i, j))
        K_by_edge[key] = K_by_edge.get(key, 0.0) + kij

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), kij in K_by_edge.items():
        G.add_edge(i, j, K=float(kij))
    if not nx.is_connected(G):
        raise ValueError("Reduced network is not connected")

    p = Pg - Pd
    p = p - p.mean()  # remove small AC-loss mismatch from the lossless model
    M = np.where(is_gen, float(inertia_gen), float(inertia_load))
    return {
        "G": G,
        "p": p,
        "M": M,
        "is_gen": is_gen,
        "baseMVA": base,
        "bus_ids": bus_ids,
        "Vm": Vm,
        "Pd": Pd,
        "Pg": Pg,
        "source": mpc.get("source"),
    }


def load_case39(data_dir: str | Path = "data"):
    path = ensure_case("case39", data_dir)
    return build_lossless_case(load_matpower_case(path), inertia_gen=0.30, inertia_load=0.02)


def load_case118(data_dir: str | Path = "data"):
    path = ensure_case("case118", data_dir)
    return build_lossless_case(load_matpower_case(path), inertia_gen=0.20, inertia_load=0.02)


if __name__ == "__main__":
    for fn in (load_case39, load_case118):
        c = fn()
        G = c["G"]
        print(fn.__name__, "n=", G.number_of_nodes(), "L=", G.number_of_edges(),
              "c=", G.number_of_edges() - G.number_of_nodes() + 1,
              "sum p=", c["p"].sum())
