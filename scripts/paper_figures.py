"""paper_figures.py -- regenerate the paper's case118 figures and Figure 1 from frozen results.

Usage:  python scripts/paper_figures.py [OUTDIR]
Inputs: data/frozen/*.csv  (written by the case118 stage code in code/).
All figures are written as both PDF and PNG.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "frozen"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7, "figure.dpi": 150,
                     "savefig.bbox": "tight", "legend.frameon": False,
                     "mathtext.fontset": "cm"})
NAVY, RED, ORANGE, GREEN, GREY = "#1b3a6b", "#c0392b", "#e67e22", "#2d7d46", "#7f8c8d"
NOISE_LABEL = {"homogeneous": "homogeneous noise", "activity_scaled": "activity-scaled noise"}
METRIC_LABEL = {"uniform": "uniform", "robust_fixed": "robust fixed", "activity_specific": "activity-specific"}


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    return out / f"{name}.pdf"


# --------------------------------------------------------------------------- #
def fig_geometry(out: Path):
    """Sketch (not a computation) of how the four constructions relate.

    Each dot is one stochastic direction i in a common eigenbasis of Q and G (literally one
    bus when noise and control channels are independent): horizontal = control authority
    g_i, vertical = noise variance q_i. Exact scalar linear solvability means all dots lie on
    one ray q = lambda g. Panel (a): the physical problem. (b) completion moves each dot
    right onto the lowest ray. (c) deflation moves each dot down onto it. (d) BOTC moves
    each dot up onto the steepest ray; unactuated dots (g_i = 0) have no finite target."""
    g = np.array([0.60, 0.50, 0.90, 1.20])
    q = np.array([0.60, 0.95, 1.30, 1.60])
    qu = np.array([0.72, 1.90])                       # unactuated directions, g = 0
    lo, hi = np.min(q / g), np.max(q / g)              # lambda_star, lambda_plus
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.3), sharex=True, sharey=True)
    xs = np.linspace(0, 2.15, 10)
    kw = dict(arrowstyle="-|>", lw=1.2, mutation_scale=10, shrinkA=3.5, shrinkB=1.5)

    def base(ax, title, show_lo=True, show_hi=False):
        if show_lo:
            ax.plot(xs, lo * xs, color=NAVY, lw=1.2)
        if show_hi:
            xh = xs[hi * xs <= 2.4]
            ax.plot(xh, hi * xh, color=RED, lw=1.1, ls="--")
        ax.scatter(g, q, s=30, color="k", zorder=5)
        ax.scatter(np.zeros_like(qu), qu, s=34, facecolor="white", edgecolor="k", zorder=6,
                   clip_on=False)
        ax.set_title(title, fontsize=8.6, loc="left")
        ax.set_xlim(-0.03, 2.15); ax.set_ylim(0, 2.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # (a) physical problem
    ax = axs[0, 0]
    base(ax, "(a) physical problem: dots not on one ray", show_lo=True, show_hi=True)
    ax.text(1.55, 1.98, r"lowest ray, slope $\lambda_\star$", color=NAVY, fontsize=7.5, ha="left")
    ax.text(1.02, 2.26, r"steepest ray, slope $\lambda^{+}$", color=RED, fontsize=7.5, ha="left")
    ax.annotate("actuated direction", xy=(g[3], q[3]), xytext=(1.30, 0.75), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY), color=GREY)
    ax.annotate("unactuated\n($g_i=0$)", xy=(0.0, qu[1]), xytext=(0.07, 1.38),
                fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY), color=GREY)

    # (b) completion
    ax = axs[0, 1]
    base(ax, "(b) completion: add authority\n     lower bound, automatic")
    for gi, qi in zip(np.r_[g, 0 * qu], np.r_[q, qu]):
        if abs(qi / max(gi, 1e-12) - lo) > 1e-9:
            ax.annotate("", xy=(qi / lo, qi), xytext=(gi, qi), arrowprops=dict(color=NAVY, **kw))
    ax.text(0.05, 2.2, "open dots receive virtual authority", fontsize=7.2, color=NAVY)

    # (c) deflation
    ax = axs[1, 0]
    base(ax, "(c) deflation: delete noise\n     lower bound, if curvature sign holds")
    for gi, qi in zip(np.r_[g, 0 * qu], np.r_[q, qu]):
        if abs(qi - lo * gi) > 1e-9:
            ax.annotate("", xy=(gi, lo * gi), xytext=(gi, qi), arrowprops=dict(color=GREEN, **kw))
    ax.text(0.05, 2.2, "open dots lose all their noise", fontsize=7.2, color=GREEN)

    # (d) BOTC
    ax = axs[1, 1]
    base(ax, "(d) BOTC: add noise (or raise price)\n     upper bound", show_lo=False, show_hi=True)
    for gi, qi in zip(g, q):
        if abs(qi - hi * gi) > 1e-9:
            ax.annotate("", xy=(gi, hi * gi), xytext=(gi, qi), arrowprops=dict(color=RED, **kw))
    for qi in qu:
        ax.plot([0.06], [qi], marker="x", color=RED, ms=6, mew=1.3)
    ax.text(0.12, 2.2, "open dots: no finite target", fontsize=7.2, color=RED)

    for ax in axs[1, :]:
        ax.set_xlabel(r"control authority $g_i$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"noise variance $q_i$")
    fig.tight_layout()
    return _save(fig, out, "fig_geometry")


# --------------------------------------------------------------------------- #
def fig_case118_certificate(out: Path):
    c = pd.read_csv(DATA / "stage5a3_certificate_summary.csv")
    # the activity-specific / activity-scaled deployed cost uses the 3072-path audit
    m = c["J_refined"].notna()
    c.loc[m, "J"] = c.loc[m, "J_refined"]
    c.loc[m, "J_se"] = c.loc[m, "J_refined_se"]
    c["gap"] = c["J"] - c["Jplus"]
    c["gap_pct"] = 100 * c["gap"] / c["J"]
    c["gap_pct_se"] = 100 * c["J_se"] / c["J"]
    trace = {("homogeneous", "uniform"): 320.0, ("activity_scaled", "uniform"): 14832.3,
             ("homogeneous", "activity_specific"): 2202.5, ("activity_scaled", "activity_specific"): 34.54,
             ("homogeneous", "robust_fixed"): 682.55, ("activity_scaled", "robust_fixed"): 682.55}
    order = ["uniform", "robust_fixed", "activity_specific"]
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0))
    w = 0.36
    for k, (noise, col) in enumerate([("homogeneous", NAVY), ("activity_scaled", ORANGE)]):
        d = c[c.noise == noise].set_index("design").loc[order]
        x = np.arange(3) + (k - 0.5) * w
        ax[0].bar(x, d.gap_pct, w, yerr=2 * d.gap_pct_se, color=col, capsize=2, label=NOISE_LABEL[noise],
                  error_kw=dict(lw=0.8))
        ax[1].bar(x, [trace[(noise, o)] for o in order], w, color=col, label=NOISE_LABEL[noise])
        for xi, v, e in zip(x, d.gap_pct, d.gap_pct_se):
            ax[0].text(xi, v + 2 * e + 0.8, f"{v:.1f}", ha="center", fontsize=7)
    for a in ax:
        a.set_xticks(range(3)); a.set_xticklabels([METRIC_LABEL[o] for o in order])
        a.set_xlabel("pricing of generator authority")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel(r"certificate gap $(J-\mathcal{J}^{+})/J$  [%]")
    ax[0].set_title(r"(a) certified suboptimality ($\pm2$ s.e.)", fontsize=9)
    ax[0].set_ylim(0, 46)
    ax[0].legend(fontsize=7.5, loc="upper right")
    ax[1].set_yscale("log"); ax[1].set_ylabel(r"completion burden $\mathfrak{I}$")
    ax[1].set_title("(b) completion burden", fontsize=9)
    fig.tight_layout()
    return _save(fig, out, "fig_case118_certificate")


# --------------------------------------------------------------------------- #
def fig_case118_frontier(out: Path):
    f = pd.read_csv(DATA / "stage5b1_frontier_final.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for (design, noise), col, mk in [(("uniform", "homogeneous"), NAVY, "o"),
                                      (("robust_fixed", "homogeneous"), NAVY, "s"),
                                      (("activity_specific", "activity_scaled"), ORANGE, "o"),
                                      (("robust_fixed", "activity_scaled"), ORANGE, "s")]:
        d = f[(f.design == design) & (f.noise == noise)].sort_values("fraction", ascending=False)
        se = d.Wplus_rep_se.fillna(0).values
        ls = "-" if mk == "o" else "--"
        ax.errorbar(d.effective_power_authority_MW / 1000, d.Wplus, yerr=2 * se, color=col, marker=mk,
                    ms=4, lw=1.3, ls=ls, capsize=2,
                    label=f"{NOISE_LABEL[noise].split()[0]}, {METRIC_LABEL[design]} pricing")
    ax.set_xlabel("authority proxy of completed control [GW]")
    ax.set_ylabel(r"completed value $\mathcal{J}^{+}_{\lambda}$")
    ax.legend(fontsize=7, loc="center right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return _save(fig, out, "fig_case118_frontier")


# --------------------------------------------------------------------------- #
def fig_case118_sparse(out: Path):
    s = pd.read_csv(DATA / "stage5b2_sparse_frontier.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for a, noise in zip(ax, ["activity_scaled", "homogeneous"]):
        for method, col, lab in [("noise", GREY, "rank by local noise"), ("shadow", NAVY, "rank by shadow value")]:
            d = s[(s.noise == noise) & (s.method.isin([method, "baseline"]))].sort_values("k")
            a.errorbar(100 * d.budget_fraction_of_full, -d.relative_delta_J_pct,
                       yerr=2 * 100 * d.delta_J_vs_k0_se / d.J.iloc[0], color=col, marker="o", ms=3.5,
                       lw=1.3, capsize=2, label=lab)
        a.axhline(0, color="k", lw=0.5)
        a.set_xlabel("added load authority [% of least completion]")
        a.set_title(NOISE_LABEL[noise], fontsize=9)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("deployed-cost reduction [%]")
    ax[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return _save(fig, out, "fig_case118_sparse")


# --------------------------------------------------------------------------- #
def fig_case118_deployed(out: Path):
    d = pd.read_csv(DATA / "stage3b_all_results.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for a, noise in zip(ax, ["homogeneous", "activity_scaled"]):
        base = d[(d.noise == noise) & (d.policy == "static")].set_index("sigma_rms").J_mean
        for pol, col, ls in [("H2_droop", GREY, "--"), ("H2_lqr", GREY, "-"),
                             ("H3_droop", NAVY, "--"), ("H3_lqr", NAVY, "-")]:
            x = d[(d.noise == noise) & (d.policy == pol)].set_index("sigma_rms").sort_index()
            red = 100 * (base.loc[x.index] - x.J_mean) / base.loc[x.index]
            a.plot(x.index, red, color=col, ls=ls, marker="o", ms=3.5,
                   label=pol.replace("_lqr", " scaled LQR").replace("_droop", " droop"))
        a.axhline(0, color="k", lw=0.5)
        a.set_xlabel(r"noise RMS $\sigma_{\rm rms}$"); a.set_title(NOISE_LABEL[noise], fontsize=9)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("cost reduction vs. static balancing [%]")
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    return _save(fig, out, "fig_case118_deployed")


def main(outdir=None):
    out = Path(outdir) if outdir else ROOT / "figures" / "generated"
    made = [fig_geometry(out), fig_case118_certificate(out), fig_case118_frontier(out),
            fig_case118_sparse(out), fig_case118_deployed(out)]
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)


# =========================================================================== #
#  IEEE two-column variants (used by the IEEE TCNS version of the paper)
# =========================================================================== #
def fig_geometry_row(out: Path):
    """Figure 1 as one row of four panels, for a full-width IEEE figure*."""
    g = np.array([0.60, 0.50, 0.90, 1.20]); q = np.array([0.60, 0.95, 1.30, 1.60])
    qu = np.array([0.72, 1.90]); lo, hi = np.min(q / g), np.max(q / g)
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.05), sharey=True)
    xs = np.linspace(0, 2.15, 10)
    kw = dict(arrowstyle="-|>", lw=0.9, mutation_scale=7, shrinkA=2.5, shrinkB=1)
    titles = ["(a) physical problem", "(b) completion: add authority",
              "(c) deflation: delete noise", "(d) BOTC: add noise"]
    sub = ["dots not on one ray", "lower bound, automatic",
           "lower bound, if curvature holds", "upper bound"]
    for k, ax in enumerate(axs):
        if k in (0, 1, 2):
            ax.plot(xs, lo * xs, color=NAVY, lw=1.0)
        if k in (0, 3):
            xh = xs[hi * xs <= 2.4]; ax.plot(xh, hi * xh, color=RED, lw=0.9, ls="--")
        ax.scatter(g, q, s=12, color="k", zorder=5)
        ax.scatter(np.zeros_like(qu), qu, s=14, facecolor="white", edgecolor="k", zorder=6,
                   clip_on=False, lw=0.8)
        ax.set_title(f"{titles[k]}\n{sub[k]}", fontsize=7, loc="left")
        ax.set_xlim(-0.03, 2.15); ax.set_ylim(0, 2.4)
        ax.tick_params(labelsize=6)
        ax.set_xlabel(r"authority $g_i$", fontsize=7, labelpad=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel(r"noise variance $q_i$", fontsize=7)
    axs[0].text(1.05, 2.22, r"slope $\lambda^{+}$", color=RED, fontsize=6.5)
    axs[0].text(1.45, 1.20, r"slope $\lambda_\star$", color=NAVY, fontsize=6.5)
    axs[0].text(1.05, 0.22, "open: unactuated\nfilled: actuated", fontsize=6, color=GREY)
    for gi, qi in zip(np.r_[g, 0 * qu], np.r_[q, qu]):
        if abs(qi / max(gi, 1e-12) - lo) > 1e-9:
            axs[1].annotate("", xy=(qi / lo, qi), xytext=(gi, qi), arrowprops=dict(color=NAVY, **kw))
        if abs(qi - lo * gi) > 1e-9:
            axs[2].annotate("", xy=(gi, lo * gi), xytext=(gi, qi), arrowprops=dict(color=GREEN, **kw))
    for gi, qi in zip(g, q):
        if abs(qi - hi * gi) > 1e-9:
            axs[3].annotate("", xy=(gi, hi * gi), xytext=(gi, qi), arrowprops=dict(color=RED, **kw))
    for qi in qu:
        axs[3].plot([0.09], [qi], marker="x", color=RED, ms=4, mew=1.0)
    axs[3].text(1.0, 0.22, "open dots:\nno finite target", fontsize=6, color=RED)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    return _save(fig, out, "fig_geometry_row")


def _cert_frame():
    c = pd.read_csv(DATA / "stage5a3_certificate_summary.csv")
    m = c["J_refined"].notna()
    c.loc[m, "J"] = c.loc[m, "J_refined"]; c.loc[m, "J_se"] = c.loc[m, "J_refined_se"]
    c["gap_pct"] = 100 * (c["J"] - c["Jplus"]) / c["J"]; c["gap_pct_se"] = 100 * c["J_se"] / c["J"]
    return c


def fig_case118_certificate_col(out: Path):
    """Certificate gap and completion burden, two stacked panels at column width."""
    c = _cert_frame()
    trace = {("homogeneous", "uniform"): 320.0, ("activity_scaled", "uniform"): 14832.3,
             ("homogeneous", "activity_specific"): 2202.5, ("activity_scaled", "activity_specific"): 34.54,
             ("homogeneous", "robust_fixed"): 682.55, ("activity_scaled", "robust_fixed"): 682.55}
    order = ["uniform", "robust_fixed", "activity_specific"]
    lab = ["uniform", "robust fixed", "noise-proportional"]
    fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.3), sharex=True)
    w = 0.36
    for k, (noise, col) in enumerate([("homogeneous", NAVY), ("activity_scaled", ORANGE)]):
        d = c[c.noise == noise].set_index("design").loc[order]
        x = np.arange(3) + (k - 0.5) * w
        ax[0].bar(x, d.gap_pct, w, yerr=2 * d.gap_pct_se, color=col, capsize=1.5,
                  error_kw=dict(lw=0.6), label=NOISE_LABEL[noise])
        for xi, v, e in zip(x, d.gap_pct, d.gap_pct_se):
            ax[0].text(xi, v + 2 * e + 0.7, f"{v:.1f}", ha="center", fontsize=6)
        ax[1].bar(x, [trace[(noise, o)] for o in order], w, color=col)
    ax[0].set_ylabel("certified gap [%]", fontsize=7); ax[0].set_ylim(0, 47)
    ax[0].legend(fontsize=6.5, loc="upper right")
    ax[1].set_yscale("log"); ax[1].set_ylabel(r"burden $\mathfrak{I}$", fontsize=7)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(lab, fontsize=7)
    for a in ax:
        a.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    return _save(fig, out, "fig_case118_certificate_col")


def fig_case118_sparse_col(out: Path):
    s = pd.read_csv(DATA / "stage5b2_sparse_frontier.csv")
    fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.2), sharex=True)
    for a, noise in zip(ax, ["activity_scaled", "homogeneous"]):
        for method, col, lab in [("noise", GREY, "rank by local noise"), ("shadow", NAVY, "rank by shadow value")]:
            d = s[(s.noise == noise) & (s.method.isin([method, "baseline"]))].sort_values("k")
            a.errorbar(100 * d.budget_fraction_of_full, -d.relative_delta_J_pct,
                       yerr=2 * 100 * d.delta_J_vs_k0_se / d.J.iloc[0], color=col, marker="o", ms=2.5,
                       lw=1.0, capsize=1.5, label=lab)
        a.axhline(0, color="k", lw=0.4); a.set_title(NOISE_LABEL[noise], fontsize=7, pad=2)
        a.set_ylabel("cost reduction [%]", fontsize=7); a.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(fontsize=6.5, loc="upper left")
    ax[1].set_xlabel("added load authority [% of least completion]", fontsize=7)
    fig.tight_layout(pad=0.3)
    return _save(fig, out, "fig_case118_sparse_col")


def main_ieee(outdir=None):
    out = Path(outdir) if outdir else ROOT / "figures" / "generated"
    for f in (fig_geometry_row, fig_case118_certificate_col, fig_case118_sparse_col):
        print("wrote", f(out))


def fig_case118_frontier_col(out: Path):
    f = pd.read_csv(DATA / "stage5b1_frontier_final.csv")
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    for (design, noise), col, mk, ls in [(("uniform", "homogeneous"), NAVY, "o", "-"),
                                          (("robust_fixed", "homogeneous"), NAVY, "s", "--"),
                                          (("activity_specific", "activity_scaled"), ORANGE, "o", "-"),
                                          (("robust_fixed", "activity_scaled"), ORANGE, "s", "--")]:
        d = f[(f.design == design) & (f.noise == noise)].sort_values("fraction", ascending=False)
        ax.errorbar(d.effective_power_authority_MW / 1000, d.Wplus, yerr=2 * d.Wplus_rep_se.fillna(0),
                    color=col, marker=mk, ms=2.5, lw=1.0, ls=ls, capsize=1.5,
                    label=f"{noise.split('_')[0]}, {METRIC_LABEL[design].replace('activity-specific','noise-prop.')}")
    ax.set_xlabel("authority proxy of completed control [GW]", fontsize=7)
    ax.set_ylabel(r"$\mathcal{J}^{+}_{\lambda}$", fontsize=7)
    ax.tick_params(labelsize=6.5); ax.legend(fontsize=5.8, loc="center right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    return _save(fig, out, "fig_case118_frontier_col")
