#!/usr/bin/env python3
"""
Make the Section 2 "money plot": maximum cosmological optical depth.

The Higgs-portal curve saturates the perturbativity and LHC Higgs-mixing
bounds, while the gravitational curve uses the analytic PQG scaling quoted in
the transfer report.
"""

import os
import sys
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import quad

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers.trinity_plotting import set_paper_style


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA = 1.0 / 137.036
V_EW = 246.22                 # GeV
M_W = 80.379                  # GeV
M_H = 125.10                  # GeV
GAMMA_H = 4.07e-3             # GeV
M_TOP = 172.76                # GeV
N_C_TOP = 3.0
Q_TOP = 2.0 / 3.0

SIN_THETA_MAX = 0.33
Y_CHI_PERT_MAX = np.sqrt(4.0 * np.pi)
Y_EFF_MAX = (ALPHA / (np.pi * V_EW)) * Y_CHI_PERT_MAX * SIN_THETA_MAX

E_GAMMA = 175.0               # GeV -- top-quark loop peak (tau_top = 4 m_top^2/|t| ~ 1
                              #        in backscattering at m_chi >> E_gamma), where the
                              #        H -> gamma gamma form factor plateau maximises.
J_COSMO = 1.37e22             # GeV / cm^2
GEV2_TO_CM2 = 3.8938e-28

# Galactic Centre NFW baseline, matching the convention in tau_vs_mchi.py.
KPC_TO_CM = 3.0857e21
RHO_S_NFW = 0.184             # GeV / cm^3
R_S_NFW_KPC = 24.42           # kpc
R_SUN_KPC = 8.5               # kpc

MCHI_MIN = 1.0                # GeV
MCHI_MAX = 1.0e19             # GeV
N_MCHI = 100

TAU_GRAV_ANCHOR_MCHI = 1.0e8  # GeV
TAU_GRAV_ANCHOR = 10.0 ** (-67.8)

FONT_PRESETS = {
    "paper": {
        "label": 18.0,
        "legend": 14.0,
        "ticks": 14.0,
    },
}


def resolve_plot_config(*, base_fontsize=None, linewidth=None,
                        fig_width=None, fig_height=None):
    """Return the paper-style plotting config used by Totani figures.

    Overrides (all optional):
      base_fontsize   — rescales FONT_PRESETS['paper'] label/legend/tick sizes
                        so that "label" ≈ base_fontsize + 1 (matches the
                        internal scaling of the trinity paper style).
      linewidth       — set_paper_style linewidth argument.
      fig_width,      — figsize override; if both provided, replaces the
      fig_height        default (6.4, 5.4).
    """
    lw = 1.6 if linewidth is None else float(linewidth)
    # Pass base_fontsize through so this figure matches the rest of the paper
    # set (Times New Roman via trinity paper styling at the same base size).
    bfs = 10.0 if base_fontsize is None else float(base_fontsize)
    set_paper_style(base_fontsize=bfs, linewidth=lw, n_colors=12, cmap_name="plasma")
    cmap = plt.get_cmap("plasma")

    fonts = dict(FONT_PRESETS["paper"])
    if base_fontsize is not None:
        # Preserve internal ratios: label ≈ base+1, legend ≈ base-1, ticks ≈ base-1
        b = float(base_fontsize)
        fonts["label"]  = b + 1.0
        fonts["legend"] = b - 1.0
        fonts["ticks"]  = b - 1.0

    if fig_width is not None and fig_height is not None:
        figsize = (float(fig_width), float(fig_height))
    else:
        figsize = (6.4, 5.4)

    return {
        "fonts": fonts,
        "figsize": figsize,
        "line_colors": {
            "higgs": cmap(0.5),
            "grav": cmap(0.2),
            "threshold": "k",
            "excluded": "0.7",
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legend-loc",
        default="center",
        help="Matplotlib legend loc string used with the legend anchor.",
    )
    parser.add_argument(
        "--legend-x",
        type=float,
        default=None,
        help="Legend x coordinate in axes fraction units. Use with --legend-y.",
    )
    parser.add_argument(
        "--legend-y",
        type=float,
        default=None,
        help="Legend y coordinate in axes fraction units. Use with --legend-x.",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="Plot style: paper, conference/dark_transparent, or conference_light/light_transparent.",
    )
    # Print-geometry overrides for PRD revtex4 column widths
    parser.add_argument("--fig-width", type=float, default=None,
                        help="Figure width [inches]. Overrides internal figsize (6.4).")
    parser.add_argument("--fig-height", type=float, default=None,
                        help="Figure height [inches]. Overrides internal figsize (5.4).")
    parser.add_argument("--base-fontsize", type=float, default=None,
                        help="Base font size for set_paper_style. Overrides FONT_PRESETS['paper'] label/legend/tick sizes proportionally.")
    parser.add_argument("--linewidth", type=float, default=None,
                        help="Line width for set_paper_style. Overrides default (1.6).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Higgs -> gamma gamma loop form factors
# ---------------------------------------------------------------------------
def loop_f(tau):
    """Analytic-continuation helper for the H -> gamma gamma form factors."""
    tau = np.asarray(tau, dtype=complex)
    out = np.zeros_like(tau, dtype=complex)

    above = tau.real >= 1.0
    if np.any(above):
        out[above] = np.arcsin(1.0 / np.sqrt(tau[above])) ** 2

    below = ~above
    if np.any(below):
        root = np.sqrt(1.0 - tau[below])
        out[below] = -0.25 * (np.log((1.0 + root) / (1.0 - root)) - 1j * np.pi) ** 2

    return out.item() if out.ndim == 0 else out


def form_factor_fermion(tau):
    """Spin-1/2 H -> gamma gamma form factor."""
    tau = np.asarray(tau, dtype=complex)
    value = -2.0 * tau * (1.0 + (1.0 - tau) * loop_f(tau))
    return value.item() if value.ndim == 0 else value


def form_factor_w(tau):
    """Spin-1 H -> gamma gamma form factor."""
    tau = np.asarray(tau, dtype=complex)
    value = 2.0 + 3.0 * tau + 3.0 * tau * (2.0 - tau) * loop_f(tau)
    return value.item() if value.ndim == 0 else value


# ---------------------------------------------------------------------------
# Scattering and optical-depth helpers
# ---------------------------------------------------------------------------
def mandelstam_s_lab(mchi, e_gamma):
    """Lab-frame s for a stationary DM target."""
    return mchi**2 + 2.0 * mchi * e_gamma


def mandelstam_t_exact(mchi, e_gamma, theta):
    """Exact elastic 2 -> 2 momentum transfer for gamma-chi scattering."""
    one_minus_cos = 1.0 - np.cos(theta)
    recoil = mchi / (mchi + e_gamma * one_minus_cos)
    return -2.0 * e_gamma**2 * one_minus_cos * recoil


def higgs_portal_dsigma_domega(theta, mchi, e_gamma=E_GAMMA, y_eff=Y_EFF_MAX):
    """Differential cross section in GeV^-2 sr^-1."""
    t = mandelstam_t_exact(mchi, e_gamma, theta)
    abs_t = max(abs(t), 1e-300)
    s = mandelstam_s_lab(mchi, e_gamma)

    tau_w = 4.0 * M_W**2 / abs_t
    tau_top = 4.0 * M_TOP**2 / abs_t
    amp = form_factor_w(tau_w) + N_C_TOP * Q_TOP**2 * form_factor_fermion(tau_top)
    amp2 = float((amp * np.conjugate(amp)).real)

    propagator = (2.0 * mchi**2 - 0.5 * t) / ((t - M_H**2) ** 2 + M_H**2 * GAMMA_H**2)
    return (
        y_eff**2
        * (3.0 * t**2 / 8.0)
        * propagator
        * amp2
        / (64.0 * np.pi**2 * s)
    )


def sigma_tot_gev2(mchi, e_gamma=E_GAMMA):
    """Total Higgs-portal cross section in GeV^-2."""
    def integrand(theta):
        return 2.0 * np.pi * np.sin(theta) * higgs_portal_dsigma_domega(
            theta, mchi, e_gamma=e_gamma
        )

    split_points = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, np.pi, 25),
                np.pi - np.logspace(-8, -1, 16),
            ]
        )
    )
    split_points = split_points[(split_points > 0.0) & (split_points < np.pi)]

    value, error = quad(
        integrand,
        0.0,
        np.pi,
        points=split_points,
        epsabs=1e-60,
        epsrel=1e-7,
        limit=300,
    )
    if error > max(1e-50, 1e-4 * abs(value)):
        print(f"Warning: large quad error at m_chi={mchi:.3e}: {error:.3e}")
    return value


def tau_higgs_portal(mchi):
    sigma_cm2 = sigma_tot_gev2(mchi) * GEV2_TO_CM2
    return (J_COSMO / mchi) * sigma_cm2


def rho_nfw(r_kpc):
    x = r_kpc / R_S_NFW_KPC
    return RHO_S_NFW / (x * (1.0 + x) ** 2)


def galactic_j_factor():
    l_los = np.concatenate(
        [
            np.linspace(0.001, 0.5, 20_000),
            np.linspace(0.5, R_SUN_KPC, 20_000),
        ]
    )
    r_los = np.sqrt(l_los**2 + R_SUN_KPC**2 - 2.0 * l_los * R_SUN_KPC)
    r_los = np.maximum(r_los, 0.001)
    return np.trapz(rho_nfw(r_los), l_los * KPC_TO_CM)


def tau_from_sigma(mchi, sigma_gev2, j_factor):
    sigma_cm2 = sigma_gev2 * GEV2_TO_CM2
    return (j_factor / mchi) * sigma_cm2


def tau_gravitational_pqg(mchi, j_factor=J_COSMO):
    tau_cosmo = TAU_GRAV_ANCHOR * (mchi / TAU_GRAV_ANCHOR_MCHI)
    return tau_cosmo * (j_factor / J_COSMO)


def monotonic_report(mchi, tau):
    diffs = np.diff(tau)
    if np.all(diffs >= 0.0):
        return "monotonic increasing over the displayed range"
    if np.all(diffs <= 0.0):
        return "monotonic decreasing over the displayed range"

    turn_idx = np.where(np.signbit(diffs[:-1]) != np.signbit(diffs[1:]))[0]
    if len(turn_idx) == 0:
        idx = int(np.argmin(diffs))
    else:
        idx = int(turn_idx[0] + 1)
    return f"turns over near m_chi = {mchi[idx]:.3e} GeV"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plot(
    mchi,
    tau_higgs_cosmo,
    tau_higgs_gal,
    tau_grav_cosmo,
    tau_grav_gal,
    output_png,
    *,
    legend_loc="center",
    legend_xy=None,
    base_fontsize=None,
    linewidth=None,
    fig_width=None,
    fig_height=None,
):
    plot_cfg = resolve_plot_config(
        base_fontsize=base_fontsize,
        linewidth=linewidth,
        fig_width=fig_width,
        fig_height=fig_height,
    )
    fonts = plot_cfg["fonts"]
    colors = plot_cfg["line_colors"]

    fig, ax = plt.subplots(figsize=plot_cfg["figsize"])

    ax.fill_between(
        mchi,
        np.maximum(tau_higgs_cosmo, tau_higgs_gal),
        1.0,
        color=colors["excluded"],
        alpha=0.15,
    )
    ax.loglog(
        mchi,
        tau_higgs_gal,
        color=colors["higgs"],
        ls="-",
        lw=2.2,
    )
    ax.loglog(
        mchi,
        tau_higgs_cosmo,
        color=colors["higgs"],
        ls="--",
        lw=2.2,
    )
    ax.loglog(
        mchi,
        tau_grav_gal,
        color=colors["grav"],
        ls="-",
        lw=2.0,
    )
    ax.loglog(
        mchi,
        tau_grav_cosmo,
        color=colors["grav"],
        ls="--",
        lw=2.0,
    )
    ax.axhline(
        1e-2,
        color=colors["threshold"],
        ls=":",
        lw=1.4,
    )

    ax.set_xlim(1.0, 1.0e19)
    ax.set_ylim(1.0e-70, 1.0)
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=fonts["label"])
    ax.set_ylabel(r"$\tau_{\rm max}$", fontsize=fonts["label"])
    ax.tick_params(axis="both", which="major", labelsize=fonts["ticks"], length=5)
    ax.tick_params(axis="both", which="minor", length=3)
    ax.grid(True, which="both", alpha=0.22)
    handles = [
        Patch(facecolor=colors["excluded"], edgecolor=colors["excluded"], alpha=0.15,
              label="Non-perturbative or excluded by LHC"),
        Line2D([0], [0], color=colors["higgs"], lw=2.2,
               label=r"Higgs portal (saturated)"),
        Line2D([0], [0], color=colors["grav"], lw=2.0,
               label=r"Gravitational (PQG)"),
        Line2D([0], [0], color="0.15", lw=1.6, ls="-",
               label="Galactic baseline"),
        Line2D([0], [0], color="0.15", lw=1.6, ls="--",
               label="Cosmological baseline"),
        Line2D([0], [0], color=colors["threshold"], lw=1.4, ls=":",
               label=r"Fermi-LAT threshold $\tau_{\rm obs} \sim 10^{-2}$"),
    ]
    # Shared legend styling (Totani make_paper_results_figures convention) so
    # this figure matches the rest of the paper set.
    LEGEND_KW = dict(frameon=True, framealpha=0.6, facecolor="white", edgecolor="0.7")
    legend_kwargs = dict(
        handles=handles,
        loc=legend_loc,
        fontsize=fonts["legend"],
        borderpad=0.4,
        **LEGEND_KW,
    )
    if legend_xy is not None:
        legend_kwargs["bbox_to_anchor"] = legend_xy
        legend_kwargs["bbox_transform"] = ax.transAxes
    ax.legend(**legend_kwargs)

    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    # Also emit a vector PDF alongside the PNG so LaTeX can include it as a
    # first-class member of the paper figure set (same font/geometry as the
    # other trinity-paper-styled PDFs).
    output_pdf = os.path.splitext(output_png)[0] + ".pdf"
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    plot_dir = os.path.join(repo_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    mchi = np.logspace(np.log10(MCHI_MIN), np.log10(MCHI_MAX), N_MCHI)
    j_gal = galactic_j_factor()
    sigma_higgs = np.array([sigma_tot_gev2(m) for m in mchi])
    tau_higgs_cosmo = tau_from_sigma(mchi, sigma_higgs, J_COSMO)
    tau_higgs_gal = tau_from_sigma(mchi, sigma_higgs, j_gal)
    tau_grav_cosmo = tau_gravitational_pqg(mchi, J_COSMO)
    tau_grav_gal = tau_gravitational_pqg(mchi, j_gal)

    output_png = os.path.join(plot_dir, "mchi_vs_tau_perturbative.png")
    output_npz = os.path.join(plot_dir, "mchi_vs_tau_perturbative.npz")
    legend_xy = None
    if args.legend_x is not None or args.legend_y is not None:
        if args.legend_x is None or args.legend_y is None:
            raise SystemExit("Use --legend-x and --legend-y together.")
        legend_xy = (args.legend_x, args.legend_y)

    make_plot(
        mchi,
        tau_higgs_cosmo,
        tau_higgs_gal,
        tau_grav_cosmo,
        tau_grav_gal,
        output_png,
        legend_loc=args.legend_loc,
        legend_xy=legend_xy,
        base_fontsize=args.base_fontsize,
        linewidth=args.linewidth,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )
    np.savez(
        output_npz,
        m_chi=mchi,
        tau_higgs=tau_higgs_cosmo,
        tau_grav=tau_grav_cosmo,
        tau_higgs_cosmo=tau_higgs_cosmo,
        tau_higgs_gal=tau_higgs_gal,
        tau_grav_cosmo=tau_grav_cosmo,
        tau_grav_gal=tau_grav_gal,
        J_cosmo=J_COSMO,
        J_gal=j_gal,
    )

    sigma_1tev_gev2 = sigma_tot_gev2(1.0e3)
    sigma_1tev_cm2 = sigma_1tev_gev2 * GEV2_TO_CM2
    tau_1tev = (J_COSMO / 1.0e3) * sigma_1tev_cm2
    tau_grav_1e8 = tau_gravitational_pqg(1.0e8)

    print(f"y_eff_max = {Y_EFF_MAX:.6e} GeV^-1")
    print(f"J_cosmo = {J_COSMO:.6e} GeV/cm^2")
    print(f"J_gal   = {j_gal:.6e} GeV/cm^2")
    print(
        "sigma_tot(m_chi=1 TeV, E_gamma=500 GeV) = "
        f"{sigma_1tev_gev2:.6e} GeV^-2 = {sigma_1tev_cm2:.6e} cm^2"
    )
    print(f"tau_higgs(m_chi=1 TeV) = {tau_1tev:.6e}")
    print(f"tau_grav(m_chi=1e8 GeV) = {tau_grav_1e8:.6e}")
    print(f"tau_grav anchor target      = {TAU_GRAV_ANCHOR:.6e}")
    print(f"Cosmological Higgs-portal curve is {monotonic_report(mchi, tau_higgs_cosmo)}.")
    print(f"Galactic Higgs-portal curve is {monotonic_report(mchi, tau_higgs_gal)}.")
    print(f"Saved {output_png}")
    print(f"Saved {output_npz}")


if __name__ == "__main__":
    main()
