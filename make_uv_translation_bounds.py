r"""
Fig 6: UV-completion parameter-space bounds.

Translates the halo-derived $(\Lambda, m_\chi)$ bounds of Sec.~IV onto the
native parameter spaces of the two illustrative UV completions considered in
Sec.~V:

* **Dark-Higgs portal** — the fermionic Rayleigh operator is generated at
  low energy by integrating out a heavy dark scalar $h'$ that mixes with the
  SM Higgs. The Wilson coefficient is
      c_r/\Lambda^3 ~ (alpha/pi v) * y_{chi H} / m_h^2 * F_loop,
  with y_{chi H} = (m_chi/v') * sin(theta). This maps the halo $\Lambda$
  bound onto the (sin theta, m_{h'}) plane at a fiducial m_chi.

* **Kinetically-mixed dark photon** — anapole/charge-radius operators are
  generated at low energy by integrating out a heavy dark photon A' with
  kinetic mixing epsilon. The Wilson coefficient is
      c_{r,a}/\Lambda^2 ~ epsilon * g_chi / M_{A'}^2,
  which maps the halo $\Lambda$ bound onto the (epsilon, M_{A'}) plane at
  fiducial g_chi.

Every existing constraint region on both planes is overlaid: for the dark
Higgs, LHC signal-strength limits on |sin theta| and the LEP mass-mixing
ceiling; for the dark photon, BaBar / NA64 / LHCb visible-mode limits and the
SN1987A energy-loss bound. The halo curve is drawn as a solid black line on
top so the "novel channel, non-competitive bound" story of the paper is
visually obvious.

Usage
-----
    python make_uv_translation_bounds.py --style paper \
        --fig-width 7.05 --fig-height 3.4 --base-fontsize 10 --linewidth 1.4

Runs standalone; the master orchestrator (make_paper_results_figures.py)
adds a --only 6 dispatch that just forwards its geometry args.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from helpers.plot_style import (  # noqa: E402
    current_style, save_figure, set_paper_style, style_sizing,
    theme_accent as _accent, theme_ink as _ink, theme_legend_kw,
)


OUTPUT_DIR = _HERE / "plots"
BOUNDARY_DIR = _HERE / "constraint_boundaries"

# Colour palette — mirrors make_paper_style_operator_overlays so the
# translation plot reads as part of the same set.
#
# COL_THIS and COL_GUIDE are the two entries chosen against a WHITE page
# (black ink, a mid-dark grey band), so they are resolved through the theme
# helpers at DRAW time rather than frozen here at import: as module constants
# they would stay black-on-black when a caller selects a dark style. The named
# accents below are bright in every theme and need no remapping.
_COL_THIS_PRINT  = "black"
_COL_GUIDE_PRINT = "#666666"   # neutral grey for reference bands / guides
COL_COLLIDER = "#FF7AC6"
COL_BEAMDUMP = "#FFD25E"
COL_ASTRO    = "#71D6FF"
COL_PERTURB  = "#B18CFF"


def _theme_palette():
    """Resolve the background-dependent palette entries for the active style."""
    return _ink(_COL_THIS_PRINT), _accent(_COL_GUIDE_PRINT)


def _resolve_sizing(base_fs, linewidth):
    """Fill in type size / line width from the active style when not given.

    The panel drawing routines derive every size from these two numbers, so
    defaulting them to the active style is what lets one call render at print
    scale for the paper and at projection scale for a talk. An explicit
    argument always wins, which is how the master figure script pins the
    published paper sizing.
    """
    sizing = style_sizing(current_style())
    return (sizing["fontsize"] if base_fs is None else base_fs,
            sizing["linewidth"] if linewidth is None else linewidth)

# Physical constants
V_EW    = 246.0          # SM Higgs vev [GeV]
ALPHA   = 1.0 / 137.036
M_H_GEV = 125.25         # SM-like Higgs mass [GeV]; sets where the
                         # dark-Higgs matching bracket (1/m_h^2 - 1/m_h'^2)
                         # vanishes and the required coupling diverges
G_STAR  = 1.0            # generic dark-sector gauge coupling for the translation


# =============================================================================
# Halo-derived scattering bound → Λ_R (scalar Rayleigh) and Λ_a (anapole)
# =============================================================================

# Tag inserted before "_90cl" when locating boundary files. Mirrors
# make_paper_results_figures.EFT_BOUNDARY_SUFFIX and is overwritten by that
# driver alongside the sensitivity-map / overlay / multi-dataset modules, so
# every figure reads one grid family. Empty = the published grids.
BOUNDARY_SUFFIX = ""


def _load_halo_lambda(operator_key: str, dm_type: str,
                       profile: str = "pixelwise_global_rho2",
                       majorana: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return the (m_chi, Lambda) 90 %CL boundary produced by the pixelwise
    halo pipeline for the named operator."""
    tag_maj = "_majorana" if majorana else ""
    fname = (f"mcmc_{profile}_halo_raw_attenuation_"
             f"{dm_type}_{operator_key}{tag_maj}{BOUNDARY_SUFFIX}_90cl.npz")
    path = BOUNDARY_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"boundary file not found: {path}")
    d = np.load(path, allow_pickle=True)
    m = np.asarray(d["mchi_GeV"], dtype=float)
    L = np.asarray(d["lambda_plot_GeV"], dtype=float)
    order = np.argsort(m)
    m, L = m[order], L[order]
    finite = np.isfinite(m) & np.isfinite(L) & (m > 0) & (L > 0)
    return m[finite], L[finite]


def dark_higgs_bound(m_hprime_GeV: np.ndarray,
                       m_chi_GeV: float = 500.0,
                       vp_GeV: float = 246.0) -> np.ndarray:
    """Halo-required |sin theta| as a function of m_h', at fixed m_chi.

    The caller multiplies by (m_chi / v_EW) to obtain the required portal
    coupling y_chi H = y_chi sin(theta), which is what Fig. 5 (left) plots.
    This is a REQUIRED coupling, not an excluded one: values above the
    LHC (x) perturbativity ceiling y_chi H <= 1.2 mean the portal cannot
    generate a halo-detectable Rayleigh coefficient at all.

    MATCHING
    --------
    Photons couple to the doublet component H, the scatterer to the singlet H'.
    With H = h cos(theta) - h' sin(theta) and H' = h sin(theta) + h' cos(theta),
    the two t-channel exchanges give, in the contact limit,

        c_r / Lambda^3 = (alpha / pi v) * y_chi * sin(theta) cos(theta)
                         * F_loop * (1/m_h^2 - 1/m_h'^2)

    Three features of this matter:

      * the bracket is a DIFFERENCE, not `1/m_h'^2` alone. It vanishes as
        m_h' -> m_h, as it must: two mass-degenerate scalars can be rotated
        into one another, so the mixing angle is unphysical there and the
        portal generates no Rayleigh coupling at any coupling strength. The
        required coupling therefore DIVERGES at m_h' = m_h = 125.25 GeV.
      * it carries sin(theta)cos(theta), not sin(theta).
      * above degeneracy the bracket tends to 1/m_h^2, so the required
        coupling PLATEAUS rather than growing as m_h'^2 without limit. The
        old form overstated it by ~570x at m_h' = 3 TeV.

    The cos(theta) enters only through the exact inversion
    sin(2 theta)/2 = B, theta = arcsin(2B)/2, which has a solution only for
    2B <= 1. Over the LHC-allowed range sin(theta) <= 0.33 one has
    cos(theta) >= 0.944, so where a solution exists the correction is at most
    5.6%; where none exists the required sin(theta) exceeds unity and the
    portal is excluded outright, which is the regime this figure is in.
    arcsin's principal branch also enforces theta <= pi/4 automatically, which
    is the physically correct root: sin(theta)cos(theta) peaks at pi/4 = 0.785,
    while the LHC bound restricts theta <= 0.336 rad, so the upper root is
    already excluded by the constraint overlaid on this panel.
    """
    m_grid, lam_grid = _load_halo_lambda("scalar_rayleigh", "scalar")
    # Closest halo bound to the fiducial scatterer mass.
    idx = int(np.argmin(np.abs(m_grid - m_chi_GeV)))
    lam_R = float(lam_grid[idx])
    F_loop = 1.0  # O(1) loop factor; absorbed into the definition of the ceiling.
    m_hp = np.asarray(m_hprime_GeV, dtype=float)

    # |1/m_h^2 - 1/m_h'^2|; only the magnitude is observable since tau ~ (c_r/Lambda^3)^2.
    with np.errstate(divide="ignore", invalid="ignore"):
        bracket = np.abs(1.0 / M_H_GEV**2 - 1.0 / m_hp**2)
        # B = required sin(theta)cos(theta) = required sin(2 theta)/2
        B = (np.pi * V_EW * vp_GeV) / (ALPHA * F_loop * m_chi_GeV) / (lam_R**3 * bracket)

    # B is the REQUIRED sin(theta)cos(theta). We return it directly rather than
    # inverting to sin(theta), because over this whole panel B >~ 3e3, so
    # 2B >= 1 everywhere and no mixing angle solves the matching at all. The
    # exact inversion would therefore return NaN/inf at every point and blank
    # the figure, discarding the very quantity it exists to show: how far the
    # required coupling sits above the achievable ceiling. Where a solution does
    # exist (B < 1/2) the difference between B and sin(theta) is at most 5.6%,
    # since cos(theta) >= 0.944 over the LHC-allowed range.
    #
    # The only genuine singularity is at m_h' = m_h, where bracket = 0 and the
    # required coupling is infinite: the portal generates no Rayleigh coupling
    # at degeneracy. np.errstate lets that pass through as +inf, so the curve
    # runs off the top of the panel there rather than being masked, which would
    # read as missing data rather than as a physical null.
    return B


def dark_photon_bound(M_Aprime_GeV: np.ndarray,
                        m_chi_GeV: float = 500.0,
                        g_chi: float = 1.0) -> np.ndarray:
    """Return the halo 90 %CL upper bound on |epsilon| as a function of M_A'.

    The Dirac anapole / charge radius Wilson coefficient generated by a
    kinetically-mixed dark photon (Eq. darkphoton_to_dim6 of the paper) is
        c_{r,a} / Lambda^2 ~ epsilon * g_chi / M_A'^2.
    Inverting for epsilon:
        |epsilon|_max = (M_A'^2 / g_chi) / Lambda_a^2.
    """
    raise RuntimeError(
        "The anapole and charge-radius operators couple through d^nu F_nu_mu, "
        "whose on-shell photon vertex vanishes identically, so the tree-level "
        "gamma chi -> gamma chi cross section is exactly zero and halo "
        "attenuation places NO bound on (epsilon, M_A'). See Sec. VI B."
    )


# =============================================================================
# External-limits envelopes (analytic approximations, from published fits)
# =============================================================================

def lhc_higgs_signal_strength() -> float:
    """LHC combined signal-strength constraint on the Higgs-portal mixing angle,
    |sin theta| <= 0.33 at 95 %CL — ATLAS + CMS Run-2 (Refs. ATLASHiggs2022,
    CMSHiggs2022)."""
    return 0.33


def dark_photon_babar_bound(M_Aprime_GeV: np.ndarray) -> np.ndarray:
    """Coarse BaBar visible-mode envelope: epsilon <= 1e-3 for
    M_A' in [0.02, 10] GeV, rising as (M_A'/10 GeV)^0.5 above 10 GeV."""
    M = np.asarray(M_Aprime_GeV, dtype=float)
    env = 1e-3 * np.ones_like(M)
    env = np.where(M > 10.0, 1e-3 * np.sqrt(M / 10.0), env)
    env = np.where((M < 0.02) | (M > 1e3), np.nan, env)
    return env


def dark_photon_na64_bound(M_Aprime_GeV: np.ndarray) -> np.ndarray:
    """Coarse NA64 invisible-mode envelope: epsilon <= 5e-4 for
    M_A' in [1 MeV, 1 GeV]."""
    M = np.asarray(M_Aprime_GeV, dtype=float)
    env = 5e-4 * np.ones_like(M)
    env = np.where((M < 1e-3) | (M > 1.0), np.nan, env)
    return env


def dark_photon_sn1987a_bound(M_Aprime_GeV: np.ndarray) -> np.ndarray:
    """Coarse SN1987A energy-loss envelope: epsilon <= 1e-6 for
    M_A' in [1e-3, 1e-1] GeV."""
    M = np.asarray(M_Aprime_GeV, dtype=float)
    env = 1e-6 * np.ones_like(M)
    env = np.where((M < 1e-3) | (M > 1e-1), np.nan, env)
    return env


# =============================================================================
# Panels
# =============================================================================

def draw_dark_higgs_panel(ax, m_chi_GeV=500.0, base_fs=None, linewidth=None):
    base_fs, linewidth = _resolve_sizing(base_fs, linewidth)
    COL_THIS, COL_GUIDE = _theme_palette()
    m_hp = np.logspace(-1, 3.5, 200)  # 0.1 GeV – 3 TeV
    sin_max = dark_higgs_bound(m_hp, m_chi_GeV=m_chi_GeV)
    ax.loglog(m_hp, sin_max, color=COL_THIS, lw=linewidth * 1.4,
              label=f"Halo scattering (this work), $m_\\chi = {m_chi_GeV:.0f}$ GeV")

    # LHC signal-strength ceiling
    sin_lhc = lhc_higgs_signal_strength()
    ax.axhline(sin_lhc, color=COL_COLLIDER, lw=linewidth, ls="-",
               label=f"LHC $\\sin\\theta \\leq {sin_lhc}$ (ATLAS/CMS)")
    ax.fill_between(m_hp, sin_lhc, 10, color=COL_COLLIDER, alpha=0.15)

    # Perturbativity ceiling: y_chi = sin(theta) * m_chi / v' <= sqrt(4 pi).
    y_pert = np.sqrt(4 * np.pi)
    sin_pert = y_pert * V_EW / m_chi_GeV
    ax.axhline(min(sin_pert, 1.0), color=COL_PERTURB, lw=linewidth, ls=":",
               label=r"Perturbativity ($y_\chi = \sqrt{4\pi}$)")

    ax.set_xlim(1e-3, 3e3)
    ax.set_ylim(1e-6, 1e12)
    ax.set_xlabel(r"$m_{h'}$ [GeV]", fontsize=base_fs)
    ax.set_ylabel(r"$|\sin\theta|$", fontsize=base_fs)
    ax.set_title(r"Dark-Higgs portal", fontsize=base_fs + 1)
    ax.tick_params(labelsize=base_fs - 2)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=base_fs - 1, loc="lower right",
              **theme_legend_kw(),
              borderpad=0.4, handlelength=2.4, handletextpad=0.5,
              labelspacing=0.4)


def ew_doublet_dipole_bound():
    """Halo 90%CL upper bound on the DM magnetic dipole moment mu(m_chi).

    The dipole Compton cross section depends only on mu = 2 c_M/Lambda
    (Eq. (IV.7)), so the halo boundary Lambda(m_chi) at c_M = 1 converts
    directly to an excluded dipole moment mu >= 2/Lambda (Sec. VI C).
    Returns (m_grid [GeV], mu_bound [GeV^-1]).
    """
    m_grid, lam_grid = _load_halo_lambda("dipole_magnetic", "fermionic")
    return m_grid, 2.0 / lam_grid


def ew_doublet_dipole_prediction(m_chi_GeV):
    """Parametric one-loop magnetic dipole of a Dirac electroweak doublet
    (higgsino-like), mu ~ e g^2 / (64 pi^2 m_chi) for m_chi >> m_W, with the
    O(1) loop function set to unity.  [GeV^-1]"""
    e_em = np.sqrt(4.0 * np.pi * ALPHA)
    g_su2 = 0.65
    return e_em * g_su2**2 / (64.0 * np.pi**2 * np.asarray(m_chi_GeV, dtype=float))


# mu_B = e / (2 m_e), the electron Bohr magneton in natural units [GeV^-1].
MU_B_INV_GEV = np.sqrt(4.0 * np.pi * ALPHA) / (2.0 * 0.51099895e-3)   # 296.3

# Lambiase et al. (arXiv:2102.04840) quote mu < 2.96e-7 mu_B for a DM magnetic
# dipole from CMB energy injection.  Their observable is the annihilation-like
# injection parameter p_ann = f alpha mu^2 / m_chi, so at fixed p_ann the bound
# on mu scales as sqrt(m_chi).  Normalising the published point onto that
# scaling gives the coefficient below; it reproduces mu < 1.6e-3 GeV^-1 at
# 0.7 TeV, which is where this panel is read.
CMB_MU_REF_MU_B = 2.96e-7                       # published bound, in mu_B
CMB_MU_SQRT_COEFF = 6.13e-5                     # [GeV^-1] at m_chi = 1 GeV
CMB_M_PRIOR_MAX = 100.0                         # upper edge of their mass prior

# PandaX-4T, Nature 618, 47 (2023), interpreted for dipole-interacting DM
# following Kavanagh, Panci & Ziegler (arXiv:1810.00033).
DD_MU_AT_700 = 1.0e-6                           # [GeV^-1] at m_chi = 0.7 TeV
DD_M_MIN = 1.0e1        # below the xenon recoil-threshold turnover the
                        # sqrt(m_chi) asymptote is invalid -- do not draw there


def cmb_dipole_bound(m_chi_GeV):
    """CMB energy-injection bound on the DM magnetic dipole moment [GeV^-1].

    mu < 6.13e-5 sqrt(m_chi/GeV), the sqrt(m_chi) scaling that p_ann = f alpha
    mu^2 / m_chi implies at fixed p_ann, normalised to the published
    mu < 2.96e-7 mu_B of Lambiase et al. (arXiv:2102.04840).
    """
    return CMB_MU_SQRT_COEFF * np.sqrt(np.asarray(m_chi_GeV, dtype=float))


def dd_dipole_bound(m_chi_GeV):
    """Direct-detection bound on the DM magnetic dipole moment [GeV^-1].

    Above the recoil-threshold turnover the event rate goes as n_chi sigma ~
    mu^2/m_chi, so the limit degrades as sqrt(m_chi) -- the same scaling the
    CMB bound has, for an unrelated reason, which is why the two curves run
    parallel on this panel.  Anchored to mu < 1e-6 GeV^-1 at 0.7 TeV
    (PandaX-4T 2023 / Kavanagh, Panci & Ziegler arXiv:1810.00033).
    """
    return DD_MU_AT_700 * np.sqrt(np.asarray(m_chi_GeV, dtype=float) / 700.0)


def draw_ew_doublet_panel(ax, base_fs=None, linewidth=None,
                          mann_window=(550.0, 720.0), col_coll=COL_COLLIDER,
                          col_this=None):
    """(m_chi, mu) plane: halo dipole-attenuation bound vs the one-loop dipole
    of a Dirac electroweak doublet -- the higgsino-like candidate class the
    companion halo fit prefers."""
    base_fs, linewidth = _resolve_sizing(base_fs, linewidth)
    _this, COL_GUIDE = _theme_palette()
    if col_this is None:
        col_this = _this
    _D_XMIN, _D_XMAX = 1e0, 1e4
    _D_YMIN, _D_YMAX = 1e-8, 1e3

    m_b, mu_b = ew_doublet_dipole_bound()
    # Everything above the halo curve is a larger dipole moment, hence more
    # scattering than the Fermi-LAT halo spectrum allows: shade it as excluded
    # with a hatched brush on the boundary, the usual exclusion-plot idiom.
    ax.fill_between(m_b, mu_b, _D_YMAX, color=col_this, alpha=0.10, lw=0, zorder=0)
    ax.fill_between(m_b, mu_b, np.minimum(mu_b * 40.0, _D_YMAX),
                    facecolor="none", hatch="////", edgecolor=col_this,
                    lw=0.0, alpha=0.40, zorder=1)
    ax.loglog(m_b, mu_b, color=col_this, lw=linewidth * 1.4, zorder=4,
              label="Halo attenuation (this work)")

    # Two competing bounds on the same plane, both STRONGER than attenuation
    # for unsplit Dirac DM (Sec. VI D).
    #
    # The CMB curve here is the ENERGY-INJECTION bound of Lambiase et al., not
    # a scattering bound. The CMB elastic-scattering constraint computed in
    # constraint_generation/cmb_constraints.py applies at the CMB temperature
    # E = T_CMB0 ~ 2.3e-13 GeV, where the dipole cross section is so suppressed
    # that mu = 2/Lambda lands at 1e5-1e12 GeV^-1, far off the top of the frame.
    #
    # Both are drawn thin and grey: the panel's argument is not that the halo
    # bound is the strongest, but that it is the only one of the three that
    # survives an inelastic splitting of the doublet.
    _sub_kw = dict(color=_ink("0.45"), lw=linewidth * 0.8, zorder=3)

    m_cmb_in = np.logspace(0, np.log10(CMB_M_PRIOR_MAX), 80)
    m_cmb_ex = np.logspace(np.log10(CMB_M_PRIOR_MAX), 4, 120)
    ax.loglog(m_cmb_in, cmb_dipole_bound(m_cmb_in), ls="-", **_sub_kw,
              label=r"CMB energy injection [Lambiase+ 2021]")
    # Dashed above the upper edge of their mass prior: the p_ann formalism is
    # mass-independent so the extrapolation is controlled, but it is an
    # extrapolation and must read as one.
    ax.loglog(m_cmb_ex, cmb_dipole_bound(m_cmb_ex), ls="--", **_sub_kw,
              label=r"CMB, extrapolated $m_\chi>100$ GeV")

    # Restricted to m_chi >= 10 GeV: below the xenon recoil-threshold turnover
    # the limit degrades far faster than sqrt(m_chi) and this curve would
    # badly overstate the low-mass direct-detection reach.
    m_dd = np.logspace(np.log10(DD_M_MIN), 4, 160)
    # Dash-DOT, deliberately not the plain dash used for the CMB extrapolation:
    # the two grey curves are otherwise easy to confuse.
    ax.loglog(m_dd, dd_dipole_bound(m_dd), ls=(0, (6, 1.6, 1, 1.6)),
              color=_ink("0.30"), lw=linewidth * 0.8, zorder=3,
              label=r"Direct detection [PandaX-4T]")

    m_grid = np.logspace(0, 4, 200)
    ax.loglog(m_grid, ew_doublet_dipole_prediction(m_grid),
              color=col_coll, lw=linewidth * 1.3, ls="-", zorder=4,
              label=r"EW doublet, $\mu \sim e g^{2}/64\pi^{2} m_\chi$")

    # Neutral grey, deliberately NOT the gold used for exclusion elsewhere in
    # the figure: this band marks a reference mass window, not a bound.
    ax.axvspan(*mann_window, color=COL_GUIDE, alpha=0.22, zorder=0,
               label=r"Halo best-fit $m_\chi^{\rm ann}$")

    # Region labels and the headline number: at the halo best-fit annihilator
    # mass the predicted doublet dipole sits this far below anything the halo
    # spectrum can currently reach. Computed here, never hard-coded.
    _m_mid = float(np.sqrt(mann_window[0] * mann_window[1]))
    _mu_bound = float(10.0 ** np.interp(np.log10(_m_mid), np.log10(m_b), np.log10(mu_b)))
    _mu_pred = float(ew_doublet_dipole_prediction(np.array([_m_mid]))[0])
    _dec = np.log10(_mu_bound / _mu_pred)

    _txt_bbox = dict(boxstyle="round,pad=0.18", fc=_ink("white"), ec="none", alpha=0.72)
    ax.text(1.3e0, 1.5e2, "Excluded by halo attenuation",
            color=col_this, fontsize=base_fs - 2.5, fontweight="bold",
            ha="left", va="center", zorder=10, bbox=_txt_bbox)

    # The halo-to-prediction gap is reported here rather than drawn: both
    # overlay curves cross that span, and halo -> direct detection spans a
    # similar number of decades, so an in-panel arrow would be ambiguous. The
    # caption carries the numbers.
    _dec_dd = np.log10(_mu_bound / float(dd_dipole_bound(_m_mid)))
    _dec_cmb = np.log10(_mu_bound / float(cmb_dipole_bound(_m_mid)))
    print(f"  [Fig5 right] at m_chi = {_m_mid:.0f} GeV: halo mu = {_mu_bound:.3e}, "
          f"CMB = {cmb_dipole_bound(_m_mid):.3e} ({_dec_cmb:.2f} dex stronger), "
          f"DD = {dd_dipole_bound(_m_mid):.3e} ({_dec_dd:.2f} dex stronger), "
          f"doublet prediction = {_mu_pred:.3e} ({_dec:.2f} dex below halo).")

    ax.set_xlim(_D_XMIN, _D_XMAX)
    ax.set_ylim(_D_YMIN, _D_YMAX)
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=base_fs)
    ax.set_ylabel(r"$\mu$ [GeV$^{-1}$]", fontsize=base_fs)
    ax.set_title(r"Electroweak-doublet dipole", fontsize=base_fs + 1)
    ax.tick_params(labelsize=base_fs - 2)
    ax.grid(True, which="both", alpha=0.2)
    # Below-plot legend -- matches the left (dark-Higgs) panel's FIG5_LEGEND_KW
    # layout in make_paper_results_figures.py so both halves of Fig 5 read the
    # same way (same fontsize, same anchor below the axes, same 2-column grid).
    # Six entries now (was three).  Kept at ncol=1 to match the left panel:
    # a second column is wider than the right axes and runs underneath the
    # dark-Higgs legend.  The driver reserves the extra vertical space.
    ax.legend(fontsize=base_fs - 2, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=1,
              **theme_legend_kw(),
              borderpad=0.35, handlelength=2.0, handletextpad=0.45,
              columnspacing=0.9, labelspacing=0.30)


def draw_dark_photon_panel(ax, m_chi_GeV=500.0, base_fs=None, linewidth=None):
    base_fs, linewidth = _resolve_sizing(base_fs, linewidth)
    COL_THIS, COL_GUIDE = _theme_palette()
    M_Ap = np.logspace(-3, 2, 200)  # 1 MeV – 100 GeV
    eps_max = dark_photon_bound(M_Ap, m_chi_GeV=m_chi_GeV, g_chi=G_STAR)
    ax.loglog(M_Ap, eps_max, color=COL_THIS, lw=linewidth * 1.4,
              label=f"Halo scattering (this work), $m_\\chi = {m_chi_GeV:.0f}$ GeV")

    # BaBar / NA64 / SN1987A envelopes
    ax.loglog(M_Ap, dark_photon_babar_bound(M_Ap), color=COL_COLLIDER, lw=linewidth,
              label="BaBar (visible mode)")
    ax.loglog(M_Ap, dark_photon_na64_bound(M_Ap), color=COL_BEAMDUMP, lw=linewidth,
              label="NA64 (invisible mode)")
    ax.loglog(M_Ap, dark_photon_sn1987a_bound(M_Ap), color=COL_ASTRO, lw=linewidth,
              ls="--", label="SN1987A (energy loss)")

    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(1e-8, 1e4)
    ax.set_xlabel(r"$M_{A'}$ [GeV]", fontsize=base_fs)
    ax.set_ylabel(r"$|\epsilon|$", fontsize=base_fs)
    ax.set_title(r"Kinetically mixed dark photon", fontsize=base_fs + 1)
    ax.tick_params(labelsize=base_fs - 2)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=base_fs - 1, loc="lower right",
              **theme_legend_kw(),
              borderpad=0.4, handlelength=2.4, handletextpad=0.5,
              labelspacing=0.4)


# =============================================================================
# Entry point
# =============================================================================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outfile", default="uv_translation_bounds",
                   help="Basename for the output PDF/PNG (default: uv_translation_bounds)")
    p.add_argument("--style", default="paper")
    p.add_argument("--fig-width", type=float, default=7.05)
    p.add_argument("--fig-height", type=float, default=3.4)
    p.add_argument("--base-fontsize", type=float, default=10)
    p.add_argument("--linewidth", type=float, default=1.4)
    p.add_argument("--mchi", type=float, default=500.0,
                   help="Fiducial scatterer mass (GeV) at which the halo bound is read off.")
    args = p.parse_args(argv)

    if args.style:
        os.environ["EFT_PLOT_STYLE"] = args.style
    set_paper_style(base_fontsize=args.base_fontsize, linewidth=args.linewidth,
                    n_colors=14, cmap_name="plasma")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Two panels: the dark-Higgs portal and the electroweak-doublet dipole.
    # There is no kinetically mixed dark-photon panel, because its matched
    # operators have identically vanishing real-photon amplitudes (Sec. VI B).
    fig, (ax_h, ax_d) = plt.subplots(1, 2, figsize=(args.fig_width, args.fig_height))
    draw_dark_higgs_panel(ax_h, m_chi_GeV=args.mchi,
                          base_fs=args.base_fontsize, linewidth=args.linewidth)
    draw_ew_doublet_panel(ax_d, base_fs=args.base_fontsize, linewidth=args.linewidth)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.22)

    save_figure(fig, str(OUTPUT_DIR / args.outfile))
    fig.savefig(str(OUTPUT_DIR / f"{args.outfile}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UV translation bounds: {OUTPUT_DIR / args.outfile}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
