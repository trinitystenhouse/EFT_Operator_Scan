#!/usr/bin/env python3
r"""
Paper result figures for "Photon-Dark Matter Elastic Scattering: An
Effective-Operator Scan and First Limits from the Galactic Halo". This is the
single source of truth for every figure in the paper. Every knob you might want to
change (font sizes, colours, LEGEND_KW, panel geometry, per-figure widths and
heights, operator lists, contour levels, legend placement, tight_layout rects,
subplots_adjust wspace/hspace) lives here in one place. The sub-modules
(make_uv_complete_tau_vs_mchi, make_sensitivity_map, make_paper_style_operator_overlays,
make_multi_dataset_overlay, make_uv_translation_bounds) supply the low-level
data loaders and per-panel drawing routines that this script calls in-process.

Figures (matches paper numbering)
---------------------------------
  f1_uv_complete_tau_vs_mchi     (--only 1) §II   perturbative UV-complete
                                            tau vs m_chi for the dark-Higgs
                                            portal and PQG channels.
  f2_sensitivity_map             (--only 2) §III  8-panel f_required heatmap
                                            covering the full operator basis.
  f3_halo_constraints            (--only 3) §IV.C 3-panel halo exclusion
                                            overlay with collider / DD / ID /
                                            cosmology bounds.
  f4_multi_dataset_overlay       (--only 4) §IV.D halo + IGRB cross-dataset
                                            overlay with optional PPPC WW/bb
                                            benchmarks at the companion best-fit
                                            annihilator masses.
  f5_uv_translation_bounds       (--only 5) §V    two-panel (sin theta, m_h')
                                            dark-Higgs and electroweak-doublet
                                            dipole translations of the halo
                                            bound onto two illustrative UV
                                            completions.

Usage
-----
  python make_paper_results_figures.py                  # all five
  python make_paper_results_figures.py --only 2 4       # selected
  python make_paper_results_figures.py --copy-to /path/to/paper/paper_plots

Editing recipes
---------------
* Global font, line width, box style, palette      -> CONFIG block below.
* Per-figure width/height                          -> FIG*_WIDTH/HEIGHT block.
* Per-figure legend contents / layout / anchors    -> the corresponding figN_*
                                                       function in this file.
* Per-figure operator list, contour levels, masses -> per-figure knobs block.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

_HERE = Path(__file__).resolve().parent          # repository root
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))                   # for core.*, helpers.*, totani_helpers.*


def _rel(p):
    """Path for display: relative to the repo when possible, else as given."""
    p = Path(p)
    try:
        return p.relative_to(_HERE)
    except ValueError:
        return p

DEFAULT_OUT_DIR = _HERE / "paper_plots"
from helpers.trinity_plotting import apply_plot_style, current_savefig_kwargs, get_cmap_colors, plasma_color  # noqa: E402

# =============================================================================
# CONFIG: style, geometry, palette
# =============================================================================
# Edit anything here to change appearance across ALL figures at once.

STYLE = "paper"

# --- Print geometry (PRD revtex4 twocolumn) --------------------------------
COL_W = 3.4      # single column \columnwidth [inches]
FULL_W = 7.05    # full width  \textwidth   [inches] (figure* environment)
BASE_FS = 10     # base_fontsize passed to set_paper_style
LINEWIDTH = 1.4  # linewidth passed to set_paper_style
ANNOT_FS = 8     # one size for dense in-panel annotations

# --- Shared legend styling (Totani convention). Every figN() uses this ------
LEGEND_KW = dict(frameon=True, framealpha=0.6, facecolor="white", edgecolor="0.7")

# --- Palette (mirrors make_paper_style_operator_overlays for consistency) ---
cols = get_cmap_colors(cmap_name="plasma", n=10, start=0, end=1)

COL_THIS_WORK = "black"
COL_COLLIDER  = "#FF7AC6"   # pink
COL_DIRECT    = "#F2D53C"   # yellow
COL_INDIRECT  = "#9B6DFF"   # purple
COL_COSMOLOGY = "#F26D0C"   # orange -- distinct from COL_DIRECT/COL_COLLIDER so cosmology curves stand out
COL_GUIDE     = "#666666"   # grey (EFT/unitarity guides)
COL_EFT_LABEL = 'cyan'   # teal accent for "EFT valid" annotation
COL_EXCLUSION = 'cyan'   # cyan for excluded regions
COL_FOLLOWUP  = "#0B7285"   # deep teal for modest exposure follow-up
COL_BEAMDUMP  = "#FFD25E"   # yellow-orange (NA64 etc.)
COL_ASTRO     = "#71D6FF"   # cyan (SN1987A)
COL_PERTURB   = "#B18CFF"   # perturbativity guide

FIG1_HIGGS_COLOR          = cols[5]
FIG1_GRAV_COLOR           = cols[2]
FIG1_THRESHOLD_COLOR      = "black"
FIG1_EXCLUSION_FILL_COLOR = "gold"
FIG1_EXCLUSION_TEXT_COLOR = "#8A6A00"

# =============================================================================
# PER-FIGURE GEOMETRY (width, height in inches)
# =============================================================================
FIG1_WIDTH,  FIG1_HEIGHT  = COL_W,  3.2   # §II   tau vs m_chi, single column + legend below
FIG2_WIDTH,  FIG2_HEIGHT  = FULL_W, 4.6   # §III  6-panel f_required grid (3 cols x 2 rows)
FIG3_WIDTH,  FIG3_HEIGHT  = FULL_W, 3.8   # §IV.C halo constraints with external bounds
FIG3_LEGEND_Y             = 0.1          # bottom-strip legend anchor; higher = closer to x-axis labels
FIG4_WIDTH,  FIG4_HEIGHT  = FULL_W, 3.8   # §IV.D multi-dataset overlay (3-col legend strip)
FIG4_LEGEND_Y             = 0.1          # bottom legend anchor; higher = closer to x-axis labels
FIG4_LEGEND_NCOL          = 2
FIG5_WIDTH,  FIG5_HEIGHT  = FULL_W, 5.0   # §V    UV-completion parameter-space bounds

# =============================================================================
# PER-FIGURE KNOBS (operator lists, contour levels, benchmark masses)
# =============================================================================
# Fig 2 sensitivity-map contour levels (multiplicative exposure boosts).
# FIG2_CONTOUR_COLOURS maps each level in FIG2_CONTOURS to a colour drawn on
# top of the heatmap in every panel; FIG2_CONTOUR_LABELS drives the legend.
# Edit either dict to recolour or rename the exposure lines across every
# panel of Fig 2 without touching make_sensitivity_map.py.
FIG2_CONTOURS = [1.0, 10.0, 50.0, 1.0e3, 1.0e6]
FIG2_CONTOUR_COLOURS = {
    1.0:   "#111111",   # black    -- current Fermi-LAT 17yr boundary (solid)
    10.0:  cols[2],   # teal     -- x10 exposure
    50.0:  cols[3],   # purple   -- CTA-scale (x50)
    1.0e3: cols[5],   # pink     -- x1000 exposure
    1.0e6: cols[8],   # cyan     -- x1e6 parametric limit
}
FIG2_CONTOUR_LABELS = {
    1.0:   r"Fermi-LAT 17yr",
    10.0:  r"$\times 10$ exposure",
    50.0:  r"CTA-scale ($\times 50$)",
    1.0e3: r"$\times 10^{3}$ exposure",
    1.0e6: r"$\times 10^{6}$ exposure",
}

# Fig 2 operator subset (6-panel main-body figure, 3 cols x 2 rows)
# The anapole and charge-radius operators couple through d^nu F_nu_mu, whose
# on-shell-photon vertex vanishes identically -- they have exactly zero
# tree-level gamma chi -> gamma chi cross section and are stated as such in
# the text rather than plotted. Their former panels are replaced by the Dirac
# Rayleigh CP-even/odd pair, so the figure still spans dim-5 (dipoles),
# dim-7 fermionic (Rayleigh), and dim-6 scalar operators.
# The Majorana Rayleigh panel shows `rayleigh_even_majorana` (matching Fig 3)
# rather than `rayleigh_full_majorana`, so the two figures display the same
# Majorana Rayleigh operator. Its in-panel title is overridden below to
# "Majorana Rayleigh Even" via FIG2_TITLE_OVERRIDES (PANEL_CONFIGS calls it
# "Rayleigh Even (Majorana)"); the pixelwise-halo npz for this operator exists.
FIG2_OPERATORS = [
    "dipole_magnetic", "dipole_electric",
    "rayleigh_even", "rayleigh_odd",
    "rayleigh_even_majorana", "scalar_rayleigh",
]

# Per-panel in-panel title overrides for Fig 2 (keeps the shared PANEL_CONFIGS
# titles untouched). Fig 3 titles the same operator "Majorana Rayleigh Even".
FIG2_TITLE_OVERRIDES = {
    "rayleigh_even_majorana": "Majorana Rayleigh Even",
}

# Fig 2 cold-dark-matter reference: vertical dotted-yellow line at the CDM
# mass floor m_chi = 1 MeV = 1e-3 GeV.
FIG2_CDM_MCHI_GEV = 1.0e-3
FIG2_CDM_COLOR    = "gold"
FIG2_CDM_LABEL    = r"CDM ($m_\chi = 1$ MeV)"

# Fig 3 halo constraint overlay (3-panel paper summary)
FIG3_PROFILES = ["pixelwise_global_rho2", "pixelwise_global_rho2.5"]
# Anapole panel replaced by Majorana Rayleigh Even -- the anapole has
# exactly zero tree-level real-photon cross section.
FIG3_OPERATORS = ["dipole_magnetic", "rayleigh_even_majorana", "scalar_rayleigh"]
FIG3_PROFILE_COLORS = {
    "pixelwise_global_rho2": COL_THIS_WORK,
    "pixelwise_global_rho2.5": plasma_color(0.4),
}

# Fig 4 multi-dataset overlay
FIG4_DATASETS = ["halo", "igrb"]           # drop 'dsph' -- profile-normalised fit degenerate
FIG4_SOURCES = ["measured", "pppc"]         # measured = self-fit; pppc = two-component ann. benchmark
# Companion Totani reanalysis best fits: m_ann = 0.55 TeV (WW), 0.72 TeV (bb).
FIG4_PPPC_MASSES = [550, 720]
FIG4_PPPC_CHANNELS = ["bb", "WW"]
FIG4_PPPC_BEST_FIT_SOURCES = {"pppc_WW_mann550", "pppc_bb_mann720"}
FIG4_HALO_PROFILE = "pixelwise_global_rho2"
FIG4_DSPH_SELECTION = "classical"

# Fig 5 UV translation.
# The companion Totani reanalysis (Ref. StenhouseGhagDeppisch2026Totani)
# argues the 20 GeV halo excess is best fit by a HEAVY annihilator
# (m_ann ~ 0.55-0.72 TeV) resonantly boosted by a SUB-GeV MEDIATOR, with a
# LIGHT SCATTERER redistributing the produced photons. So the scatterer that
# Fig 5 should read off is sub-GeV, not the annihilator mass. FIG5_MCHI_LIST
# holds every scatterer mass we want to overlay -- typically two or three
# benchmarks bracketing the light-scatterer window preferred by the halo fit.
# FIG5_MCHI is kept as a single-value alias for backward compatibility.
FIG5_MCHI_LIST  = [1.0e-2, 1.0e-1, 1.0]   # 10 MeV, 100 MeV, 1 GeV
FIG5_MCHI       = FIG5_MCHI_LIST[1]        # legacy single-value fallback (100 MeV)

# Fig 5 halo-preferred region shading on the (M_A', |eps|) panel. The
# companion analysis lands in M_A' ~ 10 to few*100 MeV to give the required
# Sommerfeld/BW boost. Shade this window so the reader can see immediately
# which slice of the plane the halo interpretation of the 20 GeV excess
# actually cares about. Set FIG5_HALO_MAP_SHADE = False to hide the band.
FIG5_HALO_MAP_SHADE   = True
FIG5_HALO_MAP_MAP     = (1.5e-2, 1.0)      # (M_A' min, max) in GeV -- full sub-GeV Sommerfeld-saturation window for TeV DM per Feng, Kaplinghat & Yu (2010)
FIG5_HALO_MAP_COLOR   = "#FF9F5A"          # warm orange -- matches Fig 3 cosmology
FIG5_HALO_MAP_ALPHA   = 0.18
FIG5_HALO_MAP_LABEL   = "Halo-preferred (this work)"
FIG5_HALO_MAP_LABEL_XY = (0.05, 0.87)      # transAxes anchor for the caption
FIG5_HALO_MAP_LABEL_COLOR = "#B04B00"
FIG5_HALO_MAP_LABEL_FS = ANNOT_FS
FIG5_PERTURB_LABEL_XY = (0.5, 0.65)
FIG5_PERTURB_LABEL_COLOR = FIG1_EXCLUSION_TEXT_COLOR
FIG5_PERTURB_LABEL_FS = ANNOT_FS
FIG5_LEGEND_KW = dict(
    fontsize=BASE_FS - 1,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    ncol=2,
    borderpad=0.35,
    handlelength=2.0,
    handletextpad=0.45,
    columnspacing=0.9,
    labelspacing=0.30,
)

# ---- Fig 5 "valid region" annotations ---------------------------------------
# Each panel has a physically ALLOWED region (below the halo curve, below the
# LHC / BaBar / etc. reference bounds) and one or more EXCLUDED regions.
# These knobs drop labelled tags inside each region so the reader can see at
# a glance which zone is which. All positions are transAxes fractions.
FIG5_DH_ALLOWED_LABEL  = "Allowed"
FIG5_DH_ALLOWED_X      = 0.88              # axes-fraction x anchor
FIG5_DH_ALLOWED_Y_FRAC = 0.18              # y = this factor * sin_lhc, definitely below LHC line
FIG5_DH_EXCLUDED_LABEL = "LHC excluded"
FIG5_DH_EXCLUDED_XY    = (0.85, 0.42)      # above the LHC line, inside the shaded band
FIG5_DH_LABEL_COLOR    = "#5A3E00"
FIG5_DH_LABEL_FS       = ANNOT_FS

FIG5_DP_ALLOWED_LABEL  = "Allowed"
FIG5_DP_ALLOWED_XY     = (0.92, 0.24)      # right edge, below all reference lines
FIG5_DP_EXCLUDED_LABEL = "BaBar / NA64 excluded"
FIG5_DP_EXCLUDED_XY    = (0.70, 0.55)      # above the reference lines
FIG5_DP_LABEL_COLOR    = "#5A3E00"
FIG5_DP_LABEL_FS       = ANNOT_FS

# =============================================================================
# FIG 2 IN-PANEL ANNOTATIONS
# =============================================================================
# Panel titles are drawn as an in-plot annotation instead of above the axes so
# every panel keeps its plot region maximised. "EFT valid" annotation likewise.
# Change the (x, y) fractions (transAxes coords: 0,0 = bottom-left, 1,1 = top-right),
# colour, weight, and font size here — nothing else needs touching.
FIG2_TITLE_XY             = (0.05, 0.94)   # top-left inside panel
FIG2_TITLE_COLOR          = "white"
FIG2_TITLE_FONTWEIGHT     = "bold"
FIG2_TITLE_FS             = BASE_FS + 1
FIG2_TITLE_HALIGN         = "left"
FIG2_TITLE_VALIGN         = "top"

FIG2_EFT_LABEL_XY         = (0.05, 0.45)   # lower-left interior
FIG2_EFT_LABEL_COLOR      = COL_EFT_LABEL
FIG2_EFT_LABEL_FONTWEIGHT = "bold"
FIG2_EFT_LABEL_FS         = ANNOT_FS
FIG2_EFT_LABEL_TEXT       = "EFT valid"

# Unitarity guide (dotted line) in-panel annotation. In every Fig 2 panel we
# evaluate the analytic unitarity curve at FIG2_UNITARITY_LABEL_MCHI [GeV]
# and drop the "Unitarity" label just above that point, rotated to sit along
# the log-log line. Nudge (x, y) in log-decades via FIG2_UNITARITY_LABEL_DX/DY,
# and rotate with FIG2_UNITARITY_LABEL_ROT.
FIG2_UNITARITY_LABEL_TEXT  = "Unitarity"
FIG2_UNITARITY_LABEL_MCHI  = 1.0e6   # GeV, x-anchor at which the curve is evaluated
FIG2_UNITARITY_LABEL_DX    = 0.0     # log-decade nudge in x from the anchor
FIG2_UNITARITY_LABEL_DY    = 0.35    # log-decade nudge in y from the curve
FIG2_UNITARITY_LABEL_COLOR = "#c4c4c4"
FIG2_UNITARITY_LABEL_FS    = ANNOT_FS - 1
FIG2_UNITARITY_LABEL_ROT   = 35      # degrees, along the log-log line

# =============================================================================
# FIG 3 IN-PANEL ANNOTATIONS (mirrors FIG 2 methodology)
# =============================================================================
# Same knob-based control as Fig 2: panel titles as in-plot text (top-left,
# white, bold), "EFT valid" label positioned inside the shaded wedge, and a
# rotated "Unitarity" label along the dotted unitarity guide. Change any of
# these constants to move / recolour / restyle every Fig 3 panel at once.
FIG3_TITLE_XY             = (0.05, 0.94)   # top-left interior -- matches FIG2/FIG4
FIG3_TITLE_COLOR          = "black"        # matches FIG4_TITLE_COLOR (both sit on a white/cyan panel, not a heatmap)
FIG3_TITLE_FONTWEIGHT     = "bold"
FIG3_TITLE_FS             = BASE_FS + 1
FIG3_TITLE_HALIGN         = "left"
FIG3_TITLE_VALIGN         = "top"

FIG3_EFT_LABEL_XY         = (0.05, 0.47)   # lower-left interior of EFT wedge
FIG3_EFT_LABEL_COLOR      = "#21A5B1"
FIG3_EFT_LABEL_FONTWEIGHT = "bold"
FIG3_EFT_LABEL_FS         = ANNOT_FS
FIG3_EFT_LABEL_TEXT       = "EFT valid"

FIG3_UNITARITY_LABEL_TEXT  = "Unitarity"
FIG3_UNITARITY_LABEL_MCHI  = 1.0e8   # GeV, x-anchor at which the curve is evaluated
FIG3_UNITARITY_LABEL_DX    = 0.0     # log-decade nudge in x
FIG3_UNITARITY_LABEL_DY    = 0.35    # log-decade nudge above the curve
FIG3_UNITARITY_LABEL_COLOR = COL_GUIDE
FIG3_UNITARITY_LABEL_FS    = ANNOT_FS - 1
FIG3_UNITARITY_LABEL_ROT   = 35      # degrees, along the log-log line

# =============================================================================
# FIG 4 IN-PANEL ANNOTATIONS (mirrors FIG 3 methodology)
# =============================================================================
# Fig 4 (multi-dataset overlay) currently draws its own title above the axes
# and an "EFT valid" annotation in the top-left corner. Below we take control
# of both from the master file: title becomes an in-panel top-left annotation,
# EFT-valid moves lower into the shaded wedge. Same knobs as
# Fig 2 / Fig 3 — edit any FIG4_TITLE_* / FIG4_EFT_LABEL_* to move / recolour.
FIG4_TITLE_XY             = (0.05, 0.94)
FIG4_TITLE_COLOR          = "black"
FIG4_TITLE_FONTWEIGHT     = "bold"
FIG4_TITLE_FS             = BASE_FS + 1
FIG4_TITLE_HALIGN         = "left"
FIG4_TITLE_VALIGN         = "top"

FIG4_EFT_LABEL_XY         = (0.05, 0.55)     # inside the shaded wedge
FIG4_EFT_LABEL_COLOR      = FIG3_EFT_LABEL_COLOR
FIG4_EFT_LABEL_FONTWEIGHT = "bold"
FIG4_EFT_LABEL_FS         = ANNOT_FS
FIG4_EFT_LABEL_TEXT       = "EFT valid"

# Optional Unitarity annotation for Fig 4 -- Fig 4 doesn't currently draw the
# unitarity curve itself; if FIG4_DRAW_UNITARITY is True we overlay both the
# curve and its rotated in-panel label so every panel matches Fig 3.
FIG4_DRAW_UNITARITY        = True
FIG4_UNITARITY_LABEL_TEXT  = "Unitarity"
FIG4_UNITARITY_LABEL_MCHI  = 1.0e8   # GeV, x-anchor
FIG4_UNITARITY_LABEL_DX    = 0.0
FIG4_UNITARITY_LABEL_DY    = 0.35
FIG4_UNITARITY_LABEL_COLOR = FIG3_UNITARITY_LABEL_COLOR
FIG4_UNITARITY_LABEL_FS    = ANNOT_FS - 1
FIG4_UNITARITY_LABEL_ROT   = 35

# Master-file-controlled dataset colours for Fig 4. These flow into the sub-
# module's DATASET_STYLES via monkey-patch at f4 runtime.
FIG4_DATASET_COLOURS = {
    "halo": FIG3_PROFILE_COLORS["pixelwise_global_rho2"],
    "igrb": FIG3_PROFILE_COLORS["pixelwise_global_rho2.5"],
    "dsph": COL_FOLLOWUP,     # unused by default, kept for completeness
}

# =============================================================================
# STYLE APPLICATION -- called once at the top of main()
# =============================================================================
def _apply_paper_style():
    """Apply the Trinity paper style once for the whole run."""
    from helpers.trinity_plotting import set_paper_style  # noqa: E402
    set_paper_style(base_fontsize=BASE_FS, linewidth=LINEWIDTH,
                    n_colors=14, cmap_name="plasma")


def _fetch(produced_by_script: Path, dest_dir: Path, new_name: str) -> Path:
    """Copy a produced PDF into dest_dir under new_name."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / new_name
    if not produced_by_script.exists():
        raise FileNotFoundError(f"expected figure not produced: {produced_by_script}")
    shutil.copy2(produced_by_script, dest)
    print(f"  wrote {_rel(dest)}")
    return dest


# =============================================================================
# FIG 1 -- §II: perturbative UV-complete tau_max vs m_chi
# =============================================================================
def f1_uv_complete_tau_vs_mchi(out_dir: Path) -> Path:
    """
    §II Fig 1: maximum cosmological / galactic optical depth tau_max(m_chi)
    for the perturbatively-consistent dark-Higgs portal and PQG channels,
    saturating y_chi <= sqrt(4pi) and sin theta <= 0.33.

    The physics (Higgs loop form factors, NFW J-factor, gravitational
    scaling) is computed by helpers imported from make_uv_complete_tau_vs_mchi.
    Everything below is layout / styling.
    """
    from make_uv_complete_tau_vs_mchi import (
        MCHI_MIN, MCHI_MAX, N_MCHI,
        J_COSMO, galactic_j_factor, sigma_tot_gev2, tau_from_sigma,
        tau_gravitational_pqg,
        GEV2_TO_CM2,
    )

    M_PL = 1.220890e19  # Planck mass [GeV]: heaviest sensible point-particle DM
    mchi = np.logspace(np.log10(MCHI_MIN), np.log10(M_PL), N_MCHI)
    j_gal = galactic_j_factor()
    sigma_higgs = np.array([sigma_tot_gev2(m) for m in mchi])
    tau_higgs_cosmo = tau_from_sigma(mchi, sigma_higgs, J_COSMO)
    tau_higgs_gal   = tau_from_sigma(mchi, sigma_higgs, j_gal)
    tau_grav_cosmo  = tau_gravitational_pqg(mchi, j_factor=J_COSMO)
    tau_grav_gal    = tau_gravitational_pqg(mchi, j_factor=j_gal)

    # --- layout ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG1_WIDTH, FIG1_HEIGHT))
    col_higgs = FIG1_HIGGS_COLOR
    col_grav  = FIG1_GRAV_COLOR
    col_thres = FIG1_THRESHOLD_COLOR
    col_shade = FIG1_EXCLUSION_FILL_COLOR
    col_shade_text = FIG1_EXCLUSION_TEXT_COLOR

    ax.fill_between(mchi, np.maximum(tau_higgs_cosmo, tau_higgs_gal), 1.0,
                    color=col_shade, alpha=0.22)
    ax.loglog(mchi, tau_higgs_gal,   color=col_higgs, ls="-",  lw=2.2)
    ax.loglog(mchi, tau_higgs_cosmo, color=col_higgs, ls="--", lw=2.2)
    ax.loglog(mchi, tau_grav_gal,    color=col_grav,  ls="-",  lw=2.0)
    ax.loglog(mchi, tau_grav_cosmo,  color=col_grav,  ls="--", lw=2.0)
    ax.axhline(1e-2, color=col_thres, ls=":", lw=1.4)
    ax.axvline(M_PL, color="0.45", ls=(0, (1, 1)), lw=1.1)
    ax.annotate(r"$M_{\rm Pl}$", xy=(M_PL, 8.0e-6), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center",
                rotation=90, fontsize=ANNOT_FS, color="0.35")

    ax.set_xlim(1.0, 1.6e19)
    ax.set_ylim(1.0e-70, 1.0)
    ax.set_xlabel(r"$m_\chi$ [GeV]")
    ax.set_ylabel(r"$\tau_{\rm max}$")
    ax.tick_params(labelsize=BASE_FS - 2, length=5)
    ax.tick_params(which="minor", length=3)
    ax.grid(True, which="both", alpha=0.22)

    handles = [
        Line2D([0], [0], color=col_higgs, lw=2.2, label=r"Higgs portal (saturated)"),
        Line2D([0], [0], color=col_grav,  lw=2.0, label=r"Gravitational (PQG)"),
        Line2D([0], [0], color="0.15", lw=1.6, ls="-",  label="Galactic baseline"),
        Line2D([0], [0], color="0.15", lw=1.6, ls="--", label="Cosmological baseline"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.35, 0.55),
              borderpad=0.4, handletextpad=0.5, labelspacing=0.35,
              **LEGEND_KW)
    ax.annotate(
        r"Fermi-LAT visual threshold $\tau_{\rm obs} \sim 10^{-2}$",
        xy=(1.0e11, 1.0e-2),
        xycoords="data",
        xytext=(0.02, 0.90),
        textcoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=ANNOT_FS,
        color=col_thres,
        )
    ax.annotate(
        "Non-perturbative\nor excluded by LHC",
        xy=(1.0e11, 1.0e-14),
        xycoords="data",
        xytext=(0.97, 0.72),
        textcoords="axes fraction",
        ha="right",
        va="center",
        fontsize=ANNOT_FS,
        color=col_shade_text,
    )

    fig.tight_layout()
    stem = out_dir / "f1_uv_complete_tau_vs_mchi"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {_rel(stem)}.pdf")
    return stem.with_suffix(".pdf")


# =============================================================================
# FIG 2 -- §III: 8-panel exposure-multiplier sensitivity map
# =============================================================================
def f2_sensitivity_map(out_dir: Path) -> Path:
    """
    §III Fig 2: 8-panel f_required(m_chi, Lambda) grid across the full EFT
    operator basis (Dirac dipoles + Majorana anapole + Rayleigh Full for
    each fermion type + Scalar Rayleigh).

    Delegates the per-panel drawing to make_sensitivity_map.plot_sensitivity_panel
    and does all layout / legend / colorbar work inline here.
    """
    from make_sensitivity_map import (
        PANEL_CONFIGS, plot_sensitivity_panel, OUTPUT_DIR as _SM_OUT,
        _unitarity_curve,          # per-operator unitarity Lambda(m_chi)
    )

    n = len(FIG2_OPERATORS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    nslots = ncols * nrows

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIG2_WIDTH, FIG2_HEIGHT),
                             squeeze=False)
    axes = axes.ravel()

    title_fs = BASE_FS + 1
    axlbl_fs = BASE_FS
    tick_fs  = BASE_FS - 2
    anno_fs  = ANNOT_FS

    contour_handles_acc = []
    last_heat = None
    plotted = []
    for i, op in enumerate(FIG2_OPERATORS):
        # We pass an empty title_override AND annotate_validity=False so that
        # the sub-module doesn't draw either — everything visible in-panel is
        # controlled from this file. See FIG2_TITLE_* and FIG2_EFT_LABEL_*
        # constants at the top of this script.
        handles, heat = plot_sensitivity_panel(
            axes[i], op, "pixelwise_global_rho2",
            model_kind="raw_attenuation",
            contour_levels=FIG2_CONTOURS,
            contour_colours=FIG2_CONTOUR_COLOURS,
            contour_labels=FIG2_CONTOUR_LABELS,
            title_override=" ",           # non-empty falsy-blank to suppress default
            title_fontsize=title_fs,
            axis_label_fontsize=axlbl_fs,
            tick_labelsize=tick_fs,
            annotate_fontsize=anno_fs,
            heatmap_cmap="plasma_r",      # reversed: yellow at f=1, dark at f=1e12
            fheatmap_min=1.0,
            fheatmap_max=1e12,
            annotate_validity=False,
            validity_fill_color=COL_EXCLUSION,
            validity_line_color=COL_EXCLUSION,
            unitarity_line_color="#c4c4c4",
        )
        # Clear whatever axes title the sub-module may still have set.
        axes[i].set_title("")

        # In-panel title (top-left, white, bold) -- edit FIG2_TITLE_* to move / recolour.
        panel_title = FIG2_TITLE_OVERRIDES.get(op, PANEL_CONFIGS[op]["title"])
        axes[i].text(
            FIG2_TITLE_XY[0], FIG2_TITLE_XY[1], panel_title,
            transform=axes[i].transAxes,
            color=FIG2_TITLE_COLOR,
            fontsize=BASE_FS,
            fontweight=FIG2_TITLE_FONTWEIGHT,
            ha=FIG2_TITLE_HALIGN, va=FIG2_TITLE_VALIGN,
            zorder=10,
        )

        # In-panel EFT-valid label -- edit FIG2_EFT_LABEL_* to move / recolour.
        axes[i].text(
            FIG2_EFT_LABEL_XY[0], FIG2_EFT_LABEL_XY[1], FIG2_EFT_LABEL_TEXT,
            transform=axes[i].transAxes,
            color=FIG2_EFT_LABEL_COLOR,
            fontsize=ANNOT_FS,
            fontweight=FIG2_EFT_LABEL_FONTWEIGHT,
            ha="left", va="bottom",
            zorder=10,
        )

        # In-panel "Unitarity" annotation along the dotted unitarity guide.
        # Evaluate the analytic unitarity curve for this operator at the
        # fiducial m_chi and offset via FIG2_UNITARITY_LABEL_D{X,Y} in log-
        # decades. Skipped for operators without a defined unitarity curve.
        cfg_op = PANEL_CONFIGS[op]["operator"]
        _lam_unit, _ = _unitarity_curve(
            cfg_op, np.array([FIG2_UNITARITY_LABEL_MCHI]))
        if _lam_unit is not None and np.isfinite(_lam_unit[0]) and _lam_unit[0] > 0:
            x_anchor = FIG2_UNITARITY_LABEL_MCHI * 10.0 ** FIG2_UNITARITY_LABEL_DX
            y_anchor = _lam_unit[0]           * 10.0 ** FIG2_UNITARITY_LABEL_DY
            axes[i].text(
                x_anchor, y_anchor, FIG2_UNITARITY_LABEL_TEXT,
                color=FIG2_UNITARITY_LABEL_COLOR,
                fontsize=FIG2_UNITARITY_LABEL_FS,
                rotation=FIG2_UNITARITY_LABEL_ROT,
                rotation_mode="anchor",
                ha="center", va="bottom",
                zorder=10,
            )

        # --- CDM reference: vertical dotted-yellow line at m_chi = 1 MeV ---
        # Marks the cold-dark-matter mass floor (m_chi = 1e-3 GeV) below which
        # a thermal relic is no longer cold. Drawn per panel; a single proxy
        # handle is added to the shared bottom legend below.
        axes[i].axvline(
            FIG2_CDM_MCHI_GEV,
            color=FIG2_CDM_COLOR, linestyle=":", linewidth=1.4,
            zorder=6,
        )

        contour_handles_acc.extend(handles)
        if heat is not None:
            last_heat = heat
        plotted.append((i, axes[i]))

    # --- axis-label consolidation: bottom row x, leftmost col y -----------
    last_in_col, first_in_row = {}, {}
    for i, _ in plotted:
        r, c = divmod(i, ncols)
        last_in_col[c] = i
        first_in_row.setdefault(r, i)
    for i, ax in plotted:
        r, c = divmod(i, ncols)
        if i != last_in_col[c]:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        if i != first_in_row[r]:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    # --- turn off any unused slots (none in a full 3x2) ------------------
    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    # --- shared colorbar --------------------------------------------------
    if last_heat is not None:
        cax = fig.add_axes([0.855, 0.20, 0.011, 0.68])
        cb = fig.colorbar(last_heat, cax=cax)
        cb.set_label(r"$f_{\rm required} = 4.61/\Delta\chi^{2}$",
                     )
        #cb.ax.tick_params(labelsize=BASE_FS - 1)

    # --- shared legend along the bottom of the figure --------------------
    if contour_handles_acc:
        seen = {}
        for h in contour_handles_acc:
            lbl = h.get_label()
            if lbl not in seen:
                seen[lbl] = h
        handles = list(seen.values())
        labels = list(seen.keys())
        # Append the CDM reference proxy so the dotted-yellow line is explained.
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color=FIG2_CDM_COLOR,
                              linestyle=":", linewidth=1.4))
        labels.append(FIG2_CDM_LABEL)
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.45, -0.1),
            ncol=int(np.ceil(len(handles) / 2)),
            title="Exposure contours",
            title_fontsize=BASE_FS,
            fontsize=BASE_FS,
            handlelength=2.6, handletextpad=0.6,
            columnspacing=1.6, borderaxespad=0.0, borderpad=0.4,
            **LEGEND_KW,
        )

    fig.tight_layout(rect=[0.0, 0.06, 0.85, 0.97])
    fig.subplots_adjust(wspace=0.08, hspace=0.12)

    stem = out_dir / "f2_sensitivity_map"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {_rel(stem)}.pdf")
    return stem.with_suffix(".pdf")


# =============================================================================
# FIG 3 -- §IV.C: 3-panel halo constraint overlay with external bounds
# =============================================================================
def f3_halo_constraints(out_dir: Path) -> Path:
    """
    §IV.C Fig 3: halo-derived 90% CL exclusion contours overlaid on external
    bounds (collider, direct detection, indirect detection, cosmology) for
    the three representative operators. Multi-profile (rho^2 + rho^{2.5}).

    Delegates the heavy constraint-loading + per-panel drawing to
    make_paper_style_operator_overlays.plot_panel and does all layout,
    legend construction, and legend styling inline here.
    """
    import make_paper_style_operator_overlays as _mp
    from make_paper_style_operator_overlays import (
        plot_panel, collect_grouped_legend, draw_bottom_grouped_legend,
        PANEL_CONFIGS as PAPER_PANEL_CONFIGS,
    )
    from make_sensitivity_map import _unitarity_curve

    # --- Propagate the master file's palette into the halo-constraints module.
    # The module captures its palette at import time in LEGEND_GROUPS and
    # elsewhere. Monkey-patch every colour constant AND rebuild any dict that
    # captured them at import so a single edit at the top of this file
    # recolours every curve, legend line, and annotation in Fig 3.
    _mp.INDIRECT_COLOR         = COL_INDIRECT
    _mp.DIRECT_COLOR           = COL_DIRECT
    _mp.COLLIDER_COLOR         = COL_COLLIDER
    _mp.COSMOLOGY_COLOR        = COL_COSMOLOGY
    _mp.BODDY_GLUSCEVIC_COLOR  = COL_COSMOLOGY   # direct-drawn CMB curve (Boddy & Gluscevic 2018)
    _mp.THIS_WORK_COLOR        = COL_THIS_WORK
    _mp.GUIDE_COLOR            = COL_GUIDE
    _mp.EFT_VALID_LABEL_COLOR  = COL_EFT_LABEL
    for _profile, _color in FIG3_PROFILE_COLORS.items():
        if _profile in _mp.DATA_DRIVEN_PROFILE_STYLES:
            _mp.DATA_DRIVEN_PROFILE_STYLES[_profile]["color"] = _color
    # Rebuild LEGEND_GROUPS with the new colours so the bottom-strip legend
    # picks them up. Labels here are what the compact-mode legend prints.
    _mp.LEGEND_GROUPS = {
        "direct_detection":   {"label": "Direct Detection",   "color": COL_DIRECT,    "linestyle": "-"},
        "collider":           {"label": "Collider",           "color": COL_COLLIDER,  "linestyle": "-"},
        "indirect_detection": {"label": "Indirect Detection", "color": COL_INDIRECT,  "linestyle": "-"},
        "cosmology":          {"label": "Cosmology",          "color": COL_COSMOLOGY, "linestyle": "-"},
        "theory":             {"label": "Theory / Validity",  "color": COL_GUIDE,     "linestyle": "-"},
        "this_work":          {"label": "This Work",          "color": COL_THIS_WORK, "linestyle": "-"},
    }

    # STYLE_HINTS drives every literature-limit line inside each panel:
    # (pretty_label, colour, linestyle) keyed by the boundary-file stem. The
    # default assigns each experiment a distinct hue in its family (LZ / XENON
    # in yellows, monojet / LEP in pinks, Fermi / H.E.S.S. / CTA in purples,
    # Planck in cyan). Here we collapse every entry to the master file's
    # group colour so the actual rendered curves match the legend swatches.
    # Line styles are preserved so LZ vs XENONnT vs Fermi vs H.E.S.S. remain
    # distinguishable. Change _STYLE_HINT_GROUPS if you want to reclassify
    # an experiment.
    _STYLE_HINT_GROUPS = {
        # Direct detection
        "xenon1t": "direct_detection",
        "xenonnt": "direct_detection",
        "xenonnt_anapole_majorana": "direct_detection",
        "hambye": "direct_detection",
        "ibarra2024_lz": "direct_detection",
        "lz2022": "direct_detection",
        "lz_magdipole": "direct_detection",
        "lz_eldipole": "direct_detection",
        "xlzd200ty": "direct_detection",
        # Collider
        "monojet": "collider",
        "lhc_monojet": "collider",
        "lep_zdecay": "collider",
        # Indirect detection
        "ams02": "indirect_detection",
        "fermilat_hess": "indirect_detection",
        "fermilat": "indirect_detection",
        "fermi_lines": "indirect_detection",
        "fermi_single_line": "indirect_detection",
        "fermi_double_line": "indirect_detection",
        "hess": "indirect_detection",
        "cta_gc": "indirect_detection",
        "cta_dsphs": "indirect_detection",
        # Cosmology / theory
        "planck": "cosmology",
        "thermal_relic": "theory",
    }
    _group_color = {
        "direct_detection":   COL_DIRECT,
        "collider":           COL_COLLIDER,
        "indirect_detection": COL_INDIRECT,
        "cosmology":           COL_COSMOLOGY,
        "theory":              COL_GUIDE,
    }
    _mp.STYLE_HINTS = {
        key: (
            entry[0],                                              # pretty label
            _group_color.get(_STYLE_HINT_GROUPS.get(key, ""), entry[1]),  # group colour, fallback to original
            entry[2],                                              # linestyle
        )
        for key, entry in _mp.STYLE_HINTS.items()
    }

    n = len(FIG3_OPERATORS)
    ncols = 3
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIG3_WIDTH, FIG3_HEIGHT))
    axes = np.atleast_1d(axes).ravel()

    # Short two-line titles so nothing overflows the PRD twocolumn panel width.
    # Edit strings here to rename any panel; the '\n' triggers a second line.
    # NOTE: keys must match FIG3_OPERATORS (the anapole panel was retired in
    # favour of the Majorana Rayleigh Even panel) and must stay in sync with
    # _FIG4_TITLES in f4_multi_dataset_overlay() -- both figures share the
    # same three operators and should render identical panel titles.
    _FIG3_TITLES = {
        "dipole_magnetic":       "Dirac Magnetic Dipole\n(dim-5)",
        "rayleigh_even_majorana": "Majorana Rayleigh Even\n(dim-7)",
        "scalar_rayleigh":       "Scalar Rayleigh\n(dim-7)",
    }

    all_handles = []
    for i, op in enumerate(FIG3_OPERATORS):
        # Same suppression trick as Fig 2: silence the sub-module's own title
        # and on-plot annotations so this file owns every text drawn in-panel.
        handles = plot_panel(
            axes[i], op,
            show_legend=False,
            data_driven_profiles=FIG3_PROFILES,
            data_driven_source_tag="halo",
            title_override=" ",                   # blank suppresses the built-in title
            title_fontsize=FIG3_TITLE_FS,
            panel_label_fontsize=BASE_FS - 2,
            axis_label_fontsize=BASE_FS,
            tick_labelsize=BASE_FS - 2,
            annotate_theory_guides=False,         # <- disables built-in EFT/Unitarity text
            include_deconvolution_ceiling=False,
            validity_fill_color=COL_EXCLUSION,
            validity_line_color=COL_EXCLUSION,
            unitarity_line_color=COL_GUIDE,
        )
        axes[i].set_title("")   # belt-and-braces clear of any residual title
        # plot_panel draws the operator symbol ($O_{...}$ / $\mathcal{O}_{...}$)
        # at (0.03, 0.93, transAxes) whenever annotate_theory_guides is False.
        # Remove any text sitting at that anchor so our own FIG3_TITLE_ label
        # sits alone in the top-left.
        for _t in list(axes[i].texts):
            _pos = _t.get_position()
            if abs(_pos[0] - 0.03) < 1e-3 and abs(_pos[1] - 0.93) < 1e-3:
                _t.remove()

        # --- FIG 3 IN-PANEL TITLE (top-left, bold -- matches FIG2/FIG4) ----
        panel_title = _FIG3_TITLES.get(op, PAPER_PANEL_CONFIGS[op]["title"])
        axes[i].text(
            FIG3_TITLE_XY[0], FIG3_TITLE_XY[1], panel_title,
            transform=axes[i].transAxes,
            color=FIG3_TITLE_COLOR,
            fontsize=FIG3_TITLE_FS,
            fontweight=FIG3_TITLE_FONTWEIGHT,
            ha=FIG3_TITLE_HALIGN, va=FIG3_TITLE_VALIGN,
            zorder=40,
        )

        # --- FIG 3 EFT-VALID LABEL (inside the shaded wedge) ---------------
        axes[i].text(
            FIG3_EFT_LABEL_XY[0], FIG3_EFT_LABEL_XY[1], FIG3_EFT_LABEL_TEXT,
            transform=axes[i].transAxes,
            color=FIG3_EFT_LABEL_COLOR,
            fontsize=FIG3_EFT_LABEL_FS,
            fontweight=FIG3_EFT_LABEL_FONTWEIGHT,
            ha="left", va="bottom",
            zorder=10,
        )

        # --- FIG 3 UNITARITY LABEL (along the dotted line) -----------------
        _cfg = PAPER_PANEL_CONFIGS[op]
        _lam_unit, _ = _unitarity_curve(_cfg["operator"],
                                        np.array([FIG3_UNITARITY_LABEL_MCHI]))
        if _lam_unit is not None and np.isfinite(_lam_unit[0]) and _lam_unit[0] > 0:
            x_anchor = FIG3_UNITARITY_LABEL_MCHI * 10.0 ** FIG3_UNITARITY_LABEL_DX
            y_anchor = _lam_unit[0]              * 10.0 ** FIG3_UNITARITY_LABEL_DY
            axes[i].text(
                x_anchor, y_anchor, FIG3_UNITARITY_LABEL_TEXT,
                color=FIG3_UNITARITY_LABEL_COLOR,
                fontsize=FIG3_UNITARITY_LABEL_FS,
                rotation=FIG3_UNITARITY_LABEL_ROT,
                rotation_mode="anchor",
                ha="center", va="bottom",
                zorder=10,
            )

        all_handles.extend(handles)

    # --- axis-label consolidation ----------------------------------------
    last_in_col, first_in_row = {}, {}
    for i in range(n):
        r, c = divmod(i, ncols)
        last_in_col[c] = i
        first_in_row.setdefault(r, i)
    for i in range(n):
        r, c = divmod(i, ncols)
        if i != last_in_col[c]:
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", labelbottom=False)
        if i != first_in_row[r]:
            axes[i].set_ylabel("")
            axes[i].tick_params(axis="y", labelleft=False)

    # --- bottom-strip legend ---------------------------------------------
    grouped = collect_grouped_legend(all_handles)
    draw_bottom_grouped_legend(
        fig, grouped, compact=True, base_fs=BASE_FS, y_anchor=FIG3_LEGEND_Y,
        ncol=4,
    )

    plt.tight_layout(rect=[0.01, 0.20, 1, 0.96], w_pad=0.6)
    fig.subplots_adjust(wspace=0.14, hspace=0.12)

    stem = out_dir / "f3_halo_constraints"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {_rel(stem)}.pdf")
    return stem.with_suffix(".pdf")


# =============================================================================
# FIG 4 -- §IV.D: multi-dataset overlay (halo + IGRB + PPPC benchmarks)
# =============================================================================
def f4_multi_dataset_overlay(out_dir: Path) -> Path:
    """
    §IV.D Fig 4: cross-dataset overlay. Solid = measured spectrum as
    intrinsic template (Phi_null = data, no DM assumed); dashed / dash-dot
    = PPPC bb / WW annihilation source at the companion best-fit
    annihilator masses (two-component dark-sector interpretation).
    """
    import make_multi_dataset_overlay as _md
    from make_multi_dataset_overlay import (
        PANEL_CONFIGS, discover_boundaries, plot_overlay_panel,
    )
    from make_sensitivity_map import (
        _unitarity_curve,
        PANEL_CONFIGS as _SM_PANEL_CONFIGS,
    )

    # --- Propagate the master palette into make_multi_dataset_overlay --------
    # Fig 4 draws each dataset in a fixed colour from DATASET_STYLES. Force
    # every entry to the master file's FIG4_DATASET_COLOURS so the actual
    # curves match the legend swatches and a single edit at the top of this
    # file recolours every panel in Fig 4.
    for _ds_key, _ds_color in FIG4_DATASET_COLOURS.items():
        if _ds_key in _md.DATASET_STYLES:
            _md.DATASET_STYLES[_ds_key]["color"] = _ds_color

    n = len(FIG3_OPERATORS)  # same 3 representative operators as Fig 3
    operators = FIG3_OPERATORS
    ncols = 3
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIG4_WIDTH, FIG4_HEIGHT),
                             squeeze=False)
    axes = axes.ravel()

    # Two-line titles matching Fig 3's convention so nothing overflows the panel.
    # NOTE: keys must match FIG3_OPERATORS (the anapole panel was retired in
    # favour of the Majorana Rayleigh Even panel; keep this dict in sync with
    # _FIG3_TITLES below or the fallback silently renders the raw operator key
    # instead of a formatted title).
    _FIG4_TITLES = {
        "dipole_magnetic":       "Dirac Magnetic Dipole\n(dim-5)",
        "rayleigh_even_majorana": "Majorana Rayleigh Even\n(dim-7)",
        "scalar_rayleigh":       "Scalar Rayleigh\n(dim-7)",
    }

    combined_handles: dict[str, Line2D] = {}
    for i, op in enumerate(operators):
        records = discover_boundaries(
            op, sources=FIG4_SOURCES,
            halo_profile=FIG4_HALO_PROFILE,
            dsph_selection=FIG4_DSPH_SELECTION,
            pppc_masses=FIG4_PPPC_MASSES,
            datasets=FIG4_DATASETS,
            pppc_channels=FIG4_PPPC_CHANNELS,
        )
        records = [
            rec for rec in records
            if not str(rec["source"]).startswith("pppc_")
            or rec["source"] in FIG4_PPPC_BEST_FIT_SOURCES
        ]
        hs = plot_overlay_panel(
            axes[i], op, records,
            title_fontsize=BASE_FS + 1,
            axis_label_fontsize=BASE_FS,
            tick_labelsize=BASE_FS - 2,
            annotate_fontsize=BASE_FS - 1,
            validity_line_color=COL_GUIDE,
            validity_fill_color=COL_EXCLUSION,
            validity_fill_alpha=0.30,
            validity_label_color=FIG4_EFT_LABEL_COLOR,
        )
        # plot_overlay_panel draws the axes title and a hardcoded "EFT valid"
        # annotation at (0.04, 0.94). Clear both so this file's FIG4_TITLE_*
        # and FIG4_EFT_LABEL_* alone control what's visible.
        axes[i].set_title("")
        for _t in list(axes[i].texts):
            if _t.get_text().strip() == "EFT valid":
                _t.remove()

        # --- FIG 4 IN-PANEL TITLE (top-left, white, bold) -----------------
        axes[i].text(
            FIG4_TITLE_XY[0], FIG4_TITLE_XY[1], _FIG4_TITLES.get(op, op),
            transform=axes[i].transAxes,
            color=FIG4_TITLE_COLOR,
            fontsize=FIG4_TITLE_FS,
            fontweight=FIG4_TITLE_FONTWEIGHT,
            ha=FIG4_TITLE_HALIGN, va=FIG4_TITLE_VALIGN,
            zorder=10,
        )
        # --- FIG 4 EFT-VALID LABEL (inside the shaded wedge) --------------
        axes[i].text(
            FIG4_EFT_LABEL_XY[0], FIG4_EFT_LABEL_XY[1], FIG4_EFT_LABEL_TEXT,
            transform=axes[i].transAxes,
            color=FIG4_EFT_LABEL_COLOR,
            fontsize=FIG4_EFT_LABEL_FS,
            fontweight=FIG4_EFT_LABEL_FONTWEIGHT,
            ha="left", va="bottom",
            zorder=10,
        )
        # --- FIG 4 UNITARITY CURVE + LABEL (optional, opt-in via flag) ----
        if FIG4_DRAW_UNITARITY:
            _cfg_op = _SM_PANEL_CONFIGS[op]["operator"]
            xg = np.logspace(-10, 12, 400)
            _lam_unit_full, _ = _unitarity_curve(_cfg_op, xg)
            if _lam_unit_full is not None:
                axes[i].plot(xg, _lam_unit_full,
                             color=COL_GUIDE, lw=1.0, ls=":", zorder=4)
                _lam_pt, _ = _unitarity_curve(
                    _cfg_op, np.array([FIG4_UNITARITY_LABEL_MCHI]))
                if _lam_pt is not None and np.isfinite(_lam_pt[0]) and _lam_pt[0] > 0:
                    x_anchor = FIG4_UNITARITY_LABEL_MCHI * 10.0 ** FIG4_UNITARITY_LABEL_DX
                    y_anchor = _lam_pt[0]              * 10.0 ** FIG4_UNITARITY_LABEL_DY
                    axes[i].text(
                        x_anchor, y_anchor, FIG4_UNITARITY_LABEL_TEXT,
                        color=FIG4_UNITARITY_LABEL_COLOR,
                        fontsize=FIG4_UNITARITY_LABEL_FS,
                        rotation=FIG4_UNITARITY_LABEL_ROT,
                        rotation_mode="anchor",
                        ha="center", va="bottom",
                        zorder=10,
                    )

        for h in hs:
            combined_handles.setdefault(h.get_label(), h)

    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    # --- axis-label consolidation ----------------------------------------
    last_in_col, first_in_row = {}, {}
    for i in range(n):
        r, c = divmod(i, ncols)
        last_in_col[c] = i
        first_in_row.setdefault(r, i)
    for i in range(n):
        r, c = divmod(i, ncols)
        if i != last_in_col[c]:
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", labelbottom=False)
        if i != first_in_row[r]:
            axes[i].set_ylabel("")
            axes[i].tick_params(axis="y", labelleft=False)

    # --- bottom legend ----------------------------------------------------
    if combined_handles:
        handles = list(combined_handles.values())
        labels  = list(combined_handles.keys())
        fig.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, FIG4_LEGEND_Y),
            ncol=min(FIG4_LEGEND_NCOL, len(handles)),
            fontsize=BASE_FS - 1,
            handlelength=2.6, handletextpad=0.5, columnspacing=1.4,
            labelspacing=0.35, alignment="left", borderpad=0.5,
            **LEGEND_KW,
        )

    fig.tight_layout(rect=[0.0, 0.20, 1.0, 0.97])
    fig.subplots_adjust(wspace=0.10, hspace=0.10)

    stem = out_dir / "f4_multi_dataset_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {_rel(stem)}.pdf")
    return stem.with_suffix(".pdf")


# =============================================================================
# FIG 5 -- §V: UV-completion parameter-space translations
# =============================================================================
def f5_uv_translation_bounds(out_dir: Path) -> Path:
    """
    §V Fig 5: UV translation of the halo bound.
    Single panel (UPDATED 2026-07-17): dark-Higgs portal (sin theta, m_h')
    plane using the scalar Rayleigh halo bound and Eq. (dh_to_rayleigh).

    The former right panel (kinetically-mixed dark photon on the
    (epsilon, M_A') plane via the anapole/charge-radius matching) has been
    REMOVED: those operators couple through d^nu F_nu_mu, whose on-shell
    photon vertex vanishes identically, so halo attenuation places NO
    tree-level bound on (epsilon, M_A'). It is replaced by the
    electroweak-doublet (higgsino-like) dipole panel: the halo dipole bound
    converted to mu = 2 c_M/Lambda against the one-loop doublet prediction.
    See §V.C/§V.D of the revised text and VERIFICATION_eft_realphoton_fix.md.
    """
    from make_uv_translation_bounds import (
        dark_higgs_bound,
        lhc_higgs_signal_strength,
        draw_ew_doublet_panel,
        V_EW,
    )

    fig, (ax_h, ax_d) = plt.subplots(1, 2, figsize=(FIG5_WIDTH, FIG5_HEIGHT))

    def _mchi_legend(m_GeV: float) -> str:
        if m_GeV >= 1.0:
            return fr"$m_\chi={m_GeV:g}\,\mathrm{{GeV}}$"
        return fr"$m_\chi={m_GeV * 1000.0:g}\,\mathrm{{MeV}}$"

    # Line-styles that distinguish the curves in FIG5_MCHI_LIST.
    _MCHI_STYLES = ["-", (0, (5, 2)), (0, (7, 2, 1, 2)), (0, (2, 1.5))]

    # --- Left panel: dark-Higgs ------------------------------------------
    # PHYSICAL y-axis (fixed 2026-07-24): plot the REQUIRED dark-portal coupling
    #   y_{chi H} = y_chi * sin(theta) = (m_chi / v') * sin(theta),
    # NOT |sin theta| itself.  The halo bounds the Rayleigh Wilson coefficient
    #   c_r / Lambda^3 ~ (alpha / pi v) * y_{chi H} / m_h'^2 * F_loop,
    # so the quantity the data actually constrains is y_{chi H}.  Plotting it
    #   (i)  keeps the axis physical -- a |sin theta| axis running up to ~1e12 is
    #        meaningless because a mixing sine cannot exceed 1; and
    #   (ii) turns the perturbativity/unitarity ceiling into a genuine HORIZONTAL
    #        line: y_chi <= sqrt(4 pi) together with |sin theta| <= 1 imposes
    #        y_{chi H} <= sqrt(4 pi), independent of m_h'.  The m_h'^2 growth that
    #        the co-author expected the perturbativity bound to track now lives
    #        (correctly) in the halo diagonals themselves.  Note the m_chi-1/m_chi
    #        factors cancel in y_{chi H}, so the three benchmark curves differ only
    #        through the m_chi-dependence of the halo scale Lambda_R.
    m_hp = np.logspace(-3, 3.5, 200)

    # Panel window, fixed up front so every annotation anchor below is
    # unambiguous. The lower limit is 1e-1: nothing in the panel lives below
    # the achievable-coupling ceiling, so the old 1e-2 floor was dead space.
    _H_XMIN, _H_XMAX = 1e-3, 3e3
    _H_YMIN, _H_YMAX = 1e-1, 1e14
    _M_MATCH = 5.0          # GeV, contact-limit validity floor

    # Perturbativity + unitarity ceiling on the portal coupling.
    #   y_chi <= sqrt(4 pi)  AND  |sin theta| <= 1   ==>   y_{chi H} <= sqrt(4 pi).
    # Adding the LHC Higgs-mixing bound |sin theta| <= 0.33 sharpens this to the
    # maximum ACHIEVABLE portal coupling y_{chi H} <= 0.33 * sqrt(4 pi) ~ 1.17.
    y_pert = np.sqrt(4 * np.pi)                          # ~ 3.545
    sin_lhc = lhc_higgs_signal_strength()               # 0.33
    y_lhc_pert = sin_lhc * y_pert                        # ~ 1.17

    _gold_edge = FIG1_EXCLUSION_TEXT_COLOR
    _xspan = np.array([_H_XMIN, _H_XMAX])

    # Excluded region: a faint wash over the whole excluded half-plane plus a
    # hatched brush along the boundary. The wash is deliberately light -- it
    # covers most of the panel, so a heavy fill buries the curves under it.
    ax_h.fill_between(_xspan, y_lhc_pert, _H_YMAX,
                      color=FIG1_EXCLUSION_FILL_COLOR, alpha=0.10, lw=0, zorder=0)
    ax_h.fill_between(_xspan, y_lhc_pert, y_lhc_pert * 14.0,
                      facecolor="none", hatch="////", edgecolor=_gold_edge,
                      lw=0.0, alpha=0.45, zorder=1)

    # --- Heavy-mediator matching validity: m_h' < 5 GeV -------------------
    # Below m_h' ~ 5 GeV the mediator is no longer heavy compared to the
    # photon energies used here, so the contact-limit (heavy-mediator)
    # matching of Eq. (VI.1) [dh_to_rayleigh] breaks down and the halo
    # diagonals in this region are extrapolations. A light wash plus an
    # explicit boundary line, so it does not muddy the exclusion shading.
    ax_h.axvspan(_H_XMIN, _M_MATCH, color=COL_GUIDE, alpha=0.13, zorder=0, lw=0)
    ax_h.axvline(_M_MATCH, color=COL_GUIDE, ls=(0, (4, 3)), lw=LINEWIDTH * 0.9, zorder=2)
    ax_h.text(1.4e-3, 2.2e12, "contact-limit extrapolation",
              ha="left", va="center",
              color="0.30", fontsize=ANNOT_FS - 1, zorder=6,
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.70))

    # One colour for every "this work" curve -- cols[5], the same halo colour
    # the right-hand panel uses -- with the benchmarks separated by linestyle
    # alone. The 100 MeV and 1 GeV curves agree to ~2% (the m_chi dependence
    # cancels in y_{chi H} except through the halo scale) and would otherwise
    # plot on top of one another, so the second is drawn wide and pale
    # underneath and the third narrow and solid-toned on top: both stay
    # visible without either being displaced or recoloured.
    _MCHI_COLORS = [cols[5]] * 3
    _MCHI_LWS    = [1.3, 2.6, 1.0]
    _MCHI_ALPHAS = [1.0, 0.40, 1.0]

    for _k, _m_chi in enumerate(FIG5_MCHI_LIST):
        _sin_theta = dark_higgs_bound(m_hp, m_chi_GeV=_m_chi)
        _y_chiH = (_m_chi / V_EW) * _sin_theta          # = y_chi * sin(theta)
        ax_h.loglog(
            m_hp, _y_chiH,
            color=_MCHI_COLORS[_k % len(_MCHI_COLORS)],
            lw=LINEWIDTH * _MCHI_LWS[_k % len(_MCHI_LWS)],
            ls=_MCHI_STYLES[_k % len(_MCHI_STYLES)],
            alpha=_MCHI_ALPHAS[_k % len(_MCHI_ALPHAS)],
            zorder=4 + _k,
            label=fr"This work, {_mchi_legend(_m_chi)}",
        )

    ax_h.axhline(y_lhc_pert, color=_gold_edge, lw=LINEWIDTH * 1.25, ls="-", zorder=5)
    ax_h.axhline(y_pert, color=_gold_edge, lw=LINEWIDTH * 0.9, ls=":", zorder=5)

    ax_h.set_xlim(_H_XMIN, _H_XMAX)
    ax_h.set_ylim(_H_YMIN, _H_YMAX)
    ax_h.set_yticks([1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14])
    ax_h.set_xlabel(r"$m_{h'}$ [GeV]", fontsize=BASE_FS)
    ax_h.set_ylabel(r"$y_{\chi H}=y_\chi\,|\sin\theta|$", fontsize=BASE_FS)
    ax_h.set_title(r"Dark-Higgs portal", fontsize=BASE_FS + 1)
    ax_h.tick_params(labelsize=BASE_FS - 2)
    ax_h.grid(True, which="major", alpha=0.18)

    # --- Region annotations (y_{chi H} plane) ---
    # The two sides of the achievable-coupling ceiling. The diagonals
    # themselves are named in the legend, so they carry no separate label.
    # Sits below the rotated decade label and to the right of the gap arrow,
    # so nothing crosses anything else. The caption carries the fuller
    # "excluded, non-perturbative" wording.
    ax_h.text(2.2e3, 2.5e2, "Excluded",
              color=FIG5_DH_LABEL_COLOR, fontsize=FIG5_DH_LABEL_FS,
              fontweight="bold", ha="right", va="center", zorder=10,
              bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72))
    ax_h.text(1.4e-3, 0.30,
              fr"LHC $\otimes$ pert.: $y_{{\chi H}}\leq{y_lhc_pert:.1f}$",
              color=FIG5_DH_LABEL_COLOR, fontsize=FIG5_DH_LABEL_FS,
              fontweight="bold", ha="left", va="center", zorder=10)

    # The headline number: how far the required coupling sits above anything
    # achievable. Measured on the LOWEST (most conservative) benchmark and
    # computed here rather than hard-coded, so it cannot go stale.
    _x_gap = 1.0e2
    _m0 = FIG5_MCHI_LIST[0]
    _y_gap_hi = float((_m0 / V_EW) * dark_higgs_bound(np.array([_x_gap]), m_chi_GeV=_m0)[0])
    _decades = np.log10(_y_gap_hi / y_lhc_pert)
    ax_h.annotate("", xy=(_x_gap, _y_gap_hi), xytext=(_x_gap, y_lhc_pert),
                  arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.1,
                                  shrinkA=0, shrinkB=0), zorder=7)
    ax_h.text(_x_gap * 1.9, np.sqrt(y_lhc_pert * _y_gap_hi),
              fr"$\gtrsim{_decades:.0f}$ decades",
              color="0.15", fontsize=ANNOT_FS, fontweight="bold",
              ha="left", va="center", rotation=90, zorder=10,
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.70))

    ax_h.legend(**{**FIG5_LEGEND_KW, "ncol": 1}, **LEGEND_KW)

    # --- Right panel: electroweak-doublet dipole (added 2026-07-17) -------
    draw_ew_doublet_panel(ax_d, base_fs=BASE_FS, linewidth=LINEWIDTH, col_coll=cols[3], col_this=cols[5])

    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.subplots_adjust(wspace=0.24)
    fig.subplots_adjust(wspace=0.22)

    stem = out_dir / "f5_uv_translation_bounds"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {_rel(stem)}.pdf")
    return stem.with_suffix(".pdf")


# =============================================================================
# Registry + entry point
# =============================================================================
FIGS = {
    1: f1_uv_complete_tau_vs_mchi,
    2: f2_sensitivity_map,
    3: f3_halo_constraints,
    4: f4_multi_dataset_overlay,
    5: f5_uv_translation_bounds,
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--only", type=int, nargs="+", choices=sorted(FIGS),
        default=sorted(FIGS),
        help="Subset of figures to build. Default: all.",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    ap.add_argument(
        "--copy-to", type=Path, default=None,
        help="Optional additional destination for the figures (e.g. a paper source tree).",
    )
    args = ap.parse_args(argv)

    print(f"[make_paper_results_figures] out_dir = {args.out_dir}")
    if args.copy_to is not None:
        print(f"[make_paper_results_figures] copy_to = {args.copy_to}")

    _apply_paper_style()

    made = []
    for n in args.only:
        print(f"\n=== fig {n} ===")
        made.append(FIGS[n](args.out_dir))

    if args.copy_to is not None:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        print(f"\n=== copying {len(made)} figure(s) to {args.copy_to} ===")
        for p in made:
            for suffix in (".pdf", ".png"):
                src = p.with_suffix(suffix)
                if src.exists():
                    dest = args.copy_to / src.name
                    shutil.copy2(src, dest)
                    print(f"  copied -> {dest.name}")

    print(f"\nBuilt {len(made)} figure(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
