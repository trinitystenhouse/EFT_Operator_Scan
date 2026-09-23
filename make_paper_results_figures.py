#!/usr/bin/env python3
r"""
Paper result figures for "Illuminating the Dark Sector -- One Scattered Photon
at a Time". Following the Totani reanalysis methodology: this is the single
source of truth for every figure in the paper. Every knob you might want to
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
  f2_sensitivity_map             (--only 2) §III  4-panel f_required heatmap,
                                            one operator per dimension with the
                                            dim-7 parity pair resolved.
  f3_halo_constraints            (--only 3) §IV.C 3-panel halo exclusion
                                            overlay with collider / DD / ID /
                                            cosmology bounds.
  f4_multi_dataset_overlay       (--only 4) §IV.D halo + IGRB cross-dataset
                                            overlay with optional PPPC WW/bb
                                            benchmarks at the companion best-fit
                                            annihilator masses.
  f5_uv_translation_bounds       (--only 5) §V    two-panel (sin theta, m_h')
                                            and (epsilon, M_A') translations of
                                            the halo bound onto the two
                                            illustrative UV completions.

Usage
-----
  python make_paper_results_figures.py                  # all five
  python make_paper_results_figures.py --only 2 4       # selected
  python make_paper_results_figures.py --copy-to ~/Documents/PhD/Year\ 2/my_work/EFT_Operator_Scan/paper_plots

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

# Select the non-interactive backend HERE, before pyplot is imported.
#
# This script only writes files, so Agg is the right backend for it -- but until
# 2026-09-21 it never said so, and picked up whatever matplotlib defaulted to.
# On this machine that is "macosx", whose text metrics differ slightly from
# Agg's, which changes the tight bounding box a figure is saved with. A full run
# happened to get Agg anyway, because Fig. 1's module calls matplotlib.use("Agg")
# when it is imported; a run that did not build Fig. 1 never imported that module
# and so was laid out under macosx. The figures were identical in content and
# differed only in their crop -- Fig. 2 came out 0.24 pt wider -- but that was
# enough that `--only N` did not reproduce the published file for any N except 1.
# Fixing the backend here makes the output independent of which figures are
# built, of the order they are built in, and of the host's default backend.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

_HERE = Path(__file__).resolve().parent          # EFT_Operator_Scan/
_REPO = _HERE.parent                             # DM_Photon_Scattering/
sys.path.insert(0, str(_REPO))                   # for helpers.trinity_plotting
sys.path.insert(0, str(_HERE))                   # for core.*

def _rel(p):
    """Path for display: relative to the repo when possible, else as given."""
    p = Path(p)
    try:
        return p.relative_to(_HERE)
    except ValueError:
        return p

DEFAULT_OUT_DIR = _HERE / "paper_plots"
from helpers.trinity_plotting import (  # noqa: E402
    add_style_argument, apply_plot_style, current_savefig_kwargs, get_cmap_colors,
    normalise_plot_style, plasma_color, scaled_figsize as _figsize, style_sizing,
    theme_accent as _accent, theme_ink as _ink, theme_legend_kw,
)

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

# The three above are the PRINT baseline. Figures here pass explicit font sizes
# and line widths (rather than leaning on rcParams), so on a slide style they
# would otherwise stay at print size while the canvas around them grew. Keep an
# immutable copy: _apply_paper_style() rescales BASE_FS / ANNOT_FS / LINEWIDTH
# from these whenever a non-paper style is selected.
_PRINT_BASE_FS, _PRINT_ANNOT_FS, _PRINT_LINEWIDTH = BASE_FS, ANNOT_FS, LINEWIDTH

# True on every non-paper (slide) style. The slide styles raise the font size by
# 1.5x while the canvas only grows by 1.35x, so type is ~11% larger RELATIVE to
# each axes box than it is in print. Every hand-placed annotation in this file
# was solved against the print ratio, so the ones with little clearance need a
# style-aware nudge. _apply_paper_style() sets this; the figN() bodies read it.
# Guarded so the paper layout is bit-for-bit what was published.
_SLIDE = False

# --- Shared legend styling (Totani convention). Every figN() uses this ------
LEGEND_KW = dict(frameon=True, framealpha=0.6, facecolor="white", edgecolor="0.7")

# --- EFT kinematic-validity ceiling ----------------------------------------
# Maximum photon energy [GeV] used to place the "EFT valid" wedge in the
# operator panels: Lambda^2 >= max(s_max, |t|_max) evaluated at this omega.
# Applies to Figs 2, 3 and 4 (the operator-basis panels). Fig 1 has no EFT
# wedge and Fig 5 draws none, so neither is affected.
#
# CAVEAT: this positions the DRAWN GUIDE only. The exclusion boundaries
# themselves were generated over the full LAT band (omega_max ~ 494 GeV, stored
# per-file as omega_max_for_validity); nothing here recomputes them. Setting a
# lower ceiling therefore draws a stricter validity line than the band the
# limits were derived over -- regenerate the boundaries if you need the two to
# correspond.
#
# CORRECTED 2026-08-11: was 200.0, the ENERGY CUT, where it should be the centre
# of the highest bin surviving that cut. The production grids record
# omega_max_for_validity = 168.9323 GeV (halo, 23 grids) and 172.2183 GeV (IGRB,
# 3 grids), and Sec. IV B quotes 168.9. Drawing at 200 put the wedge sqrt(200/
# 168.9) = 8.8% high, which is invisible at 0.037 dex but moved the wedge
# crossings a reader measures off Fig. 3 by 18-21% against the quoted values.
#
# Do NOT set this to None: load_default_omega_max() then reads the FULL LAT band
# from the default spectrum file, not each grid's stored value, so the wedge
# would be drawn far too high. The docstring above used to claim otherwise.
#
# Fig. 4 overlays halo and IGRB contours under a single wedge; the halo value is
# used, and the IGRB's own 172.2 GeV would move it by sqrt(172.2/168.9) = 0.004
# dex, well inside the line width.
EFT_OMEGA_MAX_GEV = 168.9323

# --- Boundary-grid variant ---------------------------------------------------
# Tag inserted before "_90cl" when the figures locate their boundary files.
# "_profnorm" selects the grids computed with the test statistic PROFILED over
# the source normalisation A (the method the paper describes); "" selects the
# legacy grids computed at A fixed to 1. Both sets are generated at the same
# 200 GeV ceiling, so switching this changes only the treatment of A.
#
# "_profnorm_removalfix" selects the CORRECTED grids: closed-form sigma_tot
# (no angular quadrature), the energy-migration removal criterion, and the
# template-computed w(theta). THIS IS THE DEFAULT, and it is what main.tex
# quotes: dipole 0.32 GeV at m_chi = 60 GeV, scalar Rayleigh 0.21 GeV at
# 0.22 GeV, dipole plateau 0.29 GeV, CMB elastic 13x below the halo contour
# at 1e-6 GeV.
#
# The default used to be "_profnorm", which reproduced the pre-correction
# published figures. That silently shipped figures disagreeing with the text
# by a factor of two in Lambda and by 2500x in the dipole peak mass, five
# separate times, because forgetting an environment variable is not a
# detectable event. Reproducing a superseded figure is now the thing you have
# to ask for:
#
#     EFT_BOUNDARY_SUFFIX=_profnorm  python make_paper_results_figures.py   # pre-correction
#     EFT_BOUNDARY_SUFFIX=""         python make_paper_results_figures.py   # legacy, A fixed to 1
#
# Quick check that a built Fig 3 is the corrected one: the black dipole
# contour peaks at m_chi = 60 GeV (about 0.49 of the panel width), not at
# 24 MeV (about 0.35).
EFT_BOUNDARY_SUFFIX = os.environ.get("EFT_BOUNDARY_SUFFIX", "_profnorm_removalfix")

# The IGRB grids carry their own tag: they were rescanned on the corrected
# Ackermann+ 2015 Table 3 spectrum (2026-08-13) and again on the dataset's own
# f_bin (2026-09-18). This is a DEFAULT, not an environment lookup that falls
# back to the halo tag: until 2026-09-19 Fig. 4 fell back to
# EFT_BOUNDARY_SUFFIX whenever EFT_IGRB_BOUNDARY_SUFFIX was unset, which
# silently drew the August old-spectrum IGRB grids -- a stale curve with no
# warning anywhere. _assert_igrb_f_bin() below turns any remaining mismatch into
# a loud failure rather than a quiet wrong figure.
EFT_IGRB_BOUNDARY_SUFFIX = os.environ.get(
    "EFT_IGRB_BOUNDARY_SUFFIX", "_profnorm_removalfix_igrbfix2")

# Bin geometry each dataset must be carrying (core/bin_geometry.py). Grids built
# before 2026-09-18 record no f_bin at all and used a hard-coded 0.231, the halo
# value, on every dataset.
_EXPECTED_F_BIN = {"igrb": 0.159104, "halo": 0.230548}


def _assert_boundary_f_bin(paths, dataset, *, tol=5e-5):
    """Fail loudly if a boundary file was not built with this dataset's own f.

    Checks the recorded metadata, so it catches both the pre-2026-09-18 grids
    (no f_bin recorded) and a grid built with --f-bin forcing another value.
    """
    want = _EXPECTED_F_BIN[dataset]
    bad = []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        if "f_bin_median" not in d.files:
            bad.append(f"{Path(path).name}: no f_bin recorded (built before 2026-09-18)")
            continue
        got, mode = float(d["f_bin_median"]), str(d["f_bin_mode"])
        if abs(got - want) > tol or mode != "derived":
            bad.append(f"{Path(path).name}: f_bin {got:.6f} ({mode}), expected {want:.6f} derived")
    if bad:
        raise RuntimeError(
            f"{dataset} boundary grids were not built with the dataset's own bin geometry:\n  "
            + "\n  ".join(bad)
            + "\nRegenerate them (no --f-bin) or point the suffix at grids that were.")

# --- Palette (mirrors make_paper_style_operator_overlays for consistency) ---
cols = get_cmap_colors(cmap_name="plasma", n=10, start=0, end=1)

COL_THIS_WORK = "black"
COL_COLLIDER  = "#FF7AC6"   # pink
COL_DIRECT    = "#F2D53C"   # yellow
COL_INDIRECT  = "#9B6DFF"   # purple
COL_COSMOLOGY = "#F26D0C"   # orange -- distinct from COL_DIRECT/COL_COLLIDER so cosmology curves stand out
COL_GUIDE     = "#666666"   # grey (EFT/unitarity guides)
COL_THERMAL   = "#7DDC84"   # green -- thermal-relic theory line (limits.THERMAL_COLOR)
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
FIG2_WIDTH,  FIG2_HEIGHT  = FULL_W, 4.6   # §III  4-panel f_required grid (2 cols x 2 rows)
FIG3_WIDTH,  FIG3_HEIGHT  = FULL_W, 3.9725  # §IV.C halo constraints with external bounds
#: Slide height for Fig 3, used INSTEAD of the usual _slide_height() stretch.
#: The print figure is nearly square-ish per panel (aspect ~1.8 overall); on a
#: 16:9 slide that leaves no room for a heading above it. A shorter canvas at
#: the same width turns it into a wide banner (aspect ~2.4) with space over the
#: top. Only the slide styles use this -- print keeps FIG3_HEIGHT.
FIG3_HEIGHT_SLIDE = 4.75
#: Slide WIDTH for Fig 3. At full-size legend type the 4-column strip is wider
#: than FULL_W, so bbox_inches="tight" grew the canvas sideways and the strip
#: ran over the x-axis labels. Widening the figure to the width the legend
#: actually needs puts the panels back in charge of the canvas: the axes now
#: span the same width as the strip instead of being an island inside it.
FIG3_WIDTH_SLIDE = 9.0
FIG3_LEGEND_Y             = 0.086        # bottom-strip legend anchor; higher = closer to x-axis labels
FIG4_WIDTH,  FIG4_HEIGHT  = FULL_W, 3.8   # §IV.D multi-dataset overlay (3-col legend strip)
FIG4_LEGEND_Y             = 0.1          # bottom legend anchor; higher = closer to x-axis labels
FIG4_LEGEND_NCOL          = 2
FIG5_WIDTH,  FIG5_HEIGHT  = FULL_W, 3.05   # §V    UV-completion parameter-space bounds (curves labelled in-panel, no legend strip)

# Figures 3 and 5 reserve a fixed fraction of their height for a bottom legend
# strip (rect bottom 0.233 / 0.248). The strip's contents are sized in points,
# so on a slide style they grow by 1.5x while the canvas only grows by 1.35x --
# the legend outgrows its reserved band and rides up onto the x-axis labels.
# Buying the difference back out of the axes is the WRONG fix: it re-collides
# Fig 5's hand-placed "Excluded" / "decades" annotations (see the 2026-08-07
# note in f5). Buy it with figure height instead, so the axes box keeps exactly
# the proportions every in-panel annotation was solved against.
_SLIDE_STRETCH = 1.5 / 1.35


def _slide_height(h):
    """Figure height, stretched on the slide styles to absorb a taller legend."""
    return h * _SLIDE_STRETCH if _SLIDE else h

# =============================================================================
# PER-FIGURE KNOBS (operator lists, contour levels, benchmark masses)
# =============================================================================
# Fig 2 sensitivity-map contour levels (multiplicative exposure boosts).
# FIG2_CONTOUR_COLOURS maps each level in FIG2_CONTOURS to a colour drawn on
# top of the heatmap in every panel; FIG2_CONTOUR_LABELS drives the legend.
# Edit either dict to recolour or rename the exposure lines across every
# panel of Fig 2 without touching make_sensitivity_map.py.
# The x10 and x1e3 levels were dropped: five contours crowded the panels and
# the three retained levels already bracket the range (current sensitivity,
# CTA-scale, and the parametric ceiling). FIG2_CONTOUR_COLOURS / _LABELS below
# still carry entries for the retired levels, so re-adding them here is a
# one-line change.
# DRAW_UNITARITY=0 omits the partial-wave unitarity guide from f2/f3/f4 and
# makes the shaded wedge kinematic-only. FIG2_CONTOURS drops the CTA-scale
# (x50) level when the guide is off, since that legend entry goes with it.
# Default OFF as of 2026-08-07. All four legacy partial-wave guides were
# dimensionally ill-formed (returning Lambda in GeV^1/2 or GeV^1/3) and the
# curve was removed from the manuscript text, so a figure still drawing it
# would contradict the paper. Set DRAW_UNITARITY=1 to restore the legacy
# figures. Turning it off also drops the CTA-scale (x50) contour, whose
# legend entry went with it.
DRAW_UNITARITY = os.environ.get("DRAW_UNITARITY", "0") not in ("0", "false", "False")
FIG2_CONTOURS = [1.0, 50.0, 1.0e6] if DRAW_UNITARITY else [1.0, 1.0e6]
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
# UPDATED 2026-07-17 (real-photon amplitude correction, see
# VERIFICATION_eft_realphoton_fix.md): the anapole and charge-radius operators
# couple through d^nu F_nu_mu, whose on-shell-photon vertex vanishes
# identically -- they have EXACTLY ZERO tree-level gamma chi -> gamma chi
# cross section and are stated as such in the text rather than plotted.
# Their former panels are replaced by Rayleigh panels, so the figure still
# spans dim-5 (dipoles), dim-7 fermionic (Rayleigh), and dim-6 scalar operators.
#
# Ordered by operator dimension so the reader walks up the tower: dim-5 dipole,
# dim-6 scalar Rayleigh, then the dim-7 Rayleigh pair. Keeping parity-even and
# parity-odd adjacent puts side by side the comparison the text draws, the extra
# power of |t| that separates sigma ~ omega^4 from omega^6.
#
# REDUCED 2026-08-08 from six panels to four. The two dropped panels were
# redundant, and measurably so on the production grids:
#
#   dipole_electric  peak Lambda = dipole_magnetic to 0.0000% -- the two carry
#       identical spin-averaged real-photon amplitudes (electromagnetic duality
#       rotates one into the other), so the panels were bit-identical.
#   rayleigh_full    peak Lambda = 1.0607 GeV vs 1.0567 for rayleigh_even, a
#       0.37% difference: the incoherent sum is dominated by the even piece,
#       which alone carries the 4 m_chi^2 chirality-flip term the odd operator
#       lacks. The odd panel is 26% below full and is retained on its own.
#
# NOTE: Figs 3 and 4 still show `rayleigh_full_majorana`, which is not a panel
# here. That is deliberate, not an oversight -- Fig 2 resolves the dim-7 family
# into its parity components, while Figs 3/4 carry the single combined operator
# as the representative Majorana case.
FIG2_OPERATORS = [
    "dipole_magnetic",
    "scalar_rayleigh",
    "rayleigh_even_majorana",
    "rayleigh_odd_majorana",
]

# Per-panel in-panel title overrides for Fig 2. Empty as of 2026-08-08: the
# two-line breaks existed only because "(Majorana)" ran into the panel edge at
# the old 3-column width. At 2 columns the panels are half again as wide and
# every title fits on one line, so all four now come straight from
# PANEL_CONFIGS, which uses one "Operator (DM type)" convention throughout.
FIG2_TITLE_OVERRIDES = {}

# Fig 2 cold-dark-matter reference: vertical dotted-yellow line at the CDM
# mass floor m_chi = 1 MeV = 1e-3 GeV.
FIG2_CDM_MCHI_GEV = 1.0e-3
FIG2_CDM_COLOR    = plasma_color(0.4)   # plasma pink, on Fig 2's plasma heatmap.
                                        # (No longer the rho^2.5 accent -- that
                                        # moved to a named blue; see FIG3_PROFILE_COLORS.)
                                        # used in Figs 3/4, so the same hue
                                        # never means two things across the set
FIG2_CDM_LABEL    = r"CDM ($m_\chi = 1$ MeV)"

# Fig 3 halo constraint overlay (3-panel paper summary)
FIG3_PROFILES = ["pixelwise_global_rho2", "pixelwise_global_rho2.5"]
# UPDATED 2026-07-17: anapole panel replaced by Majorana Rayleigh Even --
# the anapole has exactly zero tree-level real-photon cross section.
FIG3_OPERATORS = ["dipole_magnetic", "rayleigh_full_majorana", "scalar_rayleigh"]
# The rho^2.5 scattering curve was plasma_color(0.4). That samples the colormap,
# and the dark styles lift the sample off a brightness floor -- which moves it
# ALONG the ramp, not just up in value: hue 315 (magenta) in print became hue 4
# (salmon) on a slide, landing 20 degrees from COL_COSMOLOGY's orange, so the
# scattering curve and the two CMB curves read as one family. A named accent is
# hue-stable instead: theme_accent only raises value/desaturates along the SAME
# hue, so this blue is the same blue in both styles. Blue is also the one empty
# band in Fig 3's palette -- collider 326, direct 50, indirect 259, cosmology 25,
# the EFT wedge 180, and a minor green at 120 leave 200-250 free.
FIG3_PROFILE_COLORS = {
    "pixelwise_global_rho2": COL_THIS_WORK,
    "pixelwise_global_rho2.5": _accent("#2E6FE8"),
}

# 68% expected band on Lambda_90 (added 2026-09-14). The contours are
# median-expected (Asimov) sensitivities; the band is the toy-to-toy spread of
# the contour an experiment would actually draw. It is built by
# constraint_boundaries/toys/run_expected.py from background-only toys on the
# rho^2 posterior, with the production statistic (chi2 - chi2_min), threshold
# (4.61), grid and contour extractor, so it brackets the rho^2 curve drawn here
# and carries the same grid-resolution bias. Each band is built from toys on its
# own posterior: Fig. 3 carries one per halo profile (rho^2 and rho^2.5); Figs.
# 4 and 5 show only rho^2. A band built on grids other than EFT_BOUNDARY_SUFFIX,
# or on another profile, is refused rather than drawn. EFT_EXPECTED_BAND=0 draws
# the figures without it.
DRAW_EXPECTED_BAND = os.environ.get("EFT_EXPECTED_BAND", "1") != "0"
EXPECTED_BAND_DIR = _HERE / "constraint_boundaries" / "toys"
EXPECTED_BAND_ALPHA = 0.30
EXPECTED_BAND_OPERATOR = {                  # Fig. 3/4 panel key -> toy operator key
    "dipole_magnetic":        "dipole_magnetic",
    "rayleigh_full_majorana": "rayleigh_full",
    "scalar_rayleigh":        "scalar_rayleigh",
}


class _BandedHandle(tuple):
    """(band swatch, contour line) legend handle. matplotlib draws a tuple
    handle as its members overlaid; get_label() is here because
    draw_bottom_grouped_legend reads it off every handle."""
    def __new__(cls, swatch, line, label=None):
        obj = super().__new__(cls, (swatch, line))
        obj._label = line.get_label() if label is None else label
        return obj

    def get_label(self):
        return self._label


def _band_swatch(color):
    from matplotlib.patches import Patch
    return Patch(facecolor=color, alpha=EXPECTED_BAND_ALPHA, edgecolor="none")


def _expected_band(operator_key, profile="pixelwise_global_rho2"):
    """(m_chi, Lambda_16, Lambda_84) in raw GeV for a Fig. 3 operator, or None."""
    if not DRAW_EXPECTED_BAND:
        return None
    toy_key = EXPECTED_BAND_OPERATOR.get(operator_key, operator_key)
    tag = "" if profile == "pixelwise_global_rho2" else "_" + profile.replace("pixelwise_global_", "")
    path = EXPECTED_BAND_DIR / f"expected_band_{toy_key}{tag}.npz"
    if not path.exists():
        print(f"  [band] {_rel(path)} missing -- run constraint_boundaries/toys/run_expected.py")
        return None
    d = np.load(path, allow_pickle=True)
    if str(d["boundary_suffix"]) != EFT_BOUNDARY_SUFFIX or str(d["halo_profile"]) != profile:
        print(f"  [band] {path.name} built on {str(d['halo_profile'])}{str(d['boundary_suffix'])} "
              f"but this curve is {profile}{EFT_BOUNDARY_SUFFIX} -- not drawn")
        return None
    return (np.asarray(d["m_GeV"], float), np.asarray(d["lam_lo68"], float),
            np.asarray(d["lam_hi68"], float))


def _draw_expected_band(ax, operator_key, color, zorder, profile="pixelwise_global_rho2"):
    """Shade the 68% band on a Lambda_90 panel (Figs. 3, 4). True if drawn."""
    band = _expected_band(operator_key, profile)
    if band is None:
        return False
    m, lo, hi = band
    lo, hi = _fig3_rescale_lambda(lo, operator_key), _fig3_rescale_lambda(hi, operator_key)
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(m, np.where(ok, lo, 1.0), np.where(ok, hi, 1.0), where=ok,
                    color=color, alpha=EXPECTED_BAND_ALPHA, lw=0, zorder=zorder)
    return True

# Fig 4 multi-dataset overlay
FIG4_DATASETS = ["halo", "igrb"]           # drop 'dsph' -- profile-normalised fit degenerate
# Measured spectra only. The PPPC two-component benchmarks were dropped from
# this figure because their contours sit almost on top of the halo curve at low
# Lambda -- they added two more linestyles and two legend entries without
# separating visibly. The boundary grids still exist (and are regenerated at the
# same 200 GeV ceiling and 100x100 granularity), so re-adding "pppc" below
# restores them; the PPPC-specific knobs underneath are kept live for that.
# measured = self-fit. Add "pppc" for the two-component benchmarks; they were
# dropped from the published figure because they overlap the lower part of the
# measured-template contours. Overridable: FIG4_SOURCES="measured pppc".
FIG4_SOURCES = os.environ.get("FIG4_SOURCES", "measured").split()
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
FIG3_TITLE_FS             = BASE_FS - 1    # was BASE_FS + 1: the bold two-line block ran into the dipole panel's curves
FIG3_TITLE_HALIGN         = "left"
FIG3_TITLE_VALIGN         = "top"

FIG3_EFT_LABEL_XY         = (0.05, 0.72)   # left interior of the wedge, ABOVE the
                                           # thermalisation box (whose top edge is at
                                           # axes-fraction 0.63/0.61/0.48 per panel)
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

# -----------------------------------------------------------------------------
# FIG 3 OVERLAYS ADDED 2026-08-07
# -----------------------------------------------------------------------------
# (a) Cosmological closure of the EFT-valid wedge, all three panels.
#
# Without this the figure shades a cyan "EFT valid" region that reads as open
# parameter space, while Sec. V C argues that region is thermally excluded. The
# excluded set is the INTERSECTION of two conditions:
#
#   m_chi < 4e-4 GeV       BBN floor for any species in equilibrium with the
#                          plasma (Sabti et al., arXiv:1910.01649), and
#   Lambda < Lambda_therm  below which gamma gamma <-> chi chibar -- the crossed
#                          channel of the elastic process -- reaches Gamma/H > 1
#                          at T = 1 MeV, so the dark sector thermalises with the
#                          photon bath.
#
# A sub-MeV species in thermal contact with the photon bath is excluded, so the
# overlap of the two is excluded regardless of what the attenuation scan says.
FIG3_BBN_MCHI_MAX = 4.0e-4          # [GeV]
FIG3_LAMBDA_THERM = {               # [GeV], raw -- rescaled per panel below
    "dipole_magnetic":        600.0,   # dimension-5 dipole
    "scalar_rayleigh":        320.0,   # dimension-6 scalar Rayleigh
    "rayleigh_full_majorana":   5.0,   # dimension-7 Rayleigh
}
FIG3_THERM_LABEL   = "Thermalisation + BBN"
FIG3_THERM_HATCH   = "///"
FIG3_THERM_ALPHA   = 0.30

# (b) CMB annihilation bound and (c) direct detection, DIPOLE PANEL ONLY.
# Both constrain the magnetic dipole moment mu and have no Rayleigh analogue,
# so they are drawn only where mu = 2 c_M / Lambda is the physical variable.
# Both are converted with Lambda = 2 / mu, matching Fig. 5's right panel.
#
# CMB: make_uv_translation_bounds.cmb_dipole_bound(), the SAME function Fig. 5
# draws, so the two figures cannot disagree. It was a local e+e--only copy,
# mu < 6.13e-5 sqrt(m_chi) (Lambiase et al., arXiv:2102.04840), until
# 2026-09-14, when chi chibar -> gamma gamma (Weiner & Yavin Eq. 9, f_eff = 0.4)
# was added to it. Below ~100 GeV nothing moves; gamma gamma overtakes e+e- near
# 0.5 TeV, and above that Lambda = 2/mu turns from falling as m^-1/2 to rising
# as m^+1/4. Above m_chi ~ 2.7 TeV mu m_chi > 2 (Lambda < m_chi), where the
# two-insertion rate is outside the EFT and the curve is indicative only.
FIG3_CMB_M_PRIOR_MAX = 100.0        # [GeV] upper edge of Lambiase's mass prior
# Direct detection: make_uv_translation_bounds.dd_dipole_bound(), the digitised
# PandaX-4T limit (Nature 618, 47, Fig. 3c) that Fig. 5 also draws -- dash-dot
# over its published range from 10 GeV to 1 TeV, dashed above 1 TeV where it is
# continued as sqrt(m_chi). It replaced a local copy of an analytic estimate
# (1e-6 GeV^-1 at 0.7 TeV, 1.72x too weak) on 2026-09-14. The data file opts out
# of the constraints_data auto-plot, so plot_panel does not draw it a second time.


def _fig3_rescale_lambda(lam_raw, operator_key):
    """Put a raw Lambda [GeV] onto the panel's rescaled y axis.

    Every curve in Fig. 3 is plotted as lambda_plot_GeV = Lambda / (c^{1/n}
    f_scat^{1/p}), so a literal GeV value plotted against that axis would be
    wrong the moment either factor left unity. Route it through the same
    function the boundary grids use rather than assuming c = f_scat = 1.
    """
    import make_paper_style_operator_overlays as _mp
    from core.attenuation_eft import _paper_y_axis_values
    from constraint_generation.make_data_driven_scattering_limits import (
        operator_couplings,
    )
    cfg = _mp.PANEL_CONFIGS[operator_key]
    op, dm = cfg["operator"], cfg["dm_type"]
    c_s, c_p, c_phi = operator_couplings(op, dm)
    return _paper_y_axis_values(np.atleast_1d(np.asarray(lam_raw, dtype=float)),
                                dm, op, c_s=c_s, c_p=c_p, c_phi=c_phi)


def _fig3_draw_overlays(ax, operator_key):
    """Draw the thermalisation/BBN exclusion (all panels) plus the CMB
    annihilation and direct-detection dipole bounds (dipole panel only).

    Returns (legend_proxies, reported_values) where reported_values gives each
    new curve at m_chi = 700 GeV in the panel's rescaled units.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    proxies, reported = [], {}

    # --- (a) thermalisation + BBN, every panel --------------------------------
    lam_therm = float(_fig3_rescale_lambda(FIG3_LAMBDA_THERM[operator_key],
                                           operator_key)[0])
    reported["thermalisation"] = lam_therm
    x_edge = min(FIG3_BBN_MCHI_MAX, x_hi)
    if x_edge > x_lo and lam_therm > y_lo:
        ax.fill_between([x_lo, x_edge], [y_lo, y_lo],
                        [min(lam_therm, y_hi)] * 2,
                        facecolor="none", hatch=FIG3_THERM_HATCH,
                        edgecolor=_ink("0.35"), lw=0.0,
                        alpha=FIG3_THERM_ALPHA, zorder=2.5)
        # Solid boundary on the two closed edges so the region reads as a
        # bound, not as texture.
        ax.plot([x_lo, x_edge], [min(lam_therm, y_hi)] * 2,
                color=_ink("0.35"), lw=1.2, zorder=2.6)
        ax.plot([x_edge, x_edge], [y_lo, min(lam_therm, y_hi)],
                color=_ink("0.35"), lw=1.2, zorder=2.6)
        # Numeric label, seated in the bottom-left corner of the hatched box.
        # The dipole (600) and scalar (320) boxes sit 0.27 dex apart, about 2% of
        # a 14-decade panel, so by eye they read as the same height and have
        # already drawn a query. The number is what distinguishes them; the drawn
        # height cannot. Offsets are multiplicative because both axes are log.
        ax.text(x_lo * 10 ** 0.25, y_lo * 10 ** 0.25,
                rf"$\Lambda_{{\rm therm}}={FIG3_LAMBDA_THERM[operator_key]:g}$ GeV",
                color=_ink("0.25"), fontsize=ANNOT_FS - 2.0,
                ha="left", va="bottom", zorder=61,
                bbox=dict(boxstyle="round,pad=0.14", fc=_ink("white"),
                          ec="none", alpha=0.72))
    proxies.append(Patch(facecolor="none", hatch=FIG3_THERM_HATCH,
                         edgecolor=_ink("0.35"), label=FIG3_THERM_LABEL))

    # --- (b)+(c) dipole-only bounds ------------------------------------------
    if operator_key == "dipole_magnetic":
        m_in = np.logspace(np.log10(1.0e-3), np.log10(FIG3_CMB_M_PRIOR_MAX), 200)
        m_ex = np.logspace(np.log10(FIG3_CMB_M_PRIOR_MAX), np.log10(x_hi), 200)
        from make_uv_translation_bounds import cmb_dipole_bound
        lam_cmb = lambda m: _fig3_rescale_lambda(2.0 / cmb_dipole_bound(m), operator_key)
        ax.loglog(m_in, lam_cmb(m_in), color=COL_COSMOLOGY, lw=1.6, ls="-",
                  zorder=6)
        ax.loglog(m_ex, lam_cmb(m_ex), color=COL_COSMOLOGY, lw=1.6, ls="--",
                  zorder=6)
        reported["cmb_annihilation"] = float(lam_cmb(np.array([700.0]))[0])
        proxies.append(Line2D([0], [0], color=COL_COSMOLOGY, lw=2.0,
                              label="CMB, annihilation"))

        from make_uv_translation_bounds import (
            dd_dipole_bound, DD_M_MIN, DD_M_PUBLISHED_MAX,
        )
        lam_dd = lambda m: _fig3_rescale_lambda(2.0 / dd_dipole_bound(m), operator_key)
        m_dd = np.logspace(np.log10(DD_M_MIN), np.log10(DD_M_PUBLISHED_MAX), 200)
        ax.loglog(m_dd, lam_dd(m_dd), color=COL_DIRECT, lw=1.6,
                  ls=(0, (6, 1.6, 1, 1.6)), zorder=6)
        m_dd_ex = np.logspace(np.log10(DD_M_PUBLISHED_MAX), np.log10(x_hi), 200)
        ax.loglog(m_dd_ex, lam_dd(m_dd_ex), color=COL_DIRECT, lw=1.6, ls="--", zorder=6)
        reported["direct_detection_dipole"] = float(lam_dd(np.array([700.0]))[0])
        # Drawn but deliberately NOT in the legend: it is the same
        # experimental family as the existing "Direct detection" entry, and a
        # second entry for a scaling estimate would give it equal billing with
        # the reproduced exclusions.
    return proxies, reported


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
FIG4_TITLE_FS             = BASE_FS - 1    # matches FIG3_TITLE_FS
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
FIG4_DRAW_UNITARITY        = DRAW_UNITARITY
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
def _apply_paper_style(style="paper"):
    """Install a plotting style and re-derive every style-dependent colour.

    The CONFIG block above is evaluated at IMPORT time, before any style is
    active, so its palette is the print one. This runs after the style is
    installed and rebuilds the entries that depend on it:

      * cmap-derived colours (get_cmap_colors is lifted above the dark styles'
        brightness floor, so `cols` must be resampled);
      * black/white ink (COL_THIS_WORK, the panel-title colours);
      * dark hex accents chosen for contrast against white (the gold and brown
        caption colours), lifted along their own hue by theme_accent;
      * LEGEND_KW, splatted into every legend in this file.

    Colours drawn ON TOP of the plasma heatmap (FIG2_TITLE_COLOR,
    FIG2_UNITARITY_LABEL_COLOR) are deliberately NOT touched: their background
    is the heatmap, not the page, so they must stay light in every style.
    """
    global cols, COL_THIS_WORK, COL_GUIDE
    global FIG1_HIGGS_COLOR, FIG1_GRAV_COLOR, FIG1_THRESHOLD_COLOR
    global FIG1_EXCLUSION_TEXT_COLOR, FIG2_CONTOUR_COLOURS
    global FIG3_TITLE_COLOR, FIG3_EFT_LABEL_COLOR, FIG4_TITLE_COLOR
    global FIG5_HALO_MAP_LABEL_COLOR, FIG5_PERTURB_LABEL_COLOR
    global FIG5_DH_LABEL_COLOR, FIG5_DP_LABEL_COLOR
    global FIG3_PROFILE_COLORS, FIG4_DATASET_COLOURS

    global BASE_FS, ANNOT_FS, LINEWIDTH, FIG5_LEGEND_KW, _SLIDE
    global FIG2_TITLE_FS, FIG2_EFT_LABEL_FS, FIG2_UNITARITY_LABEL_FS
    global FIG3_TITLE_FS, FIG3_EFT_LABEL_FS, FIG3_UNITARITY_LABEL_FS
    global FIG4_TITLE_FS, FIG4_EFT_LABEL_FS, FIG4_UNITARITY_LABEL_FS
    global FIG5_HALO_MAP_LABEL_FS, FIG5_PERTURB_LABEL_FS
    global FIG5_DH_LABEL_FS, FIG5_DP_LABEL_FS

    sizing = style_sizing(style, linewidth=_PRINT_LINEWIDTH if style == "paper" else None)
    apply_plot_style(style, base_fontsize=sizing["fontsize"],
                     linewidth=sizing["linewidth"], n_colors=14, cmap_name="plasma")

    # Rescale the explicit type sizes from the print baseline. ANNOT_FS keeps
    # its ratio to BASE_FS, so dense in-panel annotations stay proportionally
    # smaller than structural text in every style.
    BASE_FS = sizing["fontsize"]
    LINEWIDTH = sizing["linewidth"]
    _SLIDE = style != "paper"
    _fs_scale = BASE_FS / _PRINT_BASE_FS
    ANNOT_FS = round(_PRINT_ANNOT_FS * _fs_scale, 1)

    FIG2_TITLE_FS = BASE_FS + 1
    # Figs. 3 and 4 titles sit on a white panel among curves, so smaller there
    # (2026-09-15 restyle). Fig. 2's sit on a dark heatmap and keep their size.
    FIG3_TITLE_FS = FIG4_TITLE_FS = BASE_FS - 1
    FIG2_EFT_LABEL_FS = FIG3_EFT_LABEL_FS = FIG4_EFT_LABEL_FS = ANNOT_FS
    FIG2_UNITARITY_LABEL_FS = FIG3_UNITARITY_LABEL_FS = \
        FIG4_UNITARITY_LABEL_FS = ANNOT_FS - 1
    FIG5_HALO_MAP_LABEL_FS = FIG5_PERTURB_LABEL_FS = ANNOT_FS
    FIG5_DH_LABEL_FS = FIG5_DP_LABEL_FS = ANNOT_FS
    FIG5_LEGEND_KW = dict(FIG5_LEGEND_KW, fontsize=BASE_FS - 1)

    cols = get_cmap_colors(cmap_name="plasma", n=10, start=0, end=1)
    COL_THIS_WORK = _ink("black")
    COL_GUIDE = _accent("#666666")
    FIG1_HIGGS_COLOR = cols[5]
    FIG1_GRAV_COLOR = cols[2]
    FIG1_THRESHOLD_COLOR = _ink("black")
    FIG1_EXCLUSION_TEXT_COLOR = _accent("#8A6A00")
    FIG2_CONTOUR_COLOURS = {
        # near-black on the page, lifted to a light grey on a dark slide
        1.0:   _accent("#111111"),   # current Fermi-LAT 17yr boundary (solid)
        10.0:  cols[2],
        50.0:  cols[3],
        1.0e3: cols[5],
        1.0e6: cols[8],
    }
    FIG3_TITLE_COLOR = _ink("black")
    FIG3_EFT_LABEL_COLOR = _accent("#21A5B1")
    FIG4_TITLE_COLOR = _ink("black")
    FIG5_HALO_MAP_LABEL_COLOR = _accent("#B04B00")
    FIG5_PERTURB_LABEL_COLOR = FIG1_EXCLUSION_TEXT_COLOR
    FIG5_DH_LABEL_COLOR = _accent("#5A3E00")
    FIG5_DP_LABEL_COLOR = _accent("#5A3E00")

    # Derived palettes: these captured the import-time colours, so rebuild them
    # from the refreshed values rather than leaving them pointing at print ink.
    FIG3_PROFILE_COLORS = {
        "pixelwise_global_rho2": COL_THIS_WORK,
        # Hue-stable across styles by construction -- see the note at the
        # import-time definition for why this is not a plasma sample.
        "pixelwise_global_rho2.5": _accent("#2E6FE8"),
    }
    FIG4_DATASET_COLOURS = {
        "halo": FIG3_PROFILE_COLORS["pixelwise_global_rho2"],
        "igrb": FIG3_PROFILE_COLORS["pixelwise_global_rho2.5"],
        "dsph": COL_FOLLOWUP,
    }

    LEGEND_KW.clear()
    LEGEND_KW.update(theme_legend_kw())


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
    fig, ax = plt.subplots(figsize=_figsize(FIG1_WIDTH, FIG1_HEIGHT))
    col_higgs = FIG1_HIGGS_COLOR
    col_grav  = FIG1_GRAV_COLOR
    col_thres = FIG1_THRESHOLD_COLOR
    col_shade = FIG1_EXCLUSION_FILL_COLOR
    col_shade_text = FIG1_EXCLUSION_TEXT_COLOR

    ax.fill_between(mchi, np.maximum(tau_higgs_cosmo, tau_higgs_gal), 1.0e3,
                    color=col_shade, alpha=0.22)
    ax.loglog(mchi, tau_higgs_gal,   color=col_higgs, ls="-",  lw=2.2)
    ax.loglog(mchi, tau_higgs_cosmo, color=col_higgs, ls="--", lw=2.2)
    ax.loglog(mchi, tau_grav_gal,    color=col_grav,  ls="-",  lw=2.0)
    ax.loglog(mchi, tau_grav_cosmo,  color=col_grav,  ls="--", lw=2.0)
    ax.axhline(2.56e-2, color=col_thres, ls=":", lw=1.4)
    ax.axvline(M_PL, color=_ink("0.45"), ls=(0, (1, 1)), lw=1.1)
    ax.annotate(r"$M_{\rm Pl}$", xy=(M_PL, 8.0e-6), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center",
                rotation=90, fontsize=ANNOT_FS, color=_ink("0.35"))

    ax.set_xlim(1.0, 1.6e19)
    # Top at 1e3, not 1: at 1 the dotted detection floor (2.6e-2) sat on the
    # frame and read as part of it.
    ax.set_ylim(1.0e-70, 1.0e3)
    ax.set_yticks([10.0 ** k for k in range(0, -71, -10)])
    ax.set_xlabel(r"$m_\chi$ [GeV]")
    ax.set_ylabel(r"$\tau_{\rm max}$")
    ax.tick_params(labelsize=BASE_FS - 2, length=5)
    ax.tick_params(which="minor", length=3)
    ax.grid(True, which="both", alpha=0.22)

    handles = [
        Line2D([0], [0], color=col_higgs, lw=2.2, label=r"Higgs portal (saturated)"),
        Line2D([0], [0], color=col_grav,  lw=2.0, label=r"Gravitational (PQG)"),
        Line2D([0], [0], color=_ink("0.15"), lw=1.6, ls="-",  label="Galactic baseline"),
        Line2D([0], [0], color=_ink("0.15"), lw=1.6, ls="--", label="Cosmological baseline"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.35, 0.55),
              borderpad=0.4, handletextpad=0.5, labelspacing=0.35,
              **LEGEND_KW)
    # One line is exactly the panel width in print. On a slide the same string
    # is ~11% wider relative to the axes and runs off the right edge, so break
    # it after "floor" there rather than shrinking it below ANNOT_FS.
    _floor_label = (
        "Fermi-LAT statistical floor\n" r"$\tau_{\rm obs} \approx 2.6\times10^{-2}$"
        if _SLIDE else
        r"Fermi-LAT statistical floor $\tau_{\rm obs} \approx 2.6\times10^{-2}$"
    )
    ax.annotate(
        _floor_label,
        xy=(1.0e11, 2.56e-2),
        xycoords="data",
        # The two-line slide variant grows upward from a "bottom" anchor and
        # would clear the axes top, so it hangs from just under the floor line
        # instead. Both variants end up in the same place on the page.
        xytext=(0.02, 0.915),
        textcoords="axes fraction",
        ha="left",
        va="top",
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
    import make_sensitivity_map as _sm
    from make_sensitivity_map import (
        PANEL_CONFIGS, plot_sensitivity_panel, OUTPUT_DIR as _SM_OUT,
        _unitarity_curve,          # per-operator unitarity Lambda(m_chi)
    )
    _sm.OMEGA_MAX_OVERRIDE = EFT_OMEGA_MAX_GEV
    _sm.BOUNDARY_SUFFIX = EFT_BOUNDARY_SUFFIX
    _sm.DRAW_UNITARITY  = DRAW_UNITARITY

    n = len(FIG2_OPERATORS)
    # 2 cols x 2 rows for the four-operator set. Fig 2 owns its own grid here;
    # make_sensitivity_map.py's ncols governs that module's standalone CLI only
    # and is deliberately left at 3.
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    nslots = ncols * nrows

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=_figsize(FIG2_WIDTH, FIG2_HEIGHT),
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
        if (DRAW_UNITARITY and _lam_unit is not None
                and np.isfinite(_lam_unit[0]) and _lam_unit[0] > 0):
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
            # Raised 2026-08-08: at 2x2 the panel block is shorter than
            # the old 3x2 and the legend was left 0.23 in adrift.
            # Raised again when the title row was dropped, to close the gap
            # it left under the x-axis labels.
            bbox_to_anchor=(0.45, 0.006),
            ncol=3,
            # No title: the legend also carries the CDM reference line, which
            # is not an exposure contour.
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
    _mp.CMB_ELASTIC_COLOR      = COL_COSMOLOGY   # direct-drawn CMB elastic curve (Wilkinson+ 2014)
    _mp.THIS_WORK_COLOR        = COL_THIS_WORK
    _mp.GUIDE_COLOR            = COL_GUIDE
    _mp.EFT_VALID_LABEL_COLOR  = COL_EFT_LABEL
    _mp.OMEGA_MAX_OVERRIDE     = EFT_OMEGA_MAX_GEV
    _mp.BOUNDARY_SUFFIX        = EFT_BOUNDARY_SUFFIX
    _mp.DRAW_UNITARITY         = DRAW_UNITARITY
    # Raw attenuation only. The spectral_reshaping grids are far sparser
    # (~100 boundary points vs 800) and do not exist at all for
    # rayleigh_full_majorana, so including them would draw a curve and a legend
    # entry in two panels of three.
    _mp.DATA_DRIVEN_MODEL_KINDS = ("raw_attenuation",)
    # That module builds its bottom-strip legend from its own module-level
    # LEGEND_KW, fixed at import before any style was active -- so on a dark
    # style it would paint a white panel across the slide. Re-theme it here,
    # alongside the colour constants above.
    _mp.LEGEND_KW.clear()
    _mp.LEGEND_KW.update(theme_legend_kw())
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
                             figsize=_figsize(
                                 FIG3_WIDTH_SLIDE if _SLIDE else FIG3_WIDTH,
                                 FIG3_HEIGHT_SLIDE if _SLIDE else FIG3_HEIGHT))
    axes = np.atleast_1d(axes).ravel()

    # Short two-line titles so nothing overflows the PRD twocolumn panel width.
    # Edit strings here to rename any panel; the '\n' triggers a second line.
    # NOTE: keys must match FIG3_OPERATORS (the anapole panel was retired in
    # favour of the Majorana Rayleigh Even panel on 2026-07-17) and must stay
    # in sync with _FIG4_TITLES in f4_multi_dataset_overlay() -- both figures
    # share the same three operators and should render identical panel titles.
    _FIG3_TITLES = {
        "dipole_magnetic":        "Magnetic Dipole (Dirac)\n(dim-5)",
        "rayleigh_full_majorana": "Rayleigh Full (Majorana)\n(dim-7)",
        "scalar_rayleigh":        "Rayleigh (Scalar)\n(dim-6)",
    }

    all_handles = []
    _fig3_proxies, _fig3_reported = [], {}
    _fig3_band = set()
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

        # --- FIG 3 PANEL TITLE (top-left in-panel, bold -- matches FIG2/FIG4) --
        # In-panel in EVERY style now. This used to move above the axes on
        # slides: the "EFT valid" label is pinned high (0.72) to clear the
        # thermalisation box, so at slide type sizes the two-line title block
        # grew down past it and overprinted "(dim-N)". The slide canvas is now
        # much taller (FIG3_HEIGHT_SLIDE), so the same block spans ~0.14 axes
        # fractions instead of ~0.24 and clears 0.72 with room to spare.
        panel_title = _FIG3_TITLES.get(op, PAPER_PANEL_CONFIGS[op]["title"])
        # Nudged up on slides only: the two-line block sits ~0.14 axes
        # fractions deep there, and at the print anchor (0.94) its second line
        # ended up only just above the "EFT valid" badge at 0.72. Print keeps
        # 0.94 -- that is the published position and it has more clearance,
        # the block being shallower at print type size.
        _title_y = FIG3_TITLE_XY[1] if not _SLIDE else 0.975
        axes[i].text(
            FIG3_TITLE_XY[0], _title_y, panel_title,
            transform=axes[i].transAxes,
            color=FIG3_TITLE_COLOR,
            fontsize=FIG3_TITLE_FS,
            fontweight=FIG3_TITLE_FONTWEIGHT,
            ha=FIG3_TITLE_HALIGN, va=FIG3_TITLE_VALIGN,
            zorder=40,
            # Lifts the title off the curves and the wedge edge it can overlap.
            bbox=dict(boxstyle="round,pad=0.15", fc=_ink("white"), ec="none", alpha=0.75),
        )

        # --- FIG 3 EFT-VALID LABEL (inside the shaded wedge) ---------------
        axes[i].text(
            FIG3_EFT_LABEL_XY[0], FIG3_EFT_LABEL_XY[1], FIG3_EFT_LABEL_TEXT,
            transform=axes[i].transAxes,
            color=FIG3_EFT_LABEL_COLOR,
            fontsize=FIG3_EFT_LABEL_FS,
            fontweight=FIG3_EFT_LABEL_FONTWEIGHT,
            ha="left", va="bottom",
            # Above everything: the wedge fill, the thermalisation hatch and
            # its boundary, every literature curve, and this work's contours.
            # The panel title sits at 40, so 60 keeps this clear of that too.
            zorder=60,
            bbox=dict(boxstyle="round,pad=0.16", fc=_ink("white"),
                      ec="none", alpha=0.70),
        )

        # --- FIG 3 UNITARITY LABEL (along the dotted line) -----------------
        _cfg = PAPER_PANEL_CONFIGS[op]
        _lam_unit, _ = _unitarity_curve(_cfg["operator"],
                                        np.array([FIG3_UNITARITY_LABEL_MCHI]))
        if (DRAW_UNITARITY and _lam_unit is not None
                and np.isfinite(_lam_unit[0]) and _lam_unit[0] > 0):
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

        # --- FIG 3 OVERLAYS (2026-08-07): thermalisation/BBN on every panel,
        # CMB-annihilation and direct-detection dipole bounds on the dipole
        # panel. Drawn after plot_panel so they sit above the wedge fill.
        _proxies, _reported = _fig3_draw_overlays(axes[i], op)
        # --- FIG 3 EXPECTED BANDS (68%), one per halo profile, beneath the lines
        for _profile in FIG3_PROFILES:
            if _draw_expected_band(axes[i], op, FIG3_PROFILE_COLORS[_profile],
                                   zorder=1.9, profile=_profile):
                _fig3_band.add(_profile)
        for _k, _v in _reported.items():
            _fig3_reported.setdefault(op, {})[_k] = _v
        for _p in _proxies:
            if _p.get_label() not in {h.get_label() for h in _fig3_proxies}:
                _fig3_proxies.append(_p)

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
    # 2026-08-07: "Cosmology" is renamed. The panel now carries TWO CMB bounds
    # that sit on opposite sides of this work's contour -- the elastic-scattering
    # limit far below it and the annihilation limit far above -- so a single
    # entry reading "Cosmology" would be ambiguous exactly where a reader looks.
    # --- group recolouring: make the curves match the legend ---------------
    # The bottom-strip legend below is a set of hand-made category proxies in
    # the COL_* palette. plot_panel, however, draws each curve in its own
    # per-experiment colour, resolved by constraints_data.limits._style_for_path
    # against that module's STYLE_HINTS table. Rewriting _mp.STYLE_HINTS above
    # never reached those curves: it rebinds a table on the overlay module,
    # which is a different object from the one _style_for_path reads. The
    # result was a legend that described no curve on the canvas -- the Planck
    # annihilation bound rendered light blue (#71D6FF) under an orange key, and
    # LZ rendered #FFB347, nearer the cosmology key than the direct-detection
    # key it belongs to.
    #
    # Recolour by constraint_type after the panels are drawn rather than by
    # mutating the table, so the result does not depend on when limits.py
    # resolves or caches a style. Line styles are left alone: LZ vs XENONnT and
    # Fermi vs H.E.S.S. stay distinguishable within a group, which is exactly
    # what a grouped legend promises.
    from constraints_data import limits as _limits
    _GROUP_COL = {
        "collider":           COL_COLLIDER,
        "direct_detection":   COL_DIRECT,
        "indirect_detection": COL_INDIRECT,
        "cosmology":          COL_COSMOLOGY,
        "thermal_relic":      COL_THERMAL,
    }
    # label -> constraint_type, from the same table that coloured the curves.
    _label_group, _canon = {}, {}
    for _lbl, _c, _ls, _ctype in _limits.STYLE_HINTS.values():
        _label_group[_lbl.strip().casefold()] = _ctype
        _canon[_lbl.strip().casefold()] = _lbl
    for _ax in axes[:n]:
        for _ln in _ax.get_lines():
            _raw = _ln.get_label()
            _key = _raw.strip().casefold()
            if _key not in _label_group:
                continue
            _grp = _label_group[_key]
            if _grp in _GROUP_COL:
                _ln.set_color(_GROUP_COL[_grp])
            # The thermal relic reaches the axes twice, from the CB0 and CW0
            # files, with labels differing only in case. Canonicalise so a
            # non-compact legend path cannot print it as two entries.
            if not _raw.startswith("_") and _raw != _canon[_key]:
                _ln.set_label(_canon[_key])

    from matplotlib.lines import Line2D as _L2D
    _fig3_categories = [
        _L2D([0], [0], color=COL_COLLIDER, lw=2.0, label="Collider"),
        _L2D([0], [0], color=COL_DIRECT, lw=2.0, label="Direct detection"),
        _L2D([0], [0], color=COL_INDIRECT, lw=2.0, label="Indirect detection"),
        # Dash-dot matches how the elastic curve is actually drawn, so the two
        # CMB entries are separable by style as well as by wording -- they share
        # the cosmology colour by design.
        _L2D([0], [0], color=COL_COSMOLOGY, lw=2.0, ls="-.",
             label="CMB, elastic scattering"),
        # Drawn in the scalar Rayleigh panel since the CB0/CW0 curves were
        # restored, but it had no legend entry at all, so the only green on
        # the canvas was unexplained. It keeps its own colour rather than a
        # group colour: it is a theory line, not an experimental exclusion.
        _L2D([0], [0], color=COL_THERMAL, lw=2.0, ls=(0, (1, 1)),
             label="Thermal relic"),
    ]
    # Explicit column placement. matplotlib fills column-major over
    # ceil(8/3) = 3 rows, so the running order fixes the columns:
    #
    #   col 1: Collider          col 2: CMB, elastic      col 3: Thermalisation
    #          Direct detection         CMB, annihilation        NFW rho^2.5
    #          Indirect detection       NFW rho^2
    #
    # The two "this work" labels are the only ones that wrap to two lines. Left
    # in their natural order they land together in one column and make it half
    # again as tall as the other two, so they are split across columns 2 and 3.
    _tw = list(grouped.get("this_work", []))
    # Swap each banded profile's entry for line-over-band rather than adding
    # entries, which would break the 3x3 column placement below. The handles
    # carry the curve colour set from FIG3_PROFILE_COLORS, so match on that.
    # Suffix kept short: ", 68% expected band" wraps the rho^2.5 label to a
    # third line, and the taller strip rides up over the x-axis labels.
    from matplotlib.colors import to_rgba as _to_rgba
    _band_of = {_to_rgba(FIG3_PROFILE_COLORS[_p]): _p for _p in _fig3_band}
    _tw = [_BandedHandle(_band_swatch(FIG3_PROFILE_COLORS[_band_of[_to_rgba(h.get_color())]]), h,
                         h.get_label() + ", 68% band")
           if hasattr(h, "get_color") and _to_rgba(h.get_color()) in _band_of else h
           for h in _tw]
    _by = {h.get_label(): h for h in _fig3_proxies}
    _fig3_ordered = (
        _fig3_categories[:3]                              # col 1
        + [_fig3_categories[3], _by.get("CMB, annihilation")]
        + _tw[:1]                                         # col 2
        + [_fig3_categories[4], _by.get(FIG3_THERM_LABEL)] + _tw[1:2]   # col 3
    )
    _fig3_ordered = [h for h in _fig3_ordered if h is not None]
    # 3 columns in every style, mirroring the three panels above. The explicit
    # ordering above is built for exactly this: it splits the two long
    # "this work" labels across separate columns so none ends up half again as
    # tall as the others under column-major fill.
    draw_bottom_grouped_legend(
        fig, grouped, compact=True,
        base_fs=BASE_FS,
        y_anchor=FIG3_LEGEND_Y, ncol=3,
        compact_handles=_fig3_ordered,
    )

    for _op, _vals in _fig3_reported.items():
        for _k, _v in sorted(_vals.items()):
            print(f"  [Fig3 {_op}] {_k} = {_v:.4g} GeV (rescaled axis units)")

    # rect bottom reserves the legend strip. The top is the same in both
    # styles again now that the panel titles sit inside the axes.
    # The legend strip is a fixed number of POINTS tall, so the fraction it
    # needs scales inversely with the canvas. The slide canvas is now short
    # (FIG3_HEIGHT_SLIDE) to leave room for a heading above the figure, so
    # 0.233 -- tuned for the tall print canvas -- left the strip riding over
    # the x-axis labels. 0.35 restores the same absolute strip height.
    plt.tight_layout(
        rect=[0.01, 0.233 if not _SLIDE else 0.24, 1, 0.96], w_pad=0.6)
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
    _md.OMEGA_MAX_OVERRIDE = EFT_OMEGA_MAX_GEV
    _md.BOUNDARY_SUFFIX    = EFT_BOUNDARY_SUFFIX
    # The IGRB grids were regenerated on the corrected Ackermann Table 3
    # spectrum and carry their own tag; the halo grids were unaffected.
    _md.DATASET_SUFFIX_OVERRIDE = {"igrb": EFT_IGRB_BOUNDARY_SUFFIX}
    _md.DRAW_UNITARITY     = DRAW_UNITARITY

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
                             figsize=_figsize(FIG4_WIDTH, FIG4_HEIGHT),
                             squeeze=False)
    axes = axes.ravel()

    # Two-line titles matching Fig 3's convention so nothing overflows the panel.
    # NOTE: keys must match FIG3_OPERATORS (the anapole panel was retired in
    # favour of the Majorana Rayleigh Even panel on 2026-07-17; keep this dict
    # in sync with _FIG3_TITLES below or the fallback silently renders the raw
    # operator key instead of a formatted title).
    _FIG4_TITLES = {
        "dipole_magnetic":        "Magnetic Dipole (Dirac)\n(dim-5)",
        "rayleigh_full_majorana": "Rayleigh Full (Majorana)\n(dim-7)",
        "scalar_rayleigh":        "Rayleigh (Scalar)\n(dim-6)",
    }

    combined_handles: dict[str, Line2D] = {}
    _fig4_band = False
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
        # Every curve this panel is about to draw must carry its own dataset's
        # bin geometry. A stale IGRB grid is otherwise indistinguishable by eye:
        # the curve simply stops 0.09 dex early and sits slightly low.
        for _ds in ("igrb", "halo"):
            _assert_boundary_f_bin([r["path"] for r in records if r["dataset"] == _ds], _ds)
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
        # 68% expected band on the halo contour; the curves sit at zorder 5.
        if _draw_expected_band(axes[i], op, _md.DATASET_STYLES["halo"]["color"], zorder=4.6):
            _fig4_band = True
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
            bbox=dict(boxstyle="round,pad=0.15", fc=_ink("white"), ec="none", alpha=0.75),
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
        if _fig4_band:
            _halo_prefix = _md.DATASET_STYLES["halo"]["label"]
            for _j, (_h, _l) in enumerate(zip(handles, labels)):
                if _l.startswith(_halo_prefix):
                    labels[_j] = _l + ", 68% expected band"
                    handles[_j] = _BandedHandle(
                        _band_swatch(_md.DATASET_STYLES["halo"]["color"]), _h, labels[_j])
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
    import make_uv_translation_bounds as _uv
    _uv.BOUNDARY_SUFFIX = EFT_BOUNDARY_SUFFIX
    from make_uv_translation_bounds import (
        dark_higgs_bound,
        lhc_higgs_signal_strength,
        draw_ew_doublet_panel,
        V_EW,
    )

    fig, (ax_h, ax_d) = plt.subplots(
        1, 2, figsize=_figsize(FIG5_WIDTH, _slide_height(FIG5_HEIGHT)))

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
    # Refine hard against the degeneracy at m_h' = m_h. The required coupling
    # diverges there; on a coarse log grid the nearest sample lands at a finite
    # height (~2e13) that is an artefact of where the grid fell, not physics.
    # These extra nodes let the curve actually reach the panel ceiling so the
    # clip below renders it as the divergence it is.
    from make_uv_translation_bounds import M_H_GEV as _MH
    _eps = _MH * np.concatenate([np.geomspace(1e-6, 1e-1, 40),
                                 -np.geomspace(1e-6, 1e-1, 40)])
    m_hp = np.unique(np.concatenate([m_hp, _MH + _eps]))

    # Panel window, fixed up front so every annotation anchor below is
    # unambiguous. The lower limit is 1e-1: nothing in the panel lives below
    # the achievable-coupling ceiling, so the old 1e-2 floor was dead space.
    _H_XMIN, _H_XMAX = 1e-3, 3e3
    # The "LHC (x) pert." label lives in the strip between the axis floor and
    # the gold ceiling at y_lhc_pert ~ 1.2 -- 1.1 decades, or 7% of the axes
    # height, which is less than a line of slide-size type. Drop the floor a
    # decade on slides so the label keeps its own clear band instead of being
    # pushed up into the hatching. Print keeps the published limits.
    _H_YMIN, _H_YMAX = 1e-2, 1e14
    # Contact-limit validity floor, PER BENCHMARK. The heavy-mediator matching
    # of Eq. (dh_to_rayleigh) needs |t| << m_h'^2, and |t|_max ~ 2 m_chi
    # omega_max, so the onset is sqrt(2 m_chi omega_max) and depends on the
    # scatterer mass. It was previously one round number (5.0 GeV) for all
    # three benchmarks. That is now the only benchmark-specific content in the
    # panel: since the corrected Lambda_R is flat in mass the three required-
    # coupling curves coincide, so without per-benchmark onsets the three
    # legend entries would carry no information at all.
    # omega_max is the highest RETAINED bin and is unmoved by the 8-bin
    # selection, since dropping the lowest bin does not touch the top of the band.
    _OMEGA_MAX = 168.9                                   # GeV
    _M_MATCH = {m: float(np.sqrt(2.0 * m * _OMEGA_MAX))  # 1.84 / 5.81 / 18.4 GeV
                for m in FIG5_MCHI_LIST}

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
    # The full-height gold wash was dropped in the 2026-09-15 restyle: laid over
    # the grey contact-limit wash it turned the left third of the panel olive.
    # The hatched brush alone marks the excluded side of the ceiling.
    ax_h.fill_between(_xspan, y_lhc_pert, y_lhc_pert * 14.0,
                      facecolor="none", hatch="////", edgecolor=_gold_edge,
                      lw=0.0, alpha=0.45, zorder=1)

    # --- Heavy-mediator matching validity: m_h' < 5 GeV -------------------
    # Below m_h' ~ 5 GeV the mediator is no longer heavy compared to the
    # photon energies used here, so the contact-limit (heavy-mediator)
    # matching of Eq. (VI.1) [dh_to_rayleigh] breaks down and the halo
    # diagonals in this region are extrapolations. A light wash plus an
    # explicit boundary line, so it does not muddy the exclusion shading.
    # Wash to the SMALLEST onset (every benchmark is an extrapolation there),
    # then one boundary line per benchmark in that benchmark's own line style.
    _m_match_lo = min(_M_MATCH.values())
    ax_h.axvspan(_H_XMIN, _m_match_lo, color=COL_GUIDE, alpha=0.13, zorder=0, lw=0)
    for _k, _m_chi in enumerate(FIG5_MCHI_LIST):
        ax_h.axvline(_M_MATCH[_m_chi], color=COL_GUIDE,
                     ls=_MCHI_STYLES[_k % len(_MCHI_STYLES)],
                     lw=LINEWIDTH * 0.9, zorder=2,
                     label=(f"Contact-limit onset, {_mchi_legend(_m_chi)}"
                            f"  ({_M_MATCH[_m_chi]:.1f} GeV)"))
        # Named on the line itself, in the gap between the gold ceiling and the
        # required-coupling curve, instead of a legend row per benchmark.
        ax_h.text(_M_MATCH[_m_chi] * 1.10, 2.0e3, _mchi_legend(_m_chi),
                  rotation=90, ha="left", va="center", fontsize=ANNOT_FS - 1.5,
                  color=_ink("0.30"), zorder=6)
    ax_h.text(1.4e-3, 2.2e12, "contact-limit extrapolation",
              ha="left", va="center",
              color=_ink("0.30"), fontsize=ANNOT_FS - 1, zorder=6,
              bbox=dict(boxstyle="round,pad=0.15", fc=_ink("white"), ec="none", alpha=0.70))

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

    # The corrected matching makes the required coupling independent of m_chi
    # except through Lambda_R, and the corrected Lambda_R is flat in mass, so
    # the three benchmark curves agree to 3 significant figures. Drawing three
    # coincident lines reads as a rendering fault; we draw ONE and say so, and
    # move the benchmark distinction entirely onto the onset markers above.
    _m0 = FIG5_MCHI_LIST[0]
    _y_chiH = (_m0 / V_EW) * dark_higgs_bound(m_hp, m_chi_GeV=_m0)
    # Clip at the panel ceiling: the peak height at m_h' = m_h is set by how
    # close the grid samples the pole, not by physics -- refine the grid and it
    # rises without limit. Clipping makes it read as the divergence it is
    # rather than as a resonance with a quotable height.
    _y_plot = np.where(np.isfinite(_y_chiH), np.minimum(_y_chiH, _H_YMAX), _H_YMAX)
    _h_this, = ax_h.loglog(
        m_hp, _y_plot, color=_MCHI_COLORS[0],
        lw=LINEWIDTH * 1.3, ls="-", alpha=1.0, zorder=4,
        label=r"This work (all three $m_\chi$ benchmarks)",
    )
    # 68% expected band, carried through the matching: y_chiH ~ Lambda_R^-3, so
    # the LOWER Lambda edge is the UPPER coupling edge. Scaled off the same
    # nearest-mass Lambda_R that dark_higgs_bound() reads, so the band brackets
    # the drawn curve exactly.
    _band_dh = _expected_band("scalar_rayleigh")
    if _band_dh is not None:
        from make_uv_translation_bounds import _load_halo_lambda
        _mg, _lg = _load_halo_lambda("scalar_rayleigh", "scalar")
        _lam_R = float(_lg[int(np.argmin(np.abs(_mg - _m0)))])
        _mb, _lo, _hi = _band_dh
        _k = int(np.argmin(np.abs(_mb - _m0)))
        _lo_R = float(_fig3_rescale_lambda(_lo[_k], "scalar_rayleigh")[0])
        _hi_R = float(_fig3_rescale_lambda(_hi[_k], "scalar_rayleigh")[0])
        with np.errstate(invalid="ignore"):
            ax_h.fill_between(m_hp, np.minimum(_y_plot * (_lam_R / _hi_R) ** 3, _H_YMAX),
                              np.minimum(_y_plot * (_lam_R / _lo_R) ** 3, _H_YMAX),
                              color=_MCHI_COLORS[0], alpha=EXPECTED_BAND_ALPHA, lw=0, zorder=3.9)
        print(f"  [Fig5 left] 68% band on y_chiH at m_chi = {_m0:g} GeV: "
              f"x{(_lam_R / _hi_R) ** 3:.3f} to x{(_lam_R / _lo_R) ** 3:.3f} "
              f"({3 * np.log10(_hi_R / _lo_R):.3f} dex)")

    ax_h.axhline(y_lhc_pert, color=_gold_edge, lw=LINEWIDTH * 1.25, ls="-", zorder=5)
    ax_h.axhline(y_pert, color=_gold_edge, lw=LINEWIDTH * 0.9, ls=":", zorder=5)
    ax_h.text(2.2e3, y_pert * 1.25, r"$\sqrt{4\pi}$", color=FIG5_DH_LABEL_COLOR,
              fontsize=ANNOT_FS - 1, ha="right", va="bottom", zorder=10,
              bbox=dict(boxstyle="round,pad=0.10", fc=_ink("white"), ec="none", alpha=0.85))
    ax_h.text(1.5e-3, 3.0e8, r"Required $y_{\chi H}$ (this work, 68% band)",
              color=_MCHI_COLORS[0], fontsize=ANNOT_FS, fontweight="bold",
              ha="left", va="center", zorder=10,
              bbox=dict(boxstyle="round,pad=0.12", fc=_ink("white"), ec="none", alpha=0.80))

    ax_h.set_xlim(_H_XMIN, _H_XMAX)
    ax_h.set_ylim(_H_YMIN, _H_YMAX)
    ax_h.set_yticks([1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14])
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
    # The rotated "decades" tag below extends further down the axis as type
    # scales up, so on the slide styles this tag has to drop with it or the two
    # boxes touch. The exponent is 0 at print sizing, making this exactly the
    # historical 2.5e2 for the paper.
    # The rotated "decades" tag grows along the axis it is rotated into, so at
    # slide sizes it spans nearly the full 15 decades and there is no height
    # left anywhere in its x-column for this box -- dropping it further just
    # lands it on the LHC label instead. Lift it clear over the top of the tag
    # instead: it still sits inside the shaded excluded region it names.
    _excl_y = (2.5e2 / 10.0 ** (2.0 * (ANNOT_FS / _PRINT_ANNOT_FS - 1.0))
               if not _SLIDE else 3.0e12)
    ax_h.text(2.2e3, _excl_y, "Excluded",
              color=FIG5_DH_LABEL_COLOR, fontsize=FIG5_DH_LABEL_FS,
              fontweight="bold", ha="right", va="center", zorder=10,
              bbox=dict(boxstyle="round,pad=0.18", fc=_ink("white"), ec="none", alpha=0.72))
    # Centred in the clear strip below the gold ceiling -- 0.30 on the print
    # axis, and the geometric middle of the extra decade the slide axis opens
    # up below it.
    _lhc_label_y = 0.12
    ax_h.text(1.4e-3, _lhc_label_y,
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
                  arrowprops=dict(arrowstyle="<->", color=_ink("0.25"), lw=1.1,
                                  shrinkA=0, shrinkB=0), zorder=7)
    # Horizontal, in dex (house style for a logarithmic offset). The rotated
    # "decades" tag it replaces ran along the arrow and crowded it.
    ax_h.text(_x_gap * 1.5, 10.0 ** (0.5 * np.log10(y_lhc_pert * _y_gap_hi) + 0.8),
              fr"$\gtrsim{_decades:.0f}$ dex",
              color=_ink("0.15"), fontsize=ANNOT_FS, fontweight="bold",
              ha="left", va="center", zorder=10,
              bbox=dict(boxstyle="round,pad=0.15", fc=_ink("white"), ec="none", alpha=0.80))
    # No legend: the curve, the onsets and both gold lines are named in-panel.

    # --- Right panel: electroweak-doublet dipole (added 2026-07-17) -------
    # mu = 2/Lambda, so the band's upper Lambda edge is its lower mu edge.
    _band_mu = _expected_band("dipole_magnetic")
    if _band_mu is not None:
        _mb, _lo, _hi = _band_mu
        _band_mu = (_mb, 2.0 / _fig3_rescale_lambda(_hi, "dipole_magnetic"),
                    2.0 / _fig3_rescale_lambda(_lo, "dipole_magnetic"), EXPECTED_BAND_ALPHA)
    draw_ew_doublet_panel(ax_d, base_fs=BASE_FS, linewidth=LINEWIDTH, col_coll=cols[3], col_this=cols[5],
                          mu_band=_band_mu, direct_labels=True,
                          col_cmb="#D9600B",    # the Fig. 3 cosmology orange, darkened for white
                          col_dd="#A88400")     # the Fig. 3 direct-detection yellow, darkened for white

    # 2026-08-07: the right legend grew from three entries to six (CMB +
    # direct-detection overlays), and both legends are single-column so neither
    # can spill sideways into the other.  The extra rows are paid for by making
    # the FIGURE taller (FIG5_HEIGHT 5.0 -> 5.569), never by taking space from
    # the axes -- shrinking the axes collided the left panel's "Excluded" and
    # "decades" annotations and pushed both legends onto the x-labels.
    # 5.569 was solved for, not guessed: tight_layout also absorbs the taller
    # legend, so the axes height is not simply (1 - rect_bottom) * H.  The value
    # reproduces the previous axes box to 2e-4 in (2.7773 x 1.8583 vs
    # 2.7773 x 1.8581), i.e. the left panel is geometrically unchanged.
    # rect bottom stays 0.248 in EVERY style. The slide styles' deeper legend is
    # paid for by _SLIDE_STRETCH on FIG5_HEIGHT above, never by taking it out of
    # the axes -- see the note there, and the 2026-08-07 note: shrinking this
    # axes is exactly what collided "Excluded" with "decades" last time.
    # No bottom legend strip since the 2026-09-15 restyle, so no reserved band.
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.24)

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
        help="Optional additional destination (e.g. Overleaf paper_plots dir).",
    )
    add_style_argument(ap)
    args = ap.parse_args(argv)

    print(f"[make_paper_results_figures] out_dir = {args.out_dir}")
    if args.copy_to is not None:
        print(f"[make_paper_results_figures] copy_to = {args.copy_to}")

    style = normalise_plot_style(args.style, default=STYLE)
    print(f"[make_paper_results_figures] style   = {style}")
    _apply_paper_style(style)

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
