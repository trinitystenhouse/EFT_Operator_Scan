"""
Δχ² / f_required sensitivity maps for γχ scattering EFT operators.

For each operator and (Λ, m_χ) point we compute

    Δχ²(m_χ, Λ) = χ²_DM(m_χ, Λ) − χ²_null

against the Fermi-LAT pixelwise halo posterior (already stored in the boundary
.npz files by make_data_driven_scattering_limits.py). The 90 % CL exclusion
threshold for a 2-parameter scan is Δχ² ≥ 4.61. Under Gaussian errors that
scale as σᵢ ∝ 1/√exposure, the multiplicative exposure boost required to
reach 90 % CL at any (m_χ, Λ) point is

    f_required(m_χ, Λ) = 4.61 / Δχ²_current(m_χ, Λ)

This script reads the chi2_grid stored in each boundary file and produces
two complementary visualisations of f_required across the (Λ, m_χ) plane:

  * a coloured heatmap of log10(f_required) — every point on the plane
    gets a number indicating "exposure multiplier needed to detect at 90 %CL"
    rather than a binary in/out;
  * multi-level contours at f = 1, 10, 50, 1000 ... corresponding to current
    Fermi-LAT 17yr, modest follow-up, CTA-scale, and far-future exposures.

Designed to mirror the panel structure of make_paper_style_operator_overlays.py
so the paper's §III/Fig 2 can be replaced one-for-one.

Usage
-----
    # All 12 operators, ρ² profile, default contour set
    python make_sensitivity_map.py --halo-profile pixelwise_global_rho2

    # Paper-summary 3-panel (Dirac dipole / Majorana anapole / scalar Rayleigh)
    python make_sensitivity_map.py --paper-summary --halo-profile pixelwise_global_rho2

    # Custom contour levels (exposure multipliers)
    python make_sensitivity_map.py --paper-summary --contours 1 10 50 1000 1e6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from helpers.trinity_plotting import save_figure, set_paper_style  # noqa: E402

BOUNDARY_DIR = _HERE / "constraint_boundaries"
OUTPUT_DIR = _HERE / "plots"

# Trinity's standard accent palette — mirrors PANEL_CONFIGS / LEGEND_GROUPS in
# make_paper_style_operator_overlays.py so the two figures read as one set.
COL_THIS_WORK = "#111111"   # solid black — current 90% CL boundary
COL_FOLLOWUP  = "#0B7285"   # deep teal   — modest exposure follow-up
COL_INDIRECT  = "#9B6DFF"   # purple      — indirect-detection / CTA-scale
COL_COLLIDER  = "#FF7AC6"   # pink        — collider / ambitious future
COL_COSMOLOGY = "#71D6FF"   # cyan        — cosmology / parametric limit
COL_VALIDITY  = "#666666"   # grey        — EFT / unitarity guides
COL_EFT_LABEL = "#008B8B"   # teal accent — "EFT valid" annotation

# Mirrors the operator metadata in make_paper_style_operator_overlays.py so
# the panel selection matches Fig 2 of the paper exactly.
PANEL_CONFIGS = {
    "dipole_magnetic":         {"title": "Magnetic Dipole (Dirac)",                       "dm_type": "fermionic", "operator": "dipole_magnetic",  "majorana": False},
    "dipole_electric":         {"title": "Electric Dipole (Dirac)",                       "dm_type": "fermionic", "operator": "dipole_electric",  "majorana": False},
    "charge_radius":           {"title": "Charge Radius (Dirac)",                         "dm_type": "fermionic", "operator": "charge_radius",    "majorana": False},
    "anapole":                 {"title": "Anapole (Dirac)",                               "dm_type": "fermionic", "operator": "anapole",          "majorana": False},
    "anapole_majorana":        {"title": "Anapole (Majorana)",                            "dm_type": "fermionic", "operator": "anapole",          "majorana": True},
    "rayleigh_even":           {"title": "Rayleigh Even (Dirac)",                         "dm_type": "fermionic", "operator": "rayleigh_even",    "majorana": False},
    "rayleigh_odd":            {"title": "Rayleigh Odd (Dirac)",                          "dm_type": "fermionic", "operator": "rayleigh_odd",     "majorana": False},
    "rayleigh_full":           {"title": "Rayleigh (Dirac)",                              "dm_type": "fermionic", "operator": "rayleigh_full",    "majorana": False},
    "rayleigh_even_majorana":  {"title": "Rayleigh Even (Majorana)",                      "dm_type": "fermionic", "operator": "rayleigh_even",    "majorana": True},
    "rayleigh_odd_majorana":   {"title": "Rayleigh Odd (Majorana)",                       "dm_type": "fermionic", "operator": "rayleigh_odd",     "majorana": True},
    "rayleigh_full_majorana":  {"title": "Rayleigh (Majorana)",                           "dm_type": "fermionic", "operator": "rayleigh_full",    "majorana": True},
    "scalar_rayleigh":         {"title": "Scalar Rayleigh",                               "dm_type": "scalar",    "operator": "scalar_rayleigh",  "majorana": False},
}

DEFAULT_OPERATORS = [
    "dipole_magnetic", "dipole_electric", "charge_radius", "anapole",
    "anapole_majorana", "rayleigh_full", "scalar_rayleigh",
]

PAPER_SUMMARY_OPERATORS = ["dipole_magnetic", "anapole_majorana", "scalar_rayleigh"]
PAPER_SUMMARY_TITLES = {
    "dipole_magnetic":  "Dirac Magnetic Dipole\n(dim-5)",
    # For Majorana χ the vector current χ̄γ^μχ vanishes, so the "vector"
    # charge-radius operator does not exist. The surviving dim-6 fermion
    # operator is the axial anapole; older DM-EFT literature calls this the
    # "axial charge radius". Same operator, two names — the caption states
    # this identity once so the panel title can stay short.
    "anapole_majorana": "Majorana Anapole\n(dim-6)",
    "scalar_rayleigh":  "Scalar Rayleigh\n(dim-7)",
}

# Default exposure-multiplier contour levels.
#   f = 1     →  currently excluded at 90% CL by Fermi-LAT 17yr
#   f = 10    →  ~10× more exposure (additional ~150yr Fermi-LAT, or modest CTA)
#   f = 50    →  CTA-scale (~50× effective area improvement at TeV vs Fermi-LAT)
#   f = 1e3   →  far-future (next-gen MeV mission with substantially better statistics)
#   f = 1e6   →  parametrically out of reach
DEFAULT_FCONTOURS = (1.0, 10.0, 50.0, 1.0e3, 1.0e6)
FCONTOUR_LABELS = {
    1.0:   r"Fermi-LAT 17yr",
    10.0:  r"$\times 10$ exposure",
    50.0:  r"CTA-scale ($\times 50$)",
    1.0e3: r"$\times 10^{3}$ exposure",
    1.0e6: r"$\times 10^{6}$ exposure",
}
FCONTOUR_COLOURS = {
    1.0:   COL_THIS_WORK,
    10.0:  COL_FOLLOWUP,
    50.0:  COL_INDIRECT,
    1.0e3: COL_COLLIDER,
    1.0e6: COL_COSMOLOGY,
}


def _resolve_boundary_path(halo_profile: str, model_kind: str, dm_type: str,
                           operator: str, majorana: bool,
                           source_tag: str = "halo") -> Path:
    majorana_suffix = "_majorana" if majorana else ""
    name = (
        f"mcmc_{halo_profile}_{source_tag}_{model_kind}_"
        f"{dm_type}_{operator}{majorana_suffix}_90cl.npz"
    )
    return BOUNDARY_DIR / name


def _paper_y_axis_values(lambda_GeV, dm_type, operator):
    """Replicate _paper_y_axis_values from core.attenuation_eft for axis labelling.

    The boundary files already store lambda_plot_GeV; the chi2_grid axis is the
    raw lambda_GeV. Operators with non-unit Wilson coefficient power are mapped
    onto Λ/C^(1/n) so the cross sections of different operator dimensions can
    be plotted on the same y-axis.
    """
    lam = np.asarray(lambda_GeV, dtype=float)
    # Dim-5 dipoles: Λ → Λ (n=1)
    if operator in ("dipole_magnetic", "dipole_electric"):
        return lam
    # Dim-6 anapole / charge radius: Λ² → (Λ²/C)^(1/2) — but C=1, so identity
    if operator in ("anapole", "charge_radius"):
        return lam
    # Dim-7 fermionic Rayleigh: Λ³ → (Λ³/C)^(1/3) — identity at C=1
    # Dim-6 scalar Rayleigh: Λ² → (Λ²/C)^(1/2) — identity at C=1
    return lam


def _f_required(chi2_grid: np.ndarray, delta_chi2_target: float = 4.61) -> np.ndarray:
    """Return f_required(m_χ, Λ) = Δχ²_target / Δχ²_current.

    Δχ² is computed relative to the best-fit point of the saved grid
    (chi2 − chi2_min). Where Δχ² is zero or non-finite f_required is +inf.
    """
    chi2 = np.asarray(chi2_grid, dtype=float)
    finite = np.isfinite(chi2)
    if not finite.any():
        return np.full_like(chi2, np.inf)
    chi2_min = np.nanmin(chi2[finite])
    delta = chi2 - chi2_min
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(delta > 0.0, float(delta_chi2_target) / delta, np.inf)
    f[~np.isfinite(chi2)] = np.nan
    return f


def _eft_validity_curve(omega_max: float, eft_factor: float, mchi_arr: np.ndarray) -> np.ndarray:
    mchi = np.asarray(mchi_arr, dtype=float)
    s_max = mchi**2 + 2.0 * mchi * omega_max
    denom = 1.0 + 2.0 * omega_max / np.where(mchi > 0, mchi, np.nan)
    t_max = 4.0 * omega_max**2 / denom
    q2_max = np.maximum(s_max, t_max)
    return np.sqrt(eft_factor * q2_max)


def _unitarity_curve(operator: str, mchi_arr: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    mchi = np.asarray(mchi_arr, dtype=float)
    x = np.where(mchi > 0, mchi, np.nan)
    if operator in ("dipole_magnetic", "dipole_electric"):
        return np.sqrt(16.0 * np.pi * x), "Unitarity (dipole)"
    if operator in ("anapole", "charge_radius"):
        return (16.0 * np.pi * x**2) ** 0.25, "Unitarity (dim-6)"
    if "rayleigh" in operator:
        return (128.0 * np.pi**2 * x**2) ** (1.0 / 6.0), "Unitarity (Rayleigh)"
    return None, None


def plot_sensitivity_panel(ax, operator_key: str, halo_profile: str,
                           *, model_kind: str = "raw_attenuation",
                           contour_levels=DEFAULT_FCONTOURS,
                           contour_colours: dict | None = None,
                           contour_labels: dict | None = None,
                           title_override=None,
                           title_fontsize=15,
                           axis_label_fontsize=13,
                           tick_labelsize=11,
                           annotate_fontsize=12,
                           heatmap_cmap: str = "plasma_r",
                           fheatmap_min: float = 1.0,
                           fheatmap_max: float = 1.0e12,
                           annotate_validity: bool = True,
                           validity_fill_color: str = "cyan",
                           validity_line_color: str | None = None,
                           unitarity_line_color: str | None = None):
    """Draw a single f_required(m_chi, Lambda) sensitivity panel.

    ``contour_colours`` maps exposure levels (float) to matplotlib colours.
    When None the module-level FCONTOUR_COLOURS is used. Pass a dict from the
    master paper-figures script to fully control the palette.
    ``contour_labels`` is analogous for the legend labels.
    """
    _colours = FCONTOUR_COLOURS if contour_colours is None else {**FCONTOUR_COLOURS, **contour_colours}
    _labels  = FCONTOUR_LABELS  if contour_labels  is None else {**FCONTOUR_LABELS,  **contour_labels}
    cfg = PANEL_CONFIGS[operator_key]
    path = _resolve_boundary_path(
        halo_profile=halo_profile,
        model_kind=model_kind,
        dm_type=cfg["dm_type"],
        operator=cfg["operator"],
        majorana=cfg["majorana"],
    )
    if not path.exists():
        ax.text(0.5, 0.5, f"Missing\n{path.name}", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return [], None

    data = np.load(path, allow_pickle=True)
    chi2 = np.asarray(data["chi2_grid"], dtype=float)
    mchi_axis = np.asarray(data["grid_mchi_GeV"], dtype=float)
    lambda_axis = np.asarray(data["grid_lambda_GeV"], dtype=float)
    eft_mask = np.asarray(data["eft_valid_mask"], dtype=bool)
    omega_max = float(data["omega_max_for_validity"])
    eft_factor = float(data["eft_kinematic_factor"])
    delta_chi2_target = float(data["delta_chi2_threshold"])

    # f_required on the saved (m_χ, Λ) grid axes
    f_req = _f_required(chi2, delta_chi2_target=delta_chi2_target)

    # Convert Λ axis to Λ/C^(1/n) (no-op at default Wilson coefficients = 1)
    lambda_plot_axis = _paper_y_axis_values(lambda_axis, cfg["dm_type"], cfg["operator"])

    # pcolormesh expects monotonic axes; the saved grids are log-spaced so this is fine
    M, L = np.meshgrid(mchi_axis, lambda_plot_axis, indexing="ij")

    f_clip = np.clip(f_req, fheatmap_min, fheatmap_max)
    heat = ax.pcolormesh(
        mchi_axis, lambda_plot_axis, f_clip.T,
        shading="auto",
        cmap=heatmap_cmap,
        norm=LogNorm(vmin=fheatmap_min, vmax=fheatmap_max),
        rasterized=True,
        zorder=1,
    )

    # Overlay contours at each requested exposure multiplier.
    # Use a Line2D proxy artist for the figure legend (contour collections
    # don't always survive the figure.legend() round-trip).
    from matplotlib.lines import Line2D
    contour_handles = []
    for f_level in contour_levels:
        try:
            colour = _colours.get(float(f_level), "white")
            lw = 3.0 if float(f_level) == 1.0 else 2.2
            ls = "-" if float(f_level) == 1.0 else "--"
            ax.contour(
                mchi_axis, lambda_plot_axis, f_req.T,
                levels=[float(f_level)],
                colors=[colour],
                linewidths=lw,
                linestyles=ls,
                zorder=5,
            )
            lbl = _labels.get(float(f_level), f"$f={f_level:g}$")
            contour_handles.append(Line2D([0], [0], color=colour, lw=lw, ls=ls, label=lbl))
        except Exception:
            continue

    # EFT-valid wedge bounded by kinematic-validity and unitarity guides
    xgrid = np.logspace(np.log10(mchi_axis.min()), np.log10(mchi_axis.max()), 400)
    lam_kin = _eft_validity_curve(omega_max, eft_factor, xgrid)
    lam_unit, unit_label = _unitarity_curve(cfg["operator"], xgrid)
    guide_color = validity_line_color or COL_VALIDITY
    unitarity_color = unitarity_line_color or guide_color
    ax.plot(xgrid, lam_kin, color=guide_color, lw=1.8, ls="--",
            label="EFT kinematic validity", zorder=4)
    if lam_unit is not None:
        ax.plot(xgrid, lam_unit, color=unitarity_color, lw=1.4, ls=":",
                label=unit_label, zorder=4)
        floor = np.maximum(lam_kin, lam_unit)
    else:
        floor = lam_kin
    ax.fill_between(
        xgrid, floor, np.full_like(floor, lambda_plot_axis.max()),
        color=validity_fill_color, alpha=0.3, zorder=2,
    )
    if annotate_validity:
        ax.text(0.04, 0.94, "EFT valid", color=COL_EFT_LABEL,
                fontsize=annotate_fontsize, ha="left", va="top",
                transform=ax.transAxes,
                fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mchi_axis.min(), mchi_axis.max())
    ax.set_ylim(lambda_plot_axis.min(), lambda_plot_axis.max())
    ax.set_title(title_override or cfg["title"], fontsize=title_fontsize)
    ax.set_xlabel(r"$m_{\chi}$ [GeV]", fontsize=axis_label_fontsize)
    ax.set_ylabel(r"$\Lambda/C^{1/n}$ [GeV]", fontsize=axis_label_fontsize)
    ax.grid(True, which="both", alpha=0.3)
    if tick_labelsize is not None:
        ax.tick_params(axis="both", which="both", labelsize=tick_labelsize)

    return contour_handles, heat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--halo-profile", default="pixelwise_global_rho2",
                        help="Halo profile tag in the boundary filename. Default: pixelwise_global_rho2.")
    parser.add_argument("--model-kind", default="raw_attenuation",
                        choices=("raw_attenuation", "spectral_reshaping"),
                        help="Which observable to read from constraint_boundaries/. Default: raw_attenuation.")
    parser.add_argument("--operators", nargs="+", default=None,
                        choices=sorted(PANEL_CONFIGS.keys()),
                        help="Operators to plot. Default: all 7 in DEFAULT_OPERATORS, or the 3-panel summary set under --paper-summary.")
    parser.add_argument("--paper-summary", action="store_true",
                        help="Compact 3-panel LaTeX-paper version mirroring the existing operator-overlay summary.")
    parser.add_argument("--contours", nargs="+", type=float, default=list(DEFAULT_FCONTOURS),
                        help="Exposure-multiplier levels to draw as contours. Default: 1 10 50 1000 1e6.")
    parser.add_argument("--fmax", type=float, default=1.0e12,
                        help="Upper colour clip for log10(f_required). Default: 1e12 (anything beyond is parametrically out of reach).")
    parser.add_argument("--fmin", type=float, default=1.0,
                        help="Lower colour clip for log10(f_required). Default: 1 (currently detectable).")
    parser.add_argument("--cmap", default="plasma_r",
                        help="Matplotlib colormap. Default: plasma_r (dark = currently testable, light = out of reach).")
    parser.add_argument("--outfile", default=None,
                        help="Output basename inside Totani_Scattering/plots/. Default depends on --paper-summary.")
    parser.add_argument("--style", default=None,
                        help="trinity_plotting style: paper / conference / conference_light. Default: paper.")
    # Print-geometry overrides (used by make_paper_results_figures.py to force
    # PRD revtex4 column widths). If None, fall back to the internal defaults.
    parser.add_argument("--fig-width", type=float, default=None,
                        help="Figure width [inches]. Overrides internal figsize.")
    parser.add_argument("--fig-height", type=float, default=None,
                        help="Figure height [inches]. Overrides internal figsize.")
    parser.add_argument("--base-fontsize", type=float, default=None,
                        help="Base font size passed to set_paper_style. Overrides internal default (13).")
    parser.add_argument("--linewidth", type=float, default=None,
                        help="Line width passed to set_paper_style. Overrides internal default (2.0).")
    args = parser.parse_args()

    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style

    if args.operators is None:
        args.operators = PAPER_SUMMARY_OPERATORS if args.paper_summary else DEFAULT_OPERATORS
    if args.outfile is None:
        suffix = "_paper_summary" if args.paper_summary else ""
        args.outfile = f"sensitivity_map_{args.halo_profile}_{args.model_kind}{suffix}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Paper style: fontsize/linewidth accept CLI overrides for PRD print geometry
    base_fs   = args.base_fontsize if args.base_fontsize is not None else 13
    linewidth = args.linewidth     if args.linewidth     is not None else 2.0
    set_paper_style(base_fontsize=base_fs, linewidth=linewidth, n_colors=14, cmap_name="plasma")

    n = len(args.operators)
    # 3 columns for both paper_summary (3 ops → 3×1) and full grid (11 ops → 3×4)
    # so both figures share a consistent column width in the final PDF.
    ncols = 3
    nrows = 1 if args.paper_summary else int(np.ceil(n / ncols))
    nslots = ncols * nrows

    # Figure geometry: CLI overrides > paper-summary default > internal-scale
    if args.fig_width is not None and args.fig_height is not None:
        figsize = (args.fig_width, args.fig_height)
    elif args.paper_summary:
        figsize = (5.0 * ncols + 1.6, 4.4)          # generous default for standalone use
    else:
        figsize = (6.0 * ncols + 1.6, 4.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    # Per-panel font sizes: scale with base_fs so PRD-tight figures don't get
    # oversized titles. Both paper_summary (3-panel Fig 2) and the full 3-col
    # grid (Fig 3) use the SAME sizes so the two figures read as one set.
    title_fs = base_fs + 1
    axlbl_fs = base_fs
    tick_fs  = base_fs - 2
    anno_fs  = base_fs - 1

    contour_handles_acc = []
    last_heat = None
    plotted_axes = []
    for i, op in enumerate(args.operators):
        handles, heat = plot_sensitivity_panel(
            axes[i], op, args.halo_profile,
            model_kind=args.model_kind,
            contour_levels=args.contours,
            title_override=PAPER_SUMMARY_TITLES.get(op) if args.paper_summary else None,
            title_fontsize=title_fs,
            axis_label_fontsize=axlbl_fs,
            tick_labelsize=tick_fs,
            annotate_fontsize=anno_fs,
            heatmap_cmap=args.cmap,
            fheatmap_min=args.fmin,
            fheatmap_max=args.fmax,
        )
        contour_handles_acc.extend(handles)
        if heat is not None:
            last_heat = heat
        plotted_axes.append((i, axes[i]))

    # Axis labels only on the bottom row (x) and leftmost column (y). The
    # last panel in each column is defined over the plotted operators, but
    # if some cell in the bottom row is an empty legend slot, we promote the
    # nearest occupied panel above it to "bottom" so it keeps its xlabel.
    last_panel_in_col = {}
    first_panel_in_row = {}
    for i, ax in plotted_axes:
        row, col = divmod(i, ncols)
        last_panel_in_col[col] = i
        first_panel_in_row.setdefault(row, i)
    # If a bottom-row slot is empty (i.e. no plotted panel with row == nrows-1
    # exists in that column), the last_panel_in_col for that column is already
    # the highest-index plotted panel — which is what we want.
    for i, ax in plotted_axes:
        row, col = divmod(i, ncols)
        if i != last_panel_in_col[col]:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        if i != first_panel_in_row[row]:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    legend_ax = None
    for j in range(n, len(axes)):
        axes[j].set_axis_off()
    if (not args.paper_summary) and n < nslots:
        legend_ax = axes[-1]

    # Shared colorbar on the right side of the figure — same sizing as titles
    # so both figures print with a consistent visual weight.
    cbar_label_fs = base_fs + 1
    cbar_tick_fs  = base_fs - 1
    if last_heat is not None:
        # Colorbar sits in a fixed-width strip at the right edge; the rotated
        # colorbar label needs ~0.55" of horizontal space to its right at
        # PRD paper widths. Move slightly leftward so the label doesn't get
        # clipped by the figure edge in bbox_inches='tight' saves.
        cax = fig.add_axes([0.82, 0.22, 0.014, 0.60] if args.paper_summary
                           else [0.87, 0.14, 0.011, 0.74])
        cb = fig.colorbar(last_heat, cax=cax)
        # Short colorbar label so the rotated text fits inside the figure edge
        # without being clipped in bbox_inches='tight' PDF saves.
        cb.set_label(r"$f_{\rm required} = 4.61/\Delta\chi^{2}$",
                     fontsize=cbar_label_fs)
        cb.ax.tick_params(labelsize=cbar_tick_fs)

    # Single shared legend with the contour labels. The full sensitivity grid
    # uses the last empty panel slot; the compact summary keeps the bottom
    # legend. Font size follows base_fs so it matches the tick / label sizing.
    legend_fs = base_fs
    # Shared legend styling (Totani make_paper_results_figures convention).
    LEGEND_KW = dict(frameon=True, framealpha=0.6, facecolor="white", edgecolor="0.7")
    leg = None
    if contour_handles_acc:
        seen = {}
        for h in contour_handles_acc:
            lbl = h.get_label()
            if lbl not in seen:
                seen[lbl] = h
        handles = list(seen.values())
        labels = list(seen.keys())
        if legend_ax is not None:
            leg = legend_ax.legend(
                handles, labels,
                loc="center",
                title="Exposure contours",
                title_fontsize=legend_fs,
                fontsize=legend_fs,
                handlelength=2.6,
                handletextpad=0.6,
                labelspacing=0.9,
                borderaxespad=0.0,
                borderpad=0.4,
                **LEGEND_KW,
            )
        else:
            leg = fig.legend(
                handles, labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(seen), 5),
                fontsize=legend_fs,
                handlelength=2.6,
                handletextpad=0.5,
                columnspacing=1.6,
                borderpad=0.4,
                **LEGEND_KW,
            )
        if leg is not None:
            frame = leg.get_frame()
            frame.set_facecolor(LEGEND_KW["facecolor"])
            frame.set_edgecolor(LEGEND_KW["edgecolor"])
            frame.set_alpha(LEGEND_KW["framealpha"])

    if args.paper_summary:
        # Right margin 0.78 leaves room for the colorbar strip (x=0.82–0.834)
        # and the rotated colorbar label to its right. wspace=0.14 gives each
        # panel title breathing room without spreading the axes out.
        fig.tight_layout(rect=[0.0, 0.16, 0.78, 0.94])
        fig.subplots_adjust(wspace=0.14, hspace=0.10)
    else:
        fig.tight_layout(rect=[0.0, 0.03, 0.85, 0.97])
        fig.subplots_adjust(wspace=0.14, hspace=0.12)

    save_figure(fig, str(OUTPUT_DIR / args.outfile))
    # Also save a PDF for direct LaTeX inclusion.
    fig.savefig(
        str(OUTPUT_DIR / f"{args.outfile}.pdf"),
        bbox_inches="tight",
        transparent=bool(plt.rcParams.get("savefig.transparent", False)),
        facecolor=plt.rcParams.get("savefig.facecolor", "auto"),
        edgecolor=plt.rcParams.get("savefig.edgecolor", "auto"),
    )
    plt.close(fig)
    print(f"Saved sensitivity map: {OUTPUT_DIR / args.outfile}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
