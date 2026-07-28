#!/usr/bin/env python3
r"""
make_uv_complete_theory_limits.py
=================================

UV-complete version of the Fig. 2 EFT-validity summary.

The original ``make_combined_fermion_scalar_tau_grid.py`` works in the
``(Lambda, m_chi)`` EFT plane.  This script keeps the same operator-panel layout
and paper plotting style, but transforms the EFT validity boundary to the
``(m_med, m_chi)`` plane at fixed UV couplings.

Matching convention
-------------------
For an s-channel or t-channel mediator we use

    Lambda = m_med / sqrt(g_SM g_chi)

so

    m_med = Lambda * sqrt(g_SM g_chi).

If your preferred convention is ``Lambda = m_med / (g_SM g_chi)``, run with
``--matching product``.

CSV overlays
------------
Collider/relic contours are read with pandas.  Files can use column names such
as ``m_med_GeV,m_chi_GeV`` or common variants like ``mmed,mchi``.  Collider
contours are shaded on the excluded side, controlled by ``:above``, ``:below``,
``:left``, or ``:right`` in the command-line spec.

Examples
--------
    python make_uv_complete_theory_limits.py \
        --collider dijet_limits.csv:Dijet:above \
        --collider MET_plus_X_limits.csv:'$E_T^{miss}+X$':above \
        --thermal thermal_relic_contour.csv \
        --g-sm 0.25 --g-chi 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from helpers.fermi_plotting import (
    CONF_ANNOT_FS,
    CONF_LABEL_FS,
    CONF_LEGEND_FS,
    CONF_TICK_FS,
    CONF_TITLE_FS,
    operator_title,
)
from helpers.trinity_plotting import save_figure, set_paper_style, set_plot_style


OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_DUMMY_DIR = Path(__file__).resolve().parent / "constraint_generation"

DEFAULT_DUMMY_COLLIDERS = (
    ("dijet_limits.csv", "Dijet", "left"),
    ("MET_plus_X_limits.csv", r"$E_T^{miss}+X$", "above"),
    ("dilepton_limits.csv", "Dilepton", "left"),
)
DEFAULT_DUMMY_THERMAL = "thermal_relic_contour.csv"

TOP_ROW_PANELS = [
    ("rayleigh_full", "dirac", "Rayleigh Full"),
    ("charge_radius", "dirac", "Charge Radius"),
    ("anapole", "dirac", "Anapole"),
    ("dipole_electric", "dirac", "EL Dipole"),
    ("dipole_magnetic", "dirac", "MA Dipole"),
]

BOTTOM_ROW_PANELS = [
    ("rayleigh_full", "majorana", "Rayleigh Full"),
    ("charge_radius", "majorana", "Charge Radius"),
    ("anapole", "majorana", "Anapole"),
    ("rayleigh_scalar", "scalar", "Scalar Rayleigh"),
    (None, None, None),
]

PAPER_COLORS = {
    "valid_fill": "#71D6FF",
    "valid_line": "#0096C7",
    "threshold": "#8B93A4",
    "thermal": "#7DDC84",
    "dijet": "#FF7AC6",
    "met": "#9B6DFF",
    "dilepton": "#F2D53C",
    "other": "#B18CFF",
}


def unitarity_lambda_curve(operator, mchi_grid):
    """Same simple unitarity guides used by make_combined_fermion_scalar_tau_grid.py."""
    mchi_grid = np.asarray(mchi_grid, dtype=float)
    xgrid = np.where(mchi_grid > 0.0, mchi_grid, np.nan)
    if operator in ("dipole_magnetic", "dipole_electric"):
        return np.sqrt(16.0 * np.pi * xgrid), "Unitarity (dipole)"
    if operator in ("charge_radius", "anapole"):
        return (16.0 * np.pi * xgrid**2) ** 0.25, "Unitarity (dim-6)"
    if "rayleigh" in str(operator):
        return (128.0 * np.pi**2 * xgrid**2) ** (1.0 / 6.0), "Unitarity (Rayleigh)"
    return None, None


def coupling_factor(g_sm: float, g_chi: float, matching: str) -> float:
    """Return f such that m_med = f * Lambda."""
    g_sm = float(g_sm)
    g_chi = float(g_chi)
    if g_sm <= 0.0 or g_chi <= 0.0:
        raise ValueError("g_SM and g_chi must be positive.")
    if matching == "sqrt":
        return float(np.sqrt(g_sm * g_chi))
    if matching == "product":
        return float(g_sm * g_chi)
    raise ValueError("matching must be 'sqrt' or 'product'.")


def lambda_to_mmed(lambda_gev, *, g_sm: float, g_chi: float, matching: str = "sqrt"):
    """Map EFT cutoff Lambda [GeV] to mediator mass m_med [GeV]."""
    return np.asarray(lambda_gev, dtype=float) * coupling_factor(g_sm, g_chi, matching)


def mmed_to_lambda(mmed_gev, *, g_sm: float, g_chi: float, matching: str = "sqrt"):
    """Map mediator mass m_med [GeV] back to EFT cutoff Lambda [GeV]."""
    return np.asarray(mmed_gev, dtype=float) / coupling_factor(g_sm, g_chi, matching)


def validity_mmed_curve(operator, mchi_grid, *, g_sm: float, g_chi: float, matching: str):
    lambda_min, label = unitarity_lambda_curve(operator, mchi_grid)
    if lambda_min is None:
        return None, None
    return lambda_to_mmed(lambda_min, g_sm=g_sm, g_chi=g_chi, matching=matching), label


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "m_med_GeV": ("m_med_GeV", "mmed_GeV", "m_med", "mmed", "Mmed", "M_med", "mediator_mass"),
        "m_chi_GeV": ("m_chi_GeV", "mchi_GeV", "m_chi", "mchi", "Mchi", "M_chi", "dm_mass"),
    }
    rename = {}
    lower_lookup = {str(c).lower(): c for c in df.columns}
    for canonical, names in aliases.items():
        for name in names:
            key = str(name).lower()
            if key in lower_lookup:
                rename[lower_lookup[key]] = canonical
                break
    df = df.rename(columns=rename)
    missing = {"m_med_GeV", "m_chi_GeV"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required mass columns: {sorted(missing)}")
    return df


def load_contour_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column ``(m_med, m_chi)`` contour using pandas."""
    df = pd.read_csv(path, comment="#")
    df = _normalise_columns(df)
    x = pd.to_numeric(df["m_med_GeV"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["m_chi_GeV"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"{path} does not contain at least two finite positive contour points.")
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    return x[order], y[order]


def parse_collider_spec(spec: str) -> dict:
    """Parse ``path[:label[:side[:color]]]``."""
    parts = [p.strip() for p in str(spec).split(":")]
    if not parts or not parts[0]:
        raise ValueError("--collider entries must start with a CSV path.")
    path = Path(parts[0]).expanduser()
    label = parts[1] if len(parts) > 1 and parts[1] else path.stem.replace("_", " ")
    side = parts[2].lower() if len(parts) > 2 and parts[2] else "above"
    color = parts[3] if len(parts) > 3 and parts[3] else None
    if side not in {"above", "below", "left", "right"}:
        raise ValueError(f"Unknown exclusion side {side!r}; use above, below, left, or right.")
    return {"path": path, "label": label, "side": side, "color": color}


def color_for_label(label: str, fallback_index: int = 0) -> str:
    lower = str(label).lower()
    if "dijet" in lower:
        return PAPER_COLORS["dijet"]
    if "met" in lower or "miss" in lower:
        return PAPER_COLORS["met"]
    if "dilepton" in lower or "lepton" in lower:
        return PAPER_COLORS["dilepton"]
    palette = [PAPER_COLORS["other"], "#FF9AD5", "#C7A6FF", "#FFE68A"]
    return palette[int(fallback_index) % len(palette)]


def shade_excluded(ax, x, y, *, side: str, color: str, label: str):
    """Shade the side of a collider contour that is experimentally excluded."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if side == "above":
        ax.fill_between(x, y, ymax, color=color, alpha=0.16, zorder=2, label=label)
    elif side == "below":
        ax.fill_between(x, ymin, y, color=color, alpha=0.16, zorder=2, label=label)
    elif side == "left":
        ax.fill_betweenx(y, xmin, x, color=color, alpha=0.16, zorder=2, label=label)
    elif side == "right":
        ax.fill_betweenx(y, x, xmax, color=color, alpha=0.16, zorder=2, label=label)


def apply_limits(ax, *, mmed_grid, mchi_grid, extra_curves):
    xs = [mmed_grid]
    ys = [mchi_grid]
    for x, y in extra_curves:
        xs.append(x)
        ys.append(y)
    x = np.concatenate([np.asarray(v, dtype=float)[np.isfinite(v) & (np.asarray(v, dtype=float) > 0)] for v in xs])
    y = np.concatenate([np.asarray(v, dtype=float)[np.isfinite(v) & (np.asarray(v, dtype=float) > 0)] for v in ys])
    ax.set_xlim(10.0 ** (np.log10(np.nanmin(x)) - 0.08), 10.0 ** (np.log10(np.nanmax(x)) + 0.08))
    ax.set_ylim(10.0 ** (np.log10(np.nanmin(y)) - 0.08), 10.0 ** (np.log10(np.nanmax(y)) + 0.08))


def draw_panel(
    ax,
    panel,
    *,
    mchi_grid,
    g_sm,
    g_chi,
    matching,
    collider_contours,
    thermal_contour,
    show_ylabel,
    show_xlabel,
):
    operator, dm_kind, _ = panel
    if operator is None:
        ax.set_axis_off()
        return []

    mmed_valid, valid_label = validity_mmed_curve(
        operator, mchi_grid, g_sm=g_sm, g_chi=g_chi, matching=matching
    )
    if mmed_valid is None:
        ax.set_axis_off()
        return []

    extra_curves = [(x, y) for x, y, _, _, _ in collider_contours]
    if thermal_contour is not None:
        extra_curves.append(thermal_contour)
    threshold_mchi = np.asarray(mchi_grid, dtype=float)
    threshold_mmed = 2.0 * threshold_mchi
    extra_curves.append((threshold_mmed, threshold_mchi))
    apply_limits(ax, mmed_grid=mmed_valid, mchi_grid=mchi_grid, extra_curves=extra_curves)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, ls=":", color="#AAAAAA", which="both")

    xmin, xmax = ax.get_xlim()
    ax.fill_betweenx(
        mchi_grid,
        mmed_valid,
        xmax,
        color=PAPER_COLORS["valid_fill"],
        alpha=0.28,
        label="Transformed EFT-valid region",
        zorder=1,
    )
    ax.plot(mmed_valid, mchi_grid, color=PAPER_COLORS["valid_line"], lw=2.0, ls="--",
            label=valid_label, zorder=5)

    handles = [
        Patch(facecolor=PAPER_COLORS["valid_fill"], edgecolor=PAPER_COLORS["valid_line"],
              alpha=0.28, label="Transformed EFT-valid region"),
        Line2D([0], [0], color=PAPER_COLORS["valid_line"], lw=2.0, ls="--", label=valid_label),
    ]

    for x, y, label, side, color in collider_contours:
        shade_excluded(ax, x, y, side=side, color=color, label=f"{label} excluded")
        ax.plot(x, y, color=color, lw=1.8, label=label, zorder=6)
        handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.16, label=f"{label} excluded"))
        handles.append(Line2D([0], [0], color=color, lw=1.8, label=label))

    visible = (threshold_mmed >= ax.get_xlim()[0]) & (threshold_mmed <= ax.get_xlim()[1])
    visible &= (threshold_mchi >= ax.get_ylim()[0]) & (threshold_mchi <= ax.get_ylim()[1])
    if np.any(visible):
        ax.plot(
            threshold_mmed[visible],
            threshold_mchi[visible],
            color=PAPER_COLORS["threshold"],
            lw=1.8,
            ls="--",
            label=r"$m_{\rm med}=2m_\chi$",
            zorder=7,
        )
        handles.append(Line2D([0], [0], color=PAPER_COLORS["threshold"], lw=1.8, ls="--",
                              label=r"$m_{\rm med}=2m_\chi$"))

    if thermal_contour is not None:
        x_th, y_th = thermal_contour
        ax.plot(
            x_th,
            y_th,
            color=PAPER_COLORS["thermal"],
            lw=2.0,
            ls="-.",
            label=r"Thermal relic $\Omega h^2=0.12$",
            zorder=8,
        )
        handles.append(Line2D([0], [0], color=PAPER_COLORS["thermal"], lw=2.0, ls="-.",
                              label=r"Thermal relic $\Omega h^2=0.12$"))

    title = "Scalar Rayleigh" if dm_kind == "scalar" else operator_title(str(operator))
    ax.set_title(title, fontsize=CONF_TITLE_FS - 6)
    if show_ylabel:
        dm_label = "Dirac" if dm_kind == "dirac" else "Majorana / Scalar"
        ax.set_ylabel(rf"{dm_label}" "\n" r"$m_\chi\,[\mathrm{GeV}]$", fontsize=CONF_LABEL_FS)
    else:
        ax.tick_params(labelleft=False)
    if show_xlabel:
        ax.set_xlabel(r"$m_{\rm med}\,[\mathrm{GeV}]$", fontsize=CONF_LABEL_FS)
    else:
        ax.tick_params(labelbottom=False)
    ax.tick_params(labelsize=CONF_TICK_FS)
    return handles


def unique_handles(handles):
    out, seen = [], set()
    for handle in handles:
        label = handle.get_label()
        if not label or label.startswith("_") or label in seen:
            continue
        seen.add(label)
        out.append(handle)
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make UV-complete mediator-mass theory-limit panels from EFT-validity curves."
    )
    parser.add_argument("--g-sm", type=float, default=0.25, help="Fixed SM coupling g_SM.")
    parser.add_argument("--g-chi", type=float, default=1.0, help="Fixed dark coupling g_chi.")
    parser.add_argument(
        "--matching",
        choices=["sqrt", "product"],
        default="sqrt",
        help="Use Lambda=m_med/sqrt(g_SM g_chi) or Lambda=m_med/(g_SM g_chi).",
    )
    parser.add_argument("--mchi-min", type=float, default=1e-6)
    parser.add_argument("--mchi-max", type=float, default=1e8)
    parser.add_argument("--n-mchi", type=int, default=500)
    parser.add_argument(
        "--collider",
        action="append",
        default=[],
        help="CSV overlay as path[:label[:excluded_side[:color]]]. Side: above, below, left, right.",
    )
    parser.add_argument("--thermal", default=None, help="CSV contour for thermal relic Ωh^2=0.12.")
    parser.add_argument(
        "--no-dummy-limits",
        action="store_true",
        help="Do not auto-overlay dummy CSV limits when no real constraints are supplied.",
    )
    parser.add_argument("--outfile", default="uv_complete_theory_limits")
    parser.add_argument("--style", default=None)
    return parser.parse_args()


def default_dummy_specs():
    specs = []
    for filename, label, side in DEFAULT_DUMMY_COLLIDERS:
        path = DEFAULT_DUMMY_DIR / filename
        if path.exists():
            specs.append(f"{path}:{label}:{side}")
    thermal = DEFAULT_DUMMY_DIR / DEFAULT_DUMMY_THERMAL
    return specs, str(thermal) if thermal.exists() else None


def main():
    args = parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style
    if args.style and str(args.style).lower() != "paper":
        set_plot_style(style=args.style, base_fontsize=13, linewidth=2.0, n_colors=14, cmap_name="plasma")
    else:
        set_paper_style(base_fontsize=9, linewidth=1.5, n_colors=14, cmap_name="plasma")

    collider_specs = list(args.collider)
    thermal_path = args.thermal
    if not args.no_dummy_limits and not collider_specs and thermal_path is None:
        collider_specs, thermal_path = default_dummy_specs()
        if collider_specs or thermal_path:
            print("Using bundled dummy UV-complete constraints. Pass --no-dummy-limits to suppress them.")

    collider_contours = []
    for idx, spec_text in enumerate(collider_specs):
        spec = parse_collider_spec(spec_text)
        x, y = load_contour_csv(spec["path"])
        color = spec["color"] or color_for_label(spec["label"], idx)
        collider_contours.append((x, y, spec["label"], spec["side"], color))

    thermal_contour = None
    if thermal_path:
        thermal_contour = load_contour_csv(thermal_path)

    mchi_grid = np.logspace(np.log10(args.mchi_min), np.log10(args.mchi_max), int(args.n_mchi))
    panels = [TOP_ROW_PANELS, BOTTOM_ROW_PANELS]

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(30.0, 14.0),
        gridspec_kw={"wspace": 0.10, "hspace": 0.14},
    )

    legend_handles = []
    for row_idx, row in enumerate(panels):
        for col_idx, panel in enumerate(row):
            handles = draw_panel(
                axes[row_idx, col_idx],
                panel,
                mchi_grid=mchi_grid,
                g_sm=args.g_sm,
                g_chi=args.g_chi,
                matching=args.matching,
                collider_contours=collider_contours,
                thermal_contour=thermal_contour,
                show_ylabel=(col_idx == 0),
                show_xlabel=(row_idx == 1),
            )
            legend_handles.extend(handles)

    matching_text = (
        r"$\Lambda=m_{\rm med}/\sqrt{g_{\rm SM}g_\chi}$"
        if args.matching == "sqrt"
        else r"$\Lambda=m_{\rm med}/(g_{\rm SM}g_\chi)$"
    )
    info_ax = axes[1, 4]
    info_ax.set_axis_off()
    info_ax.text(
        0.04,
        0.96,
        "\n".join([
            "UV-complete mediator plane",
            matching_text,
            rf"$g_{{\rm SM}}={args.g_sm:g}$",
            rf"$g_\chi={args.g_chi:g}$",
            "Collider/relic contours loaded from CSV",
        ]),
        transform=info_ax.transAxes,
        ha="left",
        va="top",
        fontsize=CONF_ANNOT_FS,
        linespacing=1.35,
    )

    fig.legend(
        handles=unique_handles(legend_handles),
        loc="upper center",
        bbox_to_anchor=(0.46, 0.992),
        ncol=4,
        fontsize=CONF_LEGEND_FS - 2,
        frameon=True,
        framealpha=0.25,
    )
    fig.subplots_adjust(top=0.88, left=0.07, right=0.86, bottom=0.10)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / args.outfile
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"Saved UV-complete theory-limit figure: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
