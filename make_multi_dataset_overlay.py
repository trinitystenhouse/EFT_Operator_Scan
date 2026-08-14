"""
Multi-dataset γχ scattering constraint overlay.

Loads 90 % CL boundaries from ``constraint_boundaries/`` for every combination
of ``(dataset, source_kind, m_ann)`` that has been generated, and overlays
them on a shared ``(Λ, m_χ)`` panel per operator. Each dataset gets a distinct
colour; each source kind gets a distinct linestyle.

Companion to ``make_sensitivity_map.py``, which shows the full f_required
heatmap for a single dataset; this script is optimised for cross-dataset
comparison of the exclusion contours.

Filename convention consumed
----------------------------
Boundary files are produced by ``make_data_driven_scattering_limits.py`` with
stems

    mcmc_{halo_profile}_{halo|pppc_bb_mann<M>}_..._90cl.npz
    igrb_ackermann2015a_{measured|pppc_bb_mann<M>}_..._90cl.npz

The dataset / source_kind / m_ann are auto-detected from the filenames.

Usage
-----
    # Paper-summary 3-panel (Dirac magnetic dipole, Majorana anapole, scalar Rayleigh)
    python make_multi_dataset_overlay.py --paper-summary

    # Full 9-panel with everything
    python make_multi_dataset_overlay.py

    # Just measured spectra, no PPPC benchmarks
    python make_multi_dataset_overlay.py --sources measured

    # Only bb-500 GeV PPPC benchmark
    python make_multi_dataset_overlay.py --sources measured pppc --pppc-masses 500
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from helpers.plot_style import save_figure, set_paper_style, theme_ink as _ink, theme_legend_kw  # noqa: E402

BOUNDARY_DIR = _HERE / "constraint_boundaries"
OUTPUT_DIR = _HERE / "plots"


# =============================================================================
# Panel configs (mirrors make_sensitivity_map.py)
# =============================================================================

PANEL_CONFIGS = {
    "dipole_magnetic":         {"title": "Magnetic Dipole (Dirac)\n(dim-5)",              "dm_type": "fermionic", "operator": "dipole_magnetic",  "majorana": False},
    "dipole_electric":         {"title": "Dirac Electric Dipole\n(dim-5)",              "dm_type": "fermionic", "operator": "dipole_electric",  "majorana": False},
    "charge_radius":           {"title": "Charge Radius\n(dim-6)",                     "dm_type": "fermionic", "operator": "charge_radius",    "majorana": False},
    "anapole":                 {"title": "Dirac Anapole\n(dim-6)",                     "dm_type": "fermionic", "operator": "anapole",          "majorana": False},
    "anapole_majorana":        {"title": "Majorana Anapole\n/ Axial CR (dim-6)",       "dm_type": "fermionic", "operator": "anapole",          "majorana": True},
    "rayleigh_even":           {"title": "Rayleigh Even (Dirac)\n(dim-7)",             "dm_type": "fermionic", "operator": "rayleigh_even",    "majorana": False},
    "rayleigh_even_majorana":  {"title": "Rayleigh Even (Majorana)\n(dim-7)",          "dm_type": "fermionic", "operator": "rayleigh_even",    "majorana": True},
    "rayleigh_odd":            {"title": "Rayleigh Odd (Dirac)\n(dim-7)",              "dm_type": "fermionic", "operator": "rayleigh_odd",     "majorana": False},
    "rayleigh_full":           {"title": "Rayleigh Full (Dirac)\n(dim-7)",             "dm_type": "fermionic", "operator": "rayleigh_full",    "majorana": False},
    "rayleigh_full_majorana":  {"title": "Rayleigh Full (Majorana)\n(dim-7)",          "dm_type": "fermionic", "operator": "rayleigh_full",    "majorana": True},
    "scalar_rayleigh":         {"title": "Rayleigh (Scalar)\n(dim-6)",                    "dm_type": "scalar",    "operator": "scalar_rayleigh",  "majorana": False},
}

PAPER_SUMMARY_OPERATORS = ["dipole_magnetic", "anapole_majorana", "scalar_rayleigh"]

# Dataset styling — mirrors the palette used in make_paper_style_operator_overlays.
# Short data-source names; the intrinsic-template mode is appended by SOURCE_STYLES.
DATASET_STYLES = {
    "halo": {
        "label":  r"GC halo",
        "color":  "#111111",  # black
    },
    "igrb": {
        "label":  r"IGRB",
        "color":  "#9B6DFF",  # purple
    },
}

# Source-kind styling: the intrinsic pre-scattering source template.
#   'measured' — Φ_null = the dataset's own SED; normalisation profiled.
#                Interpretation: "is the measured spectral shape self-consistent
#                under scattering?"  No DM model is assumed.
#   'pppc_bb_mann<M>' — Φ_null = Cirelli+ 2011 PPPC template for χχ̄→bb̄
#                annihilation at m_ann = M GeV, normalisation profiled.
#                Interpretation: "assuming the photon source IS annihilating
#                DM at (χχ̄→bb̄, m_ann=M), is the scattering-attenuated
#                version consistent with the data?"
SOURCE_STYLES = {
    "measured": {
        "linestyle": "-",
        "lw":        2.6,
        "label_suffix": ", measured spectrum",
    },
    # Companion Totani reanalysis best fits: m_ann = 0.55 TeV (W+W-) and
    # 0.72 TeV (bb). Short labels so the legend fits at PRD twocolumn width.
    # The $\Phi_{\rm null} = ...$ full form is stated in the caption.
    "pppc_bb_mann550": {
        "linestyle": (0, (5, 2)),
        "lw":        1.8,
        "label_suffix": r", PPPC $b\bar b$ 0.55 TeV",
    },
    "pppc_bb_mann720": {
        "linestyle": (0, (7, 2, 1, 2)),
        "lw":        1.8,
        "label_suffix": r", PPPC $b\bar b$ 0.72 TeV (best fit)",
    },
    "pppc_WW_mann550": {
        "linestyle": (0, (3, 1.2, 1, 1.2)),
        "lw":        1.8,
        "label_suffix": r", PPPC $W^{+}W^{-}$ 0.55 TeV (best fit)",
    },
    "pppc_WW_mann720": {
        "linestyle": (0, (1, 1.2)),
        "lw":        1.8,
        "label_suffix": r", PPPC $W^{+}W^{-}$ 0.72 TeV",
    },
    # Aliases for the 100 / 500 / 1000 GeV benchmarks.
    "pppc_bb_mann100": {
        "linestyle": (0, (2, 1.5)),
        "lw":        1.8,
        "label_suffix": r", PPPC $b\bar b$ 100 GeV",
    },
    "pppc_bb_mann500": {
        "linestyle": (0, (5, 2)),
        "lw":        1.8,
        "label_suffix": r", PPPC $b\bar b$ 500 GeV",
    },
    "pppc_bb_mann1000": {
        "linestyle": (0, (7, 2, 1, 2)),
        "lw":        1.8,
        "label_suffix": r", PPPC $b\bar b$ 1 TeV",
    },
}


# =============================================================================
# Boundary-file discovery
# =============================================================================

# Filename patterns to detect the (dataset, source_kind) from a boundary file.
_FILENAME_PATTERNS = [
    # halo/measured (the halo files carry source_tag='halo')
    (re.compile(
        r"^mcmc_(?P<profile>[^_]+(?:_[^_]+)*?)_"
        r"(?P<source>halo)_raw_attenuation_"
        r"(?P<dm>[a-z]+)_(?P<op>[a-z_]+?)(?P<maj>_majorana)?_90cl\.npz$"
     ), "halo"),
    # halo/pppc
    (re.compile(
        r"^mcmc_(?P<profile>[^_]+(?:_[^_]+)*?)_"
        r"(?P<source>pppc_(?:bb|WW)_mann\d+(?:\.\d+)?)_raw_attenuation_"
        r"(?P<dm>[a-z]+)_(?P<op>[a-z_]+?)(?P<maj>_majorana)?_90cl\.npz$"
     ), "halo"),
    # igrb/measured or igrb/pppc
    (re.compile(
        r"^igrb_(?P<profile>[^_]+)_"
        r"(?P<source>measured|pppc_(?:bb|WW)_mann\d+(?:\.\d+)?)_raw_attenuation_"
        r"(?P<dm>[a-z]+)_(?P<op>[a-z_]+?)(?P<maj>_majorana)?_90cl\.npz$"
     ), "igrb"),
]


def _parse_boundary_filename(name: str) -> dict | None:
    """Parse a boundary filename into its (dataset, profile, source, dm, op, maj) parts."""
    for pat, dataset in _FILENAME_PATTERNS:
        m = pat.match(name)
        if m:
            return {
                "dataset":  dataset,
                "profile":  m.group("profile"),
                "source":   m.group("source"),
                "dm_type":  m.group("dm"),
                "operator": m.group("op"),
                "majorana": m.group("maj") == "_majorana",
                "path":     BOUNDARY_DIR / name,
            }
    return None


# Variant tag inserted before "_90cl" when locating boundary files, so the
# figures can be built from an alternative generation (e.g. "_profnorm",
# the profiled-normalisation grids) without renaming the production set.
# Empty string = the production grids. Set by make_paper_results_figures.
BOUNDARY_SUFFIX = ""


def discover_boundaries(operator_key: str, sources: list[str],
                        halo_profile: str,
                        pppc_masses: list[int],
                        datasets: list[str] = None,
                        pppc_channels: list[str] | None = None) -> list[dict]:
    """Return every boundary-file record for the requested operator + sources.

    Parameters
    ----------
    datasets : list of {'halo', 'igrb'}, optional
        Restrict to these datasets. Default: include both.
    pppc_channels : list of {'bb', 'WW'}, optional
        Which PPPC annihilation channels to include when ``sources`` contains
        the shorthand 'pppc'. Default: ['bb', 'WW'].
    """
    cfg = PANEL_CONFIGS[operator_key]
    wanted_source_tags = _expand_wanted_sources(sources, pppc_masses, pppc_channels)
    wanted_datasets = set(datasets) if datasets else {"halo", "igrb"}
    records = []
    for path in sorted(BOUNDARY_DIR.iterdir()):
        # Only files carrying the active variant tag, and strip it before
        # parsing so the filename grammar below is unchanged.
        tail = f"{BOUNDARY_SUFFIX}_90cl.npz"
        if not path.name.endswith(tail):
            continue
        parsed = _parse_boundary_filename(
            path.name[: -len(tail)] + "_90cl.npz" if BOUNDARY_SUFFIX else path.name)
        if parsed is None:
            continue
        # _parse_boundary_filename rebuilds "path" from the name it was handed,
        # which has the variant tag stripped -- point it back at the real file.
        parsed["path"] = path
        if parsed["dataset"]  not in wanted_datasets: continue
        if parsed["dm_type"]  != cfg["dm_type"]:  continue
        if parsed["operator"] != cfg["operator"]: continue
        if bool(parsed["majorana"]) != bool(cfg["majorana"]): continue
        if parsed["source"]   not in wanted_source_tags:      continue
        if parsed["dataset"] == "halo" and parsed["profile"] != halo_profile: continue
        records.append(parsed)
    return records


def _expand_wanted_sources(sources: list[str], pppc_masses: list[int],
                            pppc_channels: list[str] | None = None) -> set[str]:
    """Turn ('measured', 'pppc') + masses + channels into concrete source_tag set.

    ``pppc_channels`` defaults to ``['bb', 'WW']`` (companion-paper best-fit
    channels). Anything else in ``sources`` is passed through verbatim.
    """
    if pppc_channels is None:
        pppc_channels = ["bb", "WW"]
    wanted = set()
    # Preserve case in channel comparisons since the WW variant is uppercase.
    lowered = [str(s).lower() for s in sources]
    for s_raw, s in zip(sources, lowered):
        if s == "measured":
            # igrb uses the 'measured' tag; halo uses 'halo'
            wanted.add("measured")
            wanted.add("halo")
        elif s == "pppc":
            for m in pppc_masses:
                for ch in pppc_channels:
                    wanted.add(f"pppc_{ch}_mann{m}")
        elif s.startswith("pppc_") and "mann" in s_raw:
            # Explicit tag like 'pppc_bb_mann500' or 'pppc_WW_mann550' — keep case.
            wanted.add(str(s_raw))
        else:
            wanted.add(str(s_raw))
    return wanted


# =============================================================================
# Plot logic
# =============================================================================

def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    m = np.asarray(d["mchi_GeV"], dtype=float)
    L = np.asarray(d["lambda_plot_GeV"], dtype=float)
    order = np.argsort(m)
    m, L = m[order], L[order]
    finite = np.isfinite(m) & np.isfinite(L) & (m > 0) & (L > 0)
    return m[finite], L[finite]


# Photon-energy ceiling [GeV] that positions the EFT kinematic-validity guide.
# None falls back to DEFAULT_OMEGA_MAX (the full LAT band). The master figure
# script sets this so every operator panel quotes one ceiling. Moves the DRAWN
# GUIDE only -- the limit curves are not recomputed.
OMEGA_MAX_OVERRIDE = None
DEFAULT_OMEGA_MAX = 494.0



def _eft_valid_curve(m_arr: np.ndarray, omega_max: float | None = None,
                      eft_factor: float = 1.0) -> np.ndarray:
    """Λ_kin such that Λ² ≥ max(s_max, |t|_max) at photon energy omega_max."""
    if omega_max is None:
        omega_max = (OMEGA_MAX_OVERRIDE if OMEGA_MAX_OVERRIDE is not None
                     else DEFAULT_OMEGA_MAX)
    m = np.asarray(m_arr, dtype=float)
    s_max = m ** 2 + 2.0 * m * omega_max
    denom = 1.0 + 2.0 * omega_max / np.where(m > 0, m, np.nan)
    t_max = 4.0 * omega_max ** 2 / denom
    return np.sqrt(eft_factor * np.maximum(s_max, t_max))


def plot_overlay_panel(ax, operator_key: str, records: list[dict],
                       *, title_fontsize=13, axis_label_fontsize=12,
                       tick_labelsize=11, annotate_fontsize=11,
                       x_lim=(1e-10, 1e12), y_lim=(1e-6, 1e8),
                       validity_line_color="#666666",
                       validity_fill_color="cyan",
                       validity_fill_alpha=0.10,
                       validity_label_color="#008B8B") -> list[Line2D]:
    """Overlay all discovered dataset/source contours onto one panel."""
    cfg = PANEL_CONFIGS[operator_key]

    # EFT-valid wedge (cyan shading + grey dashed boundary)
    xg = np.logspace(np.log10(x_lim[0]), np.log10(x_lim[1]), 400)
    lam_kin = _eft_valid_curve(xg)
    ax.plot(xg, lam_kin, color=validity_line_color, lw=1.4, ls="--",
            label="EFT kinematic validity", zorder=4)
    ax.fill_between(xg, lam_kin, np.full_like(lam_kin, y_lim[1]),
                    color=validity_fill_color, alpha=validity_fill_alpha, zorder=1)
    ax.text(0.04, 0.94, "EFT valid", color=validity_label_color,
            fontsize=annotate_fontsize, fontweight="bold",
            ha="left", va="top", transform=ax.transAxes)

    # Overlay every boundary curve
    seen_labels = set()
    handles = []
    for rec in records:
        m, L = _load_curve(rec["path"])
        if len(m) == 0:
            continue
        ds = DATASET_STYLES.get(rec["dataset"], {"color": "gray", "label": rec["dataset"]})
        src_key = "measured" if rec["source"] in ("measured", "halo") else rec["source"]
        ss = SOURCE_STYLES.get(src_key, {"linestyle": "-", "lw": 1.8, "label_suffix": ""})
        label = ds["label"] + ss["label_suffix"]
        ax.loglog(m, L, color=ds["color"], ls=ss["linestyle"], lw=ss["lw"],
                  zorder=5, label=label)
        if label not in seen_labels:
            handles.append(Line2D([0], [0], color=ds["color"], ls=ss["linestyle"],
                                  lw=ss["lw"], label=label))
            seen_labels.add(label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_title(cfg["title"], fontsize=title_fontsize)
    ax.set_xlabel(r"$m_{\chi}$ [GeV]", fontsize=axis_label_fontsize)
    ax.set_ylabel(r"$\Lambda/(c^{1/n} f_{\rm scat}^{1/p})$ [GeV]", fontsize=axis_label_fontsize)
    ax.grid(True, which="both", alpha=0.15)
    ax.tick_params(axis="both", which="both", labelsize=tick_labelsize)
    return handles


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--operators", nargs="+", default=None,
                   choices=sorted(PANEL_CONFIGS.keys()),
                   help="Operators to plot. Default: full 9-panel set, or paper-summary set.")
    p.add_argument("--paper-summary", action="store_true",
                   help="Compact 3-panel version matching Fig 2 of the paper.")
    p.add_argument("--datasets", nargs="+", default=["halo", "igrb"],
                   choices=["halo", "igrb"],
                   help="Which datasets to overlay. Default: both.")
    p.add_argument("--sources", nargs="+", default=["measured", "pppc"],
                   help="Which source kinds to include. Any of 'measured', 'pppc'.")
    p.add_argument("--pppc-masses", nargs="+", type=int, default=[550, 720],
                   help=("Which PPPC annihilator masses (GeV) to include when 'pppc' in --sources. "
                         "Default: [550, 720] — companion-paper Totani reanalysis best fits."))
    p.add_argument("--pppc-channels", nargs="+", default=["bb", "WW"],
                   choices=["bb", "WW"],
                   help="Which PPPC annihilation channels to include when 'pppc' in --sources.")
    p.add_argument("--halo-profile", default="pixelwise_global_rho2",
                   help="Halo profile tag to include. Default: pixelwise_global_rho2.")
    p.add_argument("--outfile", default=None,
                   help="Output basename (no extension). Default: auto based on --paper-summary.")
    p.add_argument("--style", default="paper",
                   help="plot_style style. Default: paper.")
    # Print-geometry overrides (used by make_paper_results_figures.py for PRD width)
    p.add_argument("--fig-width", type=float, default=None,
                   help="Figure width [inches]. Overrides internal figsize.")
    p.add_argument("--fig-height", type=float, default=None,
                   help="Figure height [inches]. Overrides internal figsize.")
    p.add_argument("--base-fontsize", type=float, default=None,
                   help="Base font size for set_paper_style. Overrides default (13).")
    p.add_argument("--linewidth", type=float, default=None,
                   help="Line width for set_paper_style. Overrides default (2.0).")
    args = p.parse_args()

    if args.style:
        os.environ["EFT_PLOT_STYLE"] = args.style
    if args.operators is None:
        args.operators = PAPER_SUMMARY_OPERATORS if args.paper_summary else list(PANEL_CONFIGS.keys())
    if args.outfile is None:
        suf = "_paper_summary" if args.paper_summary else ""
        args.outfile = f"multi_dataset_overlay{suf}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Default base_fs=10 to match the rest of the paper set (was 13, which
    # produced oversized text at PRD widths). CLI --base-fontsize overrides.
    base_fs   = args.base_fontsize if args.base_fontsize is not None else 10
    linewidth = args.linewidth     if args.linewidth     is not None else 1.4
    set_paper_style(base_fontsize=base_fs, linewidth=linewidth, n_colors=14, cmap_name="plasma")

    n = len(args.operators)
    ncols = 3 if args.paper_summary else 3
    nrows = 1 if args.paper_summary else int(np.ceil(n / ncols))
    if args.fig_width is not None and args.fig_height is not None:
        figsize = (args.fig_width, args.fig_height)
    else:
        figsize = (5.0 * ncols + 1.0, 4.4 * nrows + 1.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    print(f"Multi-dataset overlay ({n} panel{'' if n==1 else 's'})")
    print(f"  sources     : {args.sources}")
    print(f"  pppc masses : {args.pppc_masses}")
    print(f"  halo profile: {args.halo_profile}")
    print()

    combined_handles: dict[str, Line2D] = {}
    for i, op in enumerate(args.operators):
        records = discover_boundaries(
            op, sources=args.sources,
            halo_profile=args.halo_profile,
            pppc_masses=args.pppc_masses,
            datasets=args.datasets,
            pppc_channels=args.pppc_channels,
        )
        print(f"  {op:<20s}  {len(records):>3d} curve{'' if len(records)==1 else 's'} found")
        hs = plot_overlay_panel(
            axes[i], op, records,
            # Fonts scale with base_fs so this figure matches the sensitivity
            # map + halo-constraints figures at the same PRD width.
            title_fontsize=base_fs + 1,
            axis_label_fontsize=base_fs,
            tick_labelsize=base_fs - 2,
            annotate_fontsize=base_fs - 1,
        )
        for h in hs:
            combined_handles.setdefault(h.get_label(), h)

    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    # Axis labels only on the bottom row (x) and leftmost column (y). Same
    # convention as the sensitivity map and halo constraints figures so all
    # multi-panel plots read consistently.
    last_panel_in_col = {}
    first_panel_in_row = {}
    for i in range(n):
        row, col = divmod(i, ncols)
        last_panel_in_col[col] = i
        first_panel_in_row.setdefault(row, i)
    for i in range(n):
        row, col = divmod(i, ncols)
        if i != last_panel_in_col[col]:
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", labelbottom=False)
        if i != first_panel_in_row[row]:
            axes[i].set_ylabel("")
            axes[i].tick_params(axis="y", labelleft=False)

    # Shared legend styling (Totani make_paper_results_figures convention).
    LEGEND_KW = dict(**theme_legend_kw())
    # Split legend labels at the ", " between the dataset and Φ_null clauses
    # WITHOUT breaking inside a $...$ math span. Split only if both sides of
    # the comma sit entirely outside math mode.
    def _wrap_safe(s: str) -> str:
        if not s:
            return s
        # Find comma+space positions where the number of unescaped $ tokens
        # before the position is even (i.e., we are outside a math span).
        out_positions = []
        in_math = False
        for i in range(len(s) - 1):
            c = s[i]
            if c == "$":
                in_math = not in_math
            if not in_math and c == "," and s[i + 1] == " ":
                out_positions.append(i)
        if not out_positions:
            return s
        # Prefer the split that produces the most balanced two-line label.
        best = min(out_positions, key=lambda p: abs(len(s) / 2 - p))
        return s[: best + 1] + "\n" + s[best + 2 :]
    if combined_handles:
        handles = list(combined_handles.values())
        # With the concise "GC halo, PPPC bb 0.55 TeV" labels every entry fits
        # on one line, so a 3-column strip stays compact vertically.
        labels  = list(combined_handles.keys())
        leg = fig.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 0.0),
            ncol=min(3, len(handles)),
            fontsize=base_fs - 1,
            handlelength=2.6, handletextpad=0.5, columnspacing=1.4,
            labelspacing=0.35,
            alignment="left",
            borderpad=0.5,
            **LEGEND_KW,
        )
        frame = leg.get_frame()
        frame.set_facecolor(LEGEND_KW["facecolor"])
        frame.set_edgecolor(LEGEND_KW["edgecolor"])
        frame.set_alpha(LEGEND_KW["framealpha"])

    # 3-column strip for the 5–6 concise legend entries needs only ~0.20
    # bottom space. Modest inter-panel spacing so consolidated axis labels
    # have breathing room but panels still feel like one figure.
    fig.tight_layout(rect=[0.0, 0.20, 1.0, 0.97])
    fig.subplots_adjust(wspace=0.10, hspace=0.10)

    save_figure(fig, str(OUTPUT_DIR / args.outfile))
    fig.savefig(
        str(OUTPUT_DIR / f"{args.outfile}.pdf"),
        bbox_inches="tight",
        transparent=bool(plt.rcParams.get("savefig.transparent", False)),
        facecolor=plt.rcParams.get("savefig.facecolor", "auto"),
        edgecolor=plt.rcParams.get("savefig.edgecolor", "auto"),
    )
    plt.close(fig)
    print()
    print(f"Saved multi-dataset overlay: {OUTPUT_DIR / args.outfile}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
