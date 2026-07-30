import argparse
import os
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _wrap_label(s, width=34):
    """Wrap long legend labels over multiple lines so no title overflows its
    column. LaTeX math is preserved because textwrap only splits on whitespace."""
    if not s or len(s) <= width:
        return s
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from helpers.trinity_plotting import get_cmap_colors, save_figure, set_plot_style, set_paper_style
from constraints_data.limits import get_operator_limits


BOUNDARY_DIR = Path(__file__).resolve().parent / "constraint_boundaries"
CONSTRAINT_DIR = Path(__file__).resolve().parent / "constraints_data"
DETECTION_DIR = CONSTRAINT_DIR / "detection"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
PLASMA_DARK_COLORS = get_cmap_colors(cmap_name="plasma", n=22, start=0.12, end=0.98)
_FERMI_BUNDLED = Path(__file__).resolve().parent / "data" / "fermi_halo_spectrum.txt"
_FERMI_EXTERNAL = Path(__file__).resolve().parent.parent / "fermi_data" / "york" / "processed" / "spectrum_data.txt"
DEFAULT_FERMI_SPECTRUM = _FERMI_BUNDLED if _FERMI_BUNDLED.exists() else _FERMI_EXTERNAL

INDIRECT_COLOR = "#9B6DFF"
INDIRECT_ALT = "#B18CFF"
INDIRECT_DEEP = "#7F56D9"
INDIRECT_SOFT = "#C7A6FF"
DIRECT_COLOR = "#F2D53C"
DIRECT_ALT = "#FFD966"
DIRECT_DEEP = "#E3C12F"
DIRECT_SOFT = "#FFE68A"
COLLIDER_COLOR = "#FF7AC6"
COLLIDER_ALT = "#FF9AD5"
COLLIDER_DEEP = "#E754A8"
COLLIDER_SOFT = "#FFB6E2"
COSMOLOGY_COLOR = "#71D6FF"
BODDY_GLUSCEVIC_COLOR = "#71D6FF"
THERMAL_COLOR = "#59C7D8"
THIS_WORK_COLOR = "black"
THIS_WORK_ALT = "black"
THIS_WORK_DEEP = "black"
GUIDE_COLOR = "#8B93A4"
EFT_VALID_LABEL_COLOR = "#008B8B"
DATA_DRIVEN_PROFILE_STYLES = {
    "rho2": {
        "label": r"Totani $\rho^2$",
        "color": "#111111",
        "raw_ls": (0, (5, 2)),
        "reshaping_ls": "-",
        "lw": 2.8,
    },
    "global_rho2": {
        "label": r"Global $\rho^2$",
        "color": "#0B7285",
        "raw_ls": (0, (5, 2)),
        "reshaping_ls": "-",
        "lw": 2.6,
    },
    "global_rho2.5": {
        "label": r"Global $\rho^{2.5}$",
        "color": "#C92A2A",
        "raw_ls": (0, (5, 2)),
        "reshaping_ls": "-",
        "lw": 2.6,
    },
    "pixelwise_global_rho2": {
        "label": r"Scattering attenuation within NFW $\rho^2$ halo (this work)",
        "raw_attenuation_label": r"Scattering attenuation within NFW $\rho^2$ halo (this work)",
        "color": "#111111",
        "raw_ls": (0, (5, 2)),
        "reshaping_ls": "-",
        "lw": 2.8,
    },
    "pixelwise_global_rho2.5": {
        "label": r"Scattering attenuation within NFW $\rho^{2.5}$ halo (this work)",
        "raw_attenuation_label": r"Scattering attenuation within NFW $\rho^{2.5}$ halo (this work)",
        "color": "#C92A2A",
        "raw_ls": (0, (5, 2)),
        "reshaping_ls": "-",
        "lw": 2.6,
    },
}
PAPER_LABEL_OVERRIDES = {
    "rayleigh_full": r"$O_{\chi\chi FF} / O_{\chi5\chi FF}$",
}

LEGEND_GROUPS = {
    "direct_detection": {
        "label": "Direct Detection",
        "color": DIRECT_COLOR,
        "linestyle": "-",
    },
    "collider": {
        "label": "Collider",
        "color": COLLIDER_COLOR,
        "linestyle": "-",
    },
    "indirect_detection": {
        "label": "Indirect Detection (annihilation / pair production)",
        "color": INDIRECT_COLOR,
        "linestyle": "-",
    },
    "cosmology": {
        "label": "Cosmology",
        "color": COSMOLOGY_COLOR,
        "linestyle": "-",
    },
    "theory": {
        "label": "Theory / Validity",
        "color": GUIDE_COLOR,
        "linestyle": "-",
    },
    "this_work": {
        "label": "This Work",
        "color": THIS_WORK_COLOR,
        "linestyle": "-",
    },
}

GENERATED_LIMIT_LABELS = {
    "direct_detection": "Overall direct-detection limit",
    "collider": "Overall collider limit",
    "indirect_detection": "Overall indirect-detection limit",
    "deconvolution": "Overall deconvolution ceiling",
}

PANEL_CONFIGS = {
    # ── Dirac-only operators ─────────────────────────────────────────────────
    "dipole_magnetic": {
        "title": "Magnetic Dipole (Dirac)",
        "totani_file": "totani_fermionic_dipole_magnetic_90cl.npz",
        "naive_file":  "fermi_naive_fermionic_dipole_magnetic_0.01.npz",
        "constraint_subdir": "magdipole",
        "extra_constraint_files": ["files (3)/arina2020_lep_zdecay_dipole.txt"],
        "dm_type": "fermionic",
        "operator": "dipole_magnetic",
        "majorana": False,
    },
    "dipole_electric": {
        "title": "Electric Dipole (Dirac)",
        "totani_file": "totani_fermionic_dipole_electric_90cl.npz",
        "naive_file":  "fermi_naive_fermionic_dipole_electric_0.01.npz",
        "constraint_subdir": "eldipole",
        "extra_constraint_files": ["files (3)/arina2020_lep_zdecay_dipole.txt"],
        "dm_type": "fermionic",
        "operator": "dipole_electric",
        "majorana": False,
    },
    "charge_radius": {
        "title": "Charge Radius (Dirac)",
        "totani_file": "totani_fermionic_charge_radius_90cl.npz",
        "naive_file":  "fermi_naive_fermionic_charge_radius_0.01.npz",
        "constraint_subdir": "chargeradius",
        "dm_type": "fermionic",
        "operator": "charge_radius",
        "majorana": False,
    },

    # ── Dirac anapole ────────────────────────────────────────────────────────
    "anapole": {
        "title": "Anapole (Dirac)",
        "totani_file": "totani_fermionic_anapole_90cl.npz",
        "naive_file":  "fermi_naive_fermionic_anapole_0.01.npz",
        "constraint_subdir": "anapole",
        "dm_type": "fermionic",
        "operator": "anapole",
        "majorana": False,
    },

    # ── Majorana operators (dipole/charge_radius absent; anapole + Rayleigh only) ──
    "anapole_majorana": {
        "title": "Anapole / Axial Charge Radius (Majorana)",
        "totani_file": "totani_fermionic_anapole_majorana_90cl.npz",
        "naive_file":  "fermi_naive_fermionic_anapole_majorana_0.01.npz",
        "constraint_subdir": "anapole",          # shares literature constraints with Dirac anapole
        "dm_type": "fermionic",
        "operator": "anapole",
        "majorana": True,
    },
    "rayleigh_even_majorana": {
        "title": "Rayleigh Even (Majorana)",
        "totani_file": "totani_fermionic_rayleigh_even_majorana_90cl.npz",
        "naive_file":  "fermi_naive_fermionic_rayleigh_even_majorana_0.01.npz",
        "constraint_subdir": "rayleigh_even",
        "dm_type": "fermionic",
        "operator": "rayleigh_even",
        "majorana": True,
    },
    "rayleigh_odd_majorana": {
        "title": "Rayleigh Odd (Majorana)",
        "totani_file": "totani_fermionic_rayleigh_odd_majorana_90cl.npz",
        "naive_file":  "fermi_naive_fermionic_rayleigh_odd_majorana_0.01.npz",
        "constraint_subdir": "rayleigh_odd",
        "dm_type": "fermionic",
        "operator": "rayleigh_odd",
        "majorana": True,
    },
    # CP-even and CP-odd combined. Literature constraints are drawn from both
    # sub-directories, following the Dirac rayleigh_full entry below. The legacy
    # totani_file and naive_file have no full-Majorana counterparts; both are
    # optional and the panel draws the data-driven boundaries regardless.
    "rayleigh_full_majorana": {
        "title": "Rayleigh Full (Majorana)",
        "totani_file": "totani_fermionic_rayleigh_full_majorana_90cl.npz",
        "constraint_subdir": ["rayleigh_even", "rayleigh_odd"],
        "dm_type": "fermionic",
        "operator": "rayleigh_full",
        "majorana": True,
    },

    # ── Dirac Rayleigh ───────────────────────────────────────────────────────
    "rayleigh_even": {
        "title": "Rayleigh Even (Dirac)",
        "totani_file": "totani_fermionic_rayleigh_even_90cl.npz",
        "naive_file":  "fermi_naive_fermionic_rayleigh_even_0.01.npz",
        "constraint_subdir": "rayleigh_even",
        "dm_type": "fermionic",
        "operator": "rayleigh_even",
        "majorana": False,
    },
    "rayleigh_odd": {
        "title": "Rayleigh Odd (Dirac)",
        "totani_file": "totani_fermionic_rayleigh_odd_90cl.npz",
        "naive_file":  "fermi_naive_fermionic_rayleigh_odd_0.01.npz",
        "constraint_subdir": "rayleigh_odd",
        "dm_type": "fermionic",
        "operator": "rayleigh_odd",
        "majorana": False,
    },
    "rayleigh_full": {
        "title": "Rayleigh Full (Dirac)",
        "totani_file": "totani_fermionic_rayleigh_full_90cl.npz",
        "naive_file":  "fermi_naive_fermionic_rayleigh_full_0.01.npz",
        "constraint_subdir": ["rayleigh_even", "rayleigh_odd"],
        "dm_type": "fermionic",
        "operator": "rayleigh_full",
        "majorana": False,
    },

    # ── Scalar ───────────────────────────────────────────────────────────────
    "scalar_rayleigh": {
        "title": "Scalar Rayleigh",
        "totani_file": "totani_scalar_rayleigh_90cl.npz",
        "naive_file":  "fermi_naive_scalar_rayleigh_0.01.npz",
        "constraint_subdir": "rayleigh_scalar",
        "dm_type": "scalar",
        "operator": "scalar_rayleigh",
        "majorana": False,
    },
}

DEFAULT_OPERATORS = [
    "dipole_magnetic",
    "dipole_electric",
    "charge_radius",
    "anapole",
    "anapole_majorana",
    "rayleigh_full",
    "scalar_rayleigh",
]
PAPER_SUMMARY_OPERATORS = [
    "dipole_magnetic",
    "anapole_majorana",
    "scalar_rayleigh",
]
PAPER_SUMMARY_TITLES = {
    # Kept short so they fit at PRD twocolumn width without overlapping.
    # The axial-CR / anapole identity for Majorana is stated once in the
    # figure caption; there's no room to say it in each panel title.
    "dipole_magnetic":   "Dirac Magnetic Dipole\n(dim-5)",
    "anapole_majorana":  "Majorana Anapole\n(dim-6)",
    "scalar_rayleigh":   "Scalar Rayleigh\n(dim-7)",
}

STYLE_HINTS = {
    "xenon1t": ("XENON1T", "#C9A400", "-"),
    "xenonnt_anapole_majorana": ("XENONnT", "#F2D53C", "--"),
    "xenonnt": ("XENONnT", "#E2BE00", "--"),
    "hambye": ("XENON1T anapole recast", "#C9A400", "-"),
    "ibarra2024_lz": ("LZ", "#FFB347", "-."),
    "lz2022": ("LZ", "#FF9F1C", "-."),
    "lz_magdipole": ("LZ", "#FFD166", "-."),
    "lz_eldipole": ("LZ", "#FFCF70", (0, (7, 2, 1.5, 2))),
    "xlzd200ty": ("XLZD 200 ty", "#FFE68A", (0, (2, 2))),
    "monojet": ("Monojet", "#E754A8", "-"),
    "lhc_monojet": ("LHC mono-jet", "#FF5CB8", "-."),
    "lep_zdecay": ("LEP Z-decay", "#FF9AD5", (0, (3, 2))),
    "ams02": ("AMS-02", "#C7A6FF", (0, (1, 1))),
    "fermilat_hess": ("Fermi-LAT + H.E.S.S.", "#5E35B1", "-"),
    "fermilat": ("Fermi-LAT", "#8E5BFF", "-"),
    "fermi_lines": ("Fermi lines", "#8E5BFF", (0, (3, 2))),
    "fermi_single_line": ("Fermi single line", "#8E5BFF", ":"),
    "fermi_double_line": ("Fermi double line", "#8E5BFF", "-."),
    "hess": ("H.E.S.S.", "#7C4DFF", (0, (7, 2))),
    "cta_gc": ("CTA GC", "#7248D6", ":"),
    "cta_dsphs": ("CTA dSphs", "#7248D6", "--"),
    "planck": ("Planck", COSMOLOGY_COLOR, (0, (4, 2))),
    "thermal_relic": ("Thermal relic", THERMAL_COLOR, (0, (1, 1))),
}

LEGEND_ORDER = [
    "Direct Detection",
    "Collider",
    "Indirect Detection (Annihilation/Pair Production)",
    "Cosmology",
    "Theory / Validity",
    "This Work",
    "LZ",
    "XENONnT",
    "XENON1T",
    "XLZD 200 ty",
    "Monojet",
    "LEP Z-decay",
    "AMS-02",
    "Fermi-LAT + H.E.S.S.",
    "Fermi-LAT",
    "Fermi lines",
    "Fermi single line",
    "Fermi double line",
    "H.E.S.S.",
    "CTA GC",
    "CTA dSphs",
    "Planck",
    "EFT kinematic validity",
    "Unitarity",
    "Overall deconvolution ceiling",
    "Digitised Totani attenuation (legacy)",
    "This work: Totani halo scattering attenuation",
    "This work: reshaping",
    "York GC spectrum (this work, 90% CL)",
    "CMB (Boddy & Gluscevic 2018)",
]


def canonical_legend_label(label: str) -> str:
    label = str(label).strip()
    for meta in LEGEND_GROUPS.values():
        if label == meta["label"]:
            return meta["label"]
    if label.startswith("LZ"):
        return "LZ"
    if label.startswith("XENON1T"):
        return "XENON1T"
    if label.startswith("XENONnT"):
        return "XENONnT"
    if label.startswith("XLZD"):
        return "XLZD 200 ty"
    if label.startswith("Monojet") or label.startswith("LHC mono-jet"):
        return "Monojet"
    if label.startswith("LEP Z-decay"):
        return "LEP Z-decay"
    if label.startswith("AMS-02"):
        return "AMS-02"
    if label.startswith("Fermi-LAT + H.E.S.S."):
        return "Fermi-LAT + H.E.S.S."
    if label.startswith("Fermi-LAT"):
        return "Fermi-LAT"
    if label.startswith("Fermi lines"):
        return "Fermi lines"
    if label.startswith("Fermi single line"):
        return "Fermi single line"
    if label.startswith("Fermi double line"):
        return "Fermi double line"
    if label.startswith("H.E.S.S."):
        return "H.E.S.S."
    if label.startswith("CTA GC"):
        return "CTA GC"
    if label.startswith("CTA dSphs"):
        return "CTA dSphs"
    if label.startswith("Planck"):
        return "Planck"
    if label.startswith("Thermal relic") or label == "Thermal Relic":
        return "Thermal Relic"
    if label.startswith("Unitarity"):
        return "Unitarity"
    if label.startswith("This work: Totani halo scattering attenuation") or label.startswith("This work: attenuation"):
        return "This work: Totani halo scattering attenuation"
    if label.startswith("This work: reshaping"):
        return "This work: reshaping"
    if label.startswith("York GC spectrum (this work"):
        return "York GC spectrum (this work, 90% CL)"
    if label.startswith("CMB (Boddy & Gluscevic 2018)"):
        return "CMB (Boddy & Gluscevic 2018)"
    return label


def set_legend_group(handle, group: str):
    setattr(handle, "_legend_group", str(group))
    return handle


def legend_group_for_label(label: str):
    canon = canonical_legend_label(label)
    group_labels = {meta["label"]: key for key, meta in LEGEND_GROUPS.items()}
    if canon in group_labels:
        return group_labels[canon]
    if canon in ("LZ", "XENONnT", "XENON1T", "XLZD 200 ty", "Fortin & Tait"):
        return "direct_detection"
    if canon in ("Monojet", "LEP Z-decay"):
        return "collider"
    if canon in (
        "AMS-02",
        "Fermi-LAT + H.E.S.S.",
        "Fermi-LAT",
        "Fermi lines",
        "Fermi single line",
        "Fermi double line",
        "H.E.S.S.",
        "CTA GC",
        "CTA dSphs",
    ):
        return "indirect_detection"
    if canon in ("Planck", "Thermal Relic", "CMB (Boddy & Gluscevic 2018)"):
        return "cosmology"
    if canon == "Overall deconvolution ceiling":
        return "theory"
    if canon == "EFT kinematic validity":
        return "theory"
    if canon.startswith("Unitarity") or canon == "Theory / Validity":
        return "theory"
    if canon.startswith("This work") or canon.startswith("York GC spectrum"):
        return "this_work"
    return "theory"


def legend_sort_key(label: str):
    canon = canonical_legend_label(label)
    if canon in LEGEND_ORDER:
        return (0, LEGEND_ORDER.index(canon), canon)
    if any(name in canon for name in ("LZ", "XENON", "XLZD")):
        return (1, canon)
    if any(name in canon for name in ("Monojet", "LEP")):
        return (2, canon)
    if any(name in canon for name in ("Fermi", "H.E.S.S.", "CTA", "AMS-02")):
        return (3, canon)
    if any(name in canon for name in ("Planck", "Thermal relic")):
        return (4, canon)
    return (5, canon)


def collect_grouped_legend(handles):
    grouped = {group: [] for group in LEGEND_GROUPS}
    seen = {group: set() for group in LEGEND_GROUPS}
    for handle in handles:
        label = handle.get_label()
        if not label or str(label).startswith("_"):
            continue
        canon = canonical_legend_label(label)
        group = getattr(handle, "_legend_group", None) or legend_group_for_label(canon)
        if group not in grouped:
            group = legend_group_for_label(canon)
        if canon in seen[group]:
            continue
        seen[group].add(canon)
        handle.set_label(canon)
        grouped[group].append(handle)

    for group in grouped:
        grouped[group] = sorted(grouped[group], key=lambda h: legend_sort_key(h.get_label()))
    return grouped


# Shared legend styling (Totani make_paper_results_figures convention). Used
# by every fig.legend / ax.legend call in this file so all four paper figures
# read as one visual set. Multi-column bottom strip keeps frameon=False; single
# axes-level legends pick up the framed styling.
LEGEND_KW = dict(frameon=True, framealpha=0.6, facecolor="white", edgecolor="0.7")


def draw_bottom_grouped_legend(
    fig, grouped_handles, *, compact=False, base_fs=10, y_anchor=0.0,
    ncol=3,
):
    """Bottom-strip multi-column legend. Font sizes scale with base_fs so the
    figure matches the rest of the paper set at any PRD width."""
    if compact:
        # Single unified legend row at the bottom instead of three columns.
        # Concise category labels + this-work halo profiles side-by-side in
        # a boxed strip; matches Fig 5's aesthetic and takes ~half the
        # vertical space of the previous stacked 3-column layout.
        category_handles = [
            Line2D([0], [0], color=COLLIDER_COLOR, lw=2.0, label="Collider"),
            Line2D([0], [0], color=DIRECT_COLOR, lw=2.0, label="Direct detection"),
            Line2D([0], [0], color=INDIRECT_COLOR, lw=2.0,
                   label="Indirect detection"),
            Line2D([0], [0], color=COSMOLOGY_COLOR, lw=2.0, label="Cosmology"),
        ]
        this_work_handles = grouped_handles.get("this_work", [])
        # Order: category bounds first, then the halo-profile "this work" curves.
        all_handles = list(category_handles) + list(this_work_handles)
        labels = [_wrap_label(h.get_label(), width=42) for h in all_handles]
        leg = fig.legend(
            all_handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, y_anchor),
            ncol=ncol,
            fontsize=base_fs - 2,
            alignment="left",
            borderaxespad=0.0,
            borderpad=0.5,
            handlelength=2.4,
            handletextpad=0.55,
            columnspacing=1.4,
            labelspacing=0.35,
            **LEGEND_KW,
        )
        return

    # 3 columns × 2 rows keeps every group with room for a wrapped title and
    # multi-line labels. Grouping puts the two "detection" groups on the top
    # row alongside "collider", and cosmology/theory/this_work on the bottom.
    # Column x-anchors are spaced with ~0.32 width per column so long labels
    # (e.g. "Scattering attenuation within NFW ρ² halo (this work)") wrap
    # cleanly instead of overflowing.
    legend_columns = [
        # (group_key,          x_anchor, y_anchor)
        ("collider",           0.03, 0.135),
        ("direct_detection",   0.36, 0.135),
        ("indirect_detection", 0.68, 0.135),
        ("cosmology",          0.03, 0.025),
        ("theory",             0.36, 0.025),
        ("this_work",          0.68, 0.025),
    ]
    for group, xpos, ypos in legend_columns:
        handles = grouped_handles.get(group, [])
        if not handles:
            continue
        labels = [_wrap_label(h.get_label(), width=30) for h in handles]
        meta = LEGEND_GROUPS[group]
        leg = fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(xpos, ypos),
            ncol=1,
            fontsize=base_fs - 2,
            title=_wrap_label(meta["label"], width=26),
            title_fontsize=base_fs - 1,
            alignment="left",
            borderaxespad=0.0,
            borderpad=0.4,
            handlelength=2.4,
            handletextpad=0.55,
            labelspacing=0.35,
            **LEGEND_KW,
        )
        leg.get_title().set_color(meta["color"])


def npz_scalar(data, key, default=None):
    if key not in data.files:
        return default
    value = data[key]
    if hasattr(value, "shape") and value.shape == ():
        return value.item()
    return str(value)


def finite_positive_curve(mchi, lambda_plot):
    mchi = np.asarray(mchi, dtype=float)
    lambda_plot = np.asarray(lambda_plot, dtype=float)
    mask = np.isfinite(mchi) & np.isfinite(lambda_plot) & (mchi > 0.0) & (lambda_plot > 0.0)
    return mchi[mask], lambda_plot[mask], mask


def load_npz_boundary(path: Path):
    """Generic loader for any boundary saved by attenuation_eft / cmb_constraints / york_chi2."""
    data = np.load(path, allow_pickle=True)
    out = {
        "mchi":         np.asarray(data["mchi_GeV"],        dtype=float),
        "lambda_plot":  np.asarray(data["lambda_plot_GeV"], dtype=float),
        "lambda_raw":   np.asarray(data["lambda_GeV"],      dtype=float),
        "paper_label":  str(npz_scalar(data, "paper_label", "")),
        "operator":     str(npz_scalar(data, "operator", "")),
        "dm_type":      str(npz_scalar(data, "dm_type", "")),
        "validity_guides": str(npz_scalar(data, "validity_guides", "")),
        "boundary_extraction": str(npz_scalar(data, "boundary_extraction", "")),
        "omega_max_for_validity": npz_scalar(data, "omega_max_for_validity", None),
        "eft_kinematic_factor": npz_scalar(data, "eft_kinematic_factor", None),
    }
    return out


def load_totani_boundary(path: Path):
    data = np.load(path, allow_pickle=True)
    out = {
        "mchi": np.asarray(data["mchi_GeV"], dtype=float),
        "lambda_plot": np.asarray(data["lambda_plot_GeV"], dtype=float),
        "lambda_raw": np.asarray(data["lambda_GeV"], dtype=float),
        "paper_label": str(npz_scalar(data, "paper_label", "")),
        "operator": str(npz_scalar(data, "operator", "")),
        "dm_type": str(npz_scalar(data, "dm_type", "")),
    }
    stored_floor_flag = npz_scalar(data, "scan_floor_limited", None)
    mchi_finite, lambda_finite, _ = finite_positive_curve(out["mchi"], out["lambda_plot"])
    inferred_floor_flag = bool(len(lambda_finite) > 0 and np.allclose(lambda_finite, lambda_finite[0]))
    out["scan_floor_limited"] = bool(stored_floor_flag) if stored_floor_flag is not None else inferred_floor_flag
    out["has_finite_positive_curve"] = bool(len(mchi_finite) > 0)
    return out


def load_naive_boundary(path: Path):
    data = np.load(path, allow_pickle=True)
    out = {
        "mchi": np.asarray(data["mchi_GeV"], dtype=float),
        "lambda_plot": np.asarray(data["lambda_plot_GeV"], dtype=float),
        "lambda_raw": np.asarray(data["lambda_GeV"], dtype=float),
        "paper_label": str(npz_scalar(data, "paper_label", "")),
        "operator": str(npz_scalar(data, "operator", "")),
        "dm_type": str(npz_scalar(data, "dm_type", "")),
    }
    out["dip_depth"] = float(data["dip_depth"]) if ("dip_depth" in data.files) else None
    return out


def load_constraint_curve(path: Path):
    arr = np.loadtxt(path, comments="#", delimiter=None)
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 2:
        raise ValueError(f"{path} must have at least two columns: mchi_GeV lambda_plot_GeV")
    mchi, lam, _ = finite_positive_curve(arr[:, 0], arr[:, 1])
    return mchi, lam


def load_detection_boundaries(operator=None, dm_type=None, majorana=False):
    """
    Backward-compatible loader for generated detection boundaries.

    The canonical data now lives in constraints_data/limits.py as one limits
    dictionary per operator.  This wrapper keeps older call sites working.
    """
    boundaries = []
    for operator_key, cfg in PANEL_CONFIGS.items():
        if operator is not None and cfg["operator"] != operator:
            continue
        if dm_type is not None and cfg["dm_type"] != dm_type:
            continue
        if bool(cfg["majorana"]) != bool(majorana):
            continue
        for limit in get_operator_limits(operator_key, include_files=False, include_generated=True):
            if limit["constraint_type"] not in ("direct_detection", "indirect_detection", "collider"):
                continue
            data = limit["data"]
            boundaries.append((data["mchi_GeV"], data["lambda_plot_GeV"]))

    return boundaries


def validate_boundary_file(path: Path):
    required = {"mchi_GeV", "lambda_GeV", "lambda_plot_GeV", "paper_label", "operator", "dm_type"}
    if not path.exists():
        return False, f"missing {path.name}"
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        return False, f"{path.name}: unreadable ({exc})"
    missing = sorted(required - set(data.files))
    if missing:
        return False, f"{path.name}: missing keys {missing}"
    mchi, lam, _ = finite_positive_curve(data["mchi_GeV"], data["lambda_plot_GeV"])
    if len(mchi) == 0:
        return False, f"{path.name}: no finite positive curve points"
    floor = bool(np.allclose(lam, lam[0]))
    return True, (
        f"{path.name}: {len(mchi)} points, "
        f"mchi=[{mchi.min():.2e},{mchi.max():.2e}] GeV, "
        f"Lambda/C^(1/n)=[{lam.min():.2e},{lam.max():.2e}] GeV"
        + (" [scan-floor limited]" if floor else "")
    )


def print_pipeline_check(operators):
    print("Totani scattering overlay check")
    print("90% CL convention: attenuation_eft.py uses Delta chi^2 = 4.61 in (m_chi, Lambda).")
    print("Y-axis convention: saved lambda_plot_GeV is Lambda/C^(1/n), matching the paper-style overlays.\n")
    ok_all = True
    for op in operators:
        cfg = PANEL_CONFIGS[op]
        print(f"[{op}] {cfg['title']}")
        for label, filename in (
            ("Totani halo attenuation", cfg["totani_file"]),
            ("Naive Fermi reference", cfg.get("naive_file")),
        ):
            if not filename:
                continue
            ok, msg = validate_boundary_file(BOUNDARY_DIR / filename)
            ok_all = ok_all and ok
            print(f"  {'OK' if ok else 'BAD'} {label}: {msg}")

        majorana_suffix = "_majorana" if cfg["majorana"] else ""
        cmb_path = BOUNDARY_DIR / f"cmb_{cfg['dm_type']}_{cfg['operator']}{majorana_suffix}_planck2018.npz"
        if cmb_path.exists():
            ok, msg = validate_boundary_file(cmb_path)
            ok_all = ok_all and ok
            print(f"  {'OK' if ok else 'BAD'} CMB: {msg}")
        else:
            print("  INFO CMB: no curve saved for this operator")

        literature = iter_panel_constraint_files(op, cfg)
        print(f"  INFO literature overlays: {len(literature)} file(s)")
        print()
    return ok_all


def friendly_label(path: Path):
    stem = path.stem.lower()
    label = path.stem.replace("_", " ")
    color = "gray"
    linestyle = "-"

    for key, (pretty, c, ls) in STYLE_HINTS.items():
        if key in stem:
            label = pretty
            color = c
            linestyle = ls
            break

    return label, color, linestyle


def iter_constraint_files(subdir):
    if isinstance(subdir, (list, tuple)):
        paths = []
        for entry in subdir:
            paths.extend(iter_constraint_files(entry))
        return sorted({p.resolve(): p for p in paths}.values(), key=lambda p: str(p))
    folder = CONSTRAINT_DIR / subdir
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.glob("*.txt")
        if p.name.lower() != "readme.txt"
    )


def iter_panel_constraint_files(operator_key, cfg=None):
    limits = get_operator_limits(operator_key, include_generated=False)
    return [limit["path"] for limit in limits]


def data_driven_boundary_paths(cfg, *, halo_profile="rho2", source_tag="halo"):
    majorana_suffix = "_majorana" if cfg["majorana"] else ""
    stem = (
        f"mcmc_{halo_profile}_{source_tag}_{{kind}}_"
        f"{cfg['dm_type']}_{cfg['operator']}{majorana_suffix}_90cl.npz"
    )
    return {
        "raw_attenuation": BOUNDARY_DIR / stem.format(kind="raw_attenuation"),
        "spectral_reshaping": BOUNDARY_DIR / stem.format(kind="spectral_reshaping"),
    }


def data_driven_profile_style(halo_profile):
    if halo_profile in DATA_DRIVEN_PROFILE_STYLES:
        return DATA_DRIVEN_PROFILE_STYLES[halo_profile]
    return {
        "label": str(halo_profile).replace("_", " "),
        "color": THIS_WORK_COLOR,
        "raw_ls": (0, (5, 2)),
        "reshaping_ls": "-",
        "lw": 2.4,
    }


def load_default_omega_max() -> float:
    arr = np.loadtxt(str(DEFAULT_FERMI_SPECTRUM))
    return float(np.max(np.asarray(arr[:, 0], dtype=float)))


def get_s_max_lab_dmrest(mchi, omega_max):
    mchi = np.asarray(mchi, dtype=float)
    return mchi**2 + 2.0 * mchi * float(omega_max)


def get_t_abs_max_lab_dmrest(mchi, omega_max):
    mchi = np.asarray(mchi, dtype=float)
    denom = 1.0 + (2.0 * float(omega_max) / mchi)
    return 4.0 * float(omega_max) ** 2 / denom


def kinematic_eft_lambda_curve(mchi_arr, *, omega_max, eft_kinematic_factor):
    mchi_arr = np.asarray(mchi_arr, dtype=float)
    q2_max = np.maximum(
        get_s_max_lab_dmrest(mchi_arr, omega_max),
        get_t_abs_max_lab_dmrest(mchi_arr, omega_max),
    )
    return np.sqrt(float(eft_kinematic_factor) * q2_max)


def plot_panel(
    ax,
    operator_key,
    show_legend=False,
    data_driven_profiles=("rho2",),
    data_driven_source_tag="halo",
    title_override=None,
    title_fontsize=17,
    panel_label_fontsize=14,
    axis_label_fontsize=None,
    tick_labelsize=None,
    annotate_theory_guides=False,
    include_deconvolution_ceiling=True,
    validity_fill_color="cyan",
    validity_line_color=None,
    unitarity_line_color=None,
):
    cfg = PANEL_CONFIGS[operator_key]
    x_min, x_max = 1e-10, 1e12
    y_min, y_max = 1e-6, 1e8
    totani_path = BOUNDARY_DIR / cfg["totani_file"]
    # The legacy tension-scan boundary is optional. For the dipole operators the
    # legacy files were produced with direct-detection rather than real-photon
    # amplitudes and have been withdrawn; the panel draws the corrected
    # data-driven boundaries regardless.
    if totani_path.exists():
        totani = load_totani_boundary(totani_path)
    else:
        print(f"  [note] legacy boundary absent, skipping: {totani_path.name}")
        totani = None
    handles = []

    # ── york and cmb paths also need majorana suffix where applicable ────────
    majorana_suffix = "_majorana" if cfg["majorana"] else ""
    dm_type   = cfg["dm_type"]
    operator  = cfg["operator"]
    guide_color = validity_line_color or GUIDE_COLOR
    unitarity_color = unitarity_line_color or guide_color

    # Plot EFT validity guide line
    xgrid = np.logspace(np.log10(x_min), np.log10(x_max), 800)
    omega_max = load_default_omega_max()
    eft_factor = 1.0
    lam_kin = kinematic_eft_lambda_curve(xgrid, omega_max=omega_max, eft_kinematic_factor=eft_factor)
    kin_mask = np.isfinite(lam_kin) & (lam_kin >= y_min) & (lam_kin <= y_max)
    h_kin = None
    if np.any(kin_mask):
        h_kin, = ax.loglog(
            xgrid[kin_mask],
            lam_kin[kin_mask],
            color=guide_color,
            lw=1.4,
            ls="--",
            label="_nolegend_" if annotate_theory_guides else "EFT kinematic validity",
        )
        if not annotate_theory_guides:
            handles.append(set_legend_group(h_kin, "theory"))

    lam_unit = None
    unit_label = None
    if operator in ("dipole_magnetic", "dipole_electric"):
        lam_unit = np.sqrt(16 * np.pi * xgrid)
        unit_label = "Unitarity (dipole)"
    elif operator in ("charge_radius", "anapole"):
        lam_unit = (16 * np.pi * xgrid**2)**0.25
        unit_label = "Unitarity (dim-6)"
    elif "rayleigh" in operator:
        lam_unit = (128 * np.pi**2 * xgrid**2)**(1 / 6)
        unit_label = "Unitarity (Rayleigh)"

    if lam_unit is not None and np.any((lam_unit >= y_min) & (lam_unit <= y_max)):
        h_unit, = ax.loglog(
            xgrid,
            lam_unit,
            color=unitarity_color,
            lw=1.0,
            ls=":",
            label="_nolegend_" if annotate_theory_guides else unit_label,
        )
        if not annotate_theory_guides:
            handles.append(set_legend_group(h_unit, "theory"))

    # Plot physically valid region: above both the naive EFT guide and the unitarity guide.
    valid_floor = np.array(lam_kin, copy=True)
    if lam_unit is not None:
        valid_floor = np.maximum(valid_floor, lam_unit)
    valid_mask = np.isfinite(valid_floor) & (valid_floor < y_max) & (valid_floor > y_min)
    if np.any(valid_mask):
        ax.fill_between(
            xgrid[valid_mask],
            valid_floor[valid_mask],
            np.full(np.count_nonzero(valid_mask), y_max),
            color=validity_fill_color,
            alpha=0.2,
            zorder=0.5,
            label="_nolegend_",
        )
        if annotate_theory_guides:
            ax.text(
                0.03,
                0.93,
                "EFT valid",
                color=EFT_VALID_LABEL_COLOR,
                fontsize=7.0 if tick_labelsize is not None else 9.0,
                ha="left",
                va="top",
                transform=ax.transAxes,
            )
    if annotate_theory_guides and lam_unit is not None:
        unit_mask = np.isfinite(lam_unit) & (lam_unit > y_min) & (lam_unit < y_max)
        if np.any(unit_mask):
            x_unit = 2e7 if operator in ("dipole_magnetic", "dipole_electric") else 2e8
            x_unit = float(np.clip(x_unit, np.min(xgrid[unit_mask]), np.max(xgrid[unit_mask])))
            y_unit = 10.0 ** np.interp(
                np.log10(x_unit),
                np.log10(xgrid[unit_mask]),
                np.log10(lam_unit[unit_mask]),
            )
            ax.text(
                x_unit*80,
                y_unit*15,
                "Unitarity",
                color=unitarity_color,
                fontsize=7.0 if tick_labelsize is not None else 9.0,
                rotation=40,
                rotation_mode="anchor",
                ha="center",
                va="bottom",
            )

    # Plot literature constraints
    for limit in get_operator_limits(operator_key, include_generated=False):
        mchi = np.asarray(limit["data"]["mchi_GeV"], dtype=float)
        lam = np.asarray(limit["data"]["lambda_plot_GeV"], dtype=float)
        path = Path(limit["path"])
        stem_lower = path.stem.lower()

        if cfg["majorana"] and cfg["operator"] in ("rayleigh_even", "rayleigh_odd"):
            indirect_tags = ("fermi", "hess", "ams", "cta", "indirect")
            direct_tags = ("xenon", "lz", "lux", "pandax", "direct", "dd")
            if any(tag in stem_lower for tag in indirect_tags) and not any(
                tag in stem_lower for tag in direct_tags
            ):
                lam = lam * 2**(1/6)   # Majorana rate is 2x Dirac at fixed Lambda for annihilation bounds
        label = limit["label"]
        color = limit["color"]
        linestyle = limit["linestyle"]
        if len(mchi) > 0:
            h, = ax.loglog(mchi, lam, color=color, ls=linestyle, lw=1.6, label=label)
            group = "cosmology" if limit["constraint_type"] == "thermal_relic" else limit["constraint_type"]
            handles.append(set_legend_group(h, group))

    # Plot Totani boundary
    # totani_label = "Digitized Totani attenuation (legacy)"
    # if totani["scan_floor_limited"]:
    #     totani_label += " [scan-floor limited]"
    # mchi_totani, lambda_totani, _ = finite_positive_curve(totani["mchi"], totani["lambda_plot"])
    # h_totani, = ax.loglog(
    #     mchi_totani,
    #     lambda_totani,
    #     color=THIS_WORK_ALT,
    #     lw=2.5,
    #     label=totani_label,
    # )
    # handles.append(h_totani)

    # Data-driven limits from Totani_paper_check MCMC posteriors, if present.
    data_driven_specs = {
        "raw_attenuation": (
            "scattering attenuation",
            "raw_ls",
        ),
        "spectral_reshaping": (
            "scattering attenuation + spectral reshaping",
            "reshaping_ls",
        ),
    }
    plotted_eft_guide = False
    for halo_profile in data_driven_profiles:
        profile_style = data_driven_profile_style(halo_profile)
        for kind, path in data_driven_boundary_paths(
            cfg,
            halo_profile=halo_profile,
            source_tag=data_driven_source_tag,
        ).items():
            if not path.exists():
                continue
            curve = load_npz_boundary(path)
            if (
                not plotted_eft_guide
                and curve.get("omega_max_for_validity") is not None
                and curve.get("eft_kinematic_factor") is not None
            ):
                omega_max = float(curve["omega_max_for_validity"])
                eft_factor = float(curve["eft_kinematic_factor"])
                q2_max = np.maximum(
                    xgrid**2 + 2.0 * xgrid * omega_max,
                    4.0 * omega_max**2 / (1.0 + 2.0 * omega_max / xgrid),
                )
                lam_kin = np.sqrt(eft_factor * q2_max)
                kin_mask = np.isfinite(lam_kin) & (lam_kin > 0.0)
                if np.any(kin_mask):
                    h_kin, = ax.loglog(
                        xgrid[kin_mask],
                        lam_kin[kin_mask],
                        color=guide_color,
                        lw=1.4,
                        ls="--",
                        label="_nolegend_",
                    )
                    handles.append(h_kin)
                    plotted_eft_guide = True
            mchi_curve, lambda_curve, _ = finite_positive_curve(curve["mchi"], curve["lambda_plot"])
            if len(mchi_curve) == 0:
                continue
            kind_label, linestyle_key = data_driven_specs[kind]
            label = profile_style.get(
                f"{kind}_label",
                f"{profile_style['label']} 90% CL {kind_label}",
            )
            h_data, = ax.loglog(
                mchi_curve,
                lambda_curve,
                color=profile_style["color"],
                lw=profile_style["lw"],
                ls=profile_style[linestyle_key],
                label=label,
            )
            handles.append(set_legend_group(h_data, "this_work"))

    # --- York chi-squared constraint (10-deg GC circle, 50-500 GeV) ---
    york_path = BOUNDARY_DIR / f"york_{dm_type}_{operator}{majorana_suffix}_90cl.npz"
    if york_path.exists():
        york = load_npz_boundary(york_path)
        mchi_york, lambda_york, _ = finite_positive_curve(york["mchi"], york["lambda_plot"])
        if len(mchi_york) > 0:
            h_york, = ax.loglog(
                mchi_york,
                lambda_york,
                color=THIS_WORK_DEEP,
                lw=2.2,
                ls="-.",
                label="York GC spectrum (this work, 90% CL)",
            )
            handles.append(set_legend_group(h_york, "this_work"))

    # --- CMB power spectrum constraint (Planck 2018) ---
    cmb_path = BOUNDARY_DIR / f"cmb_{dm_type}_{operator}{majorana_suffix}_planck2018.npz"
    if cmb_path.exists():
        cmb = load_npz_boundary(cmb_path)
        mchi_cmb, lambda_cmb, _ = finite_positive_curve(cmb["mchi"], cmb["lambda_plot"])
        if len(mchi_cmb) > 0:
            h_cmb, = ax.loglog(
                mchi_cmb,
                lambda_cmb,
                color=BODDY_GLUSCEVIC_COLOR,
                lw=2.1,
                ls="-.",
                label="CMB (Boddy & Gluscevic 2018)",
            )
            handles.append(set_legend_group(h_cmb, "cosmology"))

    # Overlay generated detection boundaries as the top layer.
    for limit in get_operator_limits(operator_key, include_files=False, include_generated=True):
        if limit["constraint_type"] == "deconvolution" and not include_deconvolution_ceiling:
            continue
        if limit["constraint_type"] not in (
            "direct_detection",
            "indirect_detection",
            "collider",
            "deconvolution",
        ):
            continue
        mchi_det = limit["data"]["mchi_GeV"]
        lambda_det = limit["data"]["lambda_plot_GeV"]
        label = GENERATED_LIMIT_LABELS[limit["constraint_type"]]
        group = "theory" if limit["constraint_type"] == "deconvolution" else limit["constraint_type"]
        if limit["constraint_type"] == "deconvolution":
            if len(lambda_det) == 0 or not np.all(np.isfinite(lambda_det)):
                continue
            if np.allclose(lambda_det, lambda_det[0]):
                in_source_range = (xgrid >= np.nanmin(mchi_det)) & (xgrid <= np.nanmax(mchi_det))
                mchi_plot = xgrid[in_source_range]
                lambda_plot = np.full_like(mchi_plot, float(lambda_det[0]), dtype=float)
                valid_floor_plot = valid_floor[in_source_range]
            else:
                source_mask = np.isfinite(mchi_det) & np.isfinite(lambda_det) & (mchi_det > 0.0) & (lambda_det > 0.0)
                if np.count_nonzero(source_mask) < 2:
                    continue
                m_src = np.asarray(mchi_det[source_mask], dtype=float)
                l_src = np.asarray(lambda_det[source_mask], dtype=float)
                in_source_range = (xgrid >= np.min(m_src)) & (xgrid <= np.max(m_src))
                mchi_plot = xgrid[in_source_range]
                lambda_plot = 10.0 ** np.interp(np.log10(mchi_plot), np.log10(m_src), np.log10(l_src))
                valid_floor_plot = valid_floor[in_source_range]
            physical = (
                np.isfinite(mchi_plot)
                & np.isfinite(lambda_plot)
                & np.isfinite(valid_floor_plot)
                & (lambda_plot >= valid_floor_plot)
                & (lambda_plot >= y_min)
                & (lambda_plot <= y_max)
            )
            if not np.any(physical):
                continue
            mchi_det = mchi_plot[physical]
            lambda_det = lambda_plot[physical]
        h_generated, = ax.loglog(
            mchi_det,
            lambda_det,
            color=limit["color"],
            lw=2.2,
            alpha=0.95,
            zorder=30,
            label=label,
        )
        handles.append(set_legend_group(h_generated, group))

    ax.set_title(title_override or cfg["title"], fontsize=title_fontsize)
    if not annotate_theory_guides:
        panel_label = PAPER_LABEL_OVERRIDES.get(
            operator, totani["paper_label"] if totani is not None else operator)
        ax.text(0.03, 0.93, panel_label, transform=ax.transAxes, fontsize=panel_label_fontsize, va="top")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", alpha=0.22)
    ax.set_xlabel(r"$m_{\chi}$ [GeV]", fontsize=axis_label_fontsize)
    ax.set_ylabel(r"$\Lambda/C^{1/n}$ [GeV]", fontsize=axis_label_fontsize)
    if tick_labelsize is not None:
        ax.tick_params(axis="both", which="both", labelsize=tick_labelsize)

    return handles


def main():
    parser = argparse.ArgumentParser(description="Overlay Totani EFT boundaries with curves in constraints_data/")
    parser.add_argument(
        "--operators",
        nargs="+",
        default=None,
        choices=sorted(PANEL_CONFIGS.keys()),
        help="Operators to plot. Default shows the supported dipole, anapole, charge-radius, Dirac rayleigh-full, and scalar-rayleigh panels.",
    )
    parser.add_argument(
        "--outfile",
        default="totani_operator_overlays",
        help="Output basename inside Totani_Scattering/plots/",
    )
    parser.add_argument(
        "--data-driven-profiles",
        nargs="+",
        default=None,
        help=(
            "MCMC halo profiles to overlay from Totani_Scattering/constraint_boundaries. "
            "Use e.g. rho2 global_rho2 global_rho2.5 to compare Totani and global morphology constraints."
        ),
    )
    parser.add_argument(
        "--data-driven-source",
        default="halo",
        help="Source tag used in data-driven boundary filenames. Default: halo.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print a scientific/reproducibility check of the requested operator curves before plotting.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run --check and exit without writing a figure.",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="Plot style: paper, conference/dark_transparent, or conference_light/light_transparent.",
    )
    parser.add_argument(
        "--paper-summary",
        action="store_true",
        help="Make the compact 3-panel LaTeX-paper version: magnetic dipole, Majorana anapole/axial charge radius, scalar Rayleigh.",
    )
    # Print-geometry overrides for PRD revtex4 column widths
    parser.add_argument("--fig-width", type=float, default=None,
                        help="Figure width [inches]. Overrides internal figsize.")
    parser.add_argument("--fig-height", type=float, default=None,
                        help="Figure height [inches]. Overrides internal figsize.")
    parser.add_argument("--base-fontsize", type=float, default=None,
                        help="Base font size for set_paper_style. Overrides internal default (10).")
    parser.add_argument("--linewidth", type=float, default=None,
                        help="Line width for set_paper_style. Overrides internal default (1.6).")
    args = parser.parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style
    if args.operators is None:
        args.operators = PAPER_SUMMARY_OPERATORS if args.paper_summary else DEFAULT_OPERATORS
    if args.data_driven_profiles is None:
        args.data_driven_profiles = (
            ["pixelwise_global_rho2", "pixelwise_global_rho2.5"]
            if args.paper_summary
            else ["rho2"]
        )
    if args.paper_summary and args.outfile == "totani_operator_overlays":
        args.outfile = "totani_operator_overlays_paper_summary"

    if args.check or args.check_only:
        ok = print_pipeline_check(args.operators)
        if args.check_only:
            return 0 if ok else 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_fs   = args.base_fontsize if args.base_fontsize is not None else 10
    linewidth = args.linewidth     if args.linewidth     is not None else 1.6
    set_paper_style(base_fontsize=base_fs, linewidth=linewidth, n_colors=14, cmap_name="plasma")

    n = len(args.operators)
    ncols = 3 if args.paper_summary else 4
    nrows = 1 if args.paper_summary else 2
    nslots = ncols * nrows
    if n > nslots:
        layout = "3-panel paper-summary" if args.paper_summary else "fixed 4x2"
        raise ValueError(f"Requested {n} panels, but the {layout} layout only supports {nslots}.")

    if args.fig_width is not None and args.fig_height is not None:
        figsize = (args.fig_width, args.fig_height)
    else:
        figsize = (7.2, 3.25) if args.paper_summary else (6.2 * ncols, 4.9 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    all_handles = []
    # Font sizes scale with base_fs so this figure matches the sensitivity map
    # and multi-dataset overlay at the same PRD width. Paper-summary uses the
    # base sizing; the standalone full-grid mode inflates for a larger canvas.
    if args.paper_summary:
        panel_title_fs = base_fs + 1
        panel_label_fs = base_fs - 2
        panel_axlbl_fs = base_fs
        panel_tick_fs  = base_fs - 2
    else:
        panel_title_fs = 17
        panel_label_fs = 14
        panel_axlbl_fs = None
        panel_tick_fs  = None
    for i, op in enumerate(args.operators):
        handles = plot_panel(
            axes[i],
            op,
            show_legend=False,
            data_driven_profiles=args.data_driven_profiles,
            data_driven_source_tag=args.data_driven_source,
            title_override=PAPER_SUMMARY_TITLES.get(op) if args.paper_summary else None,
            title_fontsize=panel_title_fs,
            panel_label_fontsize=panel_label_fs,
            axis_label_fontsize=panel_axlbl_fs,
            tick_labelsize=panel_tick_fs,
            annotate_theory_guides=args.paper_summary,
            include_deconvolution_ceiling=not args.paper_summary,
        )
        all_handles.extend(handles)

    legend_ax = None
    if args.paper_summary:
        for j in range(n, len(axes)):
            axes[j].set_axis_off()
    elif n < nslots:
        legend_ax = axes[n]
        legend_ax.set_axis_off()
        for j in range(n + 1, len(axes)):
            axes[j].set_axis_off()
    else:
        for j in range(n, len(axes)):
            axes[j].set_axis_off()

    # Axis labels only on bottom row (x) and leftmost column (y). Same
    # convention as the sensitivity map and multi-dataset overlay so all
    # multi-panel figures read consistently.
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

    if not args.paper_summary:
        fig.suptitle("Totani halo attenuation constraints in EFT planes", fontsize=24, y=0.985)

    grouped_handles = collect_grouped_legend(all_handles)
    if legend_ax is not None:
        legend_layout = [
            ("direct_detection", 0.02, 0.97, 0.46, 0.0),
            ("indirect_detection", 0.52, 0.97, 0.46, 0.0),
            ("collider", 0.02, 0.46, 0.46, 0.0),
            ("cosmology", 0.02, 0.27, 0.46, 0.0),
            ("theory", 0.02, 0.08, 0.46, 0.0),
            ("this_work", 0.52, 0.24, 0.46, 0.0),
        ]
        for group, x_anchor, y_anchor, width, height in legend_layout:
            handles = grouped_handles.get(group, [])
            if not handles:
                continue
            labels = [h.get_label() for h in handles]
            meta = LEGEND_GROUPS[group]
            leg = legend_ax.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(x_anchor, y_anchor, width, height),
                bbox_transform=legend_ax.transAxes,
                ncol=1,
                fontsize=base_fs - 2,
                title=meta["label"],
                title_fontsize=base_fs - 1,
                borderaxespad=0.0,
                borderpad=0.4,
                handlelength=2.0,
                handletextpad=0.45,
                labelspacing=0.22,
                **LEGEND_KW,
            )
            leg.get_title().set_color(meta["color"])
            legend_ax.add_artist(leg)
    else:
        draw_bottom_grouped_legend(fig, grouped_handles, compact=args.paper_summary, base_fs=base_fs)

    if args.paper_summary:
        # Compact bottom strip (0.20) for the single boxed legend row.
        # wspace=0.14 gives two-line panel titles breathing room.
        plt.tight_layout(rect=[0.01, 0.20, 1, 0.96], w_pad=0.6)
        fig.subplots_adjust(wspace=0.14, hspace=0.12)
    else:
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_figure(fig, str(OUTPUT_DIR / args.outfile))
    # Also save a PDF copy for direct LaTeX inclusion.
    fig.savefig(
        str(OUTPUT_DIR / f"{args.outfile}.pdf"),
        bbox_inches="tight",
        transparent=bool(plt.rcParams.get("savefig.transparent", False)),
        facecolor=plt.rcParams.get("savefig.facecolor", "auto"),
        edgecolor=plt.rcParams.get("savefig.edgecolor", "auto"),
    )
    plt.close(fig)
    print(f"Saved overlay figure: {OUTPUT_DIR / args.outfile}.{{png,pdf}}")


if __name__ == "__main__":
    raise SystemExit(main())
