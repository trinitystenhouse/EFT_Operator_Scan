"""Central registry for digitised and generated constraint curves.

Each operator key maps to one dictionary containing operator metadata and a
list of limit dictionaries.  Limit entries carry their constraint category,
plot style, source path, and the loaded two-column data.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DETECTION_DIR = BASE_DIR / "detection"

INDIRECT_COLOR = "#9B6DFF"
DIRECT_COLOR = "#F2D53C"
COLLIDER_COLOR = "#FF7AC6"
COSMOLOGY_COLOR = "#71D6FF"
BODDY_GLUSCEVIC_COLOR = "#71D6FF"
THERMAL_COLOR = "#7DDC84"
THEORY_COLOR = "#8B93A4"


OPERATOR_SPECS = {
    "dipole_magnetic": {
        "operator": "dipole_magnetic",
        "operator_type": "magnetic_dipole",
        "operator_label": "Magnetic Dipole",
        "dm_type": "fermionic",
        "fermion_type": "dirac",
        "majorana": False,
        "folders": ("magdipole",),
    },
    "dipole_electric": {
        "operator": "dipole_electric",
        "operator_type": "electric_dipole",
        "operator_label": "Electric Dipole",
        "dm_type": "fermionic",
        "fermion_type": "dirac",
        "majorana": False,
        "folders": ("eldipole",),
    },
    "charge_radius": {
        "operator": "charge_radius",
        "operator_type": "charge_radius",
        "operator_label": "Charge Radius",
        "dm_type": "fermionic",
        "fermion_type": "dirac",
        "majorana": False,
        "folders": ("chargeradius",),
    },
    "anapole": {
        "operator": "anapole",
        "operator_type": "anapole",
        "operator_label": "Anapole",
        "dm_type": "fermionic",
        "fermion_type": "dirac",
        "majorana": False,
        "folders": ("anapole",),
    },
    "anapole_majorana": {
        "operator": "anapole",
        "operator_type": "anapole",
        "operator_label": "Anapole",
        "dm_type": "fermionic",
        "fermion_type": "majorana",
        "majorana": True,
        "folders": ("anapole",),
    },
    "rayleigh_even": {
        "operator": "rayleigh_even",
        "operator_type": "rayleigh_even",
        "operator_label": "Rayleigh Even",
        "dm_type": "fermionic",
        "fermion_type": "dirac",
        "majorana": False,
        "folders": ("rayleigh_even",),
    },
    "rayleigh_odd": {
        "operator": "rayleigh_odd",
        "operator_type": "rayleigh_odd",
        "operator_label": "Rayleigh Odd",
        "dm_type": "fermionic",
        "fermion_type": "dirac",
        "majorana": False,
        "folders": ("rayleigh_odd",),
    },
    "rayleigh_full": {
        "operator": "rayleigh_full",
        "operator_type": "rayleigh_full",
        "operator_label": "Rayleigh Full",
        "dm_type": "fermionic",
        "fermion_type": "dirac",
        "majorana": False,
        "folders": ("rayleigh_even", "rayleigh_odd"),
    },
    "rayleigh_even_majorana": {
        "operator": "rayleigh_even",
        "operator_type": "rayleigh_even",
        "operator_label": "Rayleigh Even",
        "dm_type": "fermionic",
        "fermion_type": "majorana",
        "majorana": True,
        "folders": ("rayleigh_even",),
    },
    "rayleigh_odd_majorana": {
        "operator": "rayleigh_odd",
        "operator_type": "rayleigh_odd",
        "operator_label": "Rayleigh Odd",
        "dm_type": "fermionic",
        "fermion_type": "majorana",
        "majorana": True,
        "folders": ("rayleigh_odd",),
    },
    "scalar_rayleigh": {
        "operator": "scalar_rayleigh",
        "operator_type": "scalar_rayleigh",
        "operator_label": "Scalar Rayleigh",
        "dm_type": "scalar",
        "fermion_type": None,
        "majorana": False,
        "folders": ("rayleigh_scalar",),
    },
}


STYLE_HINTS = {
    "xenon1t": ("XENON1T", "#C9A400", "-", "direct_detection"),
    "xenonnt_anapole_majorana": ("XENONnT", "#E2BE00", "--", "direct_detection"),
    "xenonnt": ("XENONnT", "#E2BE00", "--", "direct_detection"),
    "hambye": ("XENON1T anapole recast", "#C9A400", "-", "direct_detection"),
    "ibarra2024_lz": ("LZ", "#FFB347", "-.", "direct_detection"),
    "lz2022": ("LZ", "#FFB347", "-.", "direct_detection"),
    "lz_magdipole": ("LZ", "#FFB347", "-.", "direct_detection"),
    "lz_eldipole": ("LZ", "#FFB347", (0, (7, 2, 1.5, 2)), "direct_detection"),
    "xlzd200ty": ("XLZD 200 ty", "#FFE68A", (0, (2, 2)), "direct_detection"),
    "fortin_tait": ("Fortin & Tait", "#C9A400", (0, (3, 1)), "direct_detection"),
    "monojet": ("Monojet", "#E754A8", "-", "collider"),
    "lhc_monojet": ("LHC mono-jet", "#FF5CB8", "-.", "collider"),
    "lep_zdecay": ("LEP Z-decay", "#FF9AD5", (0, (3, 2)), "collider"),
    "ams02": ("AMS-02", "#C7A6FF", (0, (1, 1)), "indirect_detection"),
    "fermilat_hess": ("Fermi-LAT + H.E.S.S.", "#5E35B1", "-", "indirect_detection"),
    "fermilat": ("Fermi-LAT", "#8E5BFF", "-", "indirect_detection"),
    "fermi_lines": ("Fermi lines", "#8E5BFF", (0, (3, 2)), "indirect_detection"),
    "fermi_single_line": ("Fermi single line", "#8E5BFF", ":", "indirect_detection"),
    "fermi_double_line": ("Fermi double line", "#8E5BFF", "-.", "indirect_detection"),
    "hess": ("H.E.S.S.", "#7C4DFF", (0, (7, 2)), "indirect_detection"),
    "cta_gc": ("CTA GC", "#7248D6", ":", "indirect_detection"),
    "cta_dsphs": ("CTA dSphs", "#7248D6", "--", "indirect_detection"),
    "planck": ("Planck", COSMOLOGY_COLOR, (0, (4, 2)), "cosmology"),
    "thermal_relic": ("Thermal relic", THERMAL_COLOR, (0, (1, 1)), "thermal_relic"),
}


GENERATED_LIMIT_SPECS = {
    "DD_MD": {
        "operators": ("dipole_magnetic",),
        "label": "Direct detection",
        "constraint_type": "direct_detection",
        "color": DIRECT_COLOR,
        "linestyle": "-",
    },
    "DD_ED": {
        "operators": ("dipole_electric",),
        "label": "Direct detection",
        "constraint_type": "direct_detection",
        "color": DIRECT_COLOR,
        "linestyle": "-",
    },
    "DD_A": {
        "operators": ("anapole", "anapole_majorana"),
        "label": "Direct detection",
        "constraint_type": "direct_detection",
        "color": DIRECT_COLOR,
        "linestyle": "-",
    },
    "DD_CR": {
        "operators": ("charge_radius",),
        "label": "Direct detection",
        "constraint_type": "direct_detection",
        "color": DIRECT_COLOR,
        "linestyle": "-",
    },
    "ID_Rayleigh": {
        "operators": (
            "rayleigh_even",
            "rayleigh_odd",
            "rayleigh_full",
            "rayleigh_even_majorana",
            "rayleigh_odd_majorana",
        ),
        "label": "Indirect detection",
        "constraint_type": "indirect_detection",
        "color": INDIRECT_COLOR,
        "linestyle": "-",
    },
    "ID_SR": {
        "operators": ("scalar_rayleigh",),
        "label": "Indirect detection",
        "constraint_type": "indirect_detection",
        "color": INDIRECT_COLOR,
        "linestyle": "-",
    },
    "Coll_Ray": {
        "operators": (
            "rayleigh_even",
            "rayleigh_odd",
            "rayleigh_full",
            "rayleigh_even_majorana",
            "rayleigh_odd_majorana",
        ),
        "label": "Collider",
        "constraint_type": "collider",
        "color": COLLIDER_COLOR,
        "linestyle": "-",
    },
    "Coll_MD": {
        "operators": ("dipole_magnetic",),
        "label": "Collider",
        "constraint_type": "collider",
        "color": COLLIDER_COLOR,
        "linestyle": "-",
    },
    "Coll_A": {
        "operators": ("anapole", "anapole_majorana"),
        "label": "Collider",
        "constraint_type": "collider",
        "color": COLLIDER_COLOR,
        "linestyle": "-",
    },
    "deconv": {
        "operators": tuple(OPERATOR_SPECS.keys()),
        "label": "Deconvolution ceiling",
        "constraint_type": "deconvolution",
        "color": THEORY_COLOR,
        "linestyle": "--",
        "m_key": "m_dm_vals_deconv",
        "y_key": "y_vals_deconv",
    },
    "unit": {
        "operators": tuple(OPERATOR_SPECS.keys()),
        "label": "Unitarity guide",
        "constraint_type": "theory",
        "color": THEORY_COLOR,
        "linestyle": ":",
        "m_key": "m_dm_vals_unit",
        "y_key": "y_vals_unit",
    },
}


def np_array_flexible(*args, **kwargs):
    """Support detection files that use np.array(a, b, c) instead of a list."""
    if len(args) > 1:
        return np.array(list(args), **kwargs)
    return np.array(*args, **kwargs)


def finite_positive_curve(mchi, lambda_plot):
    mchi = np.asarray(mchi, dtype=float)
    lambda_plot = np.asarray(lambda_plot, dtype=float)
    mask = np.isfinite(mchi) & np.isfinite(lambda_plot) & (mchi > 0.0) & (lambda_plot > 0.0)
    return mchi[mask], lambda_plot[mask]


def load_text_curve(path: Path):
    arr = np.loadtxt(path, comments="#", delimiter=None)
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 2:
        raise ValueError(f"{path} must have at least two columns: mchi_GeV lambda_plot_GeV")
    return finite_positive_curve(arr[:, 0], arr[:, 1])


def _style_for_path(path: Path):
    stem = path.stem.lower()
    label = path.stem.replace("_", " ")
    color = THEORY_COLOR
    linestyle = "-"
    constraint_type = "other"

    for key, style in STYLE_HINTS.items():
        if key in stem:
            label, color, linestyle, constraint_type = style
            break

    return label, color, linestyle, constraint_type


def _text_limit_entry(path: Path, operator_key: str, spec: dict):
    label, color, linestyle, constraint_type = _style_for_path(path)
    mchi, lambda_plot = load_text_curve(path)
    return {
        "name": path.stem,
        "label": label,
        "operator_key": operator_key,
        "operator": spec["operator"],
        "operator_type": spec["operator_type"],
        "operator_label": spec["operator_label"],
        "dm_type": spec["dm_type"],
        "fermion_type": spec["fermion_type"],
        "majorana": spec["majorana"],
        "constraint_type": constraint_type,
        "color": color,
        "linestyle": linestyle,
        "source_kind": "text_file",
        "path": path,
        "data": {
            "mchi_GeV": mchi,
            "lambda_plot_GeV": lambda_plot,
        },
    }


def _load_detection_namespaces():
    namespaces = {}
    for py_file in sorted(DETECTION_DIR.glob("*.py")):
        namespace = {"np": np, "np_array_flexible": np_array_flexible}
        source = py_file.read_text()
        source = source.replace("np.array(", "np_array_flexible(")
        exec(source, namespace, namespace)
        namespaces[py_file.stem] = namespace
    return namespaces


def _generated_curve_source(suffix: str):
    for namespace_name, namespace in _load_detection_namespaces().items():
        spec = GENERATED_LIMIT_SPECS[suffix]
        m_key = spec.get("m_key", f"m_dm_vals_{suffix}")
        y_key = spec.get("y_key", f"y_vals_{suffix}")
        if m_key in namespace and y_key in namespace:
            return namespace_name, namespace[m_key], namespace[y_key]
    return None, None, None


def _generated_limit_entry(suffix: str, operator_key: str, operator_spec: dict, generated_spec: dict):
    namespace_name, mchi_values, lambda_values = _generated_curve_source(suffix)
    if mchi_values is None or lambda_values is None:
        return None
    mchi, lambda_plot = finite_positive_curve(mchi_values, lambda_values)
    if len(mchi) == 0:
        return None
    return {
        "name": suffix,
        "label": generated_spec["label"],
        "operator_key": operator_key,
        "operator": operator_spec["operator"],
        "operator_type": operator_spec["operator_type"],
        "operator_label": operator_spec["operator_label"],
        "dm_type": operator_spec["dm_type"],
        "fermion_type": operator_spec["fermion_type"],
        "majorana": operator_spec["majorana"],
        "constraint_type": generated_spec["constraint_type"],
        "color": generated_spec["color"],
        "linestyle": generated_spec["linestyle"],
        "source_kind": "generated_python",
        "path": DETECTION_DIR / f"{namespace_name}.py",
        "data": {
            "mchi_GeV": mchi,
            "lambda_plot_GeV": lambda_plot,
        },
    }


def _operator_text_files(spec: dict):
    paths = []
    for folder_name in spec["folders"]:
        folder = BASE_DIR / folder_name
        if not folder.exists():
            continue
        paths.extend(
            p
            for p in sorted(folder.glob("*.txt"))
            if p.name.lower() != "readme.txt"
        )

    unique_paths = sorted({p.resolve(): p for p in paths}.values(), key=lambda p: str(p))
    if spec["fermion_type"] != "majorana":
        unique_paths = [p for p in unique_paths if "majorana" not in p.stem.lower()]
    return unique_paths


def _build_limits_by_operator():
    limits_by_operator = {}
    for operator_key, spec in OPERATOR_SPECS.items():
        operator_limits = []

        for path in _operator_text_files(spec):
            operator_limits.append(_text_limit_entry(path, operator_key, spec))

        for suffix, generated_spec in GENERATED_LIMIT_SPECS.items():
            if operator_key not in generated_spec["operators"]:
                continue
            entry = _generated_limit_entry(suffix, operator_key, spec, generated_spec)
            if entry is not None:
                operator_limits.append(entry)

        limits_by_operator[operator_key] = {
            "operator_key": operator_key,
            "operator": spec["operator"],
            "operator_type": spec["operator_type"],
            "operator_label": spec["operator_label"],
            "dm_type": spec["dm_type"],
            "fermion_type": spec["fermion_type"],
            "majorana": spec["majorana"],
            "limits": operator_limits,
        }
    return limits_by_operator


LIMITS_BY_OPERATOR = _build_limits_by_operator()


def get_operator_limits(
    operator_key: str,
    *,
    operator_type: str | None = None,
    fermion_type: str | None = None,
    constraint_type: str | None = None,
    include_files: bool = True,
    include_generated: bool = True,
):
    """Return limit dictionaries for one operator, filtered by metadata."""
    operator_limits = LIMITS_BY_OPERATOR[operator_key]["limits"]
    selected = []
    for limit in operator_limits:
        if operator_type is not None and limit["operator_type"] != operator_type:
            continue
        if fermion_type is not None and limit["fermion_type"] != fermion_type:
            continue
        if constraint_type is not None and limit["constraint_type"] != constraint_type:
            continue
        if not include_files and limit["source_kind"] == "text_file":
            continue
        if not include_generated and limit["source_kind"] == "generated_python":
            continue
        selected.append(deepcopy(limit))
    return selected


def find_limits(
    *,
    operator_type: str | None = None,
    fermion_type: str | None = None,
    constraint_type: str | None = None,
    include_files: bool = True,
    include_generated: bool = True,
):
    """Search all operator dictionaries by operator, fermion, and constraint type."""
    selected = []
    for operator_key in LIMITS_BY_OPERATOR:
        selected.extend(
            get_operator_limits(
                operator_key,
                operator_type=operator_type,
                fermion_type=fermion_type,
                constraint_type=constraint_type,
                include_files=include_files,
                include_generated=include_generated,
            )
        )
    return selected
