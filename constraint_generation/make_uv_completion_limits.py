#!/usr/bin/env python3
r"""
make_uv_completion_limits.py
============================

Translate the EFT-plane photon-DM scattering limits produced by
``make_data_driven_scattering_limits.py`` into limits on the *physical*
parameters of two UV completions, and overlay them with the orthogonal
constraints those completions already face.

This is the "UV-complete constraints" section of the paper: instead of using the
Totani halo morphology to set a phenomenological EFT limit, we ask what a given
UV theory must satisfy and where the scattering channel lands relative to
collider / EDM / dark-photon bounds.

Two completions are implemented (Higgs portal is done separately):

  1. Charged-messenger loop  ->  Dirac magnetic / electric dipole (dim-5)
  2. Kinetic-mixing dark U(1)' ->  charge radius / anapole (dim-6)

For each theory the script makes two kinds of plot:

  * M_mediator vs m_chi   (coupling fixed at a benchmark)
  * g (coupling) vs M_mediator  (m_chi fixed at a benchmark)

with the scattering exclusion translated into the same plane and the relevant
external limits overlaid.

PLACEMENT
---------
This file belongs in:
    Totani_Scattering/constraint_generation/make_uv_completion_limits.py
It imports from core/ and helpers/ via the same sys.path trick as the other
constraint_generation scripts, so drop it in that folder and run from
Totani_Scattering/.

------------------------------------------------------------------------------
DERIVATION OF THE WILSON-COEFFICIENT MATCHING
------------------------------------------------------------------------------

Operator normalisation (read off from core/attenuation_eft.py):

  dipoles      |M|^2 = 4 alpha (c^2 / Lambda^2) (-t)
  charge rad./ |M|^2 = 4 alpha (c^2 / Lambda^4) (t^2)
  anapole

The explicit factor of alpha = e^2/(4 pi) means the photon field strength
F_{mu nu} in the operators is canonically normalised and the photon coupling e
sits in the amplitude. With this convention the operators are

  O_MD = (c_M / Lambda) chibar sigma^{mu nu} chi F_{mu nu}        (dim-5)
  O_ED = (c_E / Lambda) chibar i sigma^{mu nu} gamma^5 chi F_{mu nu} (dim-5)
  O_CR = (c_r / Lambda^2) chibar gamma^mu chi  d^nu F_{mu nu}     (dim-6)
  O_AP = (c_a / Lambda^2) chibar gamma^mu gamma^5 chi d^nu F_{mu nu} (dim-6)

The benchmark in the scan is c_M = c_E = c_r = c_a = 1, so the saved
``lambda_GeV`` column is directly Lambda with c = 1.

(1) Charged-messenger loop -> Dirac dipoles
-------------------------------------------
Model: heavy vector-like fermion F (mass M_F, charge Q) and heavy scalar S
(charge -Q so the DM is neutral), Yukawa-coupled to Dirac DM chi:

    L = lambda ( chibar P_R F S^dagger + h.c. )

The photon attaches to the charged F and S in a one-loop triangle. This is the
standard chirality-flipping dipole loop (identical structure to lepton g-2 and
the Hisano et al. neutralino dipole). In the m_chi << M_F, M_S limit the induced
magnetic dipole moment is

    mu_chi = (e Q |lambda|^2)/(16 pi^2) * (m_chi / M_F^2) * F(x),   x = M_S^2/M_F^2

with loop function

    F(x) = (x^2 - 1 - 2 x ln x) / (2 (x-1)^3),    F(1) = 1/6.

The dipole is chirality-flipping, hence the explicit factor m_chi (one mass
insertion on the external DM line). Writing the dipole Lagrangian as
L = -(mu_chi/2) chibar sigma chi F, matching to O_MD with c_M = 1 gives

    1/Lambda_MD = (mu_chi/2)/c_M = (e Q |lambda|^2)/(32 pi^2) (m_chi/M_F^2) F(x)

  =>  Lambda_MD(M_F, lambda, x; m_chi) = 32 pi^2 M_F^2 / [ e Q |lambda|^2 m_chi F(x) ].

Because the matched Lambda depends on m_chi, the translation to the (M_F, m_chi)
plane is NOT a constant rescaling -- it is applied point-by-point along the
scattering boundary.

Electric dipole: identical loop, but needs a relative phase between left- and
right-handed couplings. For |lambda_L| = |lambda_R| = lambda,
  mu_chi ~ Re(lambda_L lambda_R^*),  d_chi ~ Im(lambda_L lambda_R^*),
with the same F(x) and prefactor. The maximal-CP benchmark (phase = pi/2) gives
|1/Lambda_ED| equal in magnitude to the magnetic case, so the SAME map is used;
the physical difference downstream is that the electric dipole is bounded by
EDMs.

(2) Kinetic-mixing dark U(1)' -> charge radius / anapole
--------------------------------------------------------
Model: Dirac DM chi with dark charge q_chi = 1 under a dark U(1)' with gauge
coupling g_D and a massive dark photon A' (mass m_A') that kinetically mixes
with the photon with parameter epsilon:

    L = -(epsilon/2) F'_{mu nu} F^{mu nu} + g_D chibar gamma^mu chi A'_mu
        + (1/2) m_A'^2 A'_mu A'^mu .

After de-mixing, A' couples to the EM current with strength epsilon e. Integrating
out the heavy A' gives the current-current operator

    (g_D)(epsilon e)/m_A'^2  chibar gamma^mu chi  J^EM_mu .

The neutral, vector-current DM couples to an on-shell photon only through the
charge radius. Matching the q^2 -> 0 expansion of the tree exchange to O_CR
(Feynman rule contributes (c_r/Lambda^2) q^2 to the same current structure):

    c_r/Lambda^2 = g_D epsilon e / m_A'^2
  =>  Lambda_CR(m_A', epsilon, g_D) = m_A' / sqrt(g_D epsilon e)   (c_r = 1).

The anapole follows identically for an axial dark coupling
g_D chibar gamma^mu gamma^5 chi A'_mu:

    Lambda_AP(m_A', epsilon, g_D) = m_A' / sqrt(g_D epsilon e).

This map is independent of m_chi (dim-6, not chirality-flipping), so it IS a
constant rescaling of the scan axis at fixed (epsilon, g_D).

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------

    python constraint_generation/make_uv_completion_limits.py \
        --theory dipole --halo-profile rho2 --source-tag halo

    python constraint_generation/make_uv_completion_limits.py \
        --theory darkphoton --halo-profile rho2 --source-tag halo

    python constraint_generation/make_uv_completion_limits.py \
        --theory all --include-electric --include-charge-radius

Outputs (per theory) in Totani_Scattering/plots/:
    uv_<theory>_Mmed_vs_mchi.png/.pdf
    uv_<theory>_g_vs_Mmed.png/.pdf
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_WORKSPACE = _ROOT.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_WORKSPACE))

from helpers.trinity_plotting import save_figure, set_plot_style  # noqa: E402

BOUNDARY_DIR = _ROOT / "constraint_boundaries"
PLOTDIR = _ROOT / "plots"

# ---------------------------------------------------------------------------
# Physical constants (match core/attenuation_eft.py conventions)
# ---------------------------------------------------------------------------
ALPHA_EM = 1.0 / 137.035999084
E_EM = np.sqrt(4.0 * np.pi * ALPHA_EM)   # ~ 0.3028, the QED gauge coupling

# Colour scheme borrowed from make_paper_style_operator_overlays.py
COL_SCATTER = "black"
COL_COLLIDER = "#FF7AC6"
COL_EDM = "#F2D53C"
COL_DARKPHOTON = "#71D6FF"
COL_RELIC = "#7DDC84"
COL_PERTURB = "#8B93A4"
COL_GUIDE = "#B18CFF"

UV_OPERATORS = {
    "dipole_magnetic",
    "dipole_electric",
    "charge_radius",
    "anapole",
}

BOUNDARY_STYLE_HINTS = (
    ("mcmc", "Totani halo scattering", COL_SCATTER, "-"),
    ("totani_halo_exclusion", "Totani halo exclusion", "#E8EEF9", (0, (6, 2))),
    ("totani_", "Totani spectral limit", "#C9D4EA", (0, (3, 2))),
    ("cmb", "CMB", COL_DARKPHOTON, (0, (4, 2))),
    ("fermi_naive", "Fermi naive attenuation", "#B18CFF", ":"),
    ("deconv_tau", "Deconvolution tau limit", "#8B93A4", "--"),
    ("deconv_tension", "Deconvolution tension limit", "#6D7484", "-."),
)


# ===========================================================================
# 1.  Wilson-coefficient matching maps
# ===========================================================================

def dipole_loop_function(x: np.ndarray | float) -> np.ndarray:
    r"""F(x) = (x^2 - 1 - 2 x ln x) / (2 (x-1)^3), with F(1) = 1/6.

    x = M_S^2 / M_F^2.  Numerically stable near x = 1 via a series expansion.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    near = np.abs(x - 1.0) < 1e-3
    far = ~near
    if np.any(far):
        xf = x[far]
        out[far] = (xf**2 - 1.0 - 2.0 * xf * np.log(xf)) / (2.0 * (xf - 1.0) ** 3)
    if np.any(near):
        # Taylor around x = 1:  F = 1/6 - (x-1)/6 + 3(x-1)^2/20 - ...
        d = x[near] - 1.0
        out[near] = 1.0 / 6.0 - d / 6.0 + 3.0 * d**2 / 20.0
    return out


def lambda_dipole_from_uv(M_F, lam, m_chi, *, x_ratio=1.0, Q=1.0):
    r"""Matched EFT scale Lambda for the charged-messenger dipole completion.

    Lambda_MD = 32 pi^2 M_F^2 / ( e Q |lambda|^2 m_chi F(x) ).

    All masses in GeV.  Returns Lambda in GeV (with c_M = 1).
    Inputs broadcast against each other.
    """
    M_F = np.asarray(M_F, dtype=float)
    lam = np.asarray(lam, dtype=float)
    m_chi = np.asarray(m_chi, dtype=float)
    F = dipole_loop_function(x_ratio)
    num = 32.0 * np.pi**2 * M_F**2
    den = E_EM * float(Q) * lam**2 * m_chi * F
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def uv_dipole_from_lambda(Lambda, m_chi, *, x_ratio=1.0, Q=1.0, solve_for="M_F", lam=None, M_F=None):
    r"""Invert the dipole map.

    The matching is  1/Lambda = (e Q lam^2 m_chi F)/(32 pi^2 M_F^2).

    solve_for="M_F": given (Lambda, m_chi, lam) return M_F
        M_F = sqrt( e Q lam^2 m_chi F Lambda / (32 pi^2) )
    solve_for="lam": given (Lambda, m_chi, M_F) return lambda
        lam = sqrt( 32 pi^2 M_F^2 / (e Q m_chi F Lambda) )
    """
    Lambda = np.asarray(Lambda, dtype=float)
    m_chi = np.asarray(m_chi, dtype=float)
    F = dipole_loop_function(x_ratio)
    if solve_for == "M_F":
        lam = np.asarray(lam, dtype=float)
        val = E_EM * float(Q) * lam**2 * m_chi * F * Lambda / (32.0 * np.pi**2)
        with np.errstate(invalid="ignore"):
            return np.sqrt(val)
    elif solve_for == "lam":
        M_F = np.asarray(M_F, dtype=float)
        val = 32.0 * np.pi**2 * M_F**2 / (E_EM * float(Q) * m_chi * F * Lambda)
        with np.errstate(invalid="ignore"):
            return np.sqrt(val)
    raise ValueError("solve_for must be 'M_F' or 'lam'")


def lambda_darkphoton_from_uv(m_Ap, eps, g_D):
    r"""Matched EFT scale for the kinetic-mixing dark-photon completion.

    Lambda = m_A' / sqrt( g_D eps e ).  Independent of m_chi (dim-6).
    """
    m_Ap = np.asarray(m_Ap, dtype=float)
    eps = np.asarray(eps, dtype=float)
    g_D = np.asarray(g_D, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return m_Ap / np.sqrt(g_D * eps * E_EM)


def uv_darkphoton_from_lambda(Lambda, *, solve_for="m_Ap", eps=None, g_D=None, m_Ap=None):
    r"""Invert the dark-photon map  Lambda = m_A' / sqrt(g_D eps e).

    solve_for="m_Ap": given (Lambda, eps, g_D) return m_A' = Lambda sqrt(g_D eps e)
    solve_for="g_D":  given (Lambda, eps, m_Ap) return g_D = (m_A'/Lambda)^2 / (eps e)
    solve_for="eps":  given (Lambda, g_D, m_Ap) return eps = (m_A'/Lambda)^2 / (g_D e)
    """
    Lambda = np.asarray(Lambda, dtype=float)
    if solve_for == "m_Ap":
        eps = np.asarray(eps, dtype=float)
        g_D = np.asarray(g_D, dtype=float)
        return Lambda * np.sqrt(g_D * eps * E_EM)
    elif solve_for == "g_D":
        eps = np.asarray(eps, dtype=float)
        m_Ap = np.asarray(m_Ap, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (m_Ap / Lambda) ** 2 / (eps * E_EM)
    elif solve_for == "eps":
        g_D = np.asarray(g_D, dtype=float)
        m_Ap = np.asarray(m_Ap, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (m_Ap / Lambda) ** 2 / (g_D * E_EM)
    raise ValueError("solve_for must be 'm_Ap', 'g_D', or 'eps'")


# ===========================================================================
# 2.  Load the EFT-plane scattering boundary
# ===========================================================================

def load_scattering_boundary(operator: str, *, halo_profile="rho2", source_tag="halo",
                             model_kind="raw_attenuation", majorana=False):
    """Load (m_chi, Lambda) along the saved 90% CL scattering boundary.

    Returns (mchi_GeV, lambda_GeV) sorted by m_chi, finite & positive only.
    Uses lambda_GeV (raw Lambda, c=1) since the matching maps expect Lambda
    with the benchmark Wilson coefficient.
    """
    suffix = "_majorana" if majorana else ""
    fname = (
        f"mcmc_{halo_profile}_{source_tag}_{model_kind}_"
        f"fermionic_{operator}{suffix}_90cl.npz"
    )
    path = BOUNDARY_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Scattering boundary not found: {path}\n"
            f"Generate it first with:\n"
            f"  python constraint_generation/make_data_driven_scattering_limits.py "
            f"--halo-profile {halo_profile} --source {source_tag.split('_')[0]} "
            f"--dm-type fermionic --operator {operator}"
        )
    data = np.load(path, allow_pickle=True)
    mchi = np.asarray(data["mchi_GeV"], dtype=float)
    lam = np.asarray(data["lambda_GeV"], dtype=float)
    mask = np.isfinite(mchi) & np.isfinite(lam) & (mchi > 0) & (lam > 0)
    mchi, lam = mchi[mask], lam[mask]
    order = np.argsort(mchi)
    return mchi[order], lam[order]


def _scalar_text(data, key, default=""):
    if key not in data:
        return default
    value = data[key]
    try:
        value = value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)


def _operator_from_boundary(path: Path, data) -> str | None:
    op = _scalar_text(data, "operator", "")
    if op:
        return op
    stem = path.stem
    for candidate in sorted(UV_OPERATORS, key=len, reverse=True):
        if candidate in stem:
            return candidate
    return None


def _boundary_style(path: Path):
    stem = path.stem.lower()
    for needle, label, color, linestyle in BOUNDARY_STYLE_HINTS:
        if needle in stem:
            return label, color, linestyle
    label = path.stem.replace("_", " ")
    return label, COL_GUIDE, "-"


def _boundary_lambda_key(data):
    # ``lambda_GeV`` is the raw c=1 EFT scale used by the UV matching.  Older
    # deconvolution outputs used ``lambda_lim_GeV`` for the same physical axis.
    for key in ("lambda_GeV", "lambda_lim_GeV", "lambda_plot_GeV"):
        if key in data:
            return key
    return None


def _boundary_relevance(path: Path, operator: str, *, halo_profile: str, model_kind: str, majorana: bool):
    """Keep only boundary files that still mean something after UV matching."""
    stem = path.stem.lower()
    if any(skip in stem for skip in ("higgs", "rayleigh", "scalar")):
        return False
    if operator not in stem and f"{operator}_majorana" not in stem:
        return False
    if majorana and "majorana" not in stem and operator == "anapole":
        return False
    if not majorana and "majorana" in stem:
        return False
    if stem.startswith("mcmc_"):
        if f"_{halo_profile}_" not in stem or f"_{model_kind}_" not in stem:
            return False
    return True


def load_operator_boundaries(operator: str, *, halo_profile="rho2", model_kind="raw_attenuation",
                             majorana=False, skip_current=None):
    """Load all saved EFT-plane boundaries for one UV-relevant operator.

    The files in ``constraint_boundaries`` are stored as limits in the
    ``(m_chi, Lambda)`` plane.  This loader deliberately ignores operators whose
    Wilson coefficient cannot be represented by the two UV completions in this
    script (Rayleigh/scalar/Higgs) and leaves EFT-validity guide metadata alone.
    """
    curves = []
    skip_current = Path(skip_current).name if skip_current is not None else None
    for path in sorted(BOUNDARY_DIR.glob("*.npz")):
        if skip_current is not None and path.name == skip_current:
            continue
        if not _boundary_relevance(
            path, operator, halo_profile=halo_profile, model_kind=model_kind, majorana=majorana
        ):
            continue
        data = np.load(path, allow_pickle=True)
        op_in_file = _operator_from_boundary(path, data)
        if op_in_file != operator:
            continue
        lam_key = _boundary_lambda_key(data)
        if "mchi_GeV" not in data or lam_key is None:
            continue
        mchi = np.asarray(data["mchi_GeV"], dtype=float)
        lam = np.asarray(data[lam_key], dtype=float)
        mask = np.isfinite(mchi) & np.isfinite(lam) & (mchi > 0.0) & (lam > 0.0)
        if np.count_nonzero(mask) < 2:
            continue
        mchi = mchi[mask]
        lam = lam[mask]
        order = np.argsort(mchi)
        label, color, linestyle = _boundary_style(path)
        paper_label = _scalar_text(data, "paper_label", "")
        if paper_label:
            paper_label = re.sub(r"\s+", " ", paper_label).strip()
            label = paper_label.replace("90% CL", "").replace("95% CL", "").strip() or label
        curves.append({
            "path": path,
            "name": path.stem,
            "label": label,
            "color": color,
            "linestyle": linestyle,
            "mchi_GeV": mchi[order],
            "lambda_GeV": lam[order],
        })
    return curves


def _interp_log_boundary(curve, mchi_benchmark):
    mchi = np.asarray(curve["mchi_GeV"], dtype=float)
    lam = np.asarray(curve["lambda_GeV"], dtype=float)
    order = np.argsort(mchi)
    mchi = mchi[order]
    lam = lam[order]
    mb = float(mchi_benchmark)
    if mb < np.nanmin(mchi) or mb > np.nanmax(mchi):
        return None
    return 10.0 ** float(np.interp(np.log10(mb), np.log10(mchi), np.log10(lam)))


def _plot_rescaled_boundaries_mmed(
    ax, curves, transform, *, mchi_benchmark=None, max_legend=12
):
    plotted = 0
    used_labels = set()
    for curve in curves:
        mchi = curve["mchi_GeV"]
        y = transform(curve["lambda_GeV"], mchi)
        good = np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(good) < 2:
            continue
        label = curve["label"]
        if label in used_labels:
            label = curve["name"].replace("_", " ")
        used_labels.add(label)
        ax.plot(
            mchi[good], y[good],
            color=curve["color"], ls=curve["linestyle"], lw=1.55,
            alpha=0.9, label=label if plotted < max_legend else None,
            zorder=2,
        )
        plotted += 1
    return plotted


def _plot_rescaled_boundaries_coupling(
    ax, curves, transform, x_grid, *, mchi_benchmark, max_legend=12
):
    plotted = 0
    used_labels = set()
    for curve in curves:
        lam_at_mb = _interp_log_boundary(curve, mchi_benchmark)
        if lam_at_mb is None:
            continue
        y = transform(lam_at_mb, x_grid)
        good = np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(good) < 2:
            continue
        label = curve["label"]
        if label in used_labels:
            label = curve["name"].replace("_", " ")
        used_labels.add(label)
        ax.plot(
            x_grid[good], y[good],
            color=curve["color"], ls=curve["linestyle"], lw=1.55,
            alpha=0.9, label=label if plotted < max_legend else None,
            zorder=2,
        )
        plotted += 1
    return plotted


# ===========================================================================
# 3.  UV-only guide curves
# ===========================================================================
#
# The EFT validity/unitarity guide curves used elsewhere are intentionally not
# drawn here: after matching to a UV completion, the useful theory guide is the
# perturbativity ceiling on the UV coupling.


# ===========================================================================
# 4.  Plotting
# ===========================================================================

def _setup_axes(ax, xlabel, ylabel, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", alpha=0.22)


def plot_dipole(operator, *, halo_profile, source_tag, model_kind,
                lam_benchmark, mchi_benchmark, x_ratio, Q, outtag):
    """Two-panel set for the charged-messenger dipole completion."""
    mchi_b, lam_b = load_scattering_boundary(
        operator, halo_profile=halo_profile, source_tag=source_tag, model_kind=model_kind
    )
    current_path = (
        BOUNDARY_DIR
        / f"mcmc_{halo_profile}_{source_tag}_{model_kind}_fermionic_{operator}_90cl.npz"
    )
    extra_boundaries = load_operator_boundaries(
        operator, halo_profile=halo_profile, model_kind=model_kind,
        skip_current=current_path,
    )
    is_electric = operator == "dipole_electric"
    op_title = "Electric dipole" if is_electric else "Magnetic dipole"

    # ---- Panel A: M_F vs m_chi at fixed lambda ----
    M_F_scatter = uv_dipole_from_lambda(
        lam_b, mchi_b, x_ratio=x_ratio, Q=Q, solve_for="M_F", lam=lam_benchmark
    )
    good = np.isfinite(M_F_scatter) & (M_F_scatter > 0)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _setup_axes(
        ax,
        r"$m_\chi$ [GeV]",
        r"$M_F$ [GeV]",
        f"{op_title} UV completion: charged messenger\n"
        rf"$\lambda={lam_benchmark:g}$, $x=M_S^2/M_F^2={x_ratio:g}$, $Q={Q:g}$",
    )
    if np.any(good):
        ax.fill_between(
            mchi_b[good], M_F_scatter[good], M_F_scatter[good] * 0 + 1e-3,
            color=COL_SCATTER, alpha=0.12, zorder=1,
        )
        ax.plot(
            mchi_b[good], M_F_scatter[good], color=COL_SCATTER, lw=2.6,
            label="Totani halo scattering (this work)",
        )
    _plot_rescaled_boundaries_mmed(
        ax,
        extra_boundaries,
        lambda Lambda, mchi: uv_dipole_from_lambda(
            Lambda, mchi, x_ratio=x_ratio, Q=Q, solve_for="M_F", lam=lam_benchmark
        ),
    )
    ax.legend(fontsize=8.5, loc="best")
    out_a = PLOTDIR / f"uv_{outtag}_Mmed_vs_mchi"
    save_figure(fig, str(out_a))
    plt.close(fig)
    print(f"  saved {out_a}.png/.pdf")

    # ---- Panel B: lambda (coupling) vs M_F at fixed m_chi ----
    # Along the scattering boundary at the benchmark m_chi, find the Lambda by
    # interpolation, then express the boundary as lambda(M_F).
    lam_at_mb = float(np.interp(np.log10(mchi_benchmark), np.log10(mchi_b), np.log10(lam_b)))
    lam_at_mb = 10.0 ** lam_at_mb
    M_F_grid = np.logspace(1, 5, 400)   # 10 GeV - 100 TeV
    lam_coupling = uv_dipole_from_lambda(
        lam_at_mb, mchi_benchmark, x_ratio=x_ratio, Q=Q, solve_for="lam", M_F=M_F_grid
    )
    good = np.isfinite(lam_coupling) & (lam_coupling > 0)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _setup_axes(
        ax,
        r"$M_F$ [GeV]",
        r"Yukawa coupling $\lambda$",
        f"{op_title} UV completion: charged messenger\n"
        rf"$m_\chi={mchi_benchmark:g}$ GeV, $x={x_ratio:g}$, $Q={Q:g}$",
    )
    if np.any(good):
        ax.plot(M_F_grid[good], lam_coupling[good], color=COL_SCATTER, lw=2.6,
                label="Totani halo scattering (this work)")
        ax.fill_between(M_F_grid[good], lam_coupling[good], 1e3,
                        color=COL_SCATTER, alpha=0.12)
    _plot_rescaled_boundaries_coupling(
        ax,
        extra_boundaries,
        lambda Lambda, M_F: uv_dipole_from_lambda(
            Lambda, mchi_benchmark, x_ratio=x_ratio, Q=Q, solve_for="lam", M_F=M_F
        ),
        M_F_grid,
        mchi_benchmark=mchi_benchmark,
    )
    ax.axhline(np.sqrt(4 * np.pi), color=COL_PERTURB, lw=1.8, ls=":",
               label=r"Perturbativity $\lambda=\sqrt{4\pi}$")
    ax.set_ylim(1e-3, 1e2)
    ax.legend(fontsize=8.5, loc="best")
    out_b = PLOTDIR / f"uv_{outtag}_g_vs_Mmed"
    save_figure(fig, str(out_b))
    plt.close(fig)
    print(f"  saved {out_b}.png/.pdf")


def plot_darkphoton(operator, *, halo_profile, source_tag, model_kind,
                    gD_benchmark, eps_benchmark, mchi_benchmark, majorana, outtag):
    """Two-panel set for the kinetic-mixing dark-photon completion."""
    mchi_b, lam_b = load_scattering_boundary(
        operator, halo_profile=halo_profile, source_tag=source_tag,
        model_kind=model_kind, majorana=majorana,
    )
    suffix = "_majorana" if majorana else ""
    current_path = (
        BOUNDARY_DIR
        / f"mcmc_{halo_profile}_{source_tag}_{model_kind}_fermionic_{operator}{suffix}_90cl.npz"
    )
    extra_boundaries = load_operator_boundaries(
        operator, halo_profile=halo_profile, model_kind=model_kind,
        majorana=majorana, skip_current=current_path,
    )
    op_title = "Anapole" if operator == "anapole" else "Charge radius"

    # The map is m_chi-independent, so take Lambda at the benchmark m_chi.
    lam_at_mb = 10.0 ** float(
        np.interp(np.log10(mchi_benchmark), np.log10(mchi_b), np.log10(lam_b))
    )

    # ---- Panel A: m_A' vs m_chi at fixed (eps, g_D) ----
    # m_A' = Lambda(m_chi) sqrt(g_D eps e); Lambda boundary is the full curve.
    m_Ap_scatter = uv_darkphoton_from_lambda(
        lam_b, solve_for="m_Ap", eps=eps_benchmark, g_D=gD_benchmark
    )
    good = np.isfinite(m_Ap_scatter) & (m_Ap_scatter > 0)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _setup_axes(
        ax,
        r"$m_\chi$ [GeV]",
        r"$m_{A'}$ [GeV]",
        f"{op_title} UV completion: kinetic-mixing dark $U(1)'$\n"
        rf"$g_D={gD_benchmark:g}$, $\epsilon={eps_benchmark:g}$",
    )
    if np.any(good):
        ax.plot(mchi_b[good], m_Ap_scatter[good], color=COL_SCATTER, lw=2.6,
                label="Totani halo scattering (this work)")
        ax.fill_between(mchi_b[good], m_Ap_scatter[good], 1e-3,
                        color=COL_SCATTER, alpha=0.12)
    _plot_rescaled_boundaries_mmed(
        ax,
        extra_boundaries,
        lambda Lambda, _mchi: uv_darkphoton_from_lambda(
            Lambda, solve_for="m_Ap", eps=eps_benchmark, g_D=gD_benchmark
        ),
    )
    ax.legend(fontsize=8.5, loc="best")
    out_a = PLOTDIR / f"uv_{outtag}_Mmed_vs_mchi"
    save_figure(fig, str(out_a))
    plt.close(fig)
    print(f"  saved {out_a}.png/.pdf")

    # ---- Panel B: g_D vs m_A' at fixed eps (and fixed m_chi via lam_at_mb) ----
    m_Ap_grid = np.logspace(-2, 4, 400)   # 10 MeV - 10 TeV
    gD_scatter = uv_darkphoton_from_lambda(
        lam_at_mb, solve_for="g_D", eps=eps_benchmark, m_Ap=m_Ap_grid
    )
    good = np.isfinite(gD_scatter) & (gD_scatter > 0)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _setup_axes(
        ax,
        r"$m_{A'}$ [GeV]",
        r"dark gauge coupling $g_D$",
        f"{op_title} UV completion: kinetic-mixing dark $U(1)'$\n"
        rf"$\epsilon={eps_benchmark:g}$, $m_\chi={mchi_benchmark:g}$ GeV",
    )
    if np.any(good):
        ax.plot(m_Ap_grid[good], gD_scatter[good], color=COL_SCATTER, lw=2.6,
                label="Totani halo scattering (this work)")
        ax.fill_between(m_Ap_grid[good], gD_scatter[good], 1e3,
                        color=COL_SCATTER, alpha=0.12)
    _plot_rescaled_boundaries_coupling(
        ax,
        extra_boundaries,
        lambda Lambda, m_Ap: uv_darkphoton_from_lambda(
            Lambda, solve_for="g_D", eps=eps_benchmark, m_Ap=m_Ap
        ),
        m_Ap_grid,
        mchi_benchmark=mchi_benchmark,
    )
    ax.axhline(np.sqrt(4 * np.pi), color=COL_PERTURB, lw=1.8, ls=":",
               label=r"Perturbativity $g_D=\sqrt{4\pi}$")
    ax.set_ylim(1e-4, 1e2)
    ax.legend(fontsize=8.5, loc="best")
    out_b = PLOTDIR / f"uv_{outtag}_g_vs_Mmed"
    save_figure(fig, str(out_b))
    plt.close(fig)
    print(f"  saved {out_b}.png/.pdf")


# ===========================================================================
# 5.  Driver
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--theory", default="all",
                   choices=["dipole", "darkphoton", "all"],
                   help="Which UV completion(s) to plot.")
    p.add_argument("--halo-profile", default="rho2")
    p.add_argument("--source-tag", default="halo",
                   help="Source tag in the boundary filename (e.g. 'halo' or 'pppc_WW_mann700').")
    p.add_argument("--model-kind", default="raw_attenuation",
                   choices=["raw_attenuation", "spectral_reshaping"])

    # dipole benchmarks
    p.add_argument("--dipole-lambda", type=float, default=1.0,
                   help="Yukawa coupling benchmark for the M_F vs m_chi panel.")
    p.add_argument("--dipole-mchi", type=float, default=100.0,
                   help="m_chi [GeV] benchmark for the lambda vs M_F panel.")
    p.add_argument("--dipole-x", type=float, default=1.0,
                   help="x = M_S^2/M_F^2 loop-mass ratio.")
    p.add_argument("--dipole-Q", type=float, default=1.0,
                   help="Charge of the messenger fermion.")
    p.add_argument("--include-electric", action="store_true",
                   help="Also produce the electric-dipole completion plots.")

    # dark-photon benchmarks
    p.add_argument("--dp-gD", type=float, default=1.0,
                   help="Dark gauge coupling benchmark for the m_A' vs m_chi panel.")
    p.add_argument("--dp-eps", type=float, default=1e-3,
                   help="Kinetic-mixing benchmark.")
    p.add_argument("--dp-mchi", type=float, default=100.0,
                   help="m_chi [GeV] benchmark for the g_D vs m_A' panel.")
    p.add_argument("--include-charge-radius", action="store_true",
                   help="Also produce the charge-radius completion plots (Dirac).")

    p.add_argument("--style", default=None,
                   help="Plot style: paper, conference/dark_transparent, conference_light.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style
    set_plot_style(style="light", cmap_name="plasma", base_fontsize=12, linewidth=1.8)
    PLOTDIR.mkdir(parents=True, exist_ok=True)

    do_dipole = args.theory in ("dipole", "all")
    do_dp = args.theory in ("darkphoton", "all")

    if do_dipole:
        print("Charged-messenger -> magnetic dipole")
        plot_dipole(
            "dipole_magnetic",
            halo_profile=args.halo_profile, source_tag=args.source_tag,
            model_kind=args.model_kind,
            lam_benchmark=args.dipole_lambda, mchi_benchmark=args.dipole_mchi,
            x_ratio=args.dipole_x, Q=args.dipole_Q,
            outtag="dipole_magnetic",
        )
        if args.include_electric:
            print("Charged-messenger -> electric dipole")
            plot_dipole(
                "dipole_electric",
                halo_profile=args.halo_profile, source_tag=args.source_tag,
                model_kind=args.model_kind,
                lam_benchmark=args.dipole_lambda, mchi_benchmark=args.dipole_mchi,
                x_ratio=args.dipole_x, Q=args.dipole_Q,
                outtag="dipole_electric",
            )

    if do_dp:
        print("Kinetic-mixing dark photon -> anapole")
        plot_darkphoton(
            "anapole",
            halo_profile=args.halo_profile, source_tag=args.source_tag,
            model_kind=args.model_kind,
            gD_benchmark=args.dp_gD, eps_benchmark=args.dp_eps,
            mchi_benchmark=args.dp_mchi, majorana=False,
            outtag="darkphoton_anapole",
        )
        if args.include_charge_radius:
            print("Kinetic-mixing dark photon -> charge radius")
            plot_darkphoton(
                "charge_radius",
                halo_profile=args.halo_profile, source_tag=args.source_tag,
                model_kind=args.model_kind,
                gD_benchmark=args.dp_gD, eps_benchmark=args.dp_eps,
                mchi_benchmark=args.dp_mchi, majorana=False,
                outtag="darkphoton_charge_radius",
            )

    print("\nDone. Plots in", PLOTDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
