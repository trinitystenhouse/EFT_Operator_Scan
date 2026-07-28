#!/usr/bin/env python3
"""
Data-driven DM-photon scattering limits from the Totani_paper_check MCMC halo spectrum.

This replaces the old hand-digitized PHI_TOTANI/SIGMA_TOTANI arrays with the
halo-component posterior spectrum stored in Totani_paper_check/mcmc results.

Important scope
---------------
These products are halo-component transfer limits. They apply photon-DM
scattering to the extracted NFW/PPPC halo component only, not to the full
gamma-ray sky. That is useful as a template-space distortion test, but a fully
physical calculation should propagate every photon component with its own
geometry. spectral_reshaping.PhotonTransferComponent and
transfer_photon_components are the scaffold for that next step.

Two limit definitions are produced:

  raw_attenuation
      Phi_obs(E) = Phi_source(E) exp[-tau(E)].
      No angular/energy redistribution is added.

  halo_component_reshaping
      Phi_obs(E) = surviving flux + one-scatter in-scatter flux from the
      redistribution kernel in spectral_reshaping.py.

The default source model is "halo": the intrinsic source spectrum is set equal
to the MCMC-derived halo spectrum and the normalization is held fixed. This is a
model-independent distortion/preservation limit. For annihilation-model limits,
use --source pppc --channel ... --ann-mass ..., which fits the PPPC template
normalization at every scattering point.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_WORKSPACE = _ROOT.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_WORKSPACE))

from core.attenuation_eft import (  # noqa: E402
    _operator_metadata,
    _paper_y_axis_values,
    extract_90cl_boundary,
)
from core.eft_validity import (  # noqa: E402
    unitarity_lambda_curve as shared_unitarity_lambda_curve,
    validity_mask as shared_validity_mask,
)
from helpers.trinity_plotting import save_figure, set_plot_style  # noqa: E402
from core.spectral_reshaping import (  # noqa: E402
    pppc_energy_flux_template,
    scan_reshaping_chi2,
    smooth_nfw_sigma_v_from_norm,
)
from core.spectrum_source import (  # noqa: E402
    SpectrumSource,
    load_spectrum_source,
    wrap_halo_as_source,
)
from core.totani_data_loader import _MCMC_DIRS, load_halo_spectrum  # noqa: E402


OUTDIR = _ROOT / "constraint_boundaries"
PLOTDIR = _ROOT / "plots"

RUN_ALL_SPECS = [
    dict(dm_type="fermionic", operator="dipole_magnetic", majorana=False),
    dict(dm_type="fermionic", operator="dipole_electric", majorana=False),
    dict(dm_type="fermionic", operator="charge_radius", majorana=False),
    dict(dm_type="fermionic", operator="anapole", majorana=False),
    dict(dm_type="fermionic", operator="anapole", majorana=True),
    dict(dm_type="fermionic", operator="rayleigh_full", majorana=False),
    dict(dm_type="fermionic", operator="rayleigh_full", majorana=True),
    dict(dm_type="fermionic", operator="rayleigh_even", majorana=False),
    dict(dm_type="fermionic", operator="rayleigh_odd", majorana=False),
    dict(dm_type="fermionic", operator="rayleigh_even", majorana=True),
    dict(dm_type="fermionic", operator="rayleigh_odd", majorana=True),
    dict(dm_type="scalar", operator="scalar_rayleigh", majorana=False),
]

MAJORANA_FORBIDDEN = {"dipole_magnetic", "dipole_electric", "charge_radius"}


def get_s_max_lab_dmrest(mchi: float, omega_max: float) -> float:
    mchi = np.asarray(mchi, dtype=float)
    return mchi**2 + 2.0 * mchi * float(omega_max)


def get_t_abs_max_lab_dmrest(mchi: float, omega_max: float) -> float:
    mchi = np.asarray(mchi, dtype=float)
    denom = 1.0 + (2.0 * float(omega_max) / mchi)
    return 4.0 * float(omega_max)**2 / denom


def eft_validity_mask(
    mchi_grid: np.ndarray,
    lambda_grid: np.ndarray,
    *,
    omega_max: float,
    eft_kinematic_factor: float,
) -> np.ndarray:
    mchi_grid = np.asarray(mchi_grid, dtype=float)
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    q2_max = np.maximum(
        get_s_max_lab_dmrest(mchi_grid[:, None], omega_max),
        get_t_abs_max_lab_dmrest(mchi_grid[:, None], omega_max),
    )
    return (lambda_grid[None, :] ** 2) >= (float(eft_kinematic_factor) * q2_max)


def unitarity_lambda_curve(operator: str, mchi_grid: np.ndarray) -> np.ndarray:
    mchi_grid = np.asarray(mchi_grid, dtype=float)
    xgrid = np.where(mchi_grid > 0.0, mchi_grid, np.nan)
    if operator in ("dipole_magnetic", "dipole_electric"):
        return np.sqrt(16.0 * np.pi * xgrid)
    if operator in ("charge_radius", "anapole"):
        return (16.0 * np.pi * xgrid**2) ** 0.25
    if "rayleigh" in str(operator):
        return (128.0 * np.pi**2 * xgrid**2) ** (1.0 / 6.0)
    return np.full_like(xgrid, np.nan)


def validity_masks(args, energy_bins: np.ndarray) -> dict[str, np.ndarray]:
    omega_max = float(np.max(np.asarray(energy_bins, dtype=float)))
    eft_mask = shared_validity_mask(
        args.operator,
        args.m_chi_grid,
        args.lambda_grid,
        omega_max=omega_max,
        dm_type=args.dm_type,
        eft_kinematic_factor=float(args.eft_kinematic_factor),
        require_lambda_gt_mdm=bool(args.require_lambda_gt_mdm),
        include_kinematic=True,
        include_unitarity=False,
    )
    lam_unit = shared_unitarity_lambda_curve(args.operator, args.m_chi_grid)
    unitary_mask = args.lambda_grid[None, :] >= lam_unit[:, None]
    if bool(args.require_lambda_gt_mdm):
        unitary_mask &= args.lambda_grid[None, :] > args.m_chi_grid[:, None]
    return {
        "omega_max": omega_max,
        "eft_mask": np.asarray(eft_mask, dtype=bool),
        "unitary_mask": np.asarray(unitary_mask, dtype=bool),
        "combined_mask": np.asarray(eft_mask & unitary_mask, dtype=bool),
        "lam_unit": lam_unit,
    }


def masked_chi2_grid(chi2_grid: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    chi2_grid = np.asarray(chi2_grid, dtype=float)
    out = np.array(chi2_grid, copy=True)
    out[~np.asarray(valid_mask, dtype=bool)] = np.nan
    return out


def operator_couplings(operator: str, dm_type: str) -> tuple[float, float, float]:
    """Return the one-coefficient-at-a-time convention used by the overlay scripts."""
    if dm_type == "scalar" or operator in ("scalar_rayleigh", "rayleigh"):
        return 0.0, 0.0, 1.0
    if operator in ("dipole_magnetic", "charge_radius", "rayleigh_even"):
        return 1.0, 0.0, 1.0
    if operator in ("dipole_electric", "anapole", "rayleigh_odd"):
        return 0.0, 1.0, 1.0
    if operator == "rayleigh_full":
        return 1.0, 1.0, 1.0
    return 1.0, 0.0, 1.0


def scalar_str(value) -> str:
    arr = np.asarray(value)
    return str(arr.item()) if arr.shape == () else str(value)


def finite_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr.astype(np.float32)


def normalize_operator_choice(args) -> None:
    requested = str(args.operator)
    args.requested_operator = requested
    if args.dm_type == "fermionic" and bool(args.majorana) and requested == "charge_radius":
        print(
            "  [note] Majorana 'charge_radius' is being interpreted as the axial charge radius,"
            " i.e. the anapole operator, for the scattering run."
        )
        args.operator = "anapole"


def validate_operator_choice(args) -> None:
    if args.dm_type != "fermionic" or not bool(args.majorana):
        return
    if args.operator in MAJORANA_FORBIDDEN:
        allowed = "anapole, rayleigh_full, rayleigh_even, rayleigh_odd"
        raise ValueError(
            f"Operator '{args.operator}' vanishes for Majorana DM in this EFT setup. "
            f"Use one of: {allowed}."
        )


def source_spectrum(args, spectrum):
    """Build the intrinsic spectrum used before scattering.

    Two modes:

    * ``--source measured`` (aliases: ``halo``) — use the dataset's own measured
      spectrum ``spectrum.phi`` as the intrinsic pre-scattering template with the
      normalisation held fixed. This is a distortion-preservation test: which
      scattering models leave the measured spectrum's shape intact? Sensible when
      the dataset is a real detection (halo posterior, IGRB); degenerate when the
      dataset is null-like (dSph SED), in which case the profile fit trivially
      absorbs any τ by driving the intrinsic normalisation to zero.

    * ``--source pppc --channel X --ann-mass M`` — assume the observed dataset is
      sourced by DM annihilation with a PPPC(m_ann, channel) spectrum, then ask
      whether scattering-attenuated propagation reshapes it enough to be
      inconsistent with the measurement. This is the two-component dark-sector
      scenario (heavy annihilator + light scatterer) and the natural analogue
      of the standard Fermi-LAT dSph annihilation analyses for datasets whose
      SEDs are consistent with null. The intrinsic normalisation is analytically
      profiled at each grid point.

    Works uniformly for every dataset that presents a :class:`SpectrumSource`.
    """
    src_kind = str(args.source).lower()
    if src_kind in ("halo", "measured"):
        return spectrum.phi.copy(), f"measured spectrum ({spectrum.source_label})", False, None

    table = args.pppc_gamma_table
    source = pppc_energy_flux_template(
        spectrum.E_bins_GeV,
        args.ann_mass,
        channel=args.channel,
        table_path=table,
        normalise=False,
    )
    sigmav_ref = smooth_nfw_sigma_v_from_norm(1.0, args.ann_mass)
    label = f"PPPC {args.channel}, m_ann={args.ann_mass:g} GeV"
    return source, label, True, sigmav_ref


def save_boundary(
    *,
    boundary: np.ndarray,
    args,
    model_kind: str,
    source_label: str,
    halo,
    spectrum: SpectrumSource,
    chi2_grid: np.ndarray,
    norm_grid: np.ndarray,
    tau_grid: np.ndarray,
    valid_mask: np.ndarray,
    validity_meta: dict,
) -> Path | None:
    if boundary.size == 0:
        print(f"  [WARN] No {model_kind} 90% CL contour found.")
        return None

    OUTDIR.mkdir(parents=True, exist_ok=True)
    suffix = "_majorana" if args.majorana else ""

    # Filename convention:
    #   halo/measured  → mcmc_{halo_profile}_halo_{model_kind}_..._90cl.npz  (unchanged legacy)
    #   halo/pppc      → mcmc_{halo_profile}_pppc_{ch}_mann{m}_..._90cl.npz  (unchanged legacy)
    #   dsph/measured  → dsph_{selection}_measured_..._90cl.npz
    #   dsph/pppc      → dsph_{selection}_pppc_{ch}_mann{m}_..._90cl.npz
    #   igrb/measured  → igrb_ackermann2015a_measured_..._90cl.npz
    #   igrb/pppc      → igrb_ackermann2015a_pppc_{ch}_mann{m}_..._90cl.npz
    src_kind = str(args.source).lower()
    if src_kind in ("halo", "measured"):
        source_tag = "halo" if args.dataset == "halo" else "measured"
    else:  # pppc
        source_tag = f"pppc_{args.channel}_mann{args.ann_mass:g}"

    if args.dataset == "halo":
        stem = f"mcmc_{args.halo_profile}_{source_tag}"
    elif args.dataset == "igrb":
        stem = f"igrb_ackermann2015a_{source_tag}"
    elif args.dataset == "dsph":
        sel = str(args.dsph_selection).replace(",", "_").replace(" ", "")
        stem = f"dsph_{sel}_{source_tag}"
    else:
        stem = f"{args.dataset}_{source_tag}"

    out = OUTDIR / (
        f"{stem}_{model_kind}_"
        f"{args.dm_type}_{args.operator}{suffix}_90cl.npz"
    )

    lambda_raw = boundary[:, 1]
    lambda_plot = _paper_y_axis_values(
        lambda_raw,
        args.dm_type,
        args.operator,
        c_s=args.c_s,
        c_p=args.c_p,
        c_phi=args.c_phi,
    )
    meta = _operator_metadata(args.dm_type, args.operator)
    valid_chi2 = chi2_grid[np.isfinite(chi2_grid)]
    chi2_min = float(np.min(valid_chi2)) if valid_chi2.size else np.nan
    tau_max_grid = np.nanmax(tau_grid, axis=2) if tau_grid.ndim == 3 else np.full_like(chi2_grid, np.nan)

    np.savez_compressed(
        out,
        mchi_GeV=boundary[:, 0].astype(np.float32),
        lambda_GeV=lambda_raw.astype(np.float32),
        lambda_plot_GeV=lambda_plot.astype(np.float32),
        chi2_grid=finite_float32(chi2_grid),
        norm_grid=finite_float32(norm_grid),
        tau_max_grid=finite_float32(tau_max_grid),
        grid_mchi_GeV=args.m_chi_grid.astype(np.float32),
        grid_lambda_GeV=args.lambda_grid.astype(np.float32),
        paper_label=meta["paper_label"],
        dm_type=args.dm_type,
        operator=args.operator,
        model_kind=model_kind,
        dataset=args.dataset,
        data_source=spectrum.source_label,
        halo_profile=args.halo_profile if args.dataset == "halo" else "",
        nfw_label=halo.nfw_label if halo is not None else "",
        tau_prefactor_K=(
            np.float32(spectrum.tau_prefactor_K)
            if spectrum.tau_prefactor_K is not None
            else np.float32(np.nan)
        ),
        source_model=args.source,
        source_label=source_label,
        fit_normalization=bool(args.fit_normalization),
        delta_chi2_threshold=np.float32(args.delta_chi2),
        chi2_min=np.float32(chi2_min),
        chi2_grid_valid=finite_float32(masked_chi2_grid(chi2_grid, valid_mask)),
        eft_valid_mask=np.asarray(validity_meta["eft_mask"], dtype=np.uint8),
        unitary_valid_mask=np.asarray(validity_meta["unitary_mask"], dtype=np.uint8),
        combined_valid_mask=np.asarray(valid_mask, dtype=np.uint8),
        omega_max_for_validity=np.float32(validity_meta["omega_max"]),
        eft_kinematic_factor=np.float32(args.eft_kinematic_factor),
        err_mode=args.err_mode,
        positive_only=bool(not args.include_nonpositive_bins),
        c_s=np.float32(args.c_s),
        c_p=np.float32(args.c_p),
        c_phi=np.float32(args.c_phi),
        majorana=bool(args.majorana),
        max_tau_single_scatter=np.float32(args.max_tau_single_scatter),
        require_lambda_gt_mdm=bool(args.require_lambda_gt_mdm),
        validity_guides="kinematic_eft_and_unitarity",
        boundary_extraction="unfiltered_scan_contour",
    )
    print(f"  saved {model_kind}: {out}")
    return out


def make_plot(boundaries: dict[str, np.ndarray], args, halo, source_label: str, validity_meta: dict):
    PLOTDIR.mkdir(parents=True, exist_ok=True)
    set_plot_style(style="light", cmap_name="plasma", base_fontsize=13, linewidth=2.0)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {
        "raw_attenuation": "C3",
        "spectral_reshaping": "C0",
    }
    labels = {
        "raw_attenuation": "Halo-component raw attenuation only",
        "spectral_reshaping": "Halo-component transfer: survival + in-scatter",
    }

    for key, boundary in boundaries.items():
        if boundary is None or boundary.size == 0:
            continue
        y = _paper_y_axis_values(
            boundary[:, 1],
            args.dm_type,
            args.operator,
            c_s=args.c_s,
            c_p=args.c_p,
            c_phi=args.c_phi,
        )
        ax.loglog(boundary[:, 0], y, lw=2.4, color=colors.get(key), label=labels.get(key, key))

    xguide = np.logspace(np.log10(args.m_chi_grid.min()), np.log10(args.m_chi_grid.max()), 300)
    ax.loglog(xguide, xguide, color="0.35", lw=1.1, ls=":", label=r"EFT guide $\Lambda/C^{1/n}=m_\chi$")
    q2_max = np.maximum(
        get_s_max_lab_dmrest(xguide, validity_meta["omega_max"]),
        get_t_abs_max_lab_dmrest(xguide, validity_meta["omega_max"]),
    )
    lam_kin = np.sqrt(float(args.eft_kinematic_factor) * q2_max)
    kin_good = np.isfinite(lam_kin) & (lam_kin > 0.0)
    if np.any(kin_good):
        ax.loglog(xguide[kin_good], lam_kin[kin_good], color="r", lw=1.5, ls="--", label="Kinematic EFT validity")
    lam_unit = np.asarray(validity_meta["lam_unit"], dtype=float)
    unit_good = np.isfinite(lam_unit) & (lam_unit > 0.0)
    if np.any(unit_good):
        ax.loglog(args.m_chi_grid[unit_good], lam_unit[unit_good], color="0.55", lw=1.2, ls=":", label="Unitarity guide")

    meta = _operator_metadata(args.dm_type, args.operator)
    ax.set_xlabel(r"$m_\chi$ [GeV]")
    ax.set_ylabel(r"$\Lambda/C^{1/n}$ [GeV]")
    ax.set_title(
        f"Halo-component scattering limits: {args.operator}\n"
        f"{args.halo_profile}, source={source_label}"
    )
    ax.text(0.04, 0.95, scalar_str(meta["paper_label"]), transform=ax.transAxes, va="top")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    source_tag = "halo" if args.source == "halo" else f"pppc_{args.channel}_mann{args.ann_mass:g}"
    suffix = "_majorana" if args.majorana else ""
    outbase = PLOTDIR / (
        f"mcmc_{args.halo_profile}_{source_tag}_limits_"
        f"{args.dm_type}_{args.operator}{suffix}"
    )
    save_figure(fig, str(outbase))
    plt.close(fig)
    print(f"  saved plot: {outbase}.png/.pdf")


def extract_boundary_from_grid(m_grid: np.ndarray, lam_grid: np.ndarray, chi2_grid: np.ndarray, threshold: float):
    if threshold == 4.61:
        return extract_90cl_boundary(m_grid, lam_grid, chi2_grid)

    valid = np.isfinite(chi2_grid)
    if not np.any(valid):
        return np.empty((0, 2))
    shifted = chi2_grid - np.nanmin(chi2_grid) + (4.61 - float(threshold))
    return extract_90cl_boundary(m_grid, lam_grid, shifted)


def parse_args():
    p = argparse.ArgumentParser(
        description="Make raw-attenuation and spectral-reshaping limits using Totani_paper_check MCMC data.",
    )
    p.add_argument(
        "--dataset", default="halo", choices=["halo", "dsph", "igrb"],
        help="Which photon-flux measurement drives the scan. "
             "'halo' = pixel-level Fermi-LAT galactic-centre halo posterior "
             "(unchanged legacy path; requires --halo-profile). "
             "'igrb' = Ackermann+ 2015 isotropic extragalactic background. "
             "'dsph' = McDaniel 2024 dwarf-spheroidal SED stack.",
    )
    p.add_argument("--halo-profile", default="rho2", choices=sorted(_MCMC_DIRS.keys()))
    p.add_argument("--nfw-label", default=None)
    p.add_argument("--counts-path", default=None)
    p.add_argument("--err-mode", default="sym", choices=["sym", "lo", "hi", "max"])
    p.add_argument("--include-nonpositive-bins", action="store_true")
    # Dataset-specific loader options
    p.add_argument("--dsph-root", default=None,
                   help="Path to the McDaniel 2024 dSph release root "
                        "(the directory containing dSphs.csv and dSphs/SEDs/). "
                        "Only used when --dataset=dsph.")
    p.add_argument("--dsph-selection", default="all",
                   help="Which dSphs to combine at the likelihood level. "
                        "'all' or a comma-separated name list. "
                        "Only used when --dataset=dsph.")

    p.add_argument(
        "--source", default="measured", choices=["halo", "measured", "pppc"],
        help="Intrinsic pre-scattering source template. "
             "'measured' (or legacy alias 'halo') = the dataset's own SED "
             "(single-component scattering-only test). "
             "'pppc --channel X --ann-mass M' = assumed DM-annihilation source "
             "(two-component dark-sector: heavy annihilator + light scatterer). "
             "Works for --dataset {halo,dsph,igrb}.",
    )
    p.add_argument("--channel", default="WW")
    p.add_argument("--ann-mass", type=float, default=700.0)
    p.add_argument("--pppc-gamma-table", default=None)

    p.add_argument("--dm-type", default="fermionic", choices=["fermionic", "scalar"])
    p.add_argument("--operator", default="dipole_magnetic")
    p.add_argument("--majorana", action="store_true")
    p.add_argument("--c-s", type=float, default=None)
    p.add_argument("--c-p", type=float, default=None)
    p.add_argument("--c-phi", type=float, default=None)

    p.add_argument("--mchi-min", type=float, default=1e-6)
    p.add_argument("--mchi-max", type=float, default=1e8)
    p.add_argument("--nm", type=int, default=32)
    p.add_argument("--lambda-min", type=float, default=1e-3)
    p.add_argument("--lambda-max", type=float, default=1e7)
    p.add_argument("--nl", type=int, default=32)

    p.add_argument("--n-theta", type=int, default=160)
    p.add_argument("--no-roi-weight", action="store_true")
    p.add_argument("--roi-half-angle", type=float, default=60.0)
    p.add_argument("--max-tau-single-scatter", type=float, default=0.3)
    p.add_argument("--require-lambda-gt-mdm", action="store_true")
    p.add_argument("--delta-chi2", type=float, default=4.61)
    p.add_argument("--eft-kinematic-factor", type=float, default=1.0)
    p.add_argument("--run-all", action="store_true", help="Run the standard full operator set instead of a single operator.")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--style",
        default=None,
        help="Plot style: paper, conference/dark_transparent, or conference_light/light_transparent.",
    )
    return p.parse_args()


def run_single(args) -> int:
    if getattr(args, "style", None):
        os.environ["TRINITY_PLOT_STYLE"] = args.style
    normalize_operator_choice(args)
    validate_operator_choice(args)

    cs0, cp0, cphi0 = operator_couplings(args.operator, args.dm_type)
    args.c_s = cs0 if args.c_s is None else float(args.c_s)
    args.c_p = cp0 if args.c_p is None else float(args.c_p)
    args.c_phi = cphi0 if args.c_phi is None else float(args.c_phi)

    args.m_chi_grid = np.logspace(np.log10(args.mchi_min), np.log10(args.mchi_max), args.nm)
    args.lambda_grid = np.logspace(np.log10(args.lambda_min), np.log10(args.lambda_max), args.nl)

    # -------------------------------------------------------------------------
    # Dataset-agnostic spectrum loading
    # -------------------------------------------------------------------------
    # The halo path also keeps its HaloSpectrum object around so PPPC-source
    # scans (which read halo.E_bins_GeV / halo.iso_target_e2) continue to work.
    if args.dataset == "halo":
        halo = load_halo_spectrum(
            _MCMC_DIRS[args.halo_profile],
            nfw_label=args.nfw_label,
            counts_path=args.counts_path,
        )
        spectrum = wrap_halo_as_source(
            halo,
            source_label=f"MCMC halo posterior ({args.halo_profile})",
            err_mode=args.err_mode,
        )
    else:
        halo = None
        spectrum = load_spectrum_source(
            args.dataset,
            dsph_root=args.dsph_root,
            dsph_selection=args.dsph_selection,
        )

    phi_err = spectrum.phi_err_sym
    fit_mask = spectrum.finite_mask
    if not args.include_nonpositive_bins:
        fit_mask = fit_mask & spectrum.positive_mask

    E = spectrum.E_bins_GeV[fit_mask]
    phi_data = spectrum.phi[fit_mask]
    err = phi_err[fit_mask]

    # Intrinsic pre-scattering source template. Works uniformly across
    # datasets via the SpectrumSource abstraction:
    #   --source measured (or 'halo') → use the dataset's own SED (default)
    #   --source pppc               → assumed DM-annihilation template
    # See docstring of :func:`source_spectrum` for the physics interpretation
    # of each mode.
    source, source_label, source_fits_norm, sigmav_ref = source_spectrum(args, spectrum)
    source = source[fit_mask]
    args.fit_normalization = bool(source_fits_norm)

    print("Data-driven scattering limit setup")
    print(spectrum.summary())
    print(f"  fit bins       : {int(fit_mask.sum())} / {len(fit_mask)}")
    print(f"  source         : {source_label}")
    print(f"  model products : {args.dataset}-component raw_attenuation and spectral_reshaping")
    print(f"  scope          : scattering applied to the {args.dataset} line-of-sight column")
    print(f"  normalization  : {'fit analytically' if args.fit_normalization else 'fixed to measured spectrum'}")
    print(f"  validity cut   : Lambda^2 >= {float(args.eft_kinematic_factor):g} max(s_max, |t|_max), plus unitarity guide")
    if sigmav_ref is not None:
        print(f"  PPPC norm=1 maps to <sigma v>={sigmav_ref:.3e} cm^3/s")
    if spectrum.tau_prefactor_K is not None:
        print(f"  tau prefactor  : K = {spectrum.tau_prefactor_K:.3e} GeV/cm^2 (bypassing ROI integration)")

    validity_meta = validity_masks(args, E)

    result = scan_reshaping_chi2(
        args.m_chi_grid,
        args.lambda_grid,
        dm_type=args.dm_type,
        operator=args.operator,
        c_s=args.c_s,
        c_p=args.c_p,
        c_phi=args.c_phi,
        majorana=args.majorana,
        E_bins=E,
        phi_0=source,
        phi_data=phi_data,
        phi_err=err,
        n_theta=args.n_theta,
        apply_roi_weight=not args.no_roi_weight,
        roi_half_angle_deg=args.roi_half_angle,
        also_compute_attenuation=True,
        fit_normalization=args.fit_normalization,
        max_tau_single_scatter=args.max_tau_single_scatter,
        require_lambda_gt_mdm=args.require_lambda_gt_mdm,
        tau_prefactor_override=spectrum.tau_prefactor_K,
        verbose=not args.quiet,
    )

    boundaries = {}
    model_specs = {
        "raw_attenuation": ("chi2_attenuation", "norm_attenuation"),
        "spectral_reshaping": ("chi2_reshaping", "norm_reshaping"),
    }
    for model_kind, (chi_key, norm_key) in model_specs.items():
        boundary = extract_boundary_from_grid(
            args.m_chi_grid,
            args.lambda_grid,
            result[chi_key],
            args.delta_chi2,
        )
        boundaries[model_kind] = boundary
        save_boundary(
            boundary=boundary,
            args=args,
            model_kind=model_kind,
            source_label=source_label,
            halo=halo,
            spectrum=spectrum,
            chi2_grid=result[chi_key],
            norm_grid=result[norm_key],
            tau_grid=result["tau_grid"],
            valid_mask=validity_meta["combined_mask"],
            validity_meta=validity_meta,
        )

    if not args.no_plot and halo is not None:
        make_plot(boundaries, args, halo, source_label, validity_meta)

    return 0


def main():
    args = parse_args()
    if not args.run_all:
        return run_single(args)

    for spec in RUN_ALL_SPECS:
        sub_args = argparse.Namespace(**vars(args))
        sub_args.dm_type = spec["dm_type"]
        sub_args.operator = spec["operator"]
        sub_args.majorana = spec["majorana"]
        sub_args.c_s = None
        sub_args.c_p = None
        sub_args.c_phi = None
        print("")
        print("=" * 72)
        print(
            f"Running {sub_args.dm_type} / {sub_args.operator}"
            + (" / majorana" if sub_args.majorana else "")
        )
        print("=" * 72)
        run_single(sub_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
