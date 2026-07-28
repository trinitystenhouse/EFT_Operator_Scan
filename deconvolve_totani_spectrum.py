#!/usr/bin/env python3
"""
deconvolve_totani_spectrum.py
==============================
Correct approach to testing whether DM-photon EFT scattering can resolve
the Totani tension.

Conceptual inversion
--------------------
The scan_tension_resolution.py approach asked:
    "Take a PPPC template, apply scattering, does it fit the Totani data
     better and reduce the required <sigma v>?"
This fails because tau << 1 means scattering barely changes the template.

The correct approach asks:
    "If DM-photon scattering (m_chi, Lambda) HAS been acting on photons
     crossing the galactic halo, then the Totani observed spectrum is the
     POST-scattering spectrum. Deconvolve the scattering to recover the
     INTRINSIC annihilation spectrum, then fit a PPPC template to that."

Why this helps the tension
--------------------------
Scattering moves flux DOWN in energy (inscatter from high E -> low E bins).
If scattering has enhanced the 21 GeV peak, the deconvolved spectrum is
FLATTER and peaks at HIGHER energy. A flatter, higher-energy intrinsic
spectrum is consistent with:
  (a) A heavier annihilation mass m_ann (PPPC template peaks higher)
  (b) A lower required <sigma v> (because the dSph limit loosens at high mass)
This is the mechanism by which scattering could resolve the tension.

Deconvolution
-------------
The single-scatter transfer operator is:
    Phi_obs = T * Phi_intrinsic
    T_{ij} = diag(exp(-tau)) + K * diag(tau * exp(-tau))

For tau << 1, to first order:
    T^{-1} approx I - (K - I) * diag(tau)
So:
    Phi_intrinsic_recovered = Phi_obs * exp(+tau) - K * (tau * Phi_obs)
                            = Phi_obs + tau * Phi_obs - K * (tau * Phi_obs)
                            = Phi_obs + tau * (I - K) * Phi_obs

For the extended source grid (photons above the observed range):
The deconvolution on the observed bins uses the kernel projected to the
observed energies. Photons that were downscattered from above the top
Totani bin (814 GeV) into the observed range appear as additional flux
at low energies; deconvolving removes them, flattening the spectrum.

The exact (non-perturbative) deconvolution uses numpy.linalg.solve:
    T * Phi_intrinsic = Phi_obs
    Phi_intrinsic = linalg.solve(T, Phi_obs)

Output
------
For each (m_chi, Lambda):
  - Phi_intrinsic_recovered [nE] -- deconvolved spectrum
  - Best-fit m_ann, <sigma v>, chi2 from PPPC fit to recovered spectrum
  - Tension factor vs dSph at that m_ann
  - Delta tension = tension_deconvolved - tension_original

Negative delta tension means scattering RESOLVES the tension.
The (m_chi, Lambda) boundary where delta_tension = 0 is the resolution threshold.

Usage
-----
  python deconvolve_totani_spectrum.py --quick
  python deconvolve_totani_spectrum.py --all-operators --n-mchi 30 --n-lambda 30
  python deconvolve_totani_spectrum.py --operator dipole_magnetic --plot
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from helpers.trinity_plotting import current_savefig_kwargs, set_plot_style
from helpers.fermi_plotting import (
    CONF_ANNOT_FS,
    CONF_HEADER_FS,
    CONF_LABEL_FS,
    CONF_LEGEND_FS,
    CONF_TICK_FS,
    CONF_TITLE_FS,
)

from core.totani_data_loader import HaloSpectrum, _MCMC_DIRS, load_halo_spectrum
from core.spectral_reshaping import (
    ReshapingConfig,
    best_fit_normalization,
    build_dsigma_grid,
    build_kernel,
    compute_tau_spectrum,
    energy_flux_transfer_matrix,
    pppc_energy_flux_template,
    smooth_nfw_sigma_v_from_norm,
    apply_single_scatter_transfer,
)
from core.attenuation_eft import (
    dsigma_dOmega_fermionic,
    dsigma_dOmega_scalar,
    FB_TO_CM2,
    COS_THETA_MAX,
    eft_validity_lambda_curve,
    roi_tau_prefactor,
    unitarity_lambda_curve,
)
from scan_tension_resolution_old import (
    OperatorSpec,
    ALL_OPERATORS,
    operator_key,
    dsph_upper_limit,
    evaluate_pure_annihilation_mass,
    _normalise_ann_channel,
    _DEFAULT_CL_THRESHOLD,
    _DEFAULT_DELTA_CHI2_2DOF,
    _cl_from_delta_chi2_2d,
    _delta_chi2_from_cl_2d,
)


# ---------------------------------------------------------------------------
# Deconvolution
# ---------------------------------------------------------------------------

def build_transfer_matrix(
    tau:  np.ndarray,   # (nE,) optical depth per observed bin
    K:    np.ndarray,   # (nE, nE) redistribution kernel (photon number)
) -> np.ndarray:
    """
    Build the full (nE x nE) single-scatter transfer matrix T such that:
        Phi_obs = T @ Phi_intrinsic

    T_{ij} = exp(-tau_i) * delta_{ij}
           + K_{ij} * tau_j * exp(-tau_j)

    The first term is survival (photon stays at energy i).
    The second term is inscatter: a photon at energy j scatters into bin i.
    """
    nE = len(tau)
    e_tau = np.exp(-tau)                    # (nE,)
    T = np.diag(e_tau)                      # survival diagonal
    T += K * (tau * e_tau)[np.newaxis, :]   # inscatter: K_{ij} * tau_j * e^{-tau_j}
    return T


def deconvolve_spectrum(
    phi_obs:  np.ndarray,   # (nE,) observed halo spectrum
    tau:      np.ndarray,   # (nE,) optical depth
    K:        np.ndarray,   # (nE, nE) redistribution kernel
    method:   str = "exact",
) -> np.ndarray:
    """
    Recover the intrinsic annihilation spectrum from the observed post-scattering
    spectrum.

    method='exact'   : solve T @ Phi_intrinsic = Phi_obs via linalg.solve
    method='firstorder': Phi_intrinsic ~ Phi_obs + tau * (I - K) @ Phi_obs
                         Valid when tau << 1.

    Returns Phi_intrinsic_recovered (nE,).
    """
    if method == "firstorder":
        # First-order inversion: valid for tau << 1
        # T^{-1} ~ I - (K - I) * diag(tau)
        # Phi_intrinsic = Phi_obs + tau * (Phi_obs - K @ Phi_obs)
        correction = tau * (phi_obs - K @ phi_obs)
        return phi_obs + correction

    elif method == "exact":
        T = build_transfer_matrix(tau, K)
        try:
            phi_intrinsic = np.linalg.solve(T, phi_obs)
        except np.linalg.LinAlgError:
            # Fallback to first-order if matrix is singular
            correction = tau * (phi_obs - K @ phi_obs)
            phi_intrinsic = phi_obs + correction
        return phi_intrinsic

    else:
        raise ValueError(f"method must be 'exact' or 'firstorder', got {method!r}")


# ---------------------------------------------------------------------------
# PPPC template fitting over a grid of annihilation masses
# ---------------------------------------------------------------------------

def fit_pppc_to_spectrum(
    phi_target:    np.ndarray,   # (nE,) spectrum to fit
    phi_err:       np.ndarray,   # (nE,) uncertainties
    mask:          np.ndarray,   # (nE,) bool — which bins to include
    E_bins_GeV:    np.ndarray,   # (nE,) energy bin centres
    ann_masses:    np.ndarray,   # (nM,) annihilation masses to try
    ann_channel:   str = "WW",
    pppc_table:    Optional[str] = None,
    template_bank: Optional[dict] = None,
    cl_threshold:  float = 0.90,
    selection_mode: str = "max_scattering_effect",
) -> dict:
    """
    Fit PPPC annihilation templates to phi_target by scanning ann_masses
    and finding the best-fit (m_ann, norm, chi2, <sigma v>).

    Returns dict with keys:
        best_ann_mass, best_norm, best_chi2, best_sigmav,
        chi2_arr, norm_arr, sigmav_arr  (all vs ann_masses)
    """
    nM = len(ann_masses)
    chi2_arr  = np.full(nM, np.nan)
    norm_arr  = np.full(nM, np.nan)
    sigmav_arr = np.full(nM, np.nan)

    if template_bank is None:
        templates = np.full((nM, len(E_bins_GeV)), np.nan)
        valid_templates = np.zeros(nM, dtype=bool)
        for i, m_ann in enumerate(ann_masses):
            try:
                src = pppc_energy_flux_template(
                    E_bins_GeV, float(m_ann),
                    channel=ann_channel,
                    primary="gamma",
                    table_path=pppc_table,
                    normalise=False,
                )
            except Exception:
                continue
            if not np.any(src > 0):
                continue
            templates[i] = src
            valid_templates[i] = True
    else:
        templates = np.asarray(template_bank["templates"], dtype=float)
        valid_templates = np.asarray(template_bank["valid"], dtype=bool)

    fit_mask = np.asarray(mask, dtype=bool) & np.isfinite(phi_target) & np.isfinite(phi_err) & (phi_err > 0.0)
    if np.any(fit_mask):
        src_m = templates[:, fit_mask]
        y_m = np.asarray(phi_target, dtype=float)[fit_mask]
        invvar = 1.0 / np.asarray(phi_err, dtype=float)[fit_mask] ** 2
        numer = np.sum(src_m * y_m[None, :] * invvar[None, :], axis=1)
        denom = np.sum(src_m * src_m * invvar[None, :], axis=1)
        norm_arr = np.divide(
            numer,
            denom,
            out=np.full(nM, np.nan),
            where=valid_templates & np.isfinite(denom) & (denom > 0.0),
        )
        model_m = norm_arr[:, None] * src_m
        chi2_arr = np.sum((model_m - y_m[None, :]) ** 2 * invvar[None, :], axis=1)
        bad = (~valid_templates) | (~np.isfinite(norm_arr)) | (norm_arr <= 0.0)
        chi2_arr[bad] = np.nan
        norm_arr[bad] = np.nan
        for i, m_ann in enumerate(ann_masses):
            if np.isfinite(norm_arr[i]):
                sigmav_arr[i] = smooth_nfw_sigma_v_from_norm(norm_arr[i], float(m_ann))

    if not np.any(np.isfinite(chi2_arr)):
        return dict(
            best_ann_mass=np.nan, best_norm=np.nan, best_chi2=np.nan,
            best_sigmav=np.nan, best_tension=np.nan,
            chi2_arr=chi2_arr, norm_arr=norm_arr, sigmav_arr=sigmav_arr,
        )

    # Compute tension at every mass point
    tension_arr = np.full(len(ann_masses), np.nan)
    for i, m in enumerate(ann_masses):
        if not np.isfinite(sigmav_arr[i]):
            continue
        dsph_i = dsph_upper_limit(float(m), ann_channel)
        if np.isfinite(dsph_i) and dsph_i > 0:
            tension_arr[i] = sigmav_arr[i] / dsph_i

    # Select within a chi2 tolerance of the global minimum corresponding to
    # the chosen CL in the (m_ann, norm) 2D fit subspace.
    chi2_min = float(np.nanmin(chi2_arr))
    try:
        delta_chi2_window = float(_delta_chi2_from_cl_2d(float(cl_threshold)))
    except Exception:
        delta_chi2_window = np.nan
    if (not np.isfinite(delta_chi2_window)) or (delta_chi2_window <= 0.0):
        delta_chi2_window = float(_DEFAULT_DELTA_CHI2_2DOF)
    chi2_ok = np.isfinite(chi2_arr) & (chi2_arr <= chi2_min + delta_chi2_window)

    if selection_mode in ("max_scattering_effect", "min_tension"):
        if np.any(chi2_ok & np.isfinite(tension_arr)):
            # Most aggressive tension reduction while remaining CL-consistent.
            tension_in_window = np.where(
                chi2_ok & np.isfinite(tension_arr), tension_arr, np.nan
            )
            best_i = int(np.nanargmin(tension_in_window))
        else:
            # Fallback: use chi2 minimum
            best_i = int(np.nanargmin(chi2_arr))
    elif selection_mode == "best_chi2":
        best_i = int(np.nanargmin(chi2_arr))
    else:
        raise ValueError(
            f"selection_mode must be one of "
            f"{{'max_scattering_effect','min_tension','best_chi2'}}, got {selection_mode!r}"
        )

    best_m      = float(ann_masses[best_i])
    best_sv     = float(sigmav_arr[best_i])
    best_tension = float(tension_arr[best_i]) if np.isfinite(tension_arr[best_i]) else np.nan

    return dict(
        best_ann_mass=best_m,
        best_norm=float(norm_arr[best_i]),
        best_chi2=float(chi2_arr[best_i]),
        best_sigmav=best_sv,
        best_tension=best_tension,
        fit_cl_threshold=float(cl_threshold),
        fit_delta_chi2_2d=float(delta_chi2_window),
        fit_selection_mode=str(selection_mode),
        chi2_arr=chi2_arr,
        norm_arr=norm_arr,
        sigmav_arr=sigmav_arr,
        tension_arr=tension_arr,
    )


def precompute_pppc_template_bank(
    E_bins_GeV: np.ndarray,
    ann_masses: np.ndarray,
    ann_channel: str,
    pppc_table: Optional[str] = None,
) -> dict:
    """Cache PPPC templates once per channel so every scattering point reuses them."""
    templates = np.full((len(ann_masses), len(E_bins_GeV)), np.nan, dtype=float)
    valid = np.zeros(len(ann_masses), dtype=bool)
    for i, m_ann in enumerate(ann_masses):
        try:
            src = pppc_energy_flux_template(
                E_bins_GeV,
                float(m_ann),
                channel=ann_channel,
                primary="gamma",
                table_path=pppc_table,
                normalise=False,
            )
        except Exception:
            continue
        if np.any(src > 0):
            templates[i] = src
            valid[i] = True
    return {"templates": templates, "valid": valid}


def _coupling_scale_power(spec: OperatorSpec) -> float:
    """Power p where EFT cross sections scale as Lambda^-p, or y_eff^p."""
    if spec.coupling_axis == "y_eff":
        return 2.0
    if spec.name in ("dipole_magnetic", "dipole_electric"):
        return 2.0
    if spec.name in ("charge_radius", "anapole") or spec.dm_type == "scalar":
        return 4.0
    if "rayleigh" in spec.name:
        return 6.0
    return 2.0


def _scale_sigma_tot_for_coupling(
    sigma_ref: np.ndarray,
    *,
    spec: OperatorSpec,
    coupling: float,
    reference_coupling: float,
) -> np.ndarray:
    p = _coupling_scale_power(spec)
    if spec.coupling_axis == "y_eff":
        scale = (float(coupling) / float(reference_coupling)) ** p
    else:
        scale = (float(reference_coupling) / float(coupling)) ** p
    return np.asarray(sigma_ref, dtype=float) * scale


def build_scattering_shape_cache(
    *,
    spec: OperatorSpec,
    scatter_mass: float,
    reference_coupling: float,
    halo: HaloSpectrum,
    l_grid: np.ndarray,
    b_grid: np.ndarray,
    n_theta: int,
    apply_roi_weight: bool,
) -> dict:
    """Build the coupling-independent scattering kernel/sigma shape once."""
    E = halo.E_bins_GeV
    cfg = ReshapingConfig(
        m_chi=float(scatter_mass),
        Lambda=float(reference_coupling) if spec.coupling_axis == "Lambda" else 1e3,
        dm_type=spec.dm_type,
        operator=spec.name,
        c_s=spec.c_s,
        c_p=spec.c_p,
        c_phi=spec.c_phi,
        majorana=spec.majorana,
        y_eff=float(reference_coupling) if spec.coupling_axis == "y_eff" else 1.0,
        l_grid=l_grid,
        b_grid=b_grid,
        n_theta=n_theta,
        apply_roi_weight=apply_roi_weight,
        roi_half_angle_deg=60.0 if apply_roi_weight else None,
        E_bins=E,
        phi_0=halo.phi,
        phi_data=halo.phi,
        phi_err=halo.phi_err_sym,
        fit_normalization=True,
        max_tau_single_scatter=None,
        require_lambda_gt_mdm=False,
    )
    cos_theta, dsig, sigma_tot = build_dsigma_grid(cfg)
    K = build_kernel(cfg, cos_theta, dsig, sigma_tot)
    return {
        "reference_coupling": float(reference_coupling),
        "sigma_tot_ref": np.asarray(sigma_tot, dtype=float),
        "K": K,
        "roi_prefactor": float(roi_tau_prefactor(l_grid, b_grid)),
    }


# ---------------------------------------------------------------------------
# Single deconvolution point
# ---------------------------------------------------------------------------

@dataclass
class DeconvPoint:
    scatter_mass:     float
    coupling:         float
    tau_max:          float
    sigma_scat:       float
    # Deconvolved spectrum
    phi_recovered:    np.ndarray
    # Fit to deconvolved spectrum
    deconv_ann_mass:  float    # best-fit m_ann to deconvolved spectrum
    deconv_sigmav:    float    # required <sigma v> from deconvolved spectrum
    deconv_tension:   float    # deconv_sigmav / dSph(deconv_ann_mass)
    deconv_chi2:      float
    # Original Totani fit (for comparison)
    orig_ann_mass:    float
    orig_sigmav:      float
    orig_tension:     float
    orig_chi2:        float
    # Key diagnostic
    delta_tension:    float    # deconv_tension - orig_tension  (<0 = helps)
    delta_ann_mass:   float    # deconv_ann_mass - orig_ann_mass (>0 = higher mass inferred)
    eft_valid:        bool
    K:                np.ndarray
    tau:              np.ndarray


def evaluate_deconv_point(
    *,
    spec:           OperatorSpec,
    scatter_mass:   float,
    coupling:       float,
    halo:           HaloSpectrum,
    mask:           np.ndarray,
    ann_masses:     np.ndarray,
    ann_channel:    str,
    l_grid:         np.ndarray,
    b_grid:         np.ndarray,
    n_theta:        int = 200,
    apply_roi_weight: bool = True,
    max_tau:        float = 0.3,
    pppc_table:     Optional[str] = None,
    deconv_method:  str = "exact",
    template_bank:  Optional[dict] = None,
    orig_fit:       Optional[dict] = None,
    shape_cache:    Optional[dict] = None,
    fit_cl_threshold: float = 0.90,
    fit_selection_mode: str = "max_scattering_effect",
) -> DeconvPoint:
    E   = halo.E_bins_GeV
    phi = halo.phi
    err = halo.phi_err_sym
    nE  = len(E)
    _nan_arr = np.full(nE, np.nan)
    _nan_K   = np.full((nE, nE), np.nan)

    # EFT validity check
    eft_valid = True
    if spec.coupling_axis == "Lambda":
        if coupling <= scatter_mass:
            eft_valid = False
        lam_kin = float(eft_validity_lambda_curve(
            np.array([scatter_mass]), omega_max=float(np.max(E)))[0])
        lam_unit = float(unitarity_lambda_curve(
            spec.name, np.array([scatter_mass]))[0])
        if np.isfinite(lam_kin)  and coupling < lam_kin:
            eft_valid = False
        if np.isfinite(lam_unit) and coupling < lam_unit:
            eft_valid = False

    # Original Totani fit (no scattering assumed)
    orig = orig_fit
    if orig is None:
        orig = fit_pppc_to_spectrum(
            phi, err, mask, E, ann_masses, ann_channel, pppc_table,
            template_bank=template_bank,
            cl_threshold=fit_cl_threshold,
            selection_mode=fit_selection_mode,
        )

    nan_point = DeconvPoint(
        scatter_mass=scatter_mass, coupling=coupling,
        tau_max=np.nan, sigma_scat=np.nan,
        phi_recovered=_nan_arr.copy(),
        deconv_ann_mass=np.nan, deconv_sigmav=np.nan,
        deconv_tension=np.nan, deconv_chi2=np.nan,
        orig_ann_mass=orig["best_ann_mass"],
        orig_sigmav=orig["best_sigmav"],
        orig_tension=orig["best_tension"],
        orig_chi2=orig["best_chi2"],
        delta_tension=np.nan, delta_ann_mass=np.nan,
        eft_valid=eft_valid, K=_nan_K.copy(), tau=_nan_arr.copy(),
    )

    if shape_cache is None:
        shape_cache = build_scattering_shape_cache(
            spec=spec,
            scatter_mass=float(scatter_mass),
            reference_coupling=float(coupling),
            halo=halo,
            l_grid=l_grid,
            b_grid=b_grid,
            n_theta=n_theta,
            apply_roi_weight=apply_roi_weight,
        )

    sigma_tot = _scale_sigma_tot_for_coupling(
        shape_cache["sigma_tot_ref"],
        spec=spec,
        coupling=float(coupling),
        reference_coupling=float(shape_cache["reference_coupling"]),
    )
    tau = (float(shape_cache["roi_prefactor"]) / float(scatter_mass)) * np.asarray(sigma_tot, dtype=float)
    tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
    tau = np.where(tau > 0.0, tau, 0.0)
    tau_max = float(np.nanmax(np.where(np.isfinite(tau), tau, 0.0)))

    # Scattering cross section at 20 GeV for diagnostics
    sigma_scat = float(np.interp(20.0, E, sigma_tot, left=sigma_tot[0], right=sigma_tot[-1]))

    if tau_max > max_tau:
        # Multi-scatter regime — deconvolution invalid
        return DeconvPoint(
            scatter_mass=scatter_mass, coupling=coupling,
            tau_max=tau_max, sigma_scat=sigma_scat,
            phi_recovered=_nan_arr.copy(),
            deconv_ann_mass=np.nan, deconv_sigmav=np.nan,
            deconv_tension=np.nan, deconv_chi2=np.nan,
            orig_ann_mass=orig["best_ann_mass"],
            orig_sigmav=orig["best_sigmav"],
            orig_tension=orig["best_tension"],
            orig_chi2=orig["best_chi2"],
            delta_tension=np.nan, delta_ann_mass=np.nan,
            eft_valid=eft_valid, K=_nan_K.copy(), tau=tau,
        )

    K = np.asarray(shape_cache["K"], dtype=float)

    # Deconvolve: recover intrinsic spectrum from observed Totani data
    phi_intrinsic = deconvolve_spectrum(phi, tau, K, method=deconv_method)

    # Fit PPPC to deconvolved spectrum
    deconv = fit_pppc_to_spectrum(
        phi_intrinsic, err, mask, E, ann_masses, ann_channel, pppc_table,
        template_bank=template_bank,
        cl_threshold=fit_cl_threshold,
        selection_mode=fit_selection_mode,
    )

    delta_tension  = deconv["best_tension"] - orig["best_tension"] \
                     if (np.isfinite(deconv["best_tension"]) and np.isfinite(orig["best_tension"])) \
                     else np.nan
    delta_ann_mass = deconv["best_ann_mass"] - orig["best_ann_mass"] \
                     if (np.isfinite(deconv["best_ann_mass"]) and np.isfinite(orig["best_ann_mass"])) \
                     else np.nan

    return DeconvPoint(
        scatter_mass=scatter_mass, coupling=coupling,
        tau_max=tau_max, sigma_scat=sigma_scat,
        phi_recovered=phi_intrinsic,
        deconv_ann_mass=deconv["best_ann_mass"],
        deconv_sigmav=deconv["best_sigmav"],
        deconv_tension=deconv["best_tension"],
        deconv_chi2=deconv["best_chi2"],
        orig_ann_mass=orig["best_ann_mass"],
        orig_sigmav=orig["best_sigmav"],
        orig_tension=orig["best_tension"],
        orig_chi2=orig["best_chi2"],
        delta_tension=delta_tension,
        delta_ann_mass=delta_ann_mass,
        eft_valid=eft_valid, K=K, tau=tau,
    )


# ---------------------------------------------------------------------------
# 2D scan
# ---------------------------------------------------------------------------

def run_deconv_scan(
    spec:          OperatorSpec,
    halo:          HaloSpectrum,
    mask:          np.ndarray,
    scatter_masses: np.ndarray,
    couplings:     np.ndarray,
    ann_masses:    np.ndarray,
    args:          argparse.Namespace,
    ann_channel:   str,
    outdir:        Path,
) -> dict:
    E   = halo.E_bins_GeV
    nE  = len(E)
    nS  = len(scatter_masses)
    nC  = len(couplings)

    l_grid = np.linspace(-60.0, 60.0, 15)
    b_grid = np.concatenate([np.linspace(-60.0, -10.0, 8),
                              np.linspace(10.0, 60.0, 8)])

    # Grids
    tau_max_grid       = np.full((nS, nC), np.nan)
    sigma_scat_grid    = np.full((nS, nC), np.nan)
    deconv_mass_grid   = np.full((nS, nC), np.nan)
    deconv_sigmav_grid = np.full((nS, nC), np.nan)
    deconv_tension_grid= np.full((nS, nC), np.nan)
    deconv_chi2_grid   = np.full((nS, nC), np.nan)
    delta_tension_grid = np.full((nS, nC), np.nan)
    delta_mass_grid    = np.full((nS, nC), np.nan)
    eft_valid_grid     = np.zeros((nS, nC), dtype=bool)

    best_delta: Optional[DeconvPoint] = None
    best_eft: Optional[DeconvPoint] = None
    total = nS * nC
    done  = 0
    t0    = time.time()

    # PPPC templates and the original no-scattering fit are independent of
    # scattering parameters, so compute them once for the whole scan.
    template_bank = precompute_pppc_template_bank(
        E,
        ann_masses,
        ann_channel,
        args.pppc_gamma_table,
    )
    orig_fit = fit_pppc_to_spectrum(
        halo.phi, halo.phi_err_sym, mask, E, ann_masses, ann_channel,
        args.pppc_gamma_table,
        template_bank=template_bank,
        cl_threshold=args.fit_cl,
        selection_mode=args.fit_selection,
    )
    orig_tension = orig_fit["best_tension"]
    orig_mass    = orig_fit["best_ann_mass"]
    print(f"\n  Original Totani fit: m_ann={orig_mass:.0f} GeV, "
          f"<sv>={orig_fit['best_sigmav']:.2e}, tension={orig_tension:.2f}x")
    print(f"  Scanning {nS} scatter masses x {nC} couplings = {total} points")

    reference_coupling = float(couplings[0]) if len(couplings) else 1.0
    for s, m_scat in enumerate(scatter_masses):
        shape_cache = build_scattering_shape_cache(
            spec=spec,
            scatter_mass=float(m_scat),
            reference_coupling=reference_coupling,
            halo=halo,
            l_grid=l_grid,
            b_grid=b_grid,
            n_theta=args.n_theta,
            apply_roi_weight=args.apply_roi_weight,
        )
        for j, coupling in enumerate(couplings):
            pt = evaluate_deconv_point(
                spec=spec,
                scatter_mass=float(m_scat),
                coupling=float(coupling),
                halo=halo,
                mask=mask,
                ann_masses=ann_masses,
                ann_channel=ann_channel,
                l_grid=l_grid,
                b_grid=b_grid,
                n_theta=args.n_theta,
                apply_roi_weight=args.apply_roi_weight,
                max_tau=args.max_tau_single_scatter,
                pppc_table=args.pppc_gamma_table,
                deconv_method=args.deconv_method,
                template_bank=template_bank,
                orig_fit=orig_fit,
                shape_cache=shape_cache,
                fit_cl_threshold=args.fit_cl,
                fit_selection_mode=args.fit_selection,
            )

            tau_max_grid[s, j]        = pt.tau_max
            sigma_scat_grid[s, j]     = pt.sigma_scat
            deconv_mass_grid[s, j]    = pt.deconv_ann_mass
            deconv_sigmav_grid[s, j]  = pt.deconv_sigmav
            deconv_tension_grid[s, j] = pt.deconv_tension
            deconv_chi2_grid[s, j]    = pt.deconv_chi2
            delta_tension_grid[s, j]  = pt.delta_tension
            delta_mass_grid[s, j]     = pt.delta_ann_mass
            eft_valid_grid[s, j]      = pt.eft_valid

            if (np.isfinite(pt.delta_tension)
                    and (best_delta is None
                         or pt.delta_tension < best_delta.delta_tension)):
                best_delta = pt

            if (pt.eft_valid and np.isfinite(pt.delta_tension)
                    and (best_eft is None
                         or pt.delta_tension < best_eft.delta_tension)):
                best_eft = pt

            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed * (total - done) / done if done > 0 else 0
                dt = f"{pt.delta_tension:+.4f}" if np.isfinite(pt.delta_tension) else "nan"
                print(f"  {done:4d}/{total}  m_scat={m_scat:.3g} GeV  "
                      f"Lambda={coupling:.2e}  tau_max={pt.tau_max:.2e}  "
                      f"delta_tension={dt}  EFT={'Y' if pt.eft_valid else 'N'}  "
                      f"[{elapsed:.0f}s, ETA {eta:.0f}s]")

    # Best point summary
    if best_delta is not None:
        print(f"\n  Best delta_tension: {best_delta.delta_tension:+.4f}x")
        print(f"    m_scat={best_delta.scatter_mass:.3g} GeV, "
              f"Lambda={best_delta.coupling:.3e} GeV")
        print(f"    Orig: m_ann={best_delta.orig_ann_mass:.0f} GeV, "
              f"tension={best_delta.orig_tension:.2f}x")
        print(f"    Deconv: m_ann={best_delta.deconv_ann_mass:.0f} GeV, "
              f"tension={best_delta.deconv_tension:.2f}x")
        print(f"    tau_max={best_delta.tau_max:.2e}, "
              f"EFT={'valid' if best_delta.eft_valid else 'INVALID'}")

    # Find resolution threshold in EFT-valid region
    eft_mask = eft_valid_grid & np.isfinite(delta_tension_grid)
    diag_idx = None
    diag_m_scat = np.nan
    diag_coupling = np.nan
    diag_delta_tension = np.nan
    diag_deconv_mass = np.nan
    diag_deconv_tension = np.nan
    if np.any(eft_mask):
        min_dt_eft = float(np.nanmin(delta_tension_grid[eft_mask]))
        idx = np.unravel_index(
            np.nanargmin(np.where(eft_mask, delta_tension_grid, np.nan)),
            delta_tension_grid.shape)
        diag_idx = idx
        diag_m_scat = float(scatter_masses[idx[0]])
        diag_coupling = float(couplings[idx[1]])
        diag_delta_tension = float(delta_tension_grid[idx])
        diag_deconv_mass = float(deconv_mass_grid[idx]) if np.isfinite(deconv_mass_grid[idx]) else np.nan
        diag_deconv_tension = float(deconv_tension_grid[idx]) if np.isfinite(deconv_tension_grid[idx]) else np.nan
        print(f"\n  Best EFT-valid point: delta_tension={min_dt_eft:+.4f}x")
        print(f"    m_scat={diag_m_scat:.3g} GeV, "
              f"Lambda={diag_coupling:.3e} GeV")
        print(f"    Deconv tension: {deconv_tension_grid[idx]:.3f}x  "
              f"(orig: {orig_tension:.3f}x)")
    else:
        min_dt_eft = np.nan
        print("\n  No EFT-valid points with finite delta_tension found.")

    # Save
    result = dict(
        operator        = spec.name,
        dm_type         = spec.dm_type,
        majorana        = spec.majorana,
        coupling_axis   = spec.coupling_axis,
        label           = spec.label,
        ann_channel     = ann_channel,
        halo_profile    = args.halo_profile,
        deconv_method   = args.deconv_method,
        fit_cl_threshold = np.float32(args.fit_cl),
        fit_delta_chi2_2d = np.float32(orig_fit.get("fit_delta_chi2_2d", np.nan)),
        fit_selection_mode = str(args.fit_selection),
        scatter_masses_GeV    = scatter_masses.astype(np.float32),
        couplings             = couplings.astype(np.float32),
        ann_masses_GeV        = ann_masses.astype(np.float32),
        E_bins_GeV            = E.astype(np.float32),
        tau_max_grid          = tau_max_grid.astype(np.float32),
        sigma_scat_cm2        = sigma_scat_grid.astype(np.float32),
        deconv_ann_mass_GeV   = deconv_mass_grid.astype(np.float32),
        deconv_sigmav_cm3_s   = deconv_sigmav_grid.astype(np.float32),
        deconv_tension        = deconv_tension_grid.astype(np.float32),
        deconv_chi2           = deconv_chi2_grid.astype(np.float32),
        delta_tension         = delta_tension_grid.astype(np.float32),
        delta_ann_mass_GeV    = delta_mass_grid.astype(np.float32),
        eft_valid_mask        = eft_valid_grid.astype(np.uint8),
        orig_ann_mass_GeV     = np.float32(orig_mass),
        orig_sigmav_cm3_s     = np.float32(orig_fit["best_sigmav"]),
        orig_tension          = np.float32(orig_tension),
        orig_chi2             = np.float32(orig_fit["best_chi2"]),
        best_eft_delta_tension = np.float32(min_dt_eft),
        diag_best_scatter_mass_GeV = np.float32(diag_m_scat),
        diag_best_coupling = np.float32(diag_coupling),
        diag_best_delta_tension = np.float32(diag_delta_tension),
        diag_best_deconv_ann_mass_GeV = np.float32(diag_deconv_mass),
        diag_best_deconv_tension = np.float32(diag_deconv_tension),
        phi_data              = halo.phi.astype(np.float32),
        phi_err_sym           = halo.phi_err_sym.astype(np.float32),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "deconv_scan.npz"
    np.savez_compressed(str(npz_path), **result)
    print(f"  Saved: {npz_path}")

    _write_summary(result, outdir / "deconv_summary.txt")
    if args.plot:
        plot_point = best_eft if best_eft is not None else best_delta
        _make_deconv_plots(result, plot_point, halo, spec, outdir)

    return result


# ---------------------------------------------------------------------------
# Summary and plots
# ---------------------------------------------------------------------------

def _write_summary(result: dict, path: Path) -> None:
    lines = [
        "Deconvolution scan summary",
        "==========================",
        f"operator     : {result['operator']} ({result['dm_type']}, "
        f"{'Majorana' if result['majorana'] else 'Dirac/scalar'})",
        f"label        : {result['label']}",
        f"channel      : {result['ann_channel']}",
        f"halo profile : {result['halo_profile']}",
        f"deconv method: {result['deconv_method']}",
        f"fit CL window: {100.0*float(result.get('fit_cl_threshold', np.nan)):.1f}% "
        f"(Delta chi2_2D <= {float(result.get('fit_delta_chi2_2d', np.nan)):.3f})",
        f"fit selector : {result.get('fit_selection_mode', 'max_scattering_effect')}",
        "",
        "Original Totani fit (no scattering assumed):",
        f"  m_ann                  : {float(result['orig_ann_mass_GeV']):.1f} GeV",
        f"  <sigma v>_eff          : {float(result['orig_sigmav_cm3_s']):.4e} cm^3/s",
        f"  tension factor         : {float(result['orig_tension']):.3f}x",
        f"  chi2                   : {float(result['orig_chi2']):.3f}",
        "",
        "Best EFT-valid deconvolved improvement:",
        f"  min delta_tension      : {float(result['best_eft_delta_tension']):+.4f}x",
        f"  m_scat (diag point)    : {float(result.get('diag_best_scatter_mass_GeV', np.nan)):.4g} GeV",
        f"  coupling (diag point)  : {float(result.get('diag_best_coupling', np.nan)):.4g}",
        f"  m_ann (diag deconv fit): {float(result.get('diag_best_deconv_ann_mass_GeV', np.nan)):.1f} GeV",
        "  (negative = scattering assumption lowers tension after deconvolution)",
        "",
        "Interpretation:",
        "  delta_tension < 0 : assuming this scattering has occurred, the",
        "                      true annihilation spectrum requires LESS <sv>",
        "                      than Totani infers. Tension is reduced.",
        "  delta_tension > 0 : scattering makes the inferred tension WORSE.",
        "  delta_tension ~ 0 : scattering has negligible effect (tau << 1).",
    ]
    path.write_text("\n".join(lines))
    print(f"  Summary: {path}")


def _make_deconv_plots(
    result:     dict,
    best:       Optional[DeconvPoint],
    halo:       HaloSpectrum,
    spec:       OperatorSpec,
    outdir:     Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fs_title = max(13.0, CONF_TITLE_FS - 14)
    fs_label = max(12.0, CONF_LABEL_FS - 14)
    fs_tick = max(10.0, CONF_TICK_FS - 15)
    fs_leg = max(10.0, CONF_LEGEND_FS - 15)
    fs_header = max(14.0, CONF_HEADER_FS - 14)

    set_plot_style(
        style="dark_transparent",
        cmap_name="plasma",
        base_fontsize=fs_tick,
        linewidth=2.0,
        n_colors=10,
    )
    bg = plt.rcParams.get("figure.facecolor", "none")
    ax_bg = plt.rcParams.get("axes.facecolor", "none")
    text_col = plt.rcParams.get("text.color", "white")

    scatter_masses = np.asarray(result["scatter_masses_GeV"], float)
    couplings      = np.asarray(result["couplings"], float)
    delta_tension  = np.asarray(result["delta_tension"], float)
    eft_valid      = np.asarray(result["eft_valid_mask"], bool)
    E              = np.asarray(result["E_bins_GeV"], float)
    diag_m_scat = float(result.get("diag_best_scatter_mass_GeV", np.nan))
    diag_coupling = float(result.get("diag_best_coupling", np.nan))
    diag_deconv_mass = float(result.get("diag_best_deconv_ann_mass_GeV", np.nan))
    diag_delta_tension = float(result.get("diag_best_delta_tension", np.nan))
    orig_ann_mass = float(result.get("orig_ann_mass_GeV", np.nan))
    s              = 1e5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(bg)
    cmap = plt.get_cmap("plasma")
    col_main = cmap(0.72)
    col_secondary = cmap(0.35)
    col_ref = text_col

    # Panel 1: delta_tension map
    ax = axes[0]
    if scatter_masses.size > 1 and couplings.size > 1:
        S, C = np.meshgrid(scatter_masses, couplings, indexing="ij")
        vmax = max(0.01, float(np.nanpercentile(np.abs(delta_tension), 95)))
        cf = ax.contourf(
            np.log10(S), np.log10(C),
            np.clip(delta_tension, -vmax, vmax),
            levels=np.linspace(-vmax, vmax, 41),
            cmap="plasma",
        )
        plt.colorbar(cf, ax=ax, label=r"$\Delta$ tension (deconv $-$ orig)")
        ax.contour(np.log10(S), np.log10(C), delta_tension,
                   levels=[0.0], colors=[col_ref], linewidths=1.5)
        # EFT validity boundary
        ax.contour(np.log10(S), np.log10(C), eft_valid.astype(float),
                   levels=[0.5], colors=[col_secondary], linewidths=1.5, linestyles="--")
        if np.isfinite(diag_m_scat) and np.isfinite(diag_coupling) and diag_m_scat > 0 and diag_coupling > 0:
            ax.plot(
                np.log10(diag_m_scat),
                np.log10(diag_coupling),
                marker="*",
                ms=14,
                mfc="none",
                mec=col_main,
                mew=1.6,
                linestyle="None",
                label="Diagnostics EFT-best",
            )
            ax.legend(loc="best", fontsize=fs_leg)
        ax.set_xlabel(r"$\log_{10}(m_\chi/\mathrm{GeV})$", fontsize=fs_label)
        ax.set_ylabel(r"$\log_{10}(\Lambda/\mathrm{GeV})$", fontsize=fs_label)
        ax.set_title(r"$\Delta$tension (deconv $-$ orig) [black=0 line, red=EFT limit]", fontsize=fs_title)
        ax.tick_params(labelsize=fs_tick)

    # Panel 2: deconvolved spectrum at best point
    ax = axes[1]
    ax.errorbar(E, halo.phi * s,
                yerr=[halo.phi_err_lo * s, halo.phi_err_hi * s],
                fmt="o", color=col_ref, mec=col_ref, mfc="none", ms=5, lw=1.2, capsize=3,
                label="Totani observed")
    if best is not None and np.any(np.isfinite(best.phi_recovered)):
        ax.plot(E, best.phi_recovered * s, color=col_main, lw=2,
                label=rf"Deconvolved ($m_\chi$={best.scatter_mass:.2g} GeV, "
                      rf"$\Lambda$={best.coupling:.2e} GeV)")
        diag_mass_txt = (
            f"Diag masses: m_ann(orig)={orig_ann_mass:.0f} GeV, "
            f"m_ann(deconv)={diag_deconv_mass:.0f} GeV"
            if np.isfinite(orig_ann_mass) and np.isfinite(diag_deconv_mass)
            else "Diag masses: unavailable"
        )
        diag_dt_txt = (
            f"diag delta_tension={diag_delta_tension:+.4f}x"
            if np.isfinite(diag_delta_tension)
            else "diag delta_tension=nan"
        )
        ax.set_title(
            f"Best EFT-valid point (for diagnostics) | delta_tension={best.delta_tension:+.4f}x\n"
            f"Orig m_ann={best.orig_ann_mass:.0f} GeV → "
            f"Deconv m_ann={best.deconv_ann_mass:.0f} GeV | "
            f"EFT={'valid' if best.eft_valid else 'INVALID'}\n"
            f"{diag_mass_txt} | {diag_dt_txt}"
        )
    ax.set_xscale("log")
    ax.set_xlabel("E [GeV]", fontsize=fs_label)
    ax.set_ylabel(r"$E^2 dN/dE$ [$\times10^{-5}$ MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=fs_label)
    ax.axhline(0, color=col_ref, lw=0.8, alpha=0.6)
    ax.legend(fontsize=fs_leg)
    ax.tick_params(labelsize=fs_tick)

    # Panel 3: tau_max grid
    ax = axes[2]
    tau_grid = np.asarray(result["tau_max_grid"], float)
    if scatter_masses.size > 1 and couplings.size > 1:
        cf2 = ax.contourf(
            np.log10(S), np.log10(C),
            np.log10(np.where(tau_grid > 0, tau_grid, np.nan)),
            levels=40, cmap="plasma",
        )
        plt.colorbar(cf2, ax=ax, label=r"$\log_{10}(\tau_\mathrm{max})$")
        ax.contour(np.log10(S), np.log10(C), eft_valid.astype(float),
                   levels=[0.5], colors=["red"], linewidths=1.5, linestyles="--")
        ax.set_xlabel(r"$\log_{10}(m_\chi/\mathrm{GeV})$", fontsize=fs_label)
        ax.set_ylabel(r"$\log_{10}(\Lambda/\mathrm{GeV})$", fontsize=fs_label)
        ax.set_title(r"$\log_{10}(\tau_\mathrm{max})$ [red=EFT limit]", fontsize=fs_title)
        ax.tick_params(labelsize=fs_tick)

    fig.suptitle(f"{spec.label} | {result['ann_channel']} | {result['halo_profile']} | "
                 f"deconv method: {result['deconv_method']}", fontsize=fs_header)
    fig.tight_layout()
    out_path = outdir / "deconv_overview.png"
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    print(f"  Plot: {out_path}")


# ---------------------------------------------------------------------------
# 1D m_scat tension plots
# ---------------------------------------------------------------------------

_CHANNEL_COLORS = {"WW": "#F9A03F", "bb": "#7FCDBB", "tautau": "#CC4778"}
_CHANNEL_LABELS = {"WW": r"$WW$", "bb": r"$b\bar{b}$", "tautau": r"$\tau^+\tau^-$"}


def _op_colors(n: int):
    try:
        import matplotlib.pyplot as plt
        return plt.cm.plasma(np.linspace(0.15, 0.92, max(2, n)))
    except ImportError:
        return [f"C{i}" for i in range(n)]


def _base_ax(ax) -> None:
    """Apply axis styling consistent with the active repository plot style."""
    text_col = plt.rcParams.get("text.color", "white")
    grid_col = plt.rcParams.get("grid.color", text_col)
    ax.set_facecolor(plt.rcParams.get("axes.facecolor", "none"))
    for sp in ax.spines.values():
        sp.set_color(plt.rcParams.get("axes.edgecolor", text_col))
    ax.tick_params(colors=text_col)
    ax.xaxis.label.set_color(text_col)
    ax.yaxis.label.set_color(text_col)
    ax.title.set_color(text_col)
    ax.grid(True, alpha=0.22, color=grid_col)


def _add_reference_lines(ax, orig_tension: float, tension_threshold: float = 1.5) -> None:
    ax.axhline(
        orig_tension,
        color="white",
        lw=1.8,
        ls="--",
        label=rf"No scattering ($T={orig_tension:.2f}\times$)",
    )
    ax.axhline(1.0, color="lime", lw=1.4, ls="-", label="dSph limit (tension = 1)")
    ax.axhline(
        tension_threshold,
        color="cyan",
        lw=1.1,
        ls="-.",
        label=rf"Resolved cut ({tension_threshold}$\times$)",
    )
    ax.axhspan(0.0, 1.0, alpha=0.08, color="lime")


def _make_mscat_tension_plot(
    results:     list[dict],
    out_path:    Path,
    ann_channel: str,
    Lambda:      float,
) -> None:
    """One panel: tension vs m_scat for every operator in one channel."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_alpha(0.0)

    colors = _op_colors(len(results))
    orig_tension = np.nan

    for res, col in zip(results, colors):
        if not res:
            continue
        x = np.asarray(res["scatter_masses_GeV"], float)
        tension = np.asarray(res["deconv_tension"], float)
        eft_ok = np.asarray(res["eft_valid_mask"], bool)
        label = str(res["label"])

        if not np.isfinite(orig_tension):
            orig_tension = float(res["orig_tension"])
        if tension.ndim == 2:
            tension = tension[:, 0]
            eft_ok = eft_ok[:, 0]

        ax.plot(x, tension, color=col, lw=1.0, alpha=0.3)
        ax.plot(x, np.where(eft_ok, tension, np.nan), color=col, lw=2.2, label=label)
        ax.plot(x, np.where(~eft_ok, tension, np.nan), color=col, lw=1.0, ls=":", alpha=0.4)

    if np.isfinite(orig_tension):
        _add_reference_lines(ax, orig_tension)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ [GeV] (scattering DM mass)", fontsize=12)
    ax.set_ylabel(
        r"Tension $\langle\sigma v\rangle_\mathrm{eff}\,/\,\langle\sigma v\rangle_\mathrm{dSph}$",
        fontsize=11,
    )
    ch_label = _CHANNEL_LABELS.get(ann_channel, ann_channel)
    ax.set_title(
        rf"Deconvolved tension vs scattering mass | {ch_label} | $\Lambda={Lambda:.0f}$ GeV",
        fontsize=12,
    )
    ax.set_ylim(0.1, 20)
    leg = ax.legend(fontsize=8, ncol=2, loc="upper right")
    for t in leg.get_texts():
        t.set_color("white")
    _base_ax(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    print(f"  Plot: {out_path}")


def _make_channel_overlay_plot(
    results_by_channel: dict[str, list[dict]],
    out_path:           Path,
    Lambda:             float,
) -> None:
    """Panels by channel, showing tension vs m_scat for the available operators."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    channels = [ch for ch in ["WW", "bb", "tautau"] if ch in results_by_channel]
    if not channels:
        return

    fig, axes = plt.subplots(1, len(channels), figsize=(6 * len(channels), 6), sharey=True)
    if len(channels) == 1:
        axes = [axes]
    fig.patch.set_alpha(0.0)

    for ax, ch in zip(axes, channels):
        results = results_by_channel[ch]
        colors = _op_colors(len(results))
        orig_tension = np.nan

        for res, col in zip(results, colors):
            if not res:
                continue
            x = np.asarray(res["scatter_masses_GeV"], float)
            tension = np.asarray(res["deconv_tension"], float)
            eft_ok = np.asarray(res["eft_valid_mask"], bool)
            if tension.ndim == 2:
                tension = tension[:, 0]
                eft_ok = eft_ok[:, 0]
            if not np.isfinite(orig_tension):
                orig_tension = float(res["orig_tension"])

            ax.plot(x, tension, color=col, lw=0.9, alpha=0.28)
            ax.plot(x, np.where(eft_ok, tension, np.nan), color=col, lw=2.0, label=str(res["label"]))

        if np.isfinite(orig_tension):
            _add_reference_lines(ax, orig_tension)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=11)
        ax.set_title(_CHANNEL_LABELS.get(ch, ch), fontsize=13)
        ax.set_ylim(0.1, 20)
        leg = ax.legend(fontsize=7, ncol=2)
        for t in leg.get_texts():
            t.set_color("white")
        _base_ax(ax)

    axes[0].set_ylabel(
        r"Tension $\langle\sigma v\rangle_\mathrm{eff}\,/\,\langle\sigma v\rangle_\mathrm{dSph}$",
        fontsize=11,
    )
    fig.suptitle(
        rf"Deconvolved tension vs $m_\chi$ | $\Lambda={Lambda:.0f}$ GeV",
        fontsize=13,
        color="white",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    print(f"  Plot: {out_path}")


def _make_all_operators_channel_plot(
    results_by_channel: dict[str, list[dict]],
    out_path:           Path,
    Lambda:             float,
) -> None:
    """Three channel panels, each showing all operators vs m_scat."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError:
        return

    channels = [ch for ch in ["WW", "bb", "tautau"] if ch in results_by_channel]
    if not channels:
        return

    all_labels = []
    for ch in channels:
        for r in results_by_channel[ch]:
            if r and str(r["label"]) not in all_labels:
                all_labels.append(str(r["label"]))
    colors = _op_colors(len(all_labels))
    color_map = {lbl: col for lbl, col in zip(all_labels, colors)}

    fig, axes = plt.subplots(1, len(channels), figsize=(6.5 * len(channels), 6), sharey=True)
    if len(channels) == 1:
        axes = [axes]
    fig.patch.set_alpha(0.0)

    for ax, ch in zip(axes, channels):
        results = results_by_channel[ch]
        orig_tension = np.nan

        for res in results:
            if not res:
                continue
            label = str(res["label"])
            col = color_map.get(label, "white")
            x = np.asarray(res["scatter_masses_GeV"], float)
            tension = np.asarray(res["deconv_tension"], float)
            eft_ok = np.asarray(res["eft_valid_mask"], bool)
            if tension.ndim == 2:
                tension = tension[:, 0]
                eft_ok = eft_ok[:, 0]
            if not np.isfinite(orig_tension):
                orig_tension = float(res["orig_tension"])

            ax.plot(x, tension, color=col, lw=0.9, alpha=0.25)
            ax.plot(x, np.where(eft_ok, tension, np.nan), color=col, lw=2.2, label=label)
            ax.plot(x, np.where(~eft_ok, tension, np.nan), color=col, lw=1.0, ls=":", alpha=0.45)

        if np.isfinite(orig_tension):
            _add_reference_lines(ax, orig_tension)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.05, 25)
        ax.set_xlabel(r"$m_\chi$ [GeV] (scattering DM mass)", fontsize=11)
        ax.set_title(_CHANNEL_LABELS.get(ch, ch), fontsize=14, color="white")
        _base_ax(ax)

        if ax is axes[-1]:
            leg = ax.legend(fontsize=7.5, ncol=1, loc="upper right", framealpha=0.15)
            for t in leg.get_texts():
                t.set_color("white")

    axes[0].set_ylabel(
        r"Tension $\langle\sigma v\rangle_\mathrm{eff}\,/\,\langle\sigma v\rangle_\mathrm{dSph}$",
        fontsize=11,
        color="white",
    )

    solid = mlines.Line2D([], [], color="white", lw=2.2, label="Solid: EFT-valid region")
    dotted = mlines.Line2D([], [], color="white", lw=1.0, ls=":", label="Dotted: EFT-invalid region")
    fig.legend(
        handles=[solid, dotted],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        fontsize=9,
        frameon=False,
        labelcolor="white",
    )

    fig.suptitle(
        rf"EFT photon-DM operators: deconvolved tension vs $m_\chi$ | $\Lambda={Lambda:.0f}$ GeV",
        fontsize=13,
        color="white",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    print(f"  Plot: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deconvolve DM-photon scattering from Totani halo spectrum and refit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--operator", default=None,
                   choices=[s.name for s in ALL_OPERATORS] + ["all"],
                   help="Single operator. Use --all-operators for all.")
    p.add_argument("--all-operators", action="store_true", default=False)
    p.add_argument("--dm-type", default="fermionic")
    p.add_argument("--majorana", action="store_true", default=False)

    p.add_argument("--halo-profile", default="rho2",
                   choices=list(_MCMC_DIRS.keys()))
    p.add_argument("--ann-channel", default="WW")
    p.add_argument("--pppc-gamma-table", default=None)

    # Scattering DM scan (x-axis: m_chi)
    p.add_argument("--mchi-min",   type=float, default=1e-4,
                   help="Min scattering DM mass [GeV]")
    p.add_argument("--mchi-max",   type=float, default=1e6,
                   help="Max scattering DM mass [GeV]")
    p.add_argument("--n-mchi",     type=int,   default=30)

    # Coupling scan (y-axis: Lambda)
    p.add_argument("--lambda-min", type=float, default=1.0)
    p.add_argument("--lambda-max", type=float, default=1e4)
    p.add_argument("--n-lambda",   type=int,   default=30)
    p.add_argument("--fixed-lambda", "--fixed-coupling",
                   dest="fixed_coupling", type=float, default=None,
                   help="Fix Lambda [GeV] (or y_eff) for a 1D scan over m_scat. "
                        "Overrides --lambda-min/max and --n-lambda.")

    # Annihilation mass scan for PPPC fitting
    p.add_argument("--ann-mass", type=float, default=None,
                   help="Fix a single annihilation mass [GeV] for the PPPC fit scan. "
                        "If set, overrides --ann-mass-min/max and --n-ann-mass.")
    p.add_argument("--ann-mass-min", type=float, default=100.0)
    p.add_argument("--ann-mass-max", type=float, default=5000.0)
    p.add_argument("--n-ann-mass",   type=int,   default=40)
    p.add_argument("--all-ann-channels", action="store_true", default=False,
                   help="Run WW, bb, and tautau and produce channel-comparison plots.")

    # Deconvolution options
    p.add_argument("--deconv-method", default="exact",
                   choices=["exact", "firstorder"],
                   help="'exact': solve T*phi=phi_obs via linalg.solve. "
                        "'firstorder': first-order approx (valid tau<<1).")
    p.add_argument(
        "--fit-cl",
        type=float,
        default=0.90,
        help="Confidence level window around chi2 minimum for selecting the "
             "annihilation fit point (2D Delta chi2 in m_ann-normalization space).",
    )
    p.add_argument(
        "--fit-selection",
        default="max_scattering_effect",
        choices=["max_scattering_effect", "min_tension", "best_chi2"],
        help="How to choose the representative fit point within the CL window. "
             "'max_scattering_effect' and 'min_tension' pick lowest tension; "
             "'best_chi2' picks pure chi2 minimum.",
    )

    # Numerical
    p.add_argument("--n-theta",   type=int,   default=200)
    p.add_argument("--apply-roi-weight", action="store_true", default=True)
    p.add_argument("--no-roi-weight", dest="apply_roi_weight",
                   action="store_false")
    p.add_argument("--max-tau-single-scatter", type=float, default=0.3)

    p.add_argument("--quick", action="store_true", default=False,
                   help="10x10 grid, n_theta=100")
    p.add_argument("--plot",  action="store_true", default=True)
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.add_argument(
        "--style",
        default=None,
        help="Plot style: paper, conference/dark_transparent, or conference_light/light_transparent.",
    )
    p.add_argument("--output-dir", default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style

    if not (0.0 < float(args.fit_cl) < 1.0):
        raise ValueError("--fit-cl must lie strictly between 0 and 1.")

    if args.quick:
        args.n_mchi    = 10
        args.n_lambda  = 10
        args.n_theta   = 100
        args.n_ann_mass = 20
        print("[quick mode] 10x10 grid, n_theta=100")

    print(f"\nLoading halo spectrum: {args.halo_profile}")
    halo = load_halo_spectrum(_MCMC_DIRS[args.halo_profile])
    phi_err = halo.phi_err_sym
    mask = halo.positive_mask & halo.finite_mask & (phi_err > 0)
    print(f"  {int(mask.sum())} / {len(halo.E_bins_GeV)} bins in chi2 mask")

    root_out = (Path(args.output_dir) if args.output_dir
                else _HERE / "results" / "deconv_scan")

    if args.all_operators or args.operator == "all":
        specs = ALL_OPERATORS
    elif args.operator is not None:
        specs = [s for s in ALL_OPERATORS if s.name == args.operator
                 and s.dm_type == args.dm_type and s.majorana == args.majorana]
        if not specs:
            specs = [s for s in ALL_OPERATORS if s.name == args.operator]
    else:
        specs = [ALL_OPERATORS[0]]  # dipole_magnetic by default
        print("No operator specified; running dipole_magnetic (Dirac) by default.")

    scatter_masses = np.logspace(
        np.log10(args.mchi_min), np.log10(args.mchi_max), args.n_mchi)
    if args.fixed_coupling is not None:
        couplings = np.array([float(args.fixed_coupling)])
    else:
        couplings = np.logspace(
            np.log10(args.lambda_min), np.log10(args.lambda_max), args.n_lambda)
    if args.ann_mass is not None:
        ann_masses = np.array([float(args.ann_mass)])
    else:
        ann_masses = np.logspace(
            np.log10(args.ann_mass_min), np.log10(args.ann_mass_max), args.n_ann_mass)

    if getattr(args, "all_ann_channels", False):
        ann_channels = ["WW", "bb", "tautau"]
    else:
        ann_channels = [_normalise_ann_channel(args.ann_channel)]

    all_results = []
    results_by_channel: dict[str, list[dict]] = {}
    t_total = time.time()

    for ann_channel in ann_channels:
        channel_results = []
        for spec in specs:
            key = operator_key(spec)
            outdir = root_out / f"{key}_{ann_channel}_{args.halo_profile}"

            print(f"\n{'='*60}")
            print(f"  Operator: {spec.label}  ({key})  |  channel: {ann_channel}")
            print(f"{'='*60}")

            result = run_deconv_scan(
                spec=spec,
                halo=halo,
                mask=mask,
                scatter_masses=scatter_masses,
                couplings=couplings,
                ann_masses=ann_masses,
                args=args,
                ann_channel=ann_channel,
                outdir=outdir,
            )
            all_results.append(result)
            channel_results.append(result)
            results_by_channel.setdefault(ann_channel, []).append(result)

        if len(couplings) == 1 and len(scatter_masses) > 1 and len(channel_results) > 1:
            _make_mscat_tension_plot(
                channel_results,
                root_out / f"tension_vs_mscat_{ann_channel}_{args.halo_profile}.png",
                ann_channel=ann_channel,
                Lambda=float(couplings[0]),
            )

    if len(ann_channels) > 1 and len(specs) == 1 and len(couplings) == 1:
        _make_channel_overlay_plot(
            results_by_channel,
            root_out / f"tension_vs_mscat_channels_{operator_key(specs[0])}_{args.halo_profile}.png",
            Lambda=float(couplings[0]),
        )

    if len(ann_channels) > 1 and len(specs) > 1 and len(couplings) == 1:
        _make_all_operators_channel_plot(
            results_by_channel,
            root_out / f"tension_vs_mscat_all_{args.halo_profile}.png",
            Lambda=float(couplings[0]),
        )

    print(f"\n{'='*60}")
    print("DECONVOLUTION SCAN COMPLETE")
    print(f"{'='*60}")
    print(f"{'Channel':<8} {'Operator':<35} {'orig tension':>13} {'best EFT delta':>15}")
    print("-" * 75)
    for r in all_results:
        if not r:
            continue
        print(f"{str(r.get('ann_channel', '?')):<8} "
              f"{r['label']:<35} "
              f"{float(r['orig_tension']):>13.3f}x "
              f"{float(r['best_eft_delta_tension']):>+15.4f}x")
    print(f"\nTotal wall time: {time.time() - t_total:.1f} s")
    print(f"All results in: {root_out}")


if __name__ == "__main__":
    main()
