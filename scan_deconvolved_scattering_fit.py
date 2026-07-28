#!/usr/bin/env python3
"""
scan_deconvolved_scattering_fit.py
====================================
Full deconvolved scattering interpretation of the Totani halo spectrum.

User choices (CLI flags):
  1. Halo model       : --halo-profile  (rho2, rho2.5, rho1)
  2. Channel          : --ann-channel   (WW, bb, tautau)
  3. Dark sector model: --dark-sector-model  (same | different)
  4. EFT operator     : --operator
  5. Annihilator mass : --ann-mass (fixed) or --ann-mass-min/max (scan)

Workflow:
  1. Load a Totani MCMC halo posterior spectrum.
  2. Fit the observed spectrum with pure PPPC annihilation templates using the
     same weighted PPPC-template fit convention as the earlier deconvolution code.
  3. For valid EFT (m_scat, Lambda) points, build the scattering transfer
     matrix T and:
       a. Deconvolve: phi_pre = T^{-1} @ phi_obs for diagnostics
       b. Fit observed bins with A * (T_ext @ PPPC_ext), including attenuation
          and redistribution from the extended source-energy grid
  4. Produce a grid of correction plots: (phi_pre - phi_obs)/sigma per bin,
     labelled with m_ann, m_scat, Lambda.
  5. Produce a side-by-side summary:
       a. m_ann vs delta-chi2, showing original and deconvolved curves + CL line
       b. Spectrum comparison (top) + fractional residuals (bottom)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as _scipy_chi2

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.eft_validity import operator_validity_catalogue, sample_valid_lambda_grid  # noqa: E402
from core.spectral_reshaping import (  # noqa: E402
    ReshapingConfig,
    best_fit_normalization,
    build_dsigma_grid,
    build_kernel,
    configure_totani_arrays,
    energy_flux_transfer_matrix,
    pppc_energy_flux_template,
    roi_tau_prefactor,
    smooth_nfw_sigma_v_from_norm,
)
from core.dsph_limits import (  # noqa: E402
    DSph_LIMIT_SOURCES,
    dsph_limit_source,
    dsph_upper_limit,
)
from core.totani_data_loader import _MCMC_DIRS, load_halo_spectrum  # noqa: E402
from helpers.trinity_plotting import current_savefig_kwargs, set_plot_style  # noqa: E402


# ---------------------------------------------------------------------------
# Chi2 threshold from confidence level
# ---------------------------------------------------------------------------

def _delta_chi2_from_cl(cl: float, ndof: int = 2) -> float:
    return float(_scipy_chi2.ppf(cl, df=ndof))


def _select_dsph_cl(
    chi2_arr: np.ndarray,
    norm_arr: np.ndarray,
    masses_arr: np.ndarray,
    valid_arr: np.ndarray,
    *,
    cl_delta: float,
    channel: str,
    source: str = "hoof",
) -> int:
    """Index of lowest-chi2 point within cl_delta of chi2_min that is dSph-safe.

    Returns -1 if no such point exists.
    """
    finite = np.asarray(valid_arr, dtype=bool) & np.isfinite(chi2_arr)
    if not np.any(finite):
        return -1
    chi2_min = float(np.nanmin(np.where(finite, chi2_arr, np.nan)))
    cl_ok = finite & ((chi2_arr - chi2_min) <= cl_delta)
    sv = np.array([
        smooth_nfw_sigma_v_from_norm(float(norm_arr[i]), float(masses_arr[i]))
        if finite[i] else np.nan
        for i in range(len(masses_arr))
    ])
    lim = np.array([dsph_upper_limit(float(masses_arr[i]), channel, source=source) for i in range(len(masses_arr))])
    safe = np.isfinite(sv) & np.isfinite(lim) & (sv <= lim)
    combined = cl_ok & safe
    if np.any(combined):
        return int(np.nanargmin(np.where(combined, chi2_arr, np.nan)))
    return -1


def _tension_for_grid(norm_arr: np.ndarray, masses_arr: np.ndarray, finite_arr: np.ndarray, channel: str, source: str = "hoof") -> np.ndarray:
    tension = np.full(len(masses_arr), np.nan, dtype=float)
    for i, mass in enumerate(masses_arr):
        if not finite_arr[i] or not np.isfinite(norm_arr[i]):
            continue
        sigmav = smooth_nfw_sigma_v_from_norm(float(norm_arr[i]), float(mass))
        limit = dsph_upper_limit(float(mass), channel, source=source)
        if np.isfinite(sigmav) and np.isfinite(limit) and limit > 0.0:
            tension[i] = sigmav / limit
    return tension


def _select_min_tension_cl(
    chi2_arr: np.ndarray,
    norm_arr: np.ndarray,
    masses_arr: np.ndarray,
    valid_arr: np.ndarray,
    *,
    cl_delta: float,
    channel: str,
    source: str = "hoof",
) -> int:
    """Index of lowest dSph tension within cl_delta of the spectral chi2 minimum."""
    finite = np.asarray(valid_arr, dtype=bool) & np.isfinite(chi2_arr)
    if not np.any(finite):
        return -1
    chi2_min = float(np.nanmin(np.where(finite, chi2_arr, np.nan)))
    cl_ok = finite & ((chi2_arr - chi2_min) <= cl_delta)
    tension = _tension_for_grid(norm_arr, masses_arr, finite, channel, source=source)
    candidates = cl_ok & np.isfinite(tension)
    if np.any(candidates):
        return int(np.nanargmin(np.where(candidates, tension, np.nan)))
    return -1


# ---------------------------------------------------------------------------
# Template bank
# ---------------------------------------------------------------------------

@dataclass
class TemplateBank:
    masses: np.ndarray
    templates: np.ndarray
    valid: np.ndarray


def _operator_defaults(operator: str, dm_type: str | None = None) -> dict:
    catalogue = operator_validity_catalogue()
    if operator not in catalogue:
        aliases = {
            value: key
            for key, meta in catalogue.items()
            for value in meta.get("aliases", ())
        }
        operator = aliases.get(operator, operator)
    if operator not in catalogue:
        raise ValueError(f"Unknown EFT operator {operator!r}. Available: {sorted(catalogue)}")
    meta = catalogue[operator]
    coeffs = dict(meta.get("wilson_coefficients", {}))
    return {
        "operator": operator,
        "dm_type": dm_type or str(meta.get("dm_type", "fermionic")),
        "majorana": bool(
            meta.get("majorana_allowed", False) is True
            and operator in ("anapole", "rayleigh_even", "rayleigh_odd", "rayleigh_full")
        ),
        "c_s": float(coeffs.get("c_s", 0.0)),
        "c_p": float(coeffs.get("c_p", 0.0)),
        "c_phi": float(coeffs.get("c_phi", 1.0)),
    }


def _mass_grid(args: argparse.Namespace) -> np.ndarray:
    if args.ann_mass is not None and args.ann_mass_mode == "fixed":
        return np.asarray([float(args.ann_mass)], dtype=float)
    return np.logspace(
        np.log10(args.ann_mass_min), np.log10(args.ann_mass_max), int(args.n_ann_mass)
    )


def build_template_bank(
    E: np.ndarray, masses: np.ndarray, channel: str, table_path: str | None
) -> TemplateBank:
    templates = np.full((len(masses), len(E)), np.nan, dtype=float)
    valid = np.zeros(len(masses), dtype=bool)
    for i, mass in enumerate(masses):
        try:
            src = pppc_energy_flux_template(
                E, float(mass), channel=channel, primary="gamma",
                table_path=table_path, normalise=False,
            )
        except Exception:
            continue
        if np.any(np.isfinite(src) & (src > 0.0)):
            templates[i] = src
            valid[i] = True
    return TemplateBank(masses=np.asarray(masses, dtype=float), templates=templates, valid=valid)


def fit_templates(
    target: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    bank: TemplateBank,
    allowed_indices: np.ndarray | list[int] | None = None,
) -> dict:
    """Fit each PPPC template to the target spectrum; return chi2, norm, best index."""
    chi2 = np.full(len(bank.masses), np.nan, dtype=float)
    norm = np.full(len(bank.masses), np.nan, dtype=float)
    models = np.full_like(bank.templates, np.nan, dtype=float)
    fit_mask = np.asarray(mask, dtype=bool) & np.isfinite(target) & np.isfinite(err) & (err > 0.0)
    if allowed_indices is None:
        indices = range(len(bank.masses))
    else:
        indices = np.asarray(allowed_indices, dtype=int)

    for i in indices:
        if i < 0 or i >= len(bank.masses):
            continue
        src = bank.templates[i]
        if not bank.valid[i]:
            continue
        amp = best_fit_normalization(src, target, err, fit_mask)
        if not np.isfinite(amp) or amp <= 0.0:
            continue
        model = amp * src
        norm[i] = amp
        models[i] = model
        chi2[i] = float(
            np.sum(((model[fit_mask] - target[fit_mask]) / err[fit_mask]) ** 2)
        )

    best_i = int(np.nanargmin(chi2)) if np.any(np.isfinite(chi2)) else -1
    return {
        "chi2": chi2,
        "norm": norm,
        "models": models,
        "best_i": best_i,
        "best_mass": float(bank.masses[best_i]) if best_i >= 0 else np.nan,
        "best_norm": float(norm[best_i]) if best_i >= 0 else np.nan,
        "best_chi2": float(chi2[best_i]) if best_i >= 0 else np.nan,
        "best_model": (
            models[best_i].copy() if best_i >= 0 else np.full(bank.templates.shape[1], np.nan)
        ),
    }


def fit_transferred_templates(
    T_obs: np.ndarray,
    data: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    bank_ext: TemplateBank,
    n_obs: int,
    allowed_indices: np.ndarray | list[int] | None = None,
) -> dict:
    """Fit A * (T_obs @ PPPC_ext) to the observed bins.

    This is the scattering best-fit objective: the intrinsic annihilation
    template is evaluated on the extended source-energy grid, attenuated and
    redistributed by the transfer matrix, then compared only to the measured
    Totani bins.
    """
    chi2 = np.full(len(bank_ext.masses), np.nan, dtype=float)
    norm = np.full(len(bank_ext.masses), np.nan, dtype=float)
    observed_models = np.full((len(bank_ext.masses), n_obs), np.nan, dtype=float)
    intrinsic_models_obs = np.full((len(bank_ext.masses), n_obs), np.nan, dtype=float)
    fit_mask = np.asarray(mask, dtype=bool) & np.isfinite(data) & np.isfinite(err) & (err > 0.0)
    if allowed_indices is None:
        indices = range(len(bank_ext.masses))
    else:
        indices = np.asarray(allowed_indices, dtype=int)

    for i in indices:
        if i < 0 or i >= len(bank_ext.masses):
            continue
        src_ext = bank_ext.templates[i]
        if not bank_ext.valid[i]:
            continue
        transferred = T_obs @ src_ext
        if not np.all(np.isfinite(transferred)):
            continue
        amp = best_fit_normalization(transferred, data, err, fit_mask)
        if not np.isfinite(amp) or amp <= 0.0:
            continue
        model = amp * transferred
        norm[i] = amp
        observed_models[i] = model
        intrinsic_models_obs[i] = (amp * src_ext)[:n_obs]
        chi2[i] = float(np.sum(((model[fit_mask] - data[fit_mask]) / err[fit_mask]) ** 2))

    best_i = int(np.nanargmin(chi2)) if np.any(np.isfinite(chi2)) else -1
    return {
        "chi2": chi2,
        "norm": norm,
        "observed_models": observed_models,
        "intrinsic_models_obs": intrinsic_models_obs,
        "best_i": best_i,
        "best_mass": float(bank_ext.masses[best_i]) if best_i >= 0 else np.nan,
        "best_norm": float(norm[best_i]) if best_i >= 0 else np.nan,
        "best_chi2": float(chi2[best_i]) if best_i >= 0 else np.nan,
        "best_observed_model": (
            observed_models[best_i].copy() if best_i >= 0 else np.full(n_obs, np.nan)
        ),
        "best_intrinsic_model_obs": (
            intrinsic_models_obs[best_i].copy() if best_i >= 0 else np.full(n_obs, np.nan)
        ),
    }


def _log_window_mask(E: np.ndarray, center: float, width_dex: float) -> np.ndarray:
    E = np.asarray(E, dtype=float)
    center = float(center)
    if center <= 0.0 or not np.isfinite(center):
        return np.zeros_like(E, dtype=bool)
    half = max(float(width_dex), 0.0)
    return (
        np.isfinite(E)
        & (E > 0.0)
        & (np.abs(np.log10(E / center)) <= half)
    )


def _downscatter_diagnostics(
    *,
    T_in_obs: np.ndarray,
    E_obs: np.ndarray,
    E_ext: np.ndarray,
    src_ext: np.ndarray,
    src_norm: float,
    model_obs: np.ndarray,
    source_energy: float,
    target_energy: float,
    source_width_dex: float,
    target_width_dex: float,
) -> dict:
    """Quantify whether source-window photons feed the target observed bump."""
    E_obs = np.asarray(E_obs, dtype=float)
    E_ext = np.asarray(E_ext, dtype=float)
    src_ext = float(src_norm) * np.asarray(src_ext, dtype=float)
    model_obs = np.asarray(model_obs, dtype=float)

    source_mask = _log_window_mask(E_ext, source_energy, source_width_dex)
    target_mask = _log_window_mask(E_obs, target_energy, target_width_dex)
    if not np.any(source_mask) or not np.any(target_mask):
        return {
            "downscatter_peak_E_GeV": np.nan,
            "downscatter_peak_offset_dex": np.nan,
            "source_to_target_frac": np.nan,
            "target_in_from_source_frac": np.nan,
            "target_model_from_source_frac": np.nan,
            "downscatter_valid": False,
        }

    T_in_obs = np.asarray(T_in_obs, dtype=float)
    source_contrib = T_in_obs[:, source_mask] @ src_ext[source_mask]
    all_inscatter = T_in_obs @ src_ext

    if not np.any(np.isfinite(source_contrib) & (source_contrib > 0.0)):
        peak_E = np.nan
    else:
        peak_E = float(E_obs[int(np.nanargmax(source_contrib))])

    source_total = float(np.nansum(np.where(source_contrib > 0.0, source_contrib, 0.0)))
    source_to_target = float(np.nansum(
        np.where(source_contrib[target_mask] > 0.0, source_contrib[target_mask], 0.0)
    ))
    target_in_total = float(np.nansum(
        np.where(all_inscatter[target_mask] > 0.0, all_inscatter[target_mask], 0.0)
    ))
    target_model_total = float(np.nansum(
        np.where(model_obs[target_mask] > 0.0, model_obs[target_mask], 0.0)
    ))

    peak_offset = (
        float(abs(np.log10(peak_E / float(target_energy))))
        if np.isfinite(peak_E) and peak_E > 0.0 and target_energy > 0.0
        else np.nan
    )
    source_to_target_frac = source_to_target / source_total if source_total > 0.0 else np.nan
    target_in_from_source_frac = source_to_target / target_in_total if target_in_total > 0.0 else np.nan
    target_model_from_source_frac = source_to_target / target_model_total if target_model_total > 0.0 else np.nan

    return {
        "downscatter_peak_E_GeV": peak_E,
        "downscatter_peak_offset_dex": peak_offset,
        "source_to_target_frac": float(source_to_target_frac),
        "target_in_from_source_frac": float(target_in_from_source_frac),
        "target_model_from_source_frac": float(target_model_from_source_frac),
        "downscatter_valid": bool(
            np.isfinite(peak_offset)
            and np.isfinite(source_to_target_frac)
            and np.isfinite(target_in_from_source_frac)
            and np.isfinite(target_model_from_source_frac)
        ),
    }


def _downscatter_penalty(diag: dict, args: argparse.Namespace) -> tuple[float, bool]:
    if str(args.downscatter_mode) == "off":
        return 0.0, True
    if not bool(diag.get("downscatter_valid", False)):
        return np.inf, False

    sigma_peak = max(float(args.downscatter_target_window_dex), 1e-6)
    peak_offset = float(diag["downscatter_peak_offset_dex"])
    penalty = float(args.downscatter_peak_weight) * (peak_offset / sigma_peak) ** 2
    hard_ok = peak_offset <= float(args.downscatter_target_window_dex)

    frac_specs = [
        ("source_to_target_frac", float(args.min_source_to_target_frac)),
        ("target_in_from_source_frac", float(args.min_target_in_from_source_frac)),
        ("target_model_from_source_frac", float(args.min_target_model_from_source_frac)),
    ]
    for key, threshold in frac_specs:
        val = float(diag.get(key, np.nan))
        if threshold <= 0.0:
            continue
        if not np.isfinite(val):
            return np.inf, False
        if val < threshold:
            hard_ok = False
            penalty += float(args.downscatter_frac_weight) * ((threshold - val) / threshold) ** 2

    if str(args.downscatter_mode) == "hard" and not hard_ok:
        return np.inf, False
    return penalty, True


def _best_bin_contribution_table(
    *,
    E: np.ndarray,
    phi_data: np.ndarray,
    phi_err: np.ndarray,
    mask: np.ndarray,
    intrinsic_model: np.ndarray,
    recon_model: np.ndarray,
    T_obs: np.ndarray,
    T_in_obs: np.ndarray,
    tau: np.ndarray,
    out_csv: Path,
) -> str:
    """Write and return a per-bin scattering contribution table."""
    E = np.asarray(E, dtype=float)
    phi_data = np.asarray(phi_data, dtype=float)
    phi_err = np.asarray(phi_err, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    intrinsic_model = np.asarray(intrinsic_model, dtype=float)
    recon_model = np.asarray(recon_model, dtype=float)
    tau = np.asarray(tau, dtype=float)
    n_obs = len(E)

    # The stored intrinsic_model is already restricted to observed bins.  The
    # full source grid is not stored for the selected point, so decompose the
    # observed-bin part exactly and treat any extra-grid contribution as a
    # residual high-energy tail.
    T_obs_obs = np.asarray(T_obs, dtype=float)[:, :n_obs]
    T_in_obs_obs = np.asarray(T_in_obs, dtype=float)[:, :n_obs]
    survival = np.exp(-tau) * intrinsic_model
    inscatter_obs_bins = T_in_obs_obs @ intrinsic_model
    highE_or_roundoff = recon_model - survival - inscatter_obs_bins
    net_scatter_delta = recon_model - intrinsic_model
    err_safe = np.where(np.isfinite(phi_err) & (phi_err > 0.0), phi_err, np.nan)
    net_pull = net_scatter_delta / err_safe
    inscatter_pull = inscatter_obs_bins / err_safe
    attenuation_loss = survival - intrinsic_model
    attenuation_pull = attenuation_loss / err_safe

    header = (
        "k,E_GeV,fit_bin,phi_data,phi_err,intrinsic_no_scatter,"
        "survival_after_attenuation,inscatter_from_observed_bins,"
        "extra_highE_or_roundoff,total_scattered_model,"
        "attenuation_delta,net_scatter_delta,attenuation_pull,"
        "inscatter_pull,net_pull,sign"
    )
    rows = [header]
    display_lines = [
        "",
        "--- Best-fit scattering contribution by observed bin ---",
        "sign is net_scatter_delta = scattered_model - intrinsic_no_scatter",
        f"{'k':>2} {'E[GeV]':>9} {'fit':>3} {'atten/sig':>10} {'in/sig':>10} "
        f"{'net/sig':>10} {'sign':>5}",
    ]
    for k in range(n_obs):
        sign = "pos" if net_scatter_delta[k] > 0 else ("neg" if net_scatter_delta[k] < 0 else "zero")
        rows.append(
            ",".join([
                str(k),
                f"{E[k]:.9g}",
                str(int(mask[k])),
                f"{phi_data[k]:.9e}",
                f"{phi_err[k]:.9e}",
                f"{intrinsic_model[k]:.9e}",
                f"{survival[k]:.9e}",
                f"{inscatter_obs_bins[k]:.9e}",
                f"{highE_or_roundoff[k]:.9e}",
                f"{recon_model[k]:.9e}",
                f"{attenuation_loss[k]:.9e}",
                f"{net_scatter_delta[k]:.9e}",
                f"{attenuation_pull[k]:.9e}",
                f"{inscatter_pull[k]:.9e}",
                f"{net_pull[k]:.9e}",
                sign,
            ])
        )
        display_lines.append(
            f"{k:2d} {E[k]:9.3g} {int(mask[k]):3d} "
            f"{attenuation_pull[k]:10.3g} {inscatter_pull[k]:10.3g} "
            f"{net_pull[k]:10.3g} {sign:>5}"
        )

    out_csv.write_text("\n".join(rows) + "\n")
    return "\n".join(display_lines)


# ---------------------------------------------------------------------------
# Transfer matrix helpers
# ---------------------------------------------------------------------------

def transfer_matrix_energy_flux(
    tau: np.ndarray, K_photon: np.ndarray, E: np.ndarray
) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    K_energy = energy_flux_transfer_matrix(K_photon, E)
    atten = np.exp(-tau)
    return np.diag(atten) + K_energy * (tau * atten)[None, :]


def deconvolve_observed(
    phi_obs: np.ndarray, T: np.ndarray, method: str
) -> np.ndarray:
    if method == "firstorder":
        return phi_obs - (T - np.eye(T.shape[0])) @ phi_obs
    try:
        return np.linalg.solve(T, phi_obs)
    except np.linalg.LinAlgError:
        return phi_obs - (T - np.eye(T.shape[0])) @ phi_obs


def build_transfer_for_point(
    *,
    E: np.ndarray,
    phi: np.ndarray,
    err: np.ndarray,
    scatter_mass: float,
    Lambda: float,
    op: dict,
    args: argparse.Namespace,
    E_ext: np.ndarray | None = None,
) -> dict:
    """Build the transfer matrix for a given (m_scat, Lambda) point.

    If E_ext is longer than E, the kernel is built on the full extended grid.
    The square observed-bin block is returned for deconvolution diagnostics,
    while T_obs maps the full extended intrinsic spectrum into observed bins.
    """
    nObs = len(E)
    if E_ext is None or len(E_ext) <= nObs:
        E_ext = E

    # Dummy phi/err values for the extended bins (not used in fitting, only
    # needed to satisfy the ReshapingConfig shape invariant).
    nExt = len(E_ext)
    if nExt > nObs:
        phi_mean = float(np.nanmean(np.where(np.isfinite(phi), phi, np.nan)))
        err_mean  = float(np.nanmean(np.where(np.isfinite(err) & (err > 0), err, np.nan)))
        phi_ext = np.concatenate([phi, np.full(nExt - nObs, phi_mean)])
        err_ext = np.concatenate([err, np.full(nExt - nObs, err_mean)])
    else:
        phi_ext = phi
        err_ext = err

    cfg = ReshapingConfig(
        m_chi=float(scatter_mass),
        Lambda=float(Lambda),
        dm_type=str(op["dm_type"]),
        operator=str(op["operator"]),
        c_s=float(op["c_s"]),
        c_p=float(op["c_p"]),
        c_phi=float(op["c_phi"]),
        majorana=bool(args.majorana),
        n_theta=int(args.n_theta),
        apply_roi_weight=not args.no_roi_weight,
        roi_half_angle_deg=None if args.no_roi_weight else 60.0,
        E_bins=E_ext,
        phi_0=phi_ext,
        phi_data=phi_ext,
        phi_err=err_ext,
        max_tau_single_scatter=None,
        require_lambda_gt_mdm=not bool(args.allow_eft_invalid),
    )
    cos_theta, dsig, sigma_tot = build_dsigma_grid(cfg)
    K_ext = build_kernel(cfg, cos_theta, dsig, sigma_tot)
    tau_ext = (
        roi_tau_prefactor(cfg.l_grid, cfg.b_grid)
        * np.asarray(sigma_tot, dtype=float)
        / float(scatter_mass)
    )
    tau_ext = np.nan_to_num(tau_ext, nan=0.0, posinf=0.0, neginf=0.0)
    tau_ext = np.where(tau_ext > 0.0, tau_ext, 0.0)

    K_energy_ext = energy_flux_transfer_matrix(K_ext, E_ext)
    atten_ext = np.exp(-tau_ext)
    T_in_full = K_energy_ext * (tau_ext * atten_ext)[None, :]
    T_full = np.diag(atten_ext) + T_in_full

    # Observed-bin subsets used for deconvolution and the extended forward fit.
    T_obs_obs = T_full[:nObs, :nObs]   # square matrix for deconvolution
    T_obs = T_full[:nObs, :]           # observed rows, all source-energy columns
    T_in_obs = T_in_full[:nObs, :]

    return {
        "T":         T_obs_obs,          # (nObs × nObs) — deconvolution diagnostic
        "T_obs":     T_obs,              # (nObs × nExt) — attenuation + reshaping fit
        "T_in_obs":  T_in_obs,           # (nObs × nExt) — in-scatter term only
        "K":         K_ext[:nObs, :nObs],
        "tau":       tau_ext[:nObs],
        "sigma_tot": sigma_tot[:nObs],
    }


def make_scatter_mass_grid(args: argparse.Namespace, ann_best_mass: float) -> np.ndarray:
    if args.dark_sector_model == "same":
        if args.ann_mass is not None and args.ann_mass_mode == "fixed":
            return np.asarray([float(args.ann_mass)], dtype=float)
        return np.asarray([], dtype=float)
    if args.scatter_mass is not None:
        return np.asarray([float(args.scatter_mass)], dtype=float)
    lo = float(args.scatter_mass_min)
    hi = float(args.scatter_mass_max)
    n_low = int(args.n_scatter_mass_low)
    pivot = float(args.scatter_mass_low_max)
    if n_low > 0 and lo < pivot < hi:
        dense = np.logspace(np.log10(lo), np.log10(pivot), n_low, endpoint=False)
        coarse = np.logspace(np.log10(pivot), np.log10(hi), int(args.n_scatter_mass))
        return np.unique(np.concatenate([dense, coarse]))
    return np.logspace(np.log10(lo), np.log10(hi), int(args.n_scatter_mass))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_correction_grid(result: dict, out_path: Path, max_panels: int) -> None:
    """Grid of panels showing (phi_pre - phi_obs)/sigma per energy bin.

    Each panel is one valid (m_scat, Lambda) point, labelled with m_ann,
    m_scat and Lambda.  Bars are coloured by sign; ±1σ guide lines are drawn;
    the normalised cross section is overlaid on a secondary y-axis.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    set_plot_style(style="paper", cmap_name="plasma", base_fontsize=10, linewidth=1.6, n_colors=10)
    plasma = plt.cm.plasma
    neg_color = plasma(0.20)
    pos_color = plasma(0.78)
    guide_color = plasma(0.58)
    sigma_color = plasma(0.92)

    E = result["E_bins_GeV"]
    examples = result["example_points"][:max_panels]
    if not examples:
        return

    ncols = min(3, len(examples))
    nrows = int(np.ceil(len(examples) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False
    )
    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, ex in zip(axes.ravel(), examples):
        ax.set_visible(True)
        corr = ex["correction_over_sigma"]
        sigma = ex["sigma_tot_cm2"]

        colors = [pos_color if c >= 0.0 else neg_color for c in corr]
        x = np.arange(len(E))

        ax.axhline(0.0,  color=plt.rcParams.get("text.color", "black"), lw=0.9, alpha=0.8)
        ax.axhline( 1.0, color=guide_color, lw=0.8, ls="--", alpha=0.7)
        ax.axhline(-1.0, color=guide_color, lw=0.8, ls="--", alpha=0.7)
        ax.bar(x, corr, color=colors, alpha=0.85, zorder=3)

        ax2 = ax.twinx()
        sigma_norm = sigma / np.nanmax(sigma) if np.nanmax(sigma) > 0 else sigma
        ax2.plot(x, sigma_norm, color=sigma_color, lw=1.5, alpha=0.9, zorder=4)
        ax2.set_ylabel(r"$\sigma(E)$ [norm.]", fontsize=8, color=sigma_color)
        ax2.tick_params(axis="y", colors=sigma_color, labelsize=7)
        ax2.set_ylim(0, 1.25)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{e:.2g}" for e in E], rotation=55, fontsize=7)
        ax.set_xlabel(r"$E_\gamma$ [GeV]", fontsize=8)
        ax.set_ylabel(r"$\Delta\Phi/\sigma$ (deconv correction)", fontsize=8)
        ax.grid(alpha=0.2, zorder=0)

        m_ann_str = f"{ex['ann_mass']:.3g}"
        m_scat_str = f"{ex['scatter_mass']:.3g}"
        lam_str = f"{ex['Lambda']:.3g}"
        ax.set_title(
            rf"$m_\mathrm{{ann}}={m_ann_str}$ GeV"
            "\n"
            rf"$m_\mathrm{{scat}}={m_scat_str}$ GeV,  $\Lambda={lam_str}$ GeV",
            fontsize=8,
        )

    fig.suptitle(
        f"Deconvolution corrections: {result['ann_channel'].item()} channel, "
        f"{result['operator'].item()}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    print(f"  Correction grid saved: {out_path}")


def plot_best_fit_summary(result: dict, out_path: Path) -> None:
    """Side-by-side summary:
    Left  : m_ann vs delta chi2 (original and deconvolved) + CL line
    Right : energy vs flux (top) + fractional residuals (bottom)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    set_plot_style(style="paper", cmap_name="plasma", base_fontsize=10, linewidth=1.6, n_colors=10)
    plasma = plt.cm.plasma
    pure_color = plasma(0.22)
    scat_color = plasma(0.5)
    cl_color = plasma(0.92)

    E = result["E_bins_GeV"]
    data = result["phi_data"]
    mask = result["mask"].astype(bool)
    ann_masses = result["ann_masses_GeV"]

    pure_chi2 = result["pure_chi2"]
    scat_chi2 = result["best_scatter_chi2_by_ann_mass"]

    pure_min = np.nanmin(pure_chi2)
    scat_min = np.nanmin(scat_chi2[np.isfinite(scat_chi2)]) if np.any(np.isfinite(scat_chi2)) else pure_min

    pure_delta = pure_chi2 - pure_min
    scat_delta = np.where(np.isfinite(scat_chi2), scat_chi2 - scat_min, np.nan)

    cl_delta = float(result["delta_chi2_cl"])
    cl_pct = int(round(100.0 * float(result["cl"])))

    pure_best_mass = float(result["pure_best_ann_mass_GeV"])
    scat_best_mass = float(result["best_ann_mass_GeV"])
    scat_best_mscat = float(result["best_scatter_mass_GeV"])
    scat_best_lam = float(result["best_Lambda_GeV"])
    ndf_pure = int(np.sum(mask)) - 1
    best_fit_type = str(result.get("best_fit_type", "spectral"))
    pure_tension = float(result.get("pure_best_dsph_tension", np.nan))
    scat_tension = float(result.get("best_scatter_dsph_tension", np.nan))

    # Compute sigmav for labels
    pure_sigmav = smooth_nfw_sigma_v_from_norm(float(result["pure_best_norm"]), pure_best_mass)
    scat_sigmav = smooth_nfw_sigma_v_from_norm(float(result["best_scatter_norm"]), scat_best_mass)

    fig = plt.figure(figsize=(14.0, 7.0))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.2],
        height_ratios=[3.0, 1.2],
        wspace=0.30,
        hspace=0.08,
    )
    ax_chi2 = fig.add_subplot(gs[:, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_res  = fig.add_subplot(gs[1, 1], sharex=ax_spec)

    # --- Left: delta chi2 vs m_ann ---
    ax_chi2.plot(ann_masses, pure_delta, color=pure_color, lw=2.0, zorder=2,
                 label=r"Observed flux (ann. only)")
    scat_valid = np.isfinite(scat_delta)
    ax_chi2.plot(
            ann_masses[scat_valid], scat_delta[scat_valid],
            color=scat_color, lw=2.3, zorder=4, ls='--',
            label="Scattering fit (attenuated + reshaped)")
        
    ax_chi2.axhline(cl_delta, color=cl_color, lw=1.3, ls=":", alpha=0.85,
                    label=f"{cl_pct}% CL  ($\\Delta\\chi^2={cl_delta:.2f}$)")

    # Mark the selected best-fit point on the chi2 curve
    pure_best_idx = int(np.argmin(np.abs(ann_masses - pure_best_mass)))
    pure_sel_delta_val = float(pure_delta[pure_best_idx])
    ax_chi2.axvline(pure_best_mass, color=pure_color, lw=1.1, ls="--", alpha=0.7, zorder=2)
    ax_chi2.plot(pure_best_mass, pure_sel_delta_val,
                 marker="*", ms=14, color=pure_color, zorder=6,
                 label=(None if best_fit_type == "spectral" else
                        rf"Ann. selected ({cl_pct}%CL+dSph): "
                        rf"$t={pure_tension:.2f}\times$"))
    ax_chi2.annotate(
        rf"Ann. only" "\n" rf"$m_\mathrm{{ann}}={pure_best_mass:.3g}$ GeV",
        xy=(pure_best_mass, pure_sel_delta_val),
        xytext=(30, 42),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=pure_color,
        arrowprops=dict(arrowstyle="->", color=pure_color, lw=1.1, shrinkA=2, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.25", facecolor=plt.rcParams.get("axes.facecolor", "white"), edgecolor=pure_color, alpha=0.78),
        zorder=8,
    )
    if np.any(scat_valid):
        ax_chi2.axvline(scat_best_mass, color=scat_color, lw=1.1, ls="--", alpha=0.7, zorder=5)
        scat_best_idx = int(np.argmin(np.abs(ann_masses - scat_best_mass)))
        scat_sel_delta_val = float(scat_delta[scat_best_idx]) if np.isfinite(scat_delta[scat_best_idx]) else 0.0
        ax_chi2.plot(scat_best_mass, scat_sel_delta_val,
                     marker="*", ms=14, color=scat_color, zorder=6,
                     label=(None if best_fit_type == "spectral" else
                            rf"Scat. selected ({cl_pct}%CL+dSph): "
                            rf"$t={scat_tension:.2f}\times$"))
        ax_chi2.annotate(
            rf"Scattering" "\n"
            rf"$m_\mathrm{{ann}}={scat_best_mass:.3g}$ GeV" "\n"
            rf"$m_\mathrm{{scat}}={scat_best_mscat:.3g}$ GeV",
            xy=(scat_best_mass, scat_sel_delta_val),
            xytext=(65, 58),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=scat_color,
            arrowprops=dict(arrowstyle="->", color=scat_color, lw=1.1, shrinkA=2, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.25", facecolor=plt.rcParams.get("axes.facecolor", "white"), edgecolor=scat_color, alpha=0.78),
            zorder=9,
        )

    title_suffix = f" [{best_fit_type}]" if best_fit_type != "spectral" else ""
    ax_chi2.set_xscale("log")
    ax_chi2.set_xlabel(r"$m_\mathrm{ann}$ [GeV]", fontsize=11)
    ax_chi2.set_ylabel(r"$\Delta\chi^2$ (relative to minimum)", fontsize=11)
    ax_chi2.set_title(f"Spectral compatibility vs annihilator mass{title_suffix}", fontsize=10)
    ax_chi2.legend(fontsize=8)
    ax_chi2.grid(alpha=0.22)
    ax_chi2.set_ylim(bottom=0)

    # --- Right top: spectrum ---
    data_lo = np.asarray(result["phi_err_lo"])
    data_hi = np.asarray(result["phi_err_hi"])
    ax_spec.errorbar(
        E, data,
        yerr=[data_lo, data_hi],
        fmt="o", ms=4.5, color=plt.rcParams.get("text.color", "black"), ecolor="0.35",
        label="Totani halo (MCMC)", zorder=5,
    )

    pure_model = result["pure_best_model"]
    pure_tension_str = (rf", $t={pure_tension:.2f}\times\mathrm{{dSph}}$"
                        if best_fit_type == "dsph_cl" and np.isfinite(pure_tension) else "")
    ax_spec.plot(E, pure_model, color=pure_color, lw=2.0, zorder=3,
                 label=rf"Ann. only: $m_\mathrm{{ann}}={pure_best_mass:.3g}$ GeV,"
                       rf" $\langle\sigma v\rangle={pure_sigmav:.2e}$ cm$^3$/s"
                       + pure_tension_str)

    recon_model = result["best_reconvolved_model"]
    if np.any(np.isfinite(recon_model)):
        scat_tension_str = (rf", $t={scat_tension:.2f}\times\mathrm{{dSph}}$"
                            if best_fit_type == "dsph_cl" and np.isfinite(scat_tension) else "")
        ax_spec.plot(
            E, recon_model, color=scat_color, lw=2.4, ls="--", zorder=6,
            label=rf"Scattering: $m_\mathrm{{ann}}={scat_best_mass:.3g}$ GeV,"
                  rf" $m_\mathrm{{scat}}={scat_best_mscat:.3g}$ GeV,"
                  "\n"
                  rf"$\Lambda={scat_best_lam:.3g}$ GeV,"
                  rf" $\langle\sigma v\rangle={scat_sigmav:.2e}$ cm$^3$/s"
                  + scat_tension_str,
        )

    ax_spec.set_xscale("log")
    ax_spec.set_ylim(bottom=np.nanmin(data[np.isfinite(data)]) * 1.5)
    ax_spec.set_ylabel(r"$E^2 \, dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=9)
    pure_chi2_ndf = float(pure_chi2[int(np.nanargmin(pure_chi2))]) / max(ndf_pure, 1)
    sel_chi2_ndf = float(pure_chi2[pure_best_idx]) / max(ndf_pure, 1)
    ax_spec.set_title(
        rf"{result['ann_channel'].item()} | {result['operator'].item()} | "
        rf"$\chi^2/\nu={pure_chi2_ndf:.2f}$ (spectral)"
        + (rf" / ${sel_chi2_ndf:.2f}$ (selected)" if best_fit_type != "spectral" else ""),
        fontsize=9,
    )
    ax_spec.legend(fontsize=7, loc="lower right")
    ax_spec.grid(alpha=0.22)
    plt.setp(ax_spec.get_xticklabels(), visible=False)

    # --- Right bottom: fractional residuals (data - fit)/data ---
    eps = np.where(np.abs(data) > 0.0, np.abs(data), np.nan)
    ax_res.axhline(0.0, color="0.45", lw=1.0)
    ax_res.axhline(1.0,  color="0.3", lw=0.6, ls=":")
    ax_res.axhline(-1.0, color="0.3", lw=0.6, ls=":")

    res_pure = (data - pure_model) / eps
    ax_res.plot(E, res_pure, color=pure_color, marker="o", ms=3.5, lw=1.4, zorder=3,
                label="Ann. only")

    if np.any(np.isfinite(recon_model)):
        res_scat = (data - recon_model) / eps
        ax_res.plot(E, res_scat, color=scat_color, marker="s", ms=3.5, lw=1.6, zorder=6,
                    label="Scattering")

    ax_res.set_xscale("log")
    ax_res.set_xlabel(r"$E_\gamma$ [GeV]", fontsize=11)
    ax_res.set_ylabel(r"$(D - M)/D$", fontsize=10)
    ax_res.legend(fontsize=8)
    ax_res.grid(alpha=0.22)
    ax_res.set_ylim(-5, 5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    print(f"  Summary plot saved: {out_path}")


def plot_mass_shift_diagnostic(result: dict, out_path: "Path") -> None:
    """Two-panel scatter plot: δm_ann and Δχ² vs τ_max, coloured by log10(m_scat)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return
    set_plot_style(style="paper", cmap_name="plasma", base_fontsize=10, linewidth=1.6, n_colors=10)

    tau_max = np.asarray(result["scan_tau_max"], dtype=float)
    m_ann_scat = np.asarray(result["scan_best_ann_mass_GeV"], dtype=float)
    m_ann_pure = float(result["pure_best_ann_mass_GeV"])
    chi2_scat = np.asarray(result["scan_recon_chi2"], dtype=float)
    chi2_pure = float(result["pure_best_chi2"])
    m_scat = np.asarray(result["scan_scatter_mass_GeV"], dtype=float)
    cl_delta = float(result["delta_chi2_cl"])
    cl_pct = float(result.get("cl", 0.95)) * 100.0

    delta_m = m_ann_scat - m_ann_pure
    delta_chi2 = chi2_pure - chi2_scat  # positive = scattering preferred

    good = np.isfinite(tau_max) & np.isfinite(delta_m) & np.isfinite(delta_chi2) & (m_scat > 0)
    if not np.any(good):
        return

    log_mscat = np.log10(m_scat[good])
    vmin, vmax = log_mscat.min(), log_mscat.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    sc1 = ax1.scatter(
        tau_max[good], delta_m[good], c=log_mscat,
        cmap=cmap, norm=norm, s=18, alpha=0.75, linewidths=0,
    )
    ax1.axhline(0.0, color="0.5", lw=1.0, ls="--")
    ax1.set_ylabel(r"$\delta m_\mathrm{ann} = m_\mathrm{ann}^\mathrm{scat} - m_\mathrm{ann}^\mathrm{pure}$ [GeV]",
                   fontsize=9)
    ax1.set_title(
        rf"Mass shift and $\chi^2$ improvement vs $\tau_\mathrm{{max}}$"
        + f"\n(pure-ann best: {m_ann_pure:.3g} GeV)",
        fontsize=10,
    )
    ax1.grid(alpha=0.22)

    sc2 = ax2.scatter(
        tau_max[good], delta_chi2[good], c=log_mscat,
        cmap=cmap, norm=norm, s=18, alpha=0.75, linewidths=0,
    )
    ax2.axhline(0.0, color="0.5", lw=1.0, ls="--")
    ax2.axhline(cl_delta, color=plt.cm.plasma(0.92), lw=1.2, ls=":",
                label=rf"$\Delta\chi^2 = {cl_delta:.2g}$ ({cl_pct:.0f}% CL)")
    ax2.set_ylabel(r"$\Delta\chi^2 = \chi^2_\mathrm{pure} - \chi^2_\mathrm{scat}$", fontsize=9)
    ax2.set_xlabel(r"$\tau_\mathrm{max}$", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.22)

    cb = fig.colorbar(sc2, ax=[ax1, ax2], label=r"$\log_{10}(m_\mathrm{scat}/\mathrm{GeV})$")
    cb.ax.tick_params(labelsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    print(f"  Mass-shift diagnostic saved: {out_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- User choices ---
    p.add_argument("--halo-profile", default="rho2", choices=sorted(_MCMC_DIRS.keys()),
                   help="1. Halo model / NFW profile.")
    p.add_argument("--ann-channel", default="WW", choices=["WW", "bb", "tautau"],
                   help="2. Annihilation channel.")
    p.add_argument("--dark-sector-model", default="different", choices=["same", "different"],
                   help="3. Dark sector model: 'same' ties m_scat = m_ann; "
                        "'different' scans m_scat independently.")
    p.add_argument("--operator", default="dipole_magnetic",
                   help="4. EFT operator key (from eft_validity.py).")
    p.add_argument("--ann-mass", type=float, default=None,
                   help="5. Annihilator mass [GeV] (fixed). With --ann-mass-mode fixed "
                        "only this mass is fitted.")
    p.add_argument("--ann-mass-mode", default="scan", choices=["scan", "fixed"])
    # --- Annihilator mass scan ---
    p.add_argument("--ann-mass-min", type=float, default=100.0)
    p.add_argument("--ann-mass-max", type=float, default=5000.0)
    p.add_argument("--n-ann-mass", type=int, default=80)
    # --- Scatterer mass scan ---
    p.add_argument("--scatter-mass", type=float, default=None,
                   help="Fixed scatterer mass [GeV] (overrides scan if given).")
    p.add_argument("--scatter-mass-min", type=float, default=1e-4)
    p.add_argument("--scatter-mass-max", type=float, default=1e6)
    p.add_argument("--n-scatter-mass", type=int, default=16)
    p.add_argument("--scatter-mass-low-max", type=float, default=100.0,
                   help="Pivot mass [GeV] below which a denser sub-grid is added. "
                        "Only used when --n-scatter-mass-low > 0.")
    p.add_argument("--n-scatter-mass-low", type=int, default=0,
                   help="Number of extra points in the dense low-mass region "
                        "[scatter_mass_min, scatter_mass_low_max). "
                        "0 = single uniform logspace (default).")
    # --- High-energy source extension ---
    p.add_argument("--n-ext-bins", type=int, default=20,
                   help="Extra energy bins above the observed range for high-E inscatter. "
                        "0 = no extension (faster, misses inscatter from above ~800 GeV).")
    p.add_argument("--ext-energy-max", type=float, default=5000.0,
                   help="Upper energy [GeV] of the extended source grid.")
    # --- Lambda scan ---
    p.add_argument("--lambda-min", type=float, default=1.0)
    p.add_argument("--lambda-max", type=float, default=1e4)
    p.add_argument("--n-lambda", type=int, default=16)
    p.add_argument("--allow-eft-invalid", action="store_true", default=False,
                   help="Do not restrict Lambda samples to the EFT/kinematic/unitarity-valid range. "
                        "This is for diagnostics only; default scans use eft_validity.py.")
    # --- Fit options ---
    p.add_argument("--cl", type=float, default=0.95,
                   help="Confidence level for the delta-chi2 CL line on the summary plot.")
    p.add_argument("--dsph-source", default="hoof", choices=list(DSph_LIMIT_SOURCES),
                   help="dSph limit set for tension: s-wave (hoof/mcdaniel/boddy_swave) "
                        "or velocity-dependent (boddy_pwave/boddy_dwave/boddy_somm).")
    p.add_argument("--best-fit-type", default="spectral", choices=["spectral", "dsph_cl", "min_tension_cl"],
                   help="Best-fit selection criterion. "
                        "'spectral': minimum chi2 (standard). "
                        "'dsph_cl': minimum chi2 within the --cl CL window "
                        "that also satisfies the configured dSph constraints "
                        "(<sigma v> <= sigma_v_dSph). "
                        "'min_tension_cl': old deconvolution-style selector, "
                        "the lowest dSph tension inside the --cl chi2 window.")
    p.add_argument("--err-mode", default="sym", choices=["sym", "lo", "hi", "max"],
                   help="Error convention to use from MCMC posteriors.")
    p.add_argument("--include-nonpositive-bins", action="store_true", default=False,
                   help="Include bins with phi <= 0 in the chi2 fit. "
                        "Default: positive-only bins (matches Totani paper and other scripts).")
    # --- DM type / operator overrides ---
    p.add_argument("--dm-type", default=None, choices=[None, "fermionic", "scalar"])
    p.add_argument("--majorana", action="store_true", default=False)
    # --- Transfer / deconvolution ---
    p.add_argument("--deconv-method", default="exact", choices=["exact", "firstorder"])
    p.add_argument("--max-tau", type=float, default=0.3,
                   help="Skip (m_scat, Lambda) points with tau_max > this value.")
    p.add_argument("--min-tau", type=float, default=0.0,
                   help="Skip (m_scat, Lambda) points with tau_max below this value. "
                        "Use this to prevent the scattering scan from selecting "
                        "the trivial T≈I/no-scattering limit.")
    p.add_argument("--n-theta", type=int, default=300,
                   help="Number of angular integration nodes for the kernel.")
    p.add_argument("--no-roi-weight", action="store_true", default=False)
    p.add_argument("--eft-kinematic-factor", type=float, default=1.0)
    # --- Downscatter-targeted selection ---
    p.add_argument("--downscatter-mode", default="off", choices=["off", "penalty", "hard"],
                   help="Optional selector for models where photons injected near "
                        "--downscatter-source-energy feed the observed "
                        "--downscatter-target-energy bump. 'off' preserves the "
                        "ordinary chi2 objective; 'penalty' adds a chi2-like "
                        "penalty; 'hard' rejects points that miss the target.")
    p.add_argument("--downscatter-source-energy", type=float, default=20.0,
                   help="Intrinsic source energy [GeV] expected to feed the target bump.")
    p.add_argument("--downscatter-target-energy", type=float, default=3.0,
                   help="Observed energy [GeV] where the redistributed component should peak.")
    p.add_argument("--downscatter-source-window-dex", type=float, default=0.20,
                   help="Half-width in log10(E) around the source energy.")
    p.add_argument("--downscatter-target-window-dex", type=float, default=0.20,
                   help="Half-width in log10(E) around the observed target energy.")
    p.add_argument("--downscatter-peak-weight", type=float, default=10.0,
                   help="Penalty weight for the source-window in-scatter peak missing the target.")
    p.add_argument("--downscatter-frac-weight", type=float, default=10.0,
                   help="Penalty weight for failing requested downscatter fraction thresholds.")
    p.add_argument("--min-source-to-target-frac", type=float, default=0.0,
                   help="Require/penalize this minimum fraction of source-window in-scatter "
                        "to land in the target observed window.")
    p.add_argument("--min-target-in-from-source-frac", type=float, default=0.0,
                   help="Require/penalize this minimum fraction of target-window in-scatter "
                        "to originate from the source window.")
    p.add_argument("--min-target-model-from-source-frac", type=float, default=0.0,
                   help="Require/penalize this minimum fraction of the total target-window "
                        "observed model to come from source-window in-scatter.")
    # --- I/O ---
    p.add_argument("--pppc-gamma-table", default=None,
                   help="Path to PPPC gamma table (auto-discovered if None).")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (auto-named if None).")
    p.add_argument("--max-correction-panels", type=int, default=9,
                   help="Max panels in the correction grid figure.")
    p.add_argument("--quick", action="store_true", default=False,
                   help="Reduce grid sizes for a fast test run.")
    p.add_argument(
        "--style",
        default=None,
        help="Plot style: paper, conference/dark_transparent, or conference_light/light_transparent.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style

    if args.quick:
        args.n_ann_mass = min(args.n_ann_mass, 20)
        args.n_scatter_mass = min(args.n_scatter_mass, 4)
        args.n_lambda = min(args.n_lambda, 4)
        args.n_theta = min(args.n_theta, 80)

    t0 = time.time()

    # ------------------------------------------------------------------
    # 0. Configure module-level Totani arrays for the selected profile.
    #    This must run before any ReshapingConfig is constructed so that
    #    defaults and pppc_energy_flux_template normalisation are consistent.
    # ------------------------------------------------------------------
    configure_totani_arrays(_MCMC_DIRS[args.halo_profile])

    # ------------------------------------------------------------------
    # 1. Load halo spectrum
    # ------------------------------------------------------------------
    halo = load_halo_spectrum(_MCMC_DIRS[args.halo_profile])
    E = np.asarray(halo.E_bins_GeV, dtype=float)
    phi = np.asarray(halo.phi, dtype=float)
    phi_lo = np.asarray(halo.phi_err_lo, dtype=float)
    phi_hi = np.asarray(halo.phi_err_hi, dtype=float)
    err = np.asarray(
        getattr(halo, f"phi_err_{args.err_mode}", halo.phi_err_sym), dtype=float
    )

    # Use positive-only bins by default (matches Totani paper + other scripts).
    # --include-nonpositive-bins enables all finite bins.
    if args.include_nonpositive_bins:
        mask = np.asarray(halo.finite_mask & (err > 0.0), dtype=bool)
    else:
        mask = np.asarray(
            halo.positive_mask & halo.finite_mask & (err > 0.0), dtype=bool
        )

    print(f"Loaded halo profile '{args.halo_profile}': "
          f"{int(mask.sum())} / {len(E)} bins used in fit")

    # ------------------------------------------------------------------
    # 2. Build template bank and fit pure annihilation
    # ------------------------------------------------------------------
    ann_masses = _mass_grid(args)
    bank = build_template_bank(E, ann_masses, args.ann_channel, args.pppc_gamma_table)
    print(f"Template bank: {int(bank.valid.sum())} / {len(ann_masses)} masses valid "
          f"for channel {args.ann_channel}")

    # Build an optional extended energy grid for the scattering fit.  The
    # intrinsic PPPC source is evaluated on this grid, then T_obs maps it into
    # the observed Totani bins after attenuation plus redistribution.
    n_ext = int(args.n_ext_bins)
    ext_energy_max = float(args.ext_energy_max)
    if n_ext > 0 and ext_energy_max > E[-1]:
        E_above = np.logspace(np.log10(E[-1] * 1.5), np.log10(ext_energy_max), n_ext)
        E_ext = np.concatenate([E, E_above])
        bank_ext = build_template_bank(E_ext, ann_masses, args.ann_channel, args.pppc_gamma_table)
        print(f"Extended source grid: {len(E_ext)} bins up to {ext_energy_max:.0f} GeV "
              f"({n_ext} extra bins above {E[-1]:.0f} GeV)")
    else:
        E_ext = E
        bank_ext = bank
        n_ext = 0

    pure_fit = fit_templates(phi, err, mask, bank)
    if pure_fit["best_i"] < 0:
        raise RuntimeError("No finite pure-annihilation PPPC fit was found.")

    ndf = int(mask.sum()) - 1
    cl_delta = _delta_chi2_from_cl(args.cl, ndof=2)

    # Select pure-annihilation best fit according to chosen criterion
    if args.best_fit_type == "dsph_cl":
        pure_best_i = _select_dsph_cl(
            pure_fit["chi2"], pure_fit["norm"], bank.masses, bank.valid,
            cl_delta=cl_delta, channel=args.ann_channel, source=args.dsph_source,
        )
        if pure_best_i < 0:
            print("  [dsph_cl] No pure-ann point satisfies CL window + dSph; "
                  "trying min_tension_cl.")
            pure_best_i = _select_min_tension_cl(
                pure_fit["chi2"], pure_fit["norm"], bank.masses, bank.valid,
                cl_delta=cl_delta, channel=args.ann_channel, source=args.dsph_source,
            )
            if pure_best_i < 0:
                print("  [dsph_cl] No min_tension_cl point; falling back to spectral.")
                pure_best_i = pure_fit["best_i"]
    elif args.best_fit_type == "min_tension_cl":
        pure_best_i = _select_min_tension_cl(
            pure_fit["chi2"], pure_fit["norm"], bank.masses, bank.valid,
            cl_delta=cl_delta, channel=args.ann_channel, source=args.dsph_source,
        )
        if pure_best_i < 0:
            print("  [min_tension_cl] No pure-ann point inside CL window has finite dSph tension; "
                  "falling back to spectral.")
            pure_best_i = pure_fit["best_i"]
    else:
        pure_best_i = pure_fit["best_i"]

    pure_sel_mass   = float(bank.masses[pure_best_i])
    pure_sel_chi2   = float(pure_fit["chi2"][pure_best_i])
    pure_sel_norm   = float(pure_fit["norm"][pure_best_i])
    pure_sel_model  = pure_fit["models"][pure_best_i].copy()
    pure_sel_sv     = smooth_nfw_sigma_v_from_norm(pure_sel_norm, pure_sel_mass)
    pure_sel_dsph   = dsph_upper_limit(pure_sel_mass, args.ann_channel, source=args.dsph_source)
    pure_sel_tension = pure_sel_sv / pure_sel_dsph if np.isfinite(pure_sel_dsph) and pure_sel_dsph > 0 else np.nan

    print(f"Pure annihilation best fit ({args.best_fit_type}): "
          f"m_ann = {pure_sel_mass:.4g} GeV, "
          f"chi2 = {pure_sel_chi2:.4g}, "
          f"chi2/nu = {pure_sel_chi2/max(ndf,1):.3f}, "
          f"<sigma v> = {pure_sel_sv:.3e} cm^3/s, "
          f"tension = {pure_sel_tension:.3f}x dSph")

    # ------------------------------------------------------------------
    # 3. EFT operator defaults and scatterer mass grid
    # ------------------------------------------------------------------
    op = _operator_defaults(args.operator, args.dm_type)
    if args.dark_sector_model == "same":
        scatter_masses = ann_masses.copy()
    else:
        scatter_masses = make_scatter_mass_grid(args, pure_sel_mass)
    print(f"Operator: {op['operator']} ({op['dm_type']}), "
          f"scatterer masses: {len(scatter_masses)} points")

    # ------------------------------------------------------------------
    # 4. Deconvolution scan
    # ------------------------------------------------------------------
    nM = len(ann_masses)
    best_scatter_chi2_by_mass = np.full(nM, np.nan, dtype=float)
    best_scatter_recon_chi2_by_mass = np.full(nM, np.nan, dtype=float)
    best_scatter_global = None
    scan_rows: list[dict] = []
    selection_rows: list[dict] = []
    examples: list[dict] = []
    survival = {
        "scatter_masses": 0,
        "no_lambda": 0,
        "lambda_points": 0,
        "transfer_failed": 0,
        "tau_too_high": 0,
        "tau_too_low": 0,
        "fit_failed": 0,
        "downscatter_rejected": 0,
        "kept": 0,
        "tau_min_seen": np.inf,
        "tau_max_seen": -np.inf,
    }

    if args.dark_sector_model == "same":
        scatter_scan_items = [(int(i), float(m)) for i, m in enumerate(ann_masses)]
    else:
        scatter_scan_items = [(None, float(m)) for m in scatter_masses]

    for tied_ann_i, m_scat in scatter_scan_items:
        survival["scatter_masses"] += 1
        if args.allow_eft_invalid:
            lambdas = np.logspace(
                np.log10(args.lambda_min),
                np.log10(args.lambda_max),
                int(args.n_lambda),
            )
        else:
            lambdas = sample_valid_lambda_grid(
                op["operator"],
                float(m_scat),
                omega_max=float(np.max(E)),
                lambda_min=args.lambda_min,
                lambda_max=args.lambda_max,
                n_lambda=args.n_lambda,
                dm_type=op["dm_type"],
                eft_kinematic_factor=args.eft_kinematic_factor,
                require_lambda_gt_mdm=True,
            )
        if lambdas.size == 0:
            survival["no_lambda"] += 1
            continue

        for Lambda in lambdas:
            survival["lambda_points"] += 1
            try:
                tr = build_transfer_for_point(
                    E=E, phi=phi, err=err,
                    scatter_mass=float(m_scat), Lambda=float(Lambda),
                    op=op, args=args, E_ext=E_ext,
                )
            except Exception as exc:
                survival["transfer_failed"] += 1
                print(f"  Skipping m_scat={m_scat:.3g}, Lambda={Lambda:.3g}: {exc}")
                continue

            tau_max = float(np.nanmax(tr["tau"]))
            if np.isfinite(tau_max):
                survival["tau_min_seen"] = min(float(survival["tau_min_seen"]), tau_max)
                survival["tau_max_seen"] = max(float(survival["tau_max_seen"]), tau_max)
            if tau_max > args.max_tau:
                survival["tau_too_high"] += 1
                continue
            if tau_max < args.min_tau:
                survival["tau_too_low"] += 1
                continue

            # Diagnostic inverse problem: what intrinsic observed-bin spectrum
            # would exactly map back to the measured spectrum under the observed
            # block of T?
            phi_pre = deconvolve_observed(phi, tr["T"], args.deconv_method)
            if tied_ann_i is None:
                diagnostic_deconv_fit = fit_templates(phi_pre, err, mask, bank)
                scatter_fit = fit_transferred_templates(
                    tr["T_obs"], phi, err, mask, bank_ext, len(E)
                )
            else:
                diagnostic_deconv_fit = fit_templates(
                    phi_pre, err, mask, bank, allowed_indices=[tied_ann_i]
                )
                scatter_fit = fit_transferred_templates(
                    tr["T_obs"], phi, err, mask, bank_ext, len(E),
                    allowed_indices=[tied_ann_i],
                )
            if scatter_fit["best_i"] < 0:
                survival["fit_failed"] += 1
                continue

            local_best_row = None
            for mass_i, chi2_i in enumerate(scatter_fit["chi2"]):
                if not np.isfinite(chi2_i):
                    continue
                recon_model_i = scatter_fit["observed_models"][mass_i].copy()
                intrinsic_model_i = scatter_fit["intrinsic_models_obs"][mass_i].copy()
                down_diag = {
                    "downscatter_peak_E_GeV": np.nan,
                    "downscatter_peak_offset_dex": np.nan,
                    "source_to_target_frac": np.nan,
                    "target_in_from_source_frac": np.nan,
                    "target_model_from_source_frac": np.nan,
                    "downscatter_valid": False,
                }
                down_penalty = 0.0
                down_ok = True
                downscatter_was_tested = False
                if str(args.downscatter_mode) != "off":
                    downscatter_was_tested = True
                    down_diag = _downscatter_diagnostics(
                        T_in_obs=tr["T_in_obs"],
                        E_obs=E,
                        E_ext=E_ext,
                        src_ext=bank_ext.templates[mass_i],
                        src_norm=float(scatter_fit["norm"][mass_i]),
                        model_obs=recon_model_i,
                        source_energy=float(args.downscatter_source_energy),
                        target_energy=float(args.downscatter_target_energy),
                        source_width_dex=float(args.downscatter_source_window_dex),
                        target_width_dex=float(args.downscatter_target_window_dex),
                    )
                    down_penalty, down_ok = _downscatter_penalty(down_diag, args)
                recon_objective = float(chi2_i) + float(down_penalty)
                if not np.isfinite(recon_objective):
                    down_ok = False
                if not down_ok:
                    if downscatter_was_tested:
                        survival["downscatter_rejected"] += 1
                    continue

                if (
                    not np.isfinite(best_scatter_chi2_by_mass[mass_i])
                    or recon_objective < best_scatter_chi2_by_mass[mass_i]
                ):
                    best_scatter_chi2_by_mass[mass_i] = float(recon_objective)
                    best_scatter_recon_chi2_by_mass[mass_i] = float(chi2_i)

                row_i = {
                    "scatter_mass": float(m_scat),
                    "Lambda": float(Lambda),
                    "tau_max": tau_max,
                    "ann_mass": float(bank_ext.masses[mass_i]),
                    "deconv_chi2": (
                        float(diagnostic_deconv_fit["best_chi2"])
                        if diagnostic_deconv_fit["best_i"] >= 0 else np.nan
                    ),
                    "recon_chi2": float(chi2_i),
                    "recon_objective": float(recon_objective),
                    "downscatter_penalty": float(down_penalty),
                    **down_diag,
                    "scatter_norm": float(scatter_fit["norm"][mass_i]),
                    "phi_pre": phi_pre,
                    "intrinsic_model": intrinsic_model_i,
                    "recon_model": recon_model_i,
                    "T": tr["T"],
                    "T_obs": tr["T_obs"],
                    "T_in_obs": tr["T_in_obs"],
                    "tau": tr["tau"],
                    "sigma_tot": tr["sigma_tot"],
                }
                selection_rows.append(row_i)
                if (
                    local_best_row is None
                    or row_i["recon_objective"] < local_best_row["recon_objective"]
                ):
                    local_best_row = row_i

            if local_best_row is None:
                survival["fit_failed"] += 1
                continue
            survival["kept"] += 1

            row = local_best_row
            scan_rows.append(row)

            if len(examples) < args.max_correction_panels:
                err_safe = np.where(np.isfinite(err) & (err > 0), err, np.nan)
                examples.append({
                    "scatter_mass": float(m_scat),
                    "Lambda": float(Lambda),
                    "ann_mass": float(scatter_fit["best_mass"]),
                    "correction_over_sigma": (phi_pre - phi) / err_safe,
                    "sigma_tot_cm2": tr["sigma_tot"],
                })

            if (
                best_scatter_global is None
                or row["recon_objective"] < best_scatter_global["recon_objective"]
            ):
                best_scatter_global = row

    if best_scatter_global is None:
        tau_seen = (
            "none"
            if not np.isfinite(float(survival["tau_min_seen"]))
            else f"{float(survival['tau_min_seen']):.3e} to {float(survival['tau_max_seen']):.3e}"
        )
        print("\nScattering scan survival diagnostics:")
        print(f"  scatter masses tried : {survival['scatter_masses']}")
        print(f"  masses with no Lambda: {survival['no_lambda']}")
        print(f"  Lambda points tried  : {survival['lambda_points']}")
        print(f"  transfer failed      : {survival['transfer_failed']}")
        print(f"  tau too low          : {survival['tau_too_low']}  (< {args.min_tau:g})")
        print(f"  tau too high         : {survival['tau_too_high']}  (> {args.max_tau:g})")
        print(f"  fit failed           : {survival['fit_failed']}")
        print(f"  downscatter rejected : {survival['downscatter_rejected']}")
        print(f"  kept                 : {survival['kept']}")
        print(f"  tau_max seen         : {tau_seen}")
        raise RuntimeError(
            "No valid deconvolution scan points survived the validity / tau cuts.\n"
            "Try relaxing --min-tau/--max-tau, --lambda-min/max, or --scatter-mass-min/max."
        )

    # Apply non-spectral selectors for scattering (overrides spectral best if valid points exist)
    if args.best_fit_type in ("dsph_cl", "min_tension_cl"):
        objective_key = "recon_objective"
        chi2_min_scat = min(
            (r[objective_key] for r in scan_rows if np.isfinite(r[objective_key])),
            default=np.nan,
        )
        if np.isfinite(chi2_min_scat):
            cl_rows = [
                r for r in selection_rows
                if (np.isfinite(r[objective_key])
                    and r[objective_key] - chi2_min_scat <= cl_delta)
            ]
            if args.best_fit_type == "dsph_cl":
                candidate_rows = [
                    r for r in cl_rows
                    if smooth_nfw_sigma_v_from_norm(r["scatter_norm"], r["ann_mass"])
                    <= dsph_upper_limit(r["ann_mass"], args.ann_channel, source=args.dsph_source)
                ]
                selector_label = "dsph_cl"
                selector_key = lambda r: r[objective_key]
                if not candidate_rows:
                    print("  [dsph_cl] No scatter point satisfies CL + dSph; trying min_tension_cl.")
                    candidate_rows = [
                        r for r in cl_rows
                        if np.isfinite(
                            smooth_nfw_sigma_v_from_norm(r["scatter_norm"], r["ann_mass"])
                            / dsph_upper_limit(r["ann_mass"], args.ann_channel, source=args.dsph_source)
                        )
                    ]
                    selector_label = "min_tension_cl"
                    selector_key = lambda r: (
                        smooth_nfw_sigma_v_from_norm(r["scatter_norm"], r["ann_mass"])
                        / dsph_upper_limit(r["ann_mass"], args.ann_channel, source=args.dsph_source)
                    )
            else:
                candidate_rows = [
                    r for r in cl_rows
                    if np.isfinite(
                        smooth_nfw_sigma_v_from_norm(r["scatter_norm"], r["ann_mass"])
                        / dsph_upper_limit(r["ann_mass"], args.ann_channel, source=args.dsph_source)
                    )
                ]
                selector_label = "min_tension_cl"
                selector_key = lambda r: (
                    smooth_nfw_sigma_v_from_norm(r["scatter_norm"], r["ann_mass"])
                    / dsph_upper_limit(r["ann_mass"], args.ann_channel, source=args.dsph_source)
                )
            if candidate_rows:
                best_scatter_global = min(candidate_rows, key=selector_key)
                print(f"  [{selector_label}] {len(candidate_rows)} scatter points satisfy selector; "
                      f"best: m_ann={best_scatter_global['ann_mass']:.4g} GeV, "
                      f"m_scat={best_scatter_global['scatter_mass']:.4g} GeV")
            else:
                print(f"  [{args.best_fit_type}] No scatter point satisfies selector inside CL window; "
                      "falling back to spectral best.")

    scat_sel_sv = smooth_nfw_sigma_v_from_norm(
        float(best_scatter_global["scatter_norm"]), float(best_scatter_global["ann_mass"])
    )
    scat_sel_dsph = dsph_upper_limit(float(best_scatter_global["ann_mass"]), args.ann_channel, source=args.dsph_source)
    scat_sel_tension = (scat_sel_sv / scat_sel_dsph
                        if np.isfinite(scat_sel_dsph) and scat_sel_dsph > 0 else np.nan)

    # ------------------------------------------------------------------
    # 5. Outputs
    # ------------------------------------------------------------------
    op_str = op["operator"]
    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else _HERE / "results" / "deconvolved_scattering_fit"
              / f"{args.halo_profile}_{args.ann_channel}_{op_str}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    result = {
        "halo_profile":              np.array(args.halo_profile),
        "ann_channel":               np.array(args.ann_channel),
        "operator":                  np.array(op_str),
        "dark_sector_model":         np.array(args.dark_sector_model),
        "best_fit_type":             np.array(args.best_fit_type),
        "E_bins_GeV":                E.astype(np.float32),
        "phi_data":                  phi.astype(np.float32),
        "phi_err":                   err.astype(np.float32),
        "phi_err_lo":                phi_lo.astype(np.float32),
        "phi_err_hi":                phi_hi.astype(np.float32),
        "mask":                      mask.astype(np.uint8),
        "ann_masses_GeV":            ann_masses.astype(np.float32),
        "pure_chi2":                 pure_fit["chi2"].astype(np.float32),
        "pure_spectral_best_mass_GeV": np.float32(pure_fit["best_mass"]),
        "pure_spectral_best_chi2":     np.float32(pure_fit["best_chi2"]),
        "pure_best_ann_mass_GeV":    np.float32(pure_sel_mass),
        "pure_best_chi2":            np.float32(pure_sel_chi2),
        "pure_best_norm":            np.float32(pure_sel_norm),
        "pure_best_model":           pure_sel_model.astype(np.float32),
        "pure_best_dsph_tension":    np.float32(pure_sel_tension),
        "scatter_masses_GeV":        scatter_masses.astype(np.float32),
        "scan_scatter_mass_GeV":     np.asarray([r["scatter_mass"] for r in scan_rows], dtype=np.float32),
        "scan_Lambda_GeV":           np.asarray([r["Lambda"] for r in scan_rows], dtype=np.float32),
        "scan_tau_max":              np.asarray([r["tau_max"] for r in scan_rows], dtype=np.float32),
        "scan_best_ann_mass_GeV":    np.asarray([r["ann_mass"] for r in scan_rows], dtype=np.float32),
        "scan_deconv_chi2":          np.asarray([r["deconv_chi2"] for r in scan_rows], dtype=np.float32),
        "scan_recon_chi2":           np.asarray([r["recon_chi2"] for r in scan_rows], dtype=np.float32),
        "scan_recon_objective":      np.asarray([r["recon_objective"] for r in scan_rows], dtype=np.float32),
        "scan_downscatter_penalty":  np.asarray([r["downscatter_penalty"] for r in scan_rows], dtype=np.float32),
        "scan_downscatter_peak_E_GeV": np.asarray([r["downscatter_peak_E_GeV"] for r in scan_rows], dtype=np.float32),
        "scan_downscatter_peak_offset_dex": np.asarray([r["downscatter_peak_offset_dex"] for r in scan_rows], dtype=np.float32),
        "scan_source_to_target_frac": np.asarray([r["source_to_target_frac"] for r in scan_rows], dtype=np.float32),
        "scan_target_in_from_source_frac": np.asarray([r["target_in_from_source_frac"] for r in scan_rows], dtype=np.float32),
        "scan_target_model_from_source_frac": np.asarray([r["target_model_from_source_frac"] for r in scan_rows], dtype=np.float32),
        "best_scatter_chi2_by_ann_mass": best_scatter_chi2_by_mass.astype(np.float32),
        "best_scatter_recon_chi2_by_ann_mass": best_scatter_recon_chi2_by_mass.astype(np.float32),
        "best_ann_mass_GeV":         np.float32(best_scatter_global["ann_mass"]),
        "best_scatter_mass_GeV":     np.float32(best_scatter_global["scatter_mass"]),
        "best_Lambda_GeV":           np.float32(best_scatter_global["Lambda"]),
        "best_scatter_norm":         np.float32(best_scatter_global["scatter_norm"]),
        "best_deconv_chi2":          np.float32(best_scatter_global["deconv_chi2"]),
        "best_recon_chi2":           np.float32(best_scatter_global["recon_chi2"]),
        "best_recon_objective":      np.float32(best_scatter_global["recon_objective"]),
        "best_downscatter_penalty":  np.float32(best_scatter_global["downscatter_penalty"]),
        "best_downscatter_peak_E_GeV": np.float32(best_scatter_global["downscatter_peak_E_GeV"]),
        "best_downscatter_peak_offset_dex": np.float32(best_scatter_global["downscatter_peak_offset_dex"]),
        "best_source_to_target_frac": np.float32(best_scatter_global["source_to_target_frac"]),
        "best_target_in_from_source_frac": np.float32(best_scatter_global["target_in_from_source_frac"]),
        "best_target_model_from_source_frac": np.float32(best_scatter_global["target_model_from_source_frac"]),
        "downscatter_mode":          np.array(args.downscatter_mode),
        "downscatter_source_energy_GeV": np.float32(args.downscatter_source_energy),
        "downscatter_target_energy_GeV": np.float32(args.downscatter_target_energy),
        "best_phi_pre":              best_scatter_global["phi_pre"].astype(np.float32),
        "best_intrinsic_model":      best_scatter_global["intrinsic_model"].astype(np.float32),
        "best_reconvolved_model":    best_scatter_global["recon_model"].astype(np.float32),
        "best_tau":                  best_scatter_global["tau"].astype(np.float32),
        "best_sigma_tot_cm2":        best_scatter_global["sigma_tot"].astype(np.float32),
        "best_scatter_dsph_tension": np.float32(scat_sel_tension),
        "dsph_limit_source":         np.array(dsph_limit_source(args.ann_channel, source=args.dsph_source)),
        "delta_chi2_cl":             np.float32(cl_delta),
        "cl":                        np.float32(args.cl),
        "n_scan_points":             np.int32(len(scan_rows)),
        "example_points":            examples,  # not saved to npz
    }

    np.savez_compressed(
        outdir / "deconvolved_scattering_fit.npz",
        **{k: v for k, v in result.items() if k != "example_points"},
    )

    plot_correction_grid(result, outdir / "deconvolution_correction_grid.png",
                         args.max_correction_panels)
    plot_best_fit_summary(result, outdir / "best_fit_summary.png")
    plot_mass_shift_diagnostic(result, outdir / "mass_shift_diagnostic.png")
    bin_contribution_text = _best_bin_contribution_table(
        E=E,
        phi_data=phi,
        phi_err=err,
        mask=mask,
        intrinsic_model=best_scatter_global["intrinsic_model"],
        recon_model=best_scatter_global["recon_model"],
        T_obs=best_scatter_global["T_obs"],
        T_in_obs=best_scatter_global["T_in_obs"],
        tau=best_scatter_global["tau"],
        out_csv=outdir / "best_fit_scattering_bin_contributions.csv",
    )

    # ------------------------------------------------------------------
    # 6. Text summary
    # ------------------------------------------------------------------
    summary_lines = [
        "Attenuation-plus-reshaping scattering fit",
        "=" * 40,
        f"halo profile        : {args.halo_profile}",
        f"channel             : {args.ann_channel}",
        f"dark-sector model   : {args.dark_sector_model}",
        f"operator            : {op_str} ({op['dm_type']})",
        f"EFT validity        : {'not enforced' if args.allow_eft_invalid else 'enforced'}",
        f"best-fit type       : {args.best_fit_type}",
        f"dSph limits         : {dsph_limit_source(args.ann_channel, source=args.dsph_source)}",
        f"fit bins            : {int(mask.sum())} / {len(E)} "
        f"({'all finite' if args.include_nonpositive_bins else 'positive only'})",
        f"CL delta chi2       : {cl_delta:.4g} ({100 * args.cl:.1f}%)",
        f"tau window          : {args.min_tau:.4g} <= tau_max <= {args.max_tau:.4g}",
        f"downscatter mode    : {args.downscatter_mode}",
    ]
    if str(args.downscatter_mode) != "off":
        summary_lines.extend([
            f"downscatter source : {args.downscatter_source_energy:.6g} GeV "
            f"(+/- {args.downscatter_source_window_dex:.3g} dex)",
            f"downscatter target : {args.downscatter_target_energy:.6g} GeV "
            f"(+/- {args.downscatter_target_window_dex:.3g} dex)",
        ])
    summary_lines.extend([
        "",
        f"--- Pure annihilation ({args.best_fit_type}) ---",
        f"spectral best m_ann : {pure_fit['best_mass']:.6g} GeV",
        f"selected m_ann      : {pure_sel_mass:.6g} GeV",
        f"chi2                : {pure_sel_chi2:.6g}",
        f"chi2 / nu           : {pure_sel_chi2 / max(ndf, 1):.4g}",
        f"<sigma v>           : {pure_sel_sv:.4e} cm^3/s",
        f"dSph limit          : {pure_sel_dsph:.4e} cm^3/s",
        f"tension             : {pure_sel_tension:.3f}x dSph",
        "",
        f"--- Best scattering fit ({args.best_fit_type}) ---",
        f"best m_ann          : {best_scatter_global['ann_mass']:.6g} GeV",
        f"best m_scat         : {best_scatter_global['scatter_mass']:.6g} GeV",
        f"best Lambda         : {best_scatter_global['Lambda']:.6g} GeV",
        f"fit chi2            : {best_scatter_global['recon_chi2']:.6g}",
        f"selection objective : {best_scatter_global['recon_objective']:.6g}",
        f"downscatter penalty : {best_scatter_global['downscatter_penalty']:.6g}",
        f"diagnostic deconv chi2: {best_scatter_global['deconv_chi2']:.6g}",
        f"<sigma v>           : {scat_sel_sv:.4e} cm^3/s",
        f"dSph limit          : {scat_sel_dsph:.4e} cm^3/s",
        f"tension             : {scat_sel_tension:.3f}x dSph",
        f"tau_max             : {best_scatter_global['tau_max']:.4g}",
        f"downscatter peak    : {best_scatter_global['downscatter_peak_E_GeV']:.6g} GeV",
        f"source->target frac : {best_scatter_global['source_to_target_frac']:.6g}",
        f"target in from source: {best_scatter_global['target_in_from_source_frac']:.6g}",
        f"target model from source: {best_scatter_global['target_model_from_source_frac']:.6g}",
        "",
        f"scan points kept    : {len(scan_rows)}",
        f"wall time           : {time.time() - t0:.1f} s",
    ])
    summary_lines.append(bin_contribution_text)
    summary_text = "\n".join(summary_lines)
    (outdir / "summary.txt").write_text(summary_text + "\n")
    print(summary_text)
    print(f"\nOutputs saved in: {outdir}")


if __name__ == "__main__":
    main()
