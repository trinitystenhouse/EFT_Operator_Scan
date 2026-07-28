#!/usr/bin/env python3
"""
fit_totani_dm_scattering.py
===========================
Fit Totani-style WIMP parameters with photon-DM scattering.

This script follows Totani's section 4.2 workflow, but replaces the direct
annihilation spectrum with a transferred spectrum:

    PPPC E^2 dN/dE  ->  attenuation + single-scatter spectral reshaping

For each annihilation mass, the script fits the overall normalization
analytically.  For the smooth NFW-rho^2 interpretation, that normalization is
converted to <sigma v> using Totani's galactic-pole J factor.

Data source
-----------
The target spectrum (phi_data) is loaded from the Totani_paper_check MCMC
posteriors, NOT from the hard-coded PHI_TOTANI / SIGMA_TOTANI arrays in
attenuation_eft.py, which are coarse read-offs from Totani's Figure 8 in
units that were never precisely calibrated.

The MCMC files store the posterior over template coefficients f_nfw, fitted
to the actual Fermi-LAT counts via a binned Poisson likelihood on 144 spatial
cells per energy bin.  The halo spectrum is:

    E^2 dN/dE|_pole = f_nfw * iso_target_e2       [MeV cm^-2 s^-1 sr^-1]

with 1-sigma errorbars from [f_p16, f_p84].  The central value uses f_p50
(posterior median) to match Totani's Figure 8 convention — the median
guarantees p16 <= p50 <= p84 near the positivity boundary, whereas f_ml can
produce negative errorbars for near-zero bins.

Use --halo-profile to select the NFW morphology:
  rho2    (default)  rho^2,   disk excluded — Totani's primary result
  rho2.5             rho^2.5, disk excluded
  rho1               rho^1,   disk excluded

Energy axis
-----------
The 13 Totani energy bins live in the npz files as Ectr_mev.  The PPPC source
spectrum is interpolated onto the same energy grid so that the template shape,
the data, and the scattering kernel all share an identical E axis.

Normalization and <sigma v>
---------------------------
The fitted normalization A maps to <sigma v> via Totani's galactic-pole
J-factor (Eq. 4.1 of the paper):

    E^2 dN/dE|_pole = (<sigma v> / 8 pi m_chi^2) * E^2 dN/dE|_PPPC * J_pole

    <sigma v> = A * 8 pi m_chi^2 / J_pole    [cm^3 s^-1]

where A carries units of [MeV cm^-2 s^-1 sr^-1 / (E^2 dN/dE|_PPPC in GeV)].
smooth_nfw_sigma_v_from_norm handles the unit conversion.
"""

from __future__ import annotations

import argparse
import os
import sys
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

from core.totani_data_loader import (
    HaloSpectrum,
    _MCMC_DIRS,
    available_mcmc_dirs,
    load_halo_spectrum,
)
from core.spectral_reshaping import (
    ReshapingConfig,
    apply_single_scatter_transfer,
    best_fit_normalization,
    build_dsigma_grid,
    build_kernel,
    compute_tau_spectrum,
    pppc_energy_flux_template,
    smooth_nfw_sigma_v_from_norm,
)
from core.eft_validity import is_eft_point_valid
from helpers.trinity_plotting import current_savefig_kwargs, set_plot_style


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FitPoint:
    ann_mass:     float
    scatter_mass: float
    Lambda:       float
    y_eff:        float
    chi2:         float
    norm:         float
    sigmav:       float
    tau_max:      float
    model:        np.ndarray   # best-fit E^2 dN/dE at pole (transferred * norm)
    source:       np.ndarray   # PPPC template before transfer
    tau:          np.ndarray   # optical depth per energy bin
    K:            np.ndarray   # (nE, nE) redistribution kernel
    # Data arrays used for this fit point (same for all points in a scan)
    E_bins_GeV:   np.ndarray = field(default_factory=lambda: np.array([]))
    phi_data:     np.ndarray = field(default_factory=lambda: np.array([]))
    phi_err:      np.ndarray = field(default_factory=lambda: np.array([]))
    invalid_reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_log_grid(min_val: float, max_val: float, n_val: int) -> np.ndarray:
    if n_val <= 1:
        return np.array([float(min_val)])
    return np.logspace(np.log10(float(min_val)), np.log10(float(max_val)), int(n_val))


# ---------------------------------------------------------------------------
# Single fit point
# ---------------------------------------------------------------------------

def fit_one_point(
    *,
    ann_mass:     float,
    scatter_mass: float,
    Lambda:       float,
    y_eff:        float,
    args:         argparse.Namespace,
    l_grid:       np.ndarray,
    b_grid:       np.ndarray,
    # Live data from MCMC posteriors (all on the same energy axis)
    E_bins_GeV:   np.ndarray,
    phi_data:     np.ndarray,
    phi_err:      np.ndarray,
    mask:         np.ndarray,
) -> FitPoint:
    """
    Fit one (ann_mass, scatter_mass, Lambda) point.

    Parameters
    ----------
    E_bins_GeV : (nE,)  energy axis from the MCMC npz files [GeV]
    phi_data   : (nE,)  observed halo spectrum = f_p50 * iso_target_e2 [MeV cm^-2 s^-1 sr^-1]
    phi_err    : (nE,)  1-sigma errorbars (symmetric = 0.5*(p84-p16)) [same units]
    mask       : (nE,)  bool — bins to include in chi2 (phi_data > 0)

    Returns
    -------
    FitPoint
    """
    nE = len(E_bins_GeV)
    _nan_arr = np.full(nE, np.nan)
    _zero_K  = np.zeros((nE, nE))

    # EFT validity guard
    if (
        args.operator != "higgs_portal"
        and not is_eft_point_valid(
            args.operator,
            scatter_mass,
            Lambda,
            omega_max=float(np.max(E_bins_GeV)),
            dm_type=args.dm_type,
            eft_kinematic_factor=float(getattr(args, "eft_kinematic_factor", 1.0)),
            require_lambda_gt_mdm=bool(getattr(args, "require_lambda_gt_mdm", True)),
            include_kinematic=True,
            include_unitarity=True,
        )
    ):
        return FitPoint(
            ann_mass, scatter_mass, Lambda, y_eff, np.nan, np.nan, np.nan, np.nan,
            _nan_arr.copy(), _nan_arr.copy(), _nan_arr.copy(), _zero_K,
            E_bins_GeV, phi_data, phi_err,
            "eft_invalid",
        )

    # Build PPPC annihilation source spectrum on the same energy grid as the data
    source = pppc_energy_flux_template(
        E_bins_GeV,                    # use the live energy axis, not the hardcoded one
        ann_mass,
        channel=args.ann_channel,
        primary="gamma",
        table_path=args.pppc_gamma_table,
        normalise=False,
    )

    # Scattering config: phi_0 is the annihilation source, phi_data is the
    # MCMC-derived halo spectrum we are trying to fit
    cfg = ReshapingConfig(
        m_chi=scatter_mass,
        Lambda=Lambda,
        dm_type=args.dm_type,
        operator=args.operator,
        c_s=args.c_s,
        c_p=args.c_p,
        c_phi=args.c_phi,
        majorana=args.majorana,
        y_eff=y_eff,
        l_grid=l_grid,
        b_grid=b_grid,
        n_theta=args.n_theta,
        apply_roi_weight=args.apply_roi_weight,
        roi_half_angle_deg=args.roi_half_angle if args.apply_roi_weight else None,
        E_bins=E_bins_GeV,
        phi_0=source,
        phi_data=phi_data,
        phi_err=phi_err,
        fit_normalization=True,
        max_tau_single_scatter=args.max_tau_single_scatter,
        require_lambda_gt_mdm=False,  # already guarded above
    )

    # Scattering optical depth
    cos_theta, dsig, sigma_tot = build_dsigma_grid(cfg)
    tau = compute_tau_spectrum(cfg)
    tau_max = float(np.nanmax(tau))
    if not np.all(np.isfinite(sigma_tot)):
        return FitPoint(
            ann_mass, scatter_mass, Lambda, y_eff, np.nan, np.nan, np.nan, tau_max,
            _nan_arr.copy(), source, tau, _zero_K,
            E_bins_GeV, phi_data, phi_err,
            "nonfinite_sigma_tot",
        )
    if not np.all(np.isfinite(tau)):
        return FitPoint(
            ann_mass, scatter_mass, Lambda, y_eff, np.nan, np.nan, np.nan, tau_max,
            _nan_arr.copy(), source, tau, _zero_K,
            E_bins_GeV, phi_data, phi_err,
            "nonfinite_tau",
        )

    if (
        args.max_tau_single_scatter is not None
        and args.max_tau_single_scatter >= 0.0
        and np.isfinite(tau_max)
        and tau_max > args.max_tau_single_scatter
    ):
        return FitPoint(
            ann_mass, scatter_mass, Lambda, y_eff, np.nan, np.nan, np.nan, tau_max,
            _nan_arr.copy(), source, tau, _zero_K,
            E_bins_GeV, phi_data, phi_err,
            "tau_cut",
        )

    # Build redistribution kernel and apply transfer
    K = build_kernel(cfg, cos_theta, dsig, sigma_tot)
    transferred, _, _ = apply_single_scatter_transfer(source, tau, K, E_bins_GeV)
    if not np.all(np.isfinite(transferred)):
        return FitPoint(
            ann_mass, scatter_mass, Lambda, y_eff, np.nan, np.nan, np.nan, tau_max,
            _nan_arr.copy(), source, tau, K,
            E_bins_GeV, phi_data, phi_err,
            "nonfinite_transferred",
        )
    transfer_power = float(np.sum((transferred[mask] / phi_err[mask])**2))
    if transfer_power <= 0.0 or not np.isfinite(transfer_power):
        return FitPoint(
            ann_mass, scatter_mass, Lambda, y_eff, np.nan, np.nan, np.nan, tau_max,
            _nan_arr.copy(), source, tau, K,
            E_bins_GeV, phi_data, phi_err,
            "zero_transferred_template",
        )

    # Analytic WLS normalization: minimize sum[(A*T - D)^2 / sigma^2]
    norm = best_fit_normalization(transferred, phi_data, phi_err, mask)
    if not np.isfinite(norm) or norm < 0.0:
        return FitPoint(
            ann_mass, scatter_mass, Lambda, y_eff, np.nan, np.nan, np.nan, tau_max,
            _nan_arr.copy(), source, tau, K,
            E_bins_GeV, phi_data, phi_err,
            "invalid_norm",
        )

    model = norm * transferred
    residuals = (model[mask] - phi_data[mask]) / phi_err[mask]
    chi2 = float(np.sum(residuals ** 2))

    # Convert normalization to <sigma v> using Totani's pole J-factor
    sigmav = smooth_nfw_sigma_v_from_norm(norm, ann_mass)

    return FitPoint(
        ann_mass, scatter_mass, Lambda, y_eff, chi2, norm, sigmav, tau_max,
        model, source, tau, K,
        E_bins_GeV, phi_data, phi_err,
        "",
    )


# ---------------------------------------------------------------------------
# Best-fit plot
# ---------------------------------------------------------------------------

def make_best_fit_plot(
    best:      FitPoint,
    halo:      HaloSpectrum,
    args:      argparse.Namespace,
    out_path:  Path,
) -> None:
    """Plot best-fit model spectrum against the MCMC-derived halo data."""
    import matplotlib.pyplot as plt
    set_plot_style(style="paper", cmap_name="plasma", base_fontsize=10, linewidth=1.6, n_colors=8)

    E = best.E_bins_GeV
    scale = 1e5   # display in units of 1e-5 MeV cm^-2 s^-1 sr^-1

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8.0, 6.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        sharex=True,
    )

    # Data: asymmetric errorbars from MCMC posterior
    ax.errorbar(
        E, best.phi_data * scale,
        yerr=[halo.phi_err_lo * scale, halo.phi_err_hi * scale],
        fmt="ko", ms=5, lw=1.2, capsize=3,
        label=f"Totani halo (MCMC, {halo.nfw_label[:20]}…)",
    )
    ax.plot(E, best.model * scale, "C0-", lw=2.0,
            label=f"Best fit: {args.ann_channel}, "
                  fr"$m_\chi={best.ann_mass:.0f}$ GeV, "
                  fr"$\langle\sigma v\rangle={best.sigmav:.2e}$ cm$^3$/s")
    ax.plot(E, (best.norm * best.source) * scale, "C1--", lw=1.6, alpha=0.8,
            label="Intrinsic annihilation template (pre-transfer)")
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")

    ax.set_xscale("log")
    ax.set_ylabel(r"$E^2\,dN/dE\ [\times 10^{-5}\ \mathrm{MeV\,cm^{-2}\,s^{-1}\,sr^{-1}}]$",
                  fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    coupling_text = (
        rf"$y_\mathrm{{eff}}={best.y_eff:.0e}$ GeV"
        if args.operator == "higgs_portal"
        else rf"$\Lambda={best.Lambda:.0e}$ GeV"
    )
    ax.set_title(
        rf"$\chi^2={best.chi2:.2f}$, {coupling_text}, "
        rf"$\tau_\mathrm{{max}}={best.tau_max:.1e}$, "
        rf"{args.operator} ({args.dm_type})",
        fontsize=9,
    )

    # Residuals
    mask = halo.positive_mask
    resid = np.full(len(E), np.nan)
    resid[mask] = (best.model[mask] - best.phi_data[mask]) / best.phi_err[mask]
    ax_res.axhline(0, color="k", lw=0.8)
    ax_res.axhline(1, color="0.7", lw=0.6, ls="--")
    ax_res.axhline(-1, color="0.7", lw=0.6, ls="--")
    ax_res.plot(E, resid, "C0o", ms=4)
    ax_res.set_xlabel("Photon energy [GeV]", fontsize=10)
    ax_res.set_ylabel(r"$(\mathrm{model}-\mathrm{data})/\sigma$", fontsize=9)
    ax_res.set_ylim(-4, 4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit Totani-style WIMP mass and annihilation cross section "
            "with photon-DM scattering transfer. Data loaded from MCMC posteriors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data source
    p.add_argument(
        "--halo-profile",
        default="rho2",
        choices=list(_MCMC_DIRS.keys()),
        help=(
            "Which NFW profile's MCMC results to use as target data. "
            "rho2 is Totani's primary result (rho^2, disk excluded)."
        ),
    )
    p.add_argument(
        "--nfw-label", default=None,
        help="Exact NFW template label in the MCMC npz files. Auto-detected if not set.",
    )
    p.add_argument(
        "--err-mode", default="sym",
        choices=["sym", "lo", "hi", "max"],
        help=(
            "Error convention for the data. "
            "'sym' = 0.5*(p84-p16); 'lo'/'hi' = asymmetric; 'max' = conservative."
        ),
    )
    p.add_argument(
        "--counts-path", default=None,
        help="Path to counts CCUBE FITS for reading the exact energy axis. "
             "If not set, energy axis is read from the npz Ectr_mev fields.",
    )

    # Annihilation model
    p.add_argument("--ann-channel", default="WW",
                   help="PPPC annihilation channel: WW, bb, tautau, etc.")
    p.add_argument("--pppc-gamma-table", default=None,
                   help="Path to PPPC4DMID AtProduction_gammas.dat.")
    p.add_argument("--ann-mass-min", type=float, default=100.0,
                   help="Minimum annihilation WIMP mass [GeV].")
    p.add_argument("--ann-mass-max", type=float, default=2000.0,
                   help="Maximum annihilation WIMP mass [GeV].")
    p.add_argument("--n-ann-mass", type=int, default=80,
                   help="Number of mass scan points.")

    # Scattering parameters
    p.add_argument("--scatter-mass-mode", choices=["tied", "fixed"], default="tied",
                   help="'tied': scattering mass equals annihilation mass.")
    p.add_argument("--scatter-mass", type=float, default=600.0,
                   help="Fixed scattering mass [GeV] (only used if --scatter-mass-mode=fixed).")
    p.add_argument("--operator", default="dipole_magnetic",
                   choices=[
                       "dipole_magnetic", "dipole_electric",
                       "charge_radius", "anapole",
                       "rayleigh_even", "rayleigh_odd", "rayleigh_full",
                       "higgs_portal",
                   ])
    p.add_argument("--dm-type", default="fermionic", choices=["fermionic", "scalar"])
    p.add_argument("--majorana", action="store_true", default=False)
    p.add_argument("--c-s",   type=float, default=1.0)
    p.add_argument("--c-p",   type=float, default=0.0)
    p.add_argument("--c-phi", type=float, default=1.0)
    p.add_argument("--y-eff", type=float, default=1.0,
                   help="Fixed Higgs-portal y_eff [GeV], matching the York/fig6 convention.")
    p.add_argument("--scan-y-eff", action="store_true", default=False,
                   help="Scan y_eff instead of using a fixed value (Higgs portal only).")
    p.add_argument("--y-eff-min", type=float, default=1e8,
                   help="Minimum y_eff [GeV] for --scan-y-eff.")
    p.add_argument("--y-eff-max", type=float, default=1e14,
                   help="Maximum y_eff [GeV] for --scan-y-eff.")
    p.add_argument("--n-y-eff", type=int, default=50,
                   help="Number of y_eff scan points.")

    # Lambda scan
    p.add_argument("--lambda", dest="lambda_fixed", type=float, default=1e3,
                   help="Fixed Lambda [GeV] (unless --scan-lambda).")
    p.add_argument("--scan-lambda", action="store_true", default=False,
                   help="Also scan Lambda as a free parameter.")
    p.add_argument("--lambda-min",  type=float, default=10.0)
    p.add_argument("--lambda-max",  type=float, default=1e5)
    p.add_argument("--n-lambda",    type=int,   default=40)
    p.add_argument("--require-lambda-gt-mdm", action="store_true", default=True,
                   help="Mask points where Lambda <= scatter mass (EFT invalid).")
    p.add_argument("--allow-lambda-le-mdm", dest="require_lambda_gt_mdm",
                   action="store_false")
    p.add_argument("--eft-kinematic-factor", type=float, default=1.0,
                   help="Safety factor kappa in Lambda^2 >= kappa * max(s_max, |t|_max).")

    # Numerical options
    p.add_argument("--n-theta", type=int, default=600,
                   help="Angular integration nodes for dσ/dΩ.")
    p.add_argument("--apply-roi-weight",    action="store_true",  default=True)
    p.add_argument("--no-roi-weight", dest="apply_roi_weight", action="store_false")
    p.add_argument("--roi-half-angle", type=float, default=60.0)
    p.add_argument("--max-tau-single-scatter", type=float, default=0.3,
                   help="Reject points with tau_max above this; negative disables.")
    p.add_argument("--include-nonpositive-bins", action="store_true", default=False,
                   help="Include bins where phi_data <= 0 in chi2 (not recommended).")

    # Output
    p.add_argument("--output-dir", default=None,
                   help="Output directory. Defaults to results/<prefix>/.")
    p.add_argument("--output-prefix", default="totani_dm_scattering_fit",
                   help="Output prefix for file names.")
    p.add_argument(
        "--style",
        default=None,
        help="Plot style: paper, conference/dark_transparent, or conference_light/light_transparent.",
    )
    p.add_argument("--verbose", action="store_true", default=False)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style

    if args.max_tau_single_scatter is not None and args.max_tau_single_scatter < 0.0:
        args.max_tau_single_scatter = None

    # ------------------------------------------------------------------
    # Load the observed halo spectrum from the MCMC posteriors
    # ------------------------------------------------------------------
    print(f"\nLoading halo spectrum from MCMC results: {args.halo_profile}")
    mcmc_dir = _MCMC_DIRS[args.halo_profile]
    halo = load_halo_spectrum(
        mcmc_dir,
        nfw_label=args.nfw_label,
        counts_path=args.counts_path,
    )
    print(halo.summary())

    # Energy axis: use what came out of the MCMC files
    E_bins_GeV = halo.E_bins_GeV
    nE = len(E_bins_GeV)

    # Target data and errors: posterior median and symmetric 1-sigma
    phi_data = halo.phi                # f_p50 * iso_target_e2  [MeV cm^-2 s^-1 sr^-1]
    phi_err  = getattr(halo, f"phi_err_{args.err_mode}", halo.phi_err_sym)

    # Chi2 mask: bins where the data is positive and has finite errors
    if args.include_nonpositive_bins:
        mask = halo.finite_mask
    else:
        mask = halo.positive_mask & halo.finite_mask & (phi_err > 0)

    print(f"  Fitting {int(mask.sum())} / {nE} energy bins (mask: positive + finite)")

    # ------------------------------------------------------------------
    # Build scan grids
    # ------------------------------------------------------------------
    if args.scan_y_eff and args.operator != "higgs_portal":
        raise ValueError("--scan-y-eff is only meaningful with --operator higgs_portal")
    if args.scan_y_eff and args.scan_lambda:
        raise ValueError("This fitter supports one scattering scan axis at a time: use either --scan-y-eff or --scan-lambda.")

    ann_masses = parse_log_grid(args.ann_mass_min, args.ann_mass_max, args.n_ann_mass)
    lambdas = (
        parse_log_grid(args.lambda_min, args.lambda_max, args.n_lambda)
        if args.scan_lambda
        else np.array([float(args.lambda_fixed)])
    )
    y_effs = (
        parse_log_grid(args.y_eff_min, args.y_eff_max, args.n_y_eff)
        if args.scan_y_eff
        else np.array([float(args.y_eff)])
    )
    scan_axis = "y_eff" if args.scan_y_eff else "Lambda"
    scan_values = y_effs if args.scan_y_eff else lambdas

    l_grid = np.linspace(-60.0, 60.0, 15)
    b_grid = np.concatenate([
        np.linspace(-60.0, -10.0, 8),
        np.linspace(10.0, 60.0, 8),
    ])

    # ------------------------------------------------------------------
    # Grid scan
    # ------------------------------------------------------------------
    shape = (len(ann_masses), len(scan_values))
    chi2_grid      = np.full(shape, np.nan)
    sigmav_grid    = np.full(shape, np.nan)
    norm_grid      = np.full(shape, np.nan)
    tau_max_grid   = np.full(shape, np.nan)
    scatter_m_grid = np.full(shape, np.nan)
    reason_counts: dict[str, int] = {}

    best: Optional[FitPoint] = None
    total = len(ann_masses) * len(scan_values)
    done  = 0

    for i, ann_mass in enumerate(ann_masses):
        scatter_mass = (
            float(ann_mass) if args.scatter_mass_mode == "tied"
            else float(args.scatter_mass)
        )

        for j, scan_value in enumerate(scan_values):
            Lambda = float(args.lambda_fixed if args.scan_y_eff else scan_value)
            y_eff = float(scan_value if args.scan_y_eff else args.y_eff)
            point = fit_one_point(
                ann_mass=float(ann_mass),
                scatter_mass=float(scatter_mass),
                Lambda=Lambda,
                y_eff=y_eff,
                args=args,
                l_grid=l_grid,
                b_grid=b_grid,
                E_bins_GeV=E_bins_GeV,
                phi_data=phi_data,
                phi_err=phi_err,
                mask=mask,
            )

            chi2_grid[i, j]      = point.chi2
            sigmav_grid[i, j]    = point.sigmav
            norm_grid[i, j]      = point.norm
            tau_max_grid[i, j]   = point.tau_max
            scatter_m_grid[i, j] = point.scatter_mass
            if not np.isfinite(point.chi2):
                reason = point.invalid_reason or "unknown_nan"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            if np.isfinite(point.chi2) and (best is None or point.chi2 < best.chi2):
                best = point

            done += 1
            if args.verbose or done == total or done % max(1, total // 20) == 0:
                chi2_text = f"{point.chi2:.3g}" if np.isfinite(point.chi2) else "nan"
                sigmav_text = f"{point.sigmav:.2e}" if np.isfinite(point.sigmav) else "nan"
                if not np.isfinite(point.chi2) and point.invalid_reason:
                    if np.isfinite(point.tau_max):
                        chi2_text = f"{point.invalid_reason} (tau_max={point.tau_max:.2e})"
                    else:
                        chi2_text = point.invalid_reason
                print(
                    f"  {done}/{total}: m_ann={ann_mass:.0f} GeV, "
                    f"{scan_axis}={scan_value:.2e}"
                    f"{' GeV' if scan_axis in ('Lambda', 'y_eff') else ''}, "
                    f"chi2={chi2_text}, "
                    f"<sigma v>={sigmav_text} cm^3/s"
                )

    if best is None:
        if reason_counts:
            print("\nNo finite fit points. Invalid-point reasons:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {reason}: {count}")
        raise RuntimeError(
            "No finite fit points found. "
            "Check PPPC table path, mass range, tau mask, or halo profile."
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    prefix  = args.output_prefix
    outdir  = (
        Path(args.output_dir)
        if args.output_dir
        else _HERE / "results" / "totani_dm_scattering" / prefix
    )
    outdir.mkdir(parents=True, exist_ok=True)

    npz_path = outdir / "fit_grid.npz"
    np.savez_compressed(
        npz_path,
        # Grids
        ann_masses_GeV        = ann_masses.astype(np.float32),
        lambdas_GeV           = lambdas.astype(np.float32),
        y_eff_values_GeV      = y_effs.astype(np.float32),
        scan_axis             = np.array(scan_axis),
        scan_values           = scan_values.astype(np.float32),
        scatter_mass_GeV      = scatter_m_grid.astype(np.float32),
        chi2                  = chi2_grid.astype(np.float32),
        sigmav_cm3_s          = sigmav_grid.astype(np.float32),
        norm                  = norm_grid.astype(np.float32),
        tau_max               = tau_max_grid.astype(np.float32),
        # Best-fit point
        best_ann_mass_GeV     = np.float32(best.ann_mass),
        best_scatter_mass_GeV = np.float32(best.scatter_mass),
        best_lambda_GeV       = np.float32(best.Lambda),
        best_y_eff_GeV        = np.float32(best.y_eff),
        best_chi2             = np.float32(best.chi2),
        best_sigmav_cm3_s     = np.float32(best.sigmav),
        best_norm             = np.float32(best.norm),
        best_tau_max          = np.float32(best.tau_max),
        best_model            = best.model.astype(np.float32),
        best_source           = best.source.astype(np.float32),
        best_tau              = best.tau.astype(np.float32),
        # Data used
        E_bins_GeV            = E_bins_GeV.astype(np.float32),
        phi_data              = phi_data.astype(np.float32),
        phi_err               = phi_err.astype(np.float32),
        phi_p16               = halo.phi_p16.astype(np.float32),
        phi_p84               = halo.phi_p84.astype(np.float32),
        # Metadata
        channel               = np.array(args.ann_channel),
        operator              = np.array(args.operator),
        dm_type               = np.array(args.dm_type),
        halo_profile          = np.array(args.halo_profile),
        nfw_label             = np.array(halo.nfw_label),
        scatter_mass_mode     = np.array(args.scatter_mass_mode),
        err_mode              = np.array(args.err_mode),
        n_bins_fitted         = np.int32(int(mask.sum())),
    )
    print(f"\nSaved grid: {npz_path}")

    # Spectrum plot using the live MCMC data
    plot_path = outdir / "best_fit_spectrum.pdf"
    make_best_fit_plot(best, halo, args, plot_path)

    # Text summary
    ndof = int(mask.sum()) - 2 - (1 if (args.scan_lambda or args.scan_y_eff) else 0)
    best_coupling_line = (
        f"best y_eff [GeV]              : {best.y_eff:.6g}\n"
        if args.operator == "higgs_portal"
        else f"best Lambda [GeV]             : {best.Lambda:.6g}\n"
    )
    summary = (
        "Totani-style annihilation + scattering fit\n"
        "==========================================\n"
        f"data source     : MCMC posteriors, profile={args.halo_profile}\n"
        f"NFW label       : {halo.nfw_label}\n"
        f"channel         : {args.ann_channel}\n"
        f"operator        : {args.operator} ({args.dm_type})\n"
        f"err_mode        : {args.err_mode}\n"
        f"scatter mass    : {args.scatter_mass_mode}\n"
        f"bins in chi2    : {int(mask.sum())} / {nE}\n"
        f"\nbest annihilation mass [GeV]  : {best.ann_mass:.6g}\n"
        f"best scatter mass [GeV]       : {best.scatter_mass:.6g}\n"
        f"{best_coupling_line}"
        f"best <sigma v> [cm^3 s^-1]   : {best.sigmav:.6e}\n"
        f"best chi2                     : {best.chi2:.6g}\n"
        f"approx ndof                   : {ndof}\n"
        f"best tau_max                  : {best.tau_max:.6e}\n"
        f"\nnpz  : {npz_path}\n"
        f"plot : {plot_path}\n"
    )
    summary_path = outdir / "summary.txt"
    summary_path.write_text(summary)
    print(summary)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
