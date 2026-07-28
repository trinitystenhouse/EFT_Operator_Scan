#!/usr/bin/env python3
"""
run_reshaping.py
================
Driver script for the spectral-reshaping analysis of the Totani halo excess.

Runs the full pipeline for a specified operator and parameter grid, producing:
  - Diagnostic kinematics plot
  - Single benchmark: spectrum comparison (reshaping vs attenuation vs data)
  - Single benchmark: redistribution kernel heatmap
  - Single benchmark: in-scatter decomposition
  - 2D scan: delta_chi2(m_chi, Lambda) map
  - Saved scan .npz for downstream analysis

Usage
-----
    python run_reshaping.py                      # defaults: dipole_magnetic, Dirac
    python run_reshaping.py --operator anapole --majorana
    python run_reshaping.py --operator rayleigh_even --dm-type fermionic
    python run_reshaping.py --operator dipole_magnetic --quick   # coarse grid

Options are defined via argparse below. All outputs land in plots/ and
constraint_boundaries/ inside Totani_Scattering/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure the Totani_Scattering directory is on sys.path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.attenuation_eft import E_BINS_GEV, PHI_TOTANI, SIGMA_TOTANI, eft_validity_lambda_curve
from core.spectral_reshaping import (
    ReshapingConfig,
    pppc_energy_flux_template,
    smooth_nfw_sigma_v_from_norm,
    build_dsigma_grid,
    build_kernel,
    compute_tau_spectrum,
    reshaped_halo_spectrum,
    chi2_reshaping,
    chi2_attenuation_only,
    fitted_norm_reshaping,
    fitted_norm_attenuation_only,
    energy_flux_transfer_matrix,
    scan_reshaping_chi2,
    save_reshaping_scan,
    print_kinematics_summary,
)
from core.plot_reshaping import (
    plot_spectrum_comparison,
    plot_kernel_heatmap,
    plot_inscatter_decomposition,
    plot_reshaping_vs_attenuation_chi2,
    plot_energy_loss_summary,
)


def _sci_label(value: float) -> str:
    """Compact scientific-notation label for plot titles."""
    return f"{value:.0e}"


def _benchmark_validity_label(cfg: ReshapingConfig, tau: np.ndarray) -> str:
    """Human-readable validity label for benchmark diagnostic plots."""
    failures: list[str] = []

    if cfg.operator != "higgs_portal" and cfg.require_lambda_gt_mdm and cfg.Lambda <= cfg.m_chi:
        failures.append(r"$\Lambda \le m_\chi$")

    if cfg.operator != "higgs_portal":
        lam_kin = float(
            eft_validity_lambda_curve(
                np.asarray([cfg.m_chi], dtype=float),
                omega_max=float(np.max(cfg.E_bins)),
            )[0]
        )
        if np.isfinite(lam_kin) and cfg.Lambda < lam_kin:
            failures.append(rf"$\Lambda < \sqrt{{q^2_{{\max}}}} \simeq {_sci_label(lam_kin)}$ GeV")

    if cfg.max_tau_single_scatter is not None and cfg.max_tau_single_scatter >= 0.0:
        tau_max = float(np.nanmax(np.asarray(tau, dtype=float)))
        if (not np.isfinite(tau_max)) or tau_max > float(cfg.max_tau_single_scatter):
            failures.append(rf"$\tau_{{\max}} > {cfg.max_tau_single_scatter:g}$")

    if not failures:
        return "validity: EFT/one-scatter checks passed"
    return "validity: " + "; ".join(failures)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run spectral reshaping analysis for photon-DM scattering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--operator", default="dipole_magnetic",
        choices=[
            "dipole_magnetic", "dipole_electric",
            "charge_radius", "anapole",
            "rayleigh_even", "rayleigh_odd", "rayleigh_full",
            "higgs_portal",
        ],
        help="EFT operator (or 'higgs_portal' for UV-complete Higgs-mediated).",
    )
    p.add_argument("--dm-type", default="fermionic",
                   choices=["fermionic", "scalar"],
                   help="DM spin structure.")
    p.add_argument("--majorana", action="store_true", default=False,
                   help="Treat fermion as Majorana (enforces vanishing dipoles).")
    p.add_argument("--c-s", type=float, default=1.0, help="Wilson coefficient c_s.")
    p.add_argument("--c-p", type=float, default=0.0, help="Wilson coefficient c_p.")
    p.add_argument("--c-phi", type=float, default=1.0, help="Wilson coeff for scalar DM.")
    p.add_argument("--y-eff", type=float, default=1.0,
                   help="Effective Higgs-portal y_eff [GeV], matching the York/fig6 convention.")
    p.add_argument("--source-template", default="pppc",
                   choices=["pppc", "totani"],
                   help="Intrinsic source spectrum to transfer before fitting.")
    p.add_argument("--ann-channel", default="WW",
                   help="PPPC annihilation channel, e.g. WW, bb, tautau.")
    p.add_argument("--ann-primary", default="gamma",
                   help="PPPC primary species.")
    p.add_argument("--ann-mass", type=float, default=None,
                   help="Annihilating DM mass [GeV] for the PPPC source template. Defaults to --m-chi-bench.")
    p.add_argument("--pppc-gamma-table", default=None,
                   help="Path to PPPC4DMID AtProduction_gammas.dat. Defaults to $PPPC4DMID_GAMMAS or Totani_Scattering/data/.")

    # Benchmark point
    p.add_argument("--m-chi-bench", type=float, default=600.0,
                   help="Benchmark DM mass [GeV] for single-point diagnostics.")
    p.add_argument("--lambda-bench", type=float, default=1e3,
                   help="Benchmark Lambda [GeV] for single-point diagnostics.")

    # Scan grid
    p.add_argument("--m-chi-min", type=float, default=1e2,
                   help="Minimum m_chi for 2D scan [GeV].")
    p.add_argument("--m-chi-max", type=float, default=1e4,
                   help="Maximum m_chi for 2D scan [GeV].")
    p.add_argument("--lambda-min", type=float, default=1e1,
                   help="Minimum Lambda for 2D scan [GeV].")
    p.add_argument("--lambda-max", type=float, default=1e5,
                   help="Maximum Lambda for 2D scan [GeV].")
    p.add_argument("--n-m-chi", type=int, default=20,
                   help="Number of m_chi points in scan.")
    p.add_argument("--n-lambda", type=int, default=20,
                   help="Number of Lambda points in scan.")

    # Resolution
    p.add_argument("--n-theta", type=int, default=1000,
                   help="Angular integration nodes for dσ/dΩ.")
    p.add_argument("--quick", action="store_true", default=False,
                   help="Coarse grid (10x10) for fast testing.")

    # ROI
    p.add_argument("--apply-roi-weight", action="store_true", default=True,
                   help="Apply approximate geometric ROI recovery fraction to the in-scatter kernel.")
    p.add_argument("--no-roi-weight", dest="apply_roi_weight", action="store_false",
                   help="Assume all scattered photons remain in the ROI.")
    p.add_argument("--roi-half-angle", type=float, default=60.0,
                   help="ROI half-angle [deg] for recovery fraction (if --apply-roi-weight).")
    p.add_argument("--max-tau-single-scatter", type=float, default=0.3,
                   help="Mask scan points above this max optical depth; use negative to disable.")
    p.add_argument("--require-lambda-gt-mdm", action="store_true", default=True,
                   help="Require Lambda > scattering DM mass; invalid EFT points are masked.")
    p.add_argument("--allow-lambda-le-mdm", dest="require_lambda_gt_mdm", action="store_false",
                   help="Disable the Lambda > m_DM mask.")
    p.add_argument("--no-fit-normalization", action="store_true", default=False,
                   help="Do not fit the overall intrinsic annihilation-template normalization.")

    # Output
    p.add_argument("--output-dir", default=None,
                   help="Directory for outputs. Defaults to results/reshaping/<suffix>.")
    p.add_argument("--no-scan", action="store_true", default=False,
                   help="Skip the 2D scan (only run benchmark diagnostics).")
    p.add_argument("--show", action="store_true", default=False,
                   help="Call plt.show() after each figure.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.quick:
        args.n_m_chi = 10
        args.n_lambda = 10
        args.n_theta = 300
        print("[quick mode] coarse grid: 10×10, n_theta=300")

    max_tau_single_scatter = (
        None if args.max_tau_single_scatter is not None and args.max_tau_single_scatter < 0.0
        else args.max_tau_single_scatter
    )
    ann_mass = args.m_chi_bench if args.ann_mass is None else args.ann_mass

    # ROI grids (match attenuation_eft.py driver convention)
    l_grid = np.linspace(-60.0, 60.0, 15)
    b_grid = np.concatenate([
        np.linspace(-60.0, -10.0, 8),
        np.linspace(10.0, 60.0, 8),
    ])

    if args.source_template == "pppc":
        phi_source = pppc_energy_flux_template(
            E_BINS_GEV,
            ann_mass,
            channel=args.ann_channel,
            primary=args.ann_primary,
            table_path=args.pppc_gamma_table,
        )
        source_label = f"PPPC {args.ann_channel}, m_ann={ann_mass:g} GeV"
    else:
        phi_source = PHI_TOTANI.copy()
        source_label = "Totani residual source placeholder"

    # -----------------------------------------------------------------------
    # Step 0: Kinematic summary
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 0: Kinematic summary")
    print("="*60)

    cfg_bench = ReshapingConfig(
        m_chi=args.m_chi_bench,
        Lambda=args.lambda_bench,
        dm_type=args.dm_type,
        operator=args.operator,
        c_s=args.c_s,
        c_p=args.c_p,
        c_phi=args.c_phi,
        majorana=args.majorana,
        y_eff=args.y_eff,
        l_grid=l_grid,
        b_grid=b_grid,
        n_theta=args.n_theta,
        apply_roi_weight=args.apply_roi_weight,
        roi_half_angle_deg=args.roi_half_angle if args.apply_roi_weight else None,
        phi_0=phi_source,
        phi_data=PHI_TOTANI,
        phi_err=SIGMA_TOTANI,
        fit_normalization=not args.no_fit_normalization,
        max_tau_single_scatter=max_tau_single_scatter,
        require_lambda_gt_mdm=args.require_lambda_gt_mdm,
    )

    print_kinematics_summary(cfg_bench)

    m_chi_range = np.logspace(
        np.log10(args.m_chi_min), np.log10(args.m_chi_max), 5
    )
    out = plot_energy_loss_summary(
        E_BINS_GEV, list(m_chi_range),
        filename=f"energy_loss_{args.operator}.pdf",
        show=args.show,
    )
    print(f"  Saved: {out}")

    # -----------------------------------------------------------------------
    # Step 1: Benchmark spectrum
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"STEP 1: Benchmark spectrum  (m_chi={args.m_chi_bench:.2e} GeV, "
          f"Lambda={args.lambda_bench:.2e} GeV)")
    print("="*60)

    result = reshaped_halo_spectrum(cfg_bench, return_components=True)

    phi_obs_r = result["phi_obs"]
    tau = result["tau"]
    K = result["K"]
    K_energy = result["K_energy_flux"]
    phi_att = cfg_bench.phi_0 * np.exp(-tau)     # simple attenuation reference

    chi2_r = chi2_reshaping(cfg_bench)
    chi2_a = chi2_attenuation_only(cfg_bench)
    norm_r = fitted_norm_reshaping(cfg_bench)
    norm_a = fitted_norm_attenuation_only(cfg_bench)
    print(f"  chi2 (reshaping)   = {chi2_r:.4f}")
    print(f"  chi2 (attenuation) = {chi2_a:.4f}")
    print(f"  delta_chi2         = {chi2_r - chi2_a:.4f}")
    print(f"  best-fit norm reshaping   = {norm_r:.4f}")
    print(f"  best-fit norm attenuation = {norm_a:.4f}")
    if args.source_template == "pppc":
        sigv_r = smooth_nfw_sigma_v_from_norm(norm_r, ann_mass) if np.isfinite(norm_r) else np.nan
        sigv_a = smooth_nfw_sigma_v_from_norm(norm_a, ann_mass) if np.isfinite(norm_a) else np.nan
        print(f"  <sigma v> reshaping   = {sigv_r:.3e} cm^3 s^-1")
        print(f"  <sigma v> attenuation = {sigv_a:.3e} cm^3 s^-1")
    print(f"  max tau            = {tau.max():.3e}")
    print(f"  max K column sum   = {K.sum(axis=0).max():.3e}")
    print(f"  max energy-flux K column sum = {K_energy.sum(axis=0).max():.3e}")

    suffix = f"{args.operator}_{args.dm_type}"
    if args.majorana:
        suffix += "_majorana"
    if args.source_template == "pppc":
        safe_channel = "".join(ch if ch.isalnum() else "_" for ch in args.ann_channel)
        suffix += f"_pppc_{safe_channel}_mann{ann_mass:g}"
    output_dir = Path(args.output_dir) if args.output_dir else (_HERE / "results" / "reshaping" / suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    out = plot_spectrum_comparison(
        cfg_bench.E_bins, cfg_bench.phi_data, cfg_bench.phi_err,
        norm_r * phi_obs_r if np.isfinite(norm_r) else phi_obs_r,
        norm_a * phi_att if np.isfinite(norm_a) else phi_att,
        tau=tau,
        label_reshaping=f"Reshaping ({args.operator})",
        label_attenuation="Attenuation only",
        title=(
            rf"$m_\chi={_sci_label(args.m_chi_bench)}$ GeV, "
            rf"$\Lambda={_sci_label(args.lambda_bench)}$ GeV, "
            rf"{args.operator}, {source_label}" "\n"
            rf"{_benchmark_validity_label(cfg_bench, tau)}"
        ),
        filename=str(output_dir / "spectrum_comparison.pdf"),
        show=args.show,
    )
    print(f"  Saved: {out}")

    # -----------------------------------------------------------------------
    # Step 2: Redistribution kernel
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Redistribution kernel K[i,j]")
    print("="*60)

    out = plot_kernel_heatmap(
        K, cfg_bench.E_bins,
        title=(
            rf"Redistribution kernel: {args.operator}, "
            rf"$m_\chi={_sci_label(args.m_chi_bench)}$ GeV, "
            rf"$\Lambda={_sci_label(args.lambda_bench)}$ GeV" "\n"
            rf"{_benchmark_validity_label(cfg_bench, tau)}"
        ),
        filename=str(output_dir / "kernel.pdf"),
        show=args.show,
    )
    print(f"  Saved: {out}")

    # -----------------------------------------------------------------------
    # Step 3: In-scatter decomposition
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: In-scatter decomposition")
    print("="*60)

    out = plot_inscatter_decomposition(
        cfg_bench.E_bins, cfg_bench.phi_0, K_energy, tau,
        title=(
            rf"In-scatter by source bin: {args.operator}, "
            rf"$m_\chi={_sci_label(args.m_chi_bench)}$ GeV, "
            rf"$\Lambda={_sci_label(args.lambda_bench)}$ GeV, {source_label}" "\n"
            rf"{_benchmark_validity_label(cfg_bench, tau)}"
        ),
        filename=str(output_dir / "inscatter.pdf"),
        show=args.show,
    )
    print(f"  Saved: {out}")

    # -----------------------------------------------------------------------
    # Step 4: 2D scan
    # -----------------------------------------------------------------------
    if args.no_scan:
        print("\n[--no-scan specified: skipping 2D grid scan]")
        return

    print("\n" + "="*60)
    print(f"STEP 4: 2D scan  ({args.n_m_chi} × {args.n_lambda} grid)")
    print("="*60)

    m_chi_arr = np.logspace(
        np.log10(args.m_chi_min), np.log10(args.m_chi_max), args.n_m_chi
    )
    Lambda_arr = np.logspace(
        np.log10(args.lambda_min), np.log10(args.lambda_max), args.n_lambda
    )

    scan_result = scan_reshaping_chi2(
        m_chi_arr, Lambda_arr,
        dm_type=args.dm_type,
        operator=args.operator,
        c_s=args.c_s,
        c_p=args.c_p,
        c_phi=args.c_phi,
        majorana=args.majorana,
        E_bins=E_BINS_GEV,
        phi_0=phi_source,
        phi_data=PHI_TOTANI,
        phi_err=SIGMA_TOTANI,
        l_grid=l_grid,
        b_grid=b_grid,
        n_theta=args.n_theta,
        apply_roi_weight=args.apply_roi_weight,
        roi_half_angle_deg=args.roi_half_angle if args.apply_roi_weight else None,
        fit_normalization=not args.no_fit_normalization,
        max_tau_single_scatter=max_tau_single_scatter,
        require_lambda_gt_mdm=args.require_lambda_gt_mdm,
        also_compute_attenuation=True,
        verbose=True,
    )

    scan_result["source_template"] = args.source_template
    scan_result["ann_channel"] = args.ann_channel
    scan_result["ann_primary"] = args.ann_primary
    scan_result["ann_mass_GeV"] = ann_mass

    # Save
    npz_path = save_reshaping_scan(
        scan_result, output_dir / "reshaping_scan.npz"
    )
    print(f"  Saved: {npz_path}")

    # Plot delta_chi2 map
    out = plot_reshaping_vs_attenuation_chi2(
        scan_result,
        filename=str(output_dir / "delta_chi2.pdf"),
        show=args.show,
    )
    print(f"  Saved: {out}")

    # Summary statistics
    delta = scan_result["delta_chi2"]
    print(f"\n  delta_chi2 range: [{np.nanmin(delta):.10f}, {np.nanmax(delta):.10f}]")
    print(f"  Points where reshaping improves fit (delta < 0): "
          f"{np.sum(np.isfinite(delta) & (delta < 0))}/{delta.size}")
    print(f"  Points where reshaping worsens fit  (delta > 0): "
          f"{np.sum(np.isfinite(delta) & (delta > 0))}/{delta.size}")
    print(f"  Points masked by tau/invalid fits: {np.sum(~np.isfinite(delta))}/{delta.size}")

    print("\n" + "="*60)
    print("ALL STEPS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
