#!/usr/bin/env python3
"""
Set up a multi-component photon-DM scattering transfer calculation.

This is the bridge beyond the halo-component approximation used by
constraint_generation/make_data_driven_scattering_limits.py. It loads fitted
MCMC component spectra and propagates several photon components through the same
photon-DM scattering kernel, while keeping the component geometry choice
explicit.

The default geometry mode, halo-only, reproduces the conservative statement:
only NFW-like components receive the current NFW-emissivity-weighted tau(E);
all other components are passed through with tau=0. Use all-nfw-weighted only
as a diagnostic placeholder, because foregrounds/isotropic photons should have
their own optical-depth geometry in a final physical sky model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from core.spectral_reshaping import (  # noqa: E402
    PhotonTransferComponent,
    ReshapingConfig,
    compute_tau_spectrum,
    transfer_photon_components,
)
from core.totani_data_loader import _MCMC_DIRS, load_component_spectra  # noqa: E402


def operator_couplings(operator: str, dm_type: str) -> tuple[float, float, float]:
    if dm_type == "scalar" or operator in ("scalar_rayleigh", "rayleigh"):
        return 0.0, 0.0, 1.0
    if operator in ("dipole_magnetic", "charge_radius", "rayleigh_even"):
        return 1.0, 0.0, 1.0
    if operator in ("dipole_electric", "anapole", "rayleigh_odd"):
        return 0.0, 1.0, 1.0
    if operator == "rayleigh_full":
        return 1.0, 1.0, 1.0
    return 1.0, 0.0, 1.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply photon-DM transfer to several Totani MCMC components.",
    )
    p.add_argument("--halo-profile", default="rho2", choices=sorted(_MCMC_DIRS.keys()))
    p.add_argument("--labels", nargs="*", default=None,
                   help="MCMC component labels to include. Default: all labels in the files.")
    p.add_argument("--geometry-mode", choices=["halo-only", "all-nfw-weighted"], default="halo-only",
                   help="How to assign tau(E) before component-specific geometries are implemented.")
    p.add_argument("--m-chi", type=float, default=700.0)
    p.add_argument("--Lambda", type=float, default=1e3)
    p.add_argument("--dm-type", default="fermionic", choices=["fermionic", "scalar"])
    p.add_argument("--operator", default="dipole_magnetic")
    p.add_argument("--majorana", action="store_true")
    p.add_argument("--c-s", type=float, default=None)
    p.add_argument("--c-p", type=float, default=None)
    p.add_argument("--c-phi", type=float, default=None)
    p.add_argument("--n-theta", type=int, default=300)
    p.add_argument("--no-roi-weight", action="store_true")
    p.add_argument("--roi-half-angle", type=float, default=60.0)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cs0, cp0, cphi0 = operator_couplings(args.operator, args.dm_type)
    c_s = cs0 if args.c_s is None else float(args.c_s)
    c_p = cp0 if args.c_p is None else float(args.c_p)
    c_phi = cphi0 if args.c_phi is None else float(args.c_phi)

    spectra = load_component_spectra(_MCMC_DIRS[args.halo_profile], labels=args.labels)
    labels = spectra.labels
    if not labels:
        raise SystemExit("No component labels selected.")

    phi_total_input = np.zeros_like(spectra.E_bins_GeV, dtype=float)
    for label in labels:
        phi_total_input += np.nan_to_num(spectra.phi[label], nan=0.0)

    base_cfg = ReshapingConfig(
        m_chi=args.m_chi,
        Lambda=args.Lambda,
        dm_type=args.dm_type,
        operator=args.operator,
        c_s=c_s,
        c_p=c_p,
        c_phi=c_phi,
        majorana=args.majorana,
        n_theta=args.n_theta,
        apply_roi_weight=not args.no_roi_weight,
        roi_half_angle_deg=args.roi_half_angle,
        E_bins=spectra.E_bins_GeV,
        phi_0=phi_total_input,
        phi_data=phi_total_input,
        phi_err=np.ones_like(phi_total_input),
        fit_normalization=False,
        require_lambda_gt_mdm=False,
    )

    halo_tau = compute_tau_spectrum(base_cfg)
    zero_tau = np.zeros_like(halo_tau)
    components = []
    for label in labels:
        is_halo = "nfw" in label.lower()
        if args.geometry_mode == "halo-only" and not is_halo:
            tau = zero_tau
            note = "tau=0 placeholder; foreground/background geometry not assigned"
        else:
            tau = halo_tau
            note = "NFW-emissivity-weighted tau(E)"
            if not is_halo:
                note += " diagnostic placeholder, not final foreground geometry"

        components.append(
            PhotonTransferComponent(
                name=label,
                phi_0=np.nan_to_num(spectra.phi[label], nan=0.0),
                tau=tau,
                note=note,
            )
        )

    result = transfer_photon_components(base_cfg, components, return_components=True)
    comp_obs = np.asarray([piece["phi_obs"] for piece in result["components"]], dtype=float)
    comp_surv = np.asarray([piece["phi_survival"] for piece in result["components"]], dtype=float)
    comp_in = np.asarray([piece["phi_inscatter"] for piece in result["components"]], dtype=float)
    comp_tau = np.asarray([piece["tau"] for piece in result["components"]], dtype=float)
    comp_notes = np.asarray([piece["note"] for piece in result["components"]])

    out = Path(args.output) if args.output else (
        _HERE / "results" / "multicomponent_transfer"
        / f"{args.halo_profile}_{args.geometry_mode}_{args.operator}_m{args.m_chi:g}_L{args.Lambda:g}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        E_bins_GeV=spectra.E_bins_GeV.astype(np.float32),
        labels=np.asarray(labels),
        notes=comp_notes,
        phi_input_total=phi_total_input.astype(np.float32),
        phi_obs_total=result["phi_obs"].astype(np.float32),
        phi_obs_components=comp_obs.astype(np.float32),
        phi_survival_components=comp_surv.astype(np.float32),
        phi_inscatter_components=comp_in.astype(np.float32),
        tau_components=comp_tau.astype(np.float32),
        K=result["K"].astype(np.float32),
        geometry_mode=np.array(args.geometry_mode),
        halo_profile=np.array(args.halo_profile),
        operator=np.array(args.operator),
        dm_type=np.array(args.dm_type),
        m_chi_GeV=np.float32(args.m_chi),
        Lambda_GeV=np.float32(args.Lambda),
    )

    print("Multi-component transfer setup")
    print(f"  components   : {', '.join(labels)}")
    print(f"  geometry mode: {args.geometry_mode}")
    print(f"  output       : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
