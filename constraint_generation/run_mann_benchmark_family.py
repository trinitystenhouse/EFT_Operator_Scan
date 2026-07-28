#!/usr/bin/env python3
"""
Drive `make_data_driven_scattering_limits.run_single` for a family of annihilator
masses, channels, and the three Blois-paper-summary operators (dipole_magnetic
Dirac, anapole Majorana = axial charge radius, scalar Rayleigh).

Each (operator, channel, m_ann) triple becomes its own 32x32 (m_chi, Lambda)
scan with --source pppc --halo-profile rho2 and writes a unique NPZ keyed by:

    mcmc_rho2_pppc_{channel}_mann{m_ann:g}_{model_kind}_{dm_type}_{operator}{_majorana?}_90cl.npz

So nothing collides with the existing source=halo NPZs. Default behaviour is
idempotent: a run is skipped if its raw_attenuation NPZ already exists. Pass
--force to overwrite, --dry-run to list what would happen, --first-only to run
just the first job (use to time one scan before committing to the full grid).

Expected runtime per scan at the production defaults (--nm 32 --nl 32
--n-theta 160) is ~30-60 min on one core, so the full 3 x 2 x 4 = 24 grid is a
multi-hour background job. Launch with `nohup ... &` and tail the log.

The reshaping NPZ for every job is expected to come back as
"[WARN] No spectral_reshaping 90% CL contour found." -- this is a real result
(reshaping is degenerate with the data at ROI 60 deg), not a pipeline failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_WORKSPACE = _ROOT.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_WORKSPACE))

# Importing the existing pipeline so we reuse run_single() verbatim instead of
# shelling out 24 subprocesses.
from constraint_generation import make_data_driven_scattering_limits as pipeline  # noqa: E402


# ----------------------------------------------------------------------------
# Benchmark grid
# ----------------------------------------------------------------------------

OPERATOR_SPECS = [
    # Blois Fig. 2 paper-summary trio.
    {"dm_type": "fermionic", "operator": "dipole_magnetic", "majorana": False,
     "tag": "fermionic_dipole_magnetic"},
    {"dm_type": "fermionic", "operator": "anapole",         "majorana": True,
     "tag": "fermionic_anapole_majorana"},
    {"dm_type": "scalar",    "operator": "scalar_rayleigh", "majorana": False,
     "tag": "scalar_scalar_rayleigh"},
]

CHANNELS = ["WW", "bb"]

M_ANN_GEV = [100.0, 500.0, 700.0, 1000.0]


# ----------------------------------------------------------------------------
# Per-scan argument builder
# ----------------------------------------------------------------------------

def build_args(
    *,
    operator_spec: dict,
    channel: str,
    m_ann: float,
    halo_profile: str,
    nm: int,
    nl: int,
    n_theta: int,
    mchi_min: float,
    mchi_max: float,
    lambda_min: float,
    lambda_max: float,
    delta_chi2: float,
    eft_kinematic_factor: float,
    roi_half_angle: float,
    max_tau_single_scatter: float,
    err_mode: str,
    no_plot: bool,
    quiet: bool,
) -> Namespace:
    """Construct a Namespace that mirrors `make_data_driven_scattering_limits.parse_args`."""
    return Namespace(
        # Halo / data
        halo_profile=halo_profile,
        nfw_label=None,
        counts_path=None,
        err_mode=err_mode,
        include_nonpositive_bins=False,

        # Source
        source="pppc",
        channel=channel,
        ann_mass=float(m_ann),
        pppc_gamma_table=None,

        # Operator
        dm_type=operator_spec["dm_type"],
        operator=operator_spec["operator"],
        majorana=operator_spec["majorana"],
        c_s=None,
        c_p=None,
        c_phi=None,

        # Scan grid
        mchi_min=float(mchi_min),
        mchi_max=float(mchi_max),
        nm=int(nm),
        lambda_min=float(lambda_min),
        lambda_max=float(lambda_max),
        nl=int(nl),

        # Physics knobs
        n_theta=int(n_theta),
        no_roi_weight=False,
        roi_half_angle=float(roi_half_angle),
        max_tau_single_scatter=float(max_tau_single_scatter),
        require_lambda_gt_mdm=False,
        delta_chi2=float(delta_chi2),
        eft_kinematic_factor=float(eft_kinematic_factor),

        # Misc
        run_all=False,
        no_plot=bool(no_plot),
        quiet=bool(quiet),
        style=None,
    )


def expected_npz_path(args: Namespace, model_kind: str = "raw_attenuation") -> Path:
    """Mirror the `save_boundary` filename schema."""
    suffix = "_majorana" if args.majorana else ""
    source_tag = f"pppc_{args.channel}_mann{args.ann_mass:g}"
    return pipeline.OUTDIR / (
        f"mcmc_{args.halo_profile}_{source_tag}_{model_kind}_"
        f"{args.dm_type}_{args.operator}{suffix}_90cl.npz"
    )


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def enumerate_jobs():
    """Iterate the cartesian product of OPERATOR_SPECS x CHANNELS x M_ANN_GEV."""
    idx = 0
    for op_spec in OPERATOR_SPECS:
        for channel in CHANNELS:
            for m_ann in M_ANN_GEV:
                idx += 1
                yield idx, op_spec, channel, m_ann


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Grid resolution: production defaults match the existing saved NPZs.
    p.add_argument("--nm", type=int, default=32, help="mchi grid resolution.")
    p.add_argument("--nl", type=int, default=32, help="Lambda grid resolution.")
    p.add_argument("--n-theta", type=int, default=160,
                   help="Angular integration resolution for the in-scatter kernel.")
    p.add_argument("--mchi-min", type=float, default=1e-6)
    p.add_argument("--mchi-max", type=float, default=1e8)
    p.add_argument("--lambda-min", type=float, default=1e-3)
    p.add_argument("--lambda-max", type=float, default=1e7)
    p.add_argument("--delta-chi2", type=float, default=4.61,
                   help="90%% CL threshold (4.61 = 2-dof; 2.71 = 1-dof).")
    p.add_argument("--eft-kinematic-factor", type=float, default=1.0)
    p.add_argument("--roi-half-angle", type=float, default=60.0)
    p.add_argument("--max-tau-single-scatter", type=float, default=0.3)
    p.add_argument("--err-mode", default="sym", choices=["sym", "lo", "hi", "max"])
    p.add_argument("--halo-profile", default="rho2",
                   help="Halo emissivity profile; default matches the Blois figure.")

    # Driver controls
    p.add_argument("--dry-run", action="store_true",
                   help="List the 24 jobs and what NPZ each would write, then exit.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if the raw_attenuation NPZ already exists.")
    p.add_argument("--first-only", action="store_true",
                   help="Run only the first job (timing benchmark).")
    p.add_argument("--start-from", type=int, default=1,
                   help="Skip jobs 1..(start-from - 1). 1-indexed.")
    p.add_argument("--no-plot", action="store_true", default=True,
                   help="Skip the per-scan diagnostic plot (default).")
    p.add_argument("--quiet", action="store_true", default=True,
                   help="Suppress per-cell progress prints (default).")
    p.add_argument("--verbose", action="store_true",
                   help="Re-enable per-scan plotting + per-cell progress prints.")
    args = p.parse_args(argv)

    if args.verbose:
        args.no_plot = False
        args.quiet = False

    jobs = list(enumerate_jobs())
    total = len(jobs)
    print(f"Benchmark family: {len(OPERATOR_SPECS)} operators x {len(CHANNELS)} channels "
          f"x {len(M_ANN_GEV)} m_ann = {total} jobs")
    print(f"Grid per job: {args.nm} x {args.nl} (m_chi x Lambda), n_theta={args.n_theta}")
    print(f"Output dir: {pipeline.OUTDIR}")
    print()

    if args.dry_run:
        for idx, op_spec, channel, m_ann in jobs:
            scan_args = build_args(
                operator_spec=op_spec, channel=channel, m_ann=m_ann,
                halo_profile=args.halo_profile, nm=args.nm, nl=args.nl,
                n_theta=args.n_theta,
                mchi_min=args.mchi_min, mchi_max=args.mchi_max,
                lambda_min=args.lambda_min, lambda_max=args.lambda_max,
                delta_chi2=args.delta_chi2,
                eft_kinematic_factor=args.eft_kinematic_factor,
                roi_half_angle=args.roi_half_angle,
                max_tau_single_scatter=args.max_tau_single_scatter,
                err_mode=args.err_mode,
                no_plot=args.no_plot, quiet=args.quiet,
            )
            out = expected_npz_path(scan_args, "raw_attenuation")
            exists = "EXISTS" if out.exists() else "WILL WRITE"
            print(f"  [{idx:02d}/{total}] {op_spec['tag']:32s} {channel:>4s}  m_ann={m_ann:>6.0f} GeV"
                  f"  -> {exists}  {out.name}")
        return 0

    if args.first_only:
        jobs = jobs[:1]

    overall_t0 = time.time()
    ran = 0
    skipped = 0
    for idx, op_spec, channel, m_ann in jobs:
        if idx < args.start_from:
            continue
        scan_args = build_args(
            operator_spec=op_spec, channel=channel, m_ann=m_ann,
            halo_profile=args.halo_profile, nm=args.nm, nl=args.nl,
            n_theta=args.n_theta,
            mchi_min=args.mchi_min, mchi_max=args.mchi_max,
            lambda_min=args.lambda_min, lambda_max=args.lambda_max,
            delta_chi2=args.delta_chi2,
            eft_kinematic_factor=args.eft_kinematic_factor,
            roi_half_angle=args.roi_half_angle,
            max_tau_single_scatter=args.max_tau_single_scatter,
            err_mode=args.err_mode,
            no_plot=args.no_plot, quiet=args.quiet,
        )

        out = expected_npz_path(scan_args, "raw_attenuation")
        banner = (f"[{idx:02d}/{total}] {op_spec['tag']} channel={channel} "
                  f"m_ann={m_ann:g} GeV")
        if out.exists() and not args.force:
            print(f"SKIP {banner}  ({out.name} exists; pass --force to overwrite)")
            skipped += 1
            continue

        print("=" * 78)
        print(f"RUN  {banner}")
        print("=" * 78)
        t0 = time.time()
        try:
            pipeline.run_single(scan_args)
        except Exception as exc:  # noqa: BLE001 - drive-on-error policy
            print(f"FAIL {banner}: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0
        print(f"DONE {banner}  ({dt/60:.1f} min)\n")
        ran += 1

    total_dt = time.time() - overall_t0
    print(f"\nBenchmark family complete: ran {ran}, skipped {skipped}, "
          f"wall time {total_dt/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
