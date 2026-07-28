#!/usr/bin/env python3
"""Run global-profile annihilation + scattering scans over channels/operators.

Example
-------
python Totani_Scattering/run_global_scattering_grid.py \
  --halo-profile global_rho2 \
  --ann-mass-min 1 \
  --ann-mass-max 5000 \
  --n-ann-mass 160 \
  --scatter-mass-min 1e-6 \
  --scatter-mass-max 30 \
  --n-scatter-mass 40 \
  --lambda-min 1e-3 \
  --lambda-max 1e5 \
  --n-lambda 60
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parent

CHANNELS = ["WW", "bb", "tautau"]
OPERATORS = [
    "dipole_magnetic",
    "dipole_electric",
    "charge_radius",
    "anapole",
    "rayleigh_even",
    "rayleigh_odd",
    "rayleigh_full",
    "scalar_rayleigh",
]


def _parse_list(raw: str, allowed: Iterable[str], *, name: str) -> List[str]:
    allowed = list(allowed)
    text = str(raw).strip()
    if text == "all":
        return allowed
    out = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise SystemExit(f"Unknown {name}: {bad}. Allowed: {allowed} or all")
    return out


def _operator_extra_args(operator: str) -> List[str]:
    if operator == "scalar_rayleigh":
        return ["--dm-type", "scalar"]
    return []


def _run(cmd: List[str], *, dry_run: bool) -> int:
    print("\n" + "=" * 96)
    print(" ".join(cmd))
    print("=" * 96)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(REPO_DIR), check=False).returncode


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--halo-profile", default="global_rho2")
    p.add_argument("--channels", default="all", help="all or comma list: WW,bb,tautau")
    p.add_argument("--operators", default="all", help="all or comma-separated operator list")
    p.add_argument("--dark-sector-model", default="different", choices=["same", "different"])
    p.add_argument("--best-fit-type", default="spectral", choices=["spectral", "dsph_cl", "min_tension_cl"])

    p.add_argument("--ann-mass-min", type=float, default=1.0)
    p.add_argument("--ann-mass-max", type=float, default=5000.0)
    p.add_argument("--n-ann-mass", type=int, default=160)

    p.add_argument("--scatter-mass-min", type=float, default=1e-6)
    p.add_argument("--scatter-mass-max", type=float, default=30.0)
    p.add_argument("--n-scatter-mass", type=int, default=80)
    p.add_argument("--scatter-mass-low-max", type=float, default=100.0)
    p.add_argument("--n-scatter-mass-low", type=int, default=0)

    p.add_argument("--lambda-min", type=float, default=1e-3)
    p.add_argument("--lambda-max", type=float, default=1e5)
    p.add_argument("--n-lambda", type=int, default=60)
    p.add_argument("--allow-eft-invalid", action="store_true")

    p.add_argument("--n-ext-bins", type=int, default=30)
    p.add_argument("--ext-energy-max", type=float, default=5000.0)
    p.add_argument("--min-tau", type=float, default=0.0)
    p.add_argument("--max-tau", type=float, default=0.3)
    p.add_argument("--cl", type=float, default=0.95)
    p.add_argument("--err-mode", default="sym", choices=["sym", "lo", "hi", "max"])
    p.add_argument("--include-nonpositive-bins", action="store_true")

    p.add_argument("--downscatter-mode", default="off", choices=["off", "penalty", "hard"])
    p.add_argument("--downscatter-source-energy", type=float, default=20.0)
    p.add_argument("--downscatter-target-energy", type=float, default=3.0)
    p.add_argument("--downscatter-source-window-dex", type=float, default=0.20)
    p.add_argument("--downscatter-target-window-dex", type=float, default=0.20)
    p.add_argument("--downscatter-peak-weight", type=float, default=10.0)
    p.add_argument("--downscatter-frac-weight", type=float, default=10.0)
    p.add_argument("--min-source-to-target-frac", type=float, default=0.0)
    p.add_argument("--min-target-in-from-source-frac", type=float, default=0.0)
    p.add_argument("--min-target-model-from-source-frac", type=float, default=0.0)

    p.add_argument(
        "--output-root",
        default=str(HERE / "results" / "deconvolved_scattering_fit_global_mmin1"),
        help="Each scan writes to output-root/{halo}_{channel}_{operator}.",
    )
    p.add_argument(
        "--compile-output-dir",
        default=str(HERE / "results" / "compiled_best_fits_global_mmin1"),
    )
    p.add_argument(
        "--annihilation-dir",
        default=str(HERE / "results" / "tension_scan_global_mmin1"),
        help="Used only by compile_best_fits.py after the scattering scans.",
    )
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Rerun scans even if summary.txt already exists.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--style", default=None)
    args = p.parse_args()

    channels = _parse_list(args.channels, CHANNELS, name="channel")
    operators = _parse_list(args.operators, OPERATORS, name="operator")
    output_root = Path(args.output_root)

    failures: list[tuple[str, str, int]] = []
    for channel in channels:
        for operator in operators:
            outdir = output_root / f"{args.halo_profile}_{channel}_{operator}"
            summary = outdir / "summary.txt"
            if summary.exists() and not args.overwrite:
                print(f"\n[skip] {summary} exists. Pass --overwrite to rerun.")
                continue

            cmd = [
                args.python,
                str(HERE / "scan_deconvolved_scattering_fit.py"),
                "--halo-profile", args.halo_profile,
                "--ann-channel", channel,
                "--dark-sector-model", args.dark_sector_model,
                "--operator", operator,
                "--best-fit-type", args.best_fit_type,
                "--ann-mass-min", str(args.ann_mass_min),
                "--ann-mass-max", str(args.ann_mass_max),
                "--n-ann-mass", str(args.n_ann_mass),
                "--scatter-mass-min", str(args.scatter_mass_min),
                "--scatter-mass-max", str(args.scatter_mass_max),
                "--n-scatter-mass", str(args.n_scatter_mass),
                "--scatter-mass-low-max", str(args.scatter_mass_low_max),
                "--n-scatter-mass-low", str(args.n_scatter_mass_low),
                "--lambda-min", str(args.lambda_min),
                "--lambda-max", str(args.lambda_max),
                "--n-lambda", str(args.n_lambda),
                "--n-ext-bins", str(args.n_ext_bins),
                "--ext-energy-max", str(args.ext_energy_max),
                "--min-tau", str(args.min_tau),
                "--max-tau", str(args.max_tau),
                "--cl", str(args.cl),
                "--err-mode", args.err_mode,
                "--output-dir", str(outdir),
            ]
            cmd.extend(_operator_extra_args(operator))
            if args.allow_eft_invalid:
                cmd.append("--allow-eft-invalid")
            if args.include_nonpositive_bins:
                cmd.append("--include-nonpositive-bins")
            if args.style:
                cmd.extend(["--style", args.style])
            if args.downscatter_mode != "off":
                cmd.extend(
                    [
                        "--downscatter-mode", args.downscatter_mode,
                        "--downscatter-source-energy", str(args.downscatter_source_energy),
                        "--downscatter-target-energy", str(args.downscatter_target_energy),
                        "--downscatter-source-window-dex", str(args.downscatter_source_window_dex),
                        "--downscatter-target-window-dex", str(args.downscatter_target_window_dex),
                        "--downscatter-peak-weight", str(args.downscatter_peak_weight),
                        "--downscatter-frac-weight", str(args.downscatter_frac_weight),
                        "--min-source-to-target-frac", str(args.min_source_to_target_frac),
                        "--min-target-in-from-source-frac", str(args.min_target_in_from_source_frac),
                        "--min-target-model-from-source-frac", str(args.min_target_model_from_source_frac),
                    ]
                )

            code = _run(cmd, dry_run=args.dry_run)
            if code != 0:
                failures.append((channel, operator, code))
                if args.stop_on_failure:
                    break
        if failures and args.stop_on_failure:
            break

    if not args.no_compile:
        compile_cmd = [
            args.python,
            str(HERE / "compile_best_fits.py"),
            "--annihilation-dir", str(args.annihilation_dir),
            "--scattering-dir", str(output_root),
            "--scattering-prefix", f"{args.halo_profile}_",
            "--output-dir", str(args.compile_output_dir),
        ]
        code = _run(compile_cmd, dry_run=args.dry_run)
        if code != 0:
            failures.append(("compile", "compile_best_fits", code))

    if failures:
        print("\nFailures:")
        for channel, operator, code in failures:
            print(f"  {channel} {operator}: exit {code}")
        return 1

    print("\nAll requested scans completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
