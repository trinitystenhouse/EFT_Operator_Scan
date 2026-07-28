#!/usr/bin/env python3
"""Generate toy UV-complete constraint contours for plotting tests.

The real paper plots should use digitised experimental contours.  These files
are only shaped examples for checking that
``Totani_Scattering/make_uv_complete_theory_limits.py`` can read and overlay
CSV limits.

Output columns are intentionally named ``m_med_GeV`` and ``m_chi_GeV`` because
the UV plotting script recognises those directly.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


OUTDIR = Path(__file__).resolve().parent


def logspace(start_exp: float, stop_exp: float, n: int) -> list[float]:
    if n < 2:
        raise ValueError("logspace needs n >= 2")
    step = (float(stop_exp) - float(start_exp)) / float(n - 1)
    return [10.0 ** (float(start_exp) + i * step) for i in range(int(n))]


def write_contour(path: Path, m_med, m_chi) -> None:
    m_med = [float(x) for x in m_med]
    m_chi = [float(y) for y in m_chi]
    if len(m_med) != len(m_chi):
        raise ValueError(f"{path.name}: m_med and m_chi must have the same shape")
    if len(m_med) < 2:
        raise ValueError(f"{path.name}: contour must be a 1D array with at least two points")

    rows = [
        (x, y)
        for x, y in zip(m_med, m_chi)
        if math.isfinite(x) and math.isfinite(y) and x > 0.0 and y > 0.0
    ]
    if len(rows) < 2:
        raise ValueError(f"{path.name}: contour has fewer than two finite positive points")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m_med_GeV", "m_chi_GeV"])
        for x, y in rows:
            writer.writerow([f"{x:.8g}", f"{y:.8g}"])
    print(f"Wrote {path}")


def make_dijet_contour():
    # A vertical-ish resonance bound: excludes mediator masses below/around the
    # contour over a broad m_chi range.  Use side "left" or "above" depending on
    # what visual convention you want to test.
    m_chi = logspace(0, 3.2, 80)
    m_med = [2400.0 / (1.0 + (m / 850.0) ** 1.4) + 240.0 for m in m_chi]
    return m_med, m_chi


def make_met_plus_x_contour():
    # A missing-energy contour that roughly follows the on-shell region and
    # weakens for heavier DM.
    m_chi = logspace(0, 3.1, 90)
    m_med = [
        2.0 * m + 1350.0 / (1.0 + (m / 260.0) ** 1.15) + 180.0
        for m in m_chi
    ]
    return m_med, m_chi


def make_dilepton_contour():
    m_chi = logspace(0, 3.0, 70)
    m_med = [3200.0 / (1.0 + (m / 1100.0) ** 1.8) + 350.0 for m in m_chi]
    return m_med, m_chi


def make_thermal_relic_contour():
    m_chi = logspace(0, 3.4, 120)
    m_med = [
        2.15 * m * (1.0 + 0.22 * math.exp(-((math.log10(m) - 2.0) ** 2) / 0.55))
        for m in m_chi
    ]
    return m_med, m_chi


def main() -> int:
    write_contour(OUTDIR / "dijet_limits.csv", *make_dijet_contour())
    write_contour(OUTDIR / "MET_plus_X_limits.csv", *make_met_plus_x_contour())
    write_contour(OUTDIR / "dilepton_limits.csv", *make_dilepton_contour())
    write_contour(OUTDIR / "thermal_relic_contour.csv", *make_thermal_relic_contour())

    print("\nExample:")
    print(
        "  python Totani_Scattering/make_uv_complete_theory_limits.py "
        "--collider Totani_Scattering/constraint_generation/dijet_limits.csv:Dijet:left "
        "--collider 'Totani_Scattering/constraint_generation/MET_plus_X_limits.csv:$E_T^{miss}+X:above' "
        "--thermal Totani_Scattering/constraint_generation/thermal_relic_contour.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
