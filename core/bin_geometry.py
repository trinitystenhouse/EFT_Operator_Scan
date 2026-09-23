"""Bin geometry of a measured spectrum: the energy-migration fraction f_bin.

WHY THIS MODULE EXISTS
----------------------
``f_bin`` is the fractional energy loss that moves a photon out of the energy
bin it was measured in. It sets the removal threshold used by the attenuation
arm,

    u_E = f / (1 - f),        u = omega/omega' - 1,

so it is a property of the DATASET'S BINNING, not of the operator or of the
scatterer. Until 2026-09-18 it was a hard-coded default of 0.231 on
``ReshapingConfig``, which is the HALO value (native log grid ratio 1.689030,
half-width sqrt(1.689030), f = 1 - 1.689030^(-1/2) = 0.230548, rounded to 0.231
in the source). Every non-halo run therefore used the halo's binning. The IGRB
grid of Ackermann+ 2015 is spaced by 10^(1/8) = 1.414214 per bin, for which
f = 0.159104 and u_E = 0.189226, against the 0.300390 that was used.

PER-BIN, NOT SCALAR
-------------------
The criterion is applied to a photon in a particular bin, so the quantity the
physics needs is f_i, the half-width of THAT bin, and f is returned per bin.
A single scalar f is defensible only when the grid is exactly geometric, where
every f_i is the same number; on a non-uniform grid a scalar would apply one
bin's width to every other bin, which is the same class of error as applying the
halo's width to the IGRB. Both production grids here are geometric to 1e-4 in
ratio, so the two agree in practice -- but the per-bin form is what is computed,
and the scalar summary is reported for the record only.

EDGE CONVENTION
---------------
Bin edges are the log-midpoints between neighbouring bin centres, with the two
outer edges reflected, which is the same construction the reshaping kernel uses
for bin membership (``core.kinematics``, two copies there). Using it here keeps
the removal threshold of the attenuation arm and the bin bookkeeping of the
reshaping arm on one definition of "the bin".

    E_lo_i = exp( (log E_{i-1} + log E_i) / 2 ),     f_i = 1 - E_lo_i / E_i

For a geometric grid of ratio R this is exactly f = 1 - R^(-1/2).
"""

from __future__ import annotations

import numpy as np

__all__ = ["log_bin_edges", "f_bin_per_bin", "bin_geometry_summary"]


def log_bin_edges(E_bins) -> tuple[np.ndarray, np.ndarray]:
    """(E_lo, E_hi) log-midpoint edges of the bins centred on ``E_bins``."""
    E = np.asarray(E_bins, dtype=float)
    if E.ndim != 1 or E.size < 2:
        raise ValueError("E_bins must be a 1-D array of at least two bin centres")
    if np.any(E <= 0) or np.any(np.diff(E) <= 0):
        raise ValueError("E_bins must be positive and strictly increasing")
    logE = np.log(E)
    edges = np.empty(E.size + 1)
    edges[1:-1] = 0.5 * (logE[:-1] + logE[1:])
    edges[0] = logE[0] - 0.5 * (logE[1] - logE[0])
    edges[-1] = logE[-1] + 0.5 * (logE[-1] - logE[-2])
    return np.exp(edges[:-1]), np.exp(edges[1:])


def f_bin_per_bin(E_bins) -> np.ndarray:
    """Fractional energy loss that drops a photon below its own bin's lower edge."""
    E = np.asarray(E_bins, dtype=float)
    E_lo, _ = log_bin_edges(E)
    return 1.0 - E_lo / E


def bin_geometry_summary(E_bins) -> dict:
    """Per-bin f, the neighbour ratios, and whether one scalar f would do.

    ``uniform`` is True when every neighbour ratio agrees to 1e-6 relative, i.e.
    the grid is geometric and the per-bin f collapses to a single number.
    """
    E = np.asarray(E_bins, dtype=float)
    f = f_bin_per_bin(E)
    ratio = E[1:] / E[:-1]
    r_med = float(np.median(ratio))
    spread = float(np.max(np.abs(ratio / r_med - 1.0))) if ratio.size else 0.0
    return {
        "f_bin": f,
        "f_bin_median": float(np.median(f)),
        "f_bin_min": float(np.min(f)),
        "f_bin_max": float(np.max(f)),
        "bin_ratio_median": r_med,
        "bin_ratio_min": float(np.min(ratio)) if ratio.size else np.nan,
        "bin_ratio_max": float(np.max(ratio)) if ratio.size else np.nan,
        "uniform": bool(spread <= 1e-6),
        "n_bins": int(E.size),
    }
