"""
ROI recovery fraction w(theta), computed from the template rather than adopted.

w(theta) = the intensity-weighted fraction of ROI solid angle for which a
deflection of theta, at uniformly random azimuth, leaves the photon still inside
the ROI. It is a 2D integral over the same (l, b) grid used by
attenuation_eft.roi_tau_prefactor, weighted by the rho^2 template, and it does
NOT depend on operator, mass, or photon energy -- so it is computed once per ROI
geometry and cached.

WHY THIS REPLACES A CHOSEN ANGLE
--------------------------------
The previous treatment, kinematics.roi_recovery_fraction, models a circular cone
of half-angle alpha with the photon at the cone's CENTRE, with alpha = 60 deg
taken from the box half-width in l. The ROI actually integrated is a box:
|l| <= 60, 10 <= |b| <= 60, with a 20-deg-wide disk cut through the middle. That
cone model has no calibration behind it, and its own docstring concedes the
point: "For the Totani halo template, which is extended and fills the full ROI,
the right treatment averages over source positions."

This module does that average. The computed curve is not a cone of any radius:

    theta      5 deg   10 deg   20 deg   60 deg   120 deg
    computed   0.895   0.807    0.660    0.403    0.019
    cone(60)   1.000   1.000    1.000    0.500    0.000
    cone(8.5)  0.688   0.448    0.000    0.000    0.000

It falls immediately -- 4.5% loss by 2 deg -- because ~44% of the ROI's solid
angle lies within a few degrees of the sharp |b| = 10 deg disk edge, which no
cone represents. But it has a long tail: a 90 deg deflection still retains 15%,
because the ROI is 120 deg wide in longitude. Neither cone brackets it.

NUMERICS
--------
Sampling must be CELL-CENTRED. Putting samples on b = +-10, +-60 or l = +-60
exactly gives w(0) = 0.93 instead of 1 -- a 7% error at zero deflection, purely
from boundary cells losing half their azimuths at infinitesimal theta. With
cell centres, w(theta -> 0) -> 1 exactly and the curve converges to ~1e-3 by
240 x 200 x 256; the residual grid sensitivity is confined to theta <~ 1 deg,
where w is within 2% of unity anyway.

ASSUMPTIONS
-----------
1. Deflection is azimuthally uniform about the incoming direction (correct for
   an unpolarised scatter).
2. Every sightline's photons originate inside the ROI. True for the halo
   template. NOT true for the IGRB -- see igrb_recovery_fraction below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_CACHE_DIR = Path(__file__).resolve().parent.parent / "constraint_boundaries"
_CACHE = {}

# The ROI actually integrated by roi_tau_prefactor / ReshapingConfig defaults.
DEFAULT_L_RANGE = (-60.0, 60.0)
DEFAULT_B_RANGE = (10.0, 60.0)


def _roi_mask(l, b, l_range=DEFAULT_L_RANGE, b_range=DEFAULT_B_RANGE):
    return ((l >= l_range[0]) & (l <= l_range[1])
            & (np.abs(b) >= b_range[0]) & (np.abs(b) <= b_range[1]))


def compute_w_curve(theta_deg, *, nl=120, nb=100, nphi=128, n_J=150,
                    l_range=DEFAULT_L_RANGE, b_range=DEFAULT_B_RANGE):
    """Template-weighted recovery fraction at each theta [deg]. Cell-centred."""
    from core.attenuation_eft import compute_J_los

    le = np.linspace(l_range[0], l_range[1], nl + 1)
    l = 0.5 * (le[:-1] + le[1:])
    be = np.linspace(b_range[0], b_range[1], nb + 1)
    bp = 0.5 * (be[:-1] + be[1:])
    b = np.concatenate([-bp[::-1], bp])
    L, B = np.meshgrid(l, b, indexing="xy")

    J2 = np.array([[compute_J_los(float(x), float(y), power=2, n_points=n_J)
                    for x in l] for y in b])
    W = J2 * np.cos(np.deg2rad(B))
    W = np.where(np.isfinite(W) & (W > 0), W, 0.0)

    Wf = W.ravel()
    Lf, Bf = np.deg2rad(L.ravel()), np.deg2rad(B.ravel())
    n = np.stack([np.cos(Bf) * np.cos(Lf), np.cos(Bf) * np.sin(Lf), np.sin(Bf)], -1)
    eb = np.stack([-np.sin(Bf) * np.cos(Lf), -np.sin(Bf) * np.sin(Lf), np.cos(Bf)], -1)
    el = np.stack([-np.sin(Lf), np.cos(Lf), np.zeros_like(Lf)], -1)

    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    cp, sp = np.cos(phi)[:, None, None], np.sin(phi)[:, None, None]

    out = np.empty(len(theta_deg), dtype=float)
    for i, th in enumerate(np.asarray(theta_deg, dtype=float)):
        t = np.deg2rad(th)
        npr = np.cos(t) * n[None] + np.sin(t) * (cp * eb[None] + sp * el[None])
        bpr = np.rad2deg(np.arcsin(np.clip(npr[..., 2], -1.0, 1.0)))
        lpr = np.rad2deg(np.arctan2(npr[..., 1], npr[..., 0]))
        out[i] = np.sum(_roi_mask(lpr, bpr, l_range, b_range).mean(0) * Wf) / np.sum(Wf)
    return out


def _default_theta_nodes():
    return np.concatenate([[0.0], np.geomspace(1e-3, 180.0, 180)])


def halo_recovery_curve(*, rebuild=False, **kw):
    """Cached (theta_deg, w) for the halo ROI. Computed once, ~20 s."""
    key = "halo"
    if not rebuild and key in _CACHE:
        return _CACHE[key]
    path = _CACHE_DIR / "roi_recovery_halo.npz"
    if path.exists() and not rebuild:
        d = np.load(path)
        _CACHE[key] = (d["theta_deg"], d["w"])
        return _CACHE[key]
    th = _default_theta_nodes()
    w = compute_w_curve(th, **kw)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, theta_deg=th, w=w)
    _CACHE[key] = (th, w)
    return _CACHE[key]


def halo_recovery_fraction(theta_deg):
    """w(theta) for the halo ROI, interpolated on the cached curve."""
    th, w = halo_recovery_curve()
    return np.interp(np.asarray(theta_deg, dtype=float), th, w)


def igrb_recovery_fraction(theta_deg):
    """w(theta) = 1 identically for the IGRB.  This is a derivation, not a guard.

    The IGRB is an ISOTROPIC source observed against a scattering medium of the
    cosmological MEAN density. Both the source intensity and the scatterer
    distribution are, by construction, independent of direction. Radiative
    transfer then gives exact compensation: for every photon scattered out of a
    given line of sight through angle theta, the surrounding sky delivers a
    statistically identical photon scattered INTO that line of sight through
    -theta, because the specific intensity is the same in both directions and
    the scattering is azimuthally symmetric. There is no boundary anywhere for a
    photon to be deflected across.

    Angular escape therefore removes nothing from the IGRB measurement, and
    ENERGY MIGRATION is the only removal channel: a scattered photon is lost
    only if it leaves its energy bin, which is a statement about the Compton
    relation and not about geometry.

    The halo case differs precisely because the halo template is anisotropic and
    the ROI has edges -- most importantly the |b| = 10 deg disk cut, which
    accounts for ~44% of the ROI solid angle at close range.
    """
    return np.ones_like(np.asarray(theta_deg, dtype=float))


def recovery_fraction(theta_deg, dataset="halo"):
    """Dispatch on dataset: 'halo' uses the computed template curve, 'igrb' = 1."""
    ds = str(dataset).lower()
    if ds in ("halo", "measured"):
        return halo_recovery_fraction(theta_deg)
    if ds == "igrb":
        return igrb_recovery_fraction(theta_deg)
    raise KeyError(f"no recovery model for dataset {dataset!r}")
