"""
kinematics.py
=============
Photon-DM Compton-like scattering kinematics in the DM rest frame.

All energies in GeV throughout. The heavy-DM regime (E_gamma << m_chi)
is the physically relevant one for the Totani halo excess, where
E_gamma ~ 1-800 GeV and m_chi ~ 0.5-0.8 TeV, giving E/m_chi ~ O(0.1).

Public API
----------
scattered_energy(E, theta, m_chi)      : E'(E, theta, m_chi)
recoil_fraction(E, theta, m_chi)       : (E - E') / E
jacobian_dOmega_dE(E, theta, m_chi)    : |dOmega/dE'| for change-of-variables
max_energy_loss_fraction(E, m_chi)     : Delta E / E at theta = pi (backscatter)
omega_out_grid(E, cos_theta, m_chi)    : vectorised E' over a (nE, nTheta) grid
build_redistribution_matrix_exact(E_bins, m_chi, dsigma_dOmega, w_fn=None)
    : assemble the (nE_out, nE_in) kernel K[i,j] = probability that a photon
      born in energy bin j is observed in bin i after one scatter, Eq. (II.13).

Notes
-----
Kinematics are exact (not heavy-DM expanded). The expansion parameter is
    eps = E / m_chi
which ranges from ~0.002 at 1 GeV / 500 GeV to ~1.6 at 800 GeV / 500 GeV,
so the full expressions are required across the Totani energy range.

Sign conventions: theta in [0, pi], cos_theta in [-1, 1].
Forward scattering (theta = 0) preserves photon energy exactly.
Backscattering (theta = pi) gives maximum energy loss.
"""

import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Core kinematics
# ---------------------------------------------------------------------------

def scattered_energy(
    E: ArrayLike,
    theta: ArrayLike,
    m_chi: float,
) -> np.ndarray:
    """
    Scattered photon energy E'(E, theta, m_chi) from Compton-like kinematics
    with electron mass replaced by m_chi.

        E' = E / (1 + (E / m_chi) * (1 - cos(theta)))

    Parameters
    ----------
    E : float or array
        Incoming photon energy [GeV].
    theta : float or array
        Scattering angle [rad], broadcast-compatible with E.
    m_chi : float
        DM mass [GeV].

    Returns
    -------
    E_out : ndarray
        Scattered photon energy [GeV], same shape as broadcast(E, theta).
    """
    E = np.asarray(E, dtype=float)
    theta = np.asarray(theta, dtype=float)
    denom = 1.0 + (E / m_chi) * (1.0 - np.cos(theta))
    return E / denom


def recoil_fraction(
    E: ArrayLike,
    theta: ArrayLike,
    m_chi: float,
) -> np.ndarray:
    """
    Fractional energy loss (E - E') / E for a single scatter.

    Ranges from 0 at forward scatter to ~2E/m_chi at backscatter (heavy-DM limit).

    Parameters
    ----------
    E, theta, m_chi : as in scattered_energy.

    Returns
    -------
    delta : ndarray  in [0, 1)
    """
    E_out = scattered_energy(E, theta, m_chi)
    return (np.asarray(E, dtype=float) - E_out) / np.asarray(E, dtype=float)


def max_energy_loss_fraction(E: ArrayLike, m_chi: float) -> np.ndarray:
    """
    Maximum fractional energy loss (backscatter, theta = pi).

        Delta_max / E = 2 * (E/m_chi) / (1 + 2 * E/m_chi)

    In the heavy-DM limit (E << m_chi) this approaches 2E/m_chi.

    Parameters
    ----------
    E : float or array   [GeV]
    m_chi : float        [GeV]

    Returns
    -------
    frac : ndarray   dimensionless, in [0, 1)
    """
    E = np.asarray(E, dtype=float)
    eps = E / m_chi
    return 2.0 * eps / (1.0 + 2.0 * eps)


def scattered_energy_grid(
    E_bins: np.ndarray,
    cos_theta_vals: np.ndarray,
    m_chi: float,
) -> np.ndarray:
    """
    Build a 2D grid of scattered energies.

    Parameters
    ----------
    E_bins : (nE,)           Photon energies [GeV]
    cos_theta_vals : (nTh,)  cos(theta) values in [-1, 1]
    m_chi : float            DM mass [GeV]

    Returns
    -------
    E_out : (nE, nTh)   E'[i, k] = scattered energy for E_bins[i], cos_theta[k]
    """
    E_bins = np.asarray(E_bins, dtype=float)[:, None]      # (nE, 1)
    cos_th = np.asarray(cos_theta_vals, dtype=float)[None, :]  # (1, nTh)
    denom = 1.0 + (E_bins / m_chi) * (1.0 - cos_th)
    return E_bins / denom                                   # (nE, nTh)


def jacobian_dE_dcostheta(
    E: ArrayLike,
    cos_theta: ArrayLike,
    m_chi: float,
) -> np.ndarray:
    """
    Jacobian |dE' / d(cos theta)| for the change-of-variables
    d(cos theta) -> dE'.

    From E' = E / [1 + (E/m)(1 - cos theta)]:

        dE'/d(cos theta) = E^2 / (m_chi * denom^2)   > 0

    Used to convert dσ/dΩ into dσ/dE' via:

        dσ/dE' = (2π / dE'/d(cosθ)) * dσ/dΩ

    Parameters
    ----------
    E : float or array   [GeV]
    cos_theta : float or array   dimensionless
    m_chi : float   [GeV]

    Returns
    -------
    jac : ndarray   [GeV^-1 sr]  (positive definite)
    """
    E = np.asarray(E, dtype=float)
    cos_theta = np.asarray(cos_theta, dtype=float)
    denom = 1.0 + (E / m_chi) * (1.0 - cos_theta)
    return E**2 / (m_chi * denom**2)


# ---------------------------------------------------------------------------
# Convenience: flux-normalised in-scatter term
# ---------------------------------------------------------------------------

def inscatter_flux(
    Phi_0: np.ndarray,
    K: np.ndarray,
    tau: np.ndarray,
    E_bins: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the in-scatter contribution to the observed spectrum.

    For each output energy bin i, photons that were born at higher energies j
    and scattered into bin i contribute:

        Phi_in[i] = sum_{j > i} K[i,j] * tau[j] * Phi_0[j] * exp(-tau[j])

    The factor tau[j] * exp(-tau[j]) is the single-scatter probability in the
    optically thin regime. For tau << 1, this approximates to tau[j] * Phi_0[j].

    Parameters
    ----------
    Phi_0 : (nE,)   unattenuated source spectrum (arbitrary units)
    K : (nE, nE)    photon-number redistribution matrix from build_redistribution_matrix
    tau : (nE,)     optical depth per energy bin
    E_bins : (nE,) optional
        If provided, Phi_0 is treated as an energy-flux-like spectrum
        (approximately E^2 dN/dE) and K is converted with an E_i/E_j factor
        before being applied. If omitted, Phi_0 is treated as photon number.

    Returns
    -------
    Phi_in : (nE,)  in-scatter contribution, same units as Phi_0
    """
    Phi_0 = np.asarray(Phi_0, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    if E_bins is not None:
        E_bins = np.asarray(E_bins, dtype=float)
        K = K * (E_bins[:, None] / E_bins[None, :])

    # Weight source by single-scatter probability
    scatter_weight = tau * np.exp(-tau)          # (nE,)  peaks at tau=1
    return K @ (scatter_weight * Phi_0)          # (nE,)


def reshaped_spectrum(
    Phi_0: np.ndarray,
    K: np.ndarray,
    tau: np.ndarray,
    E_bins: np.ndarray | None = None,
) -> np.ndarray:
    """
    Full reshaping model for the observed spectrum.

    Phi_obs[i] = Phi_0[i] * exp(-tau[i])              [survival term]
               + sum_{j>=i} K[i,j] * tau[j] * Phi_0[j] * exp(-tau[j])   [in-scatter]

    This is the single-scatter approximation. It is valid when tau << 1
    everywhere (the Totani halo case: tau ~ 10^{-3} to 10^{-6}).

    For completeness, the diagonal term K[i,i] * tau[i] * Phi_0[i] * exp(-tau[i])
    represents photons that scatter but remain in the same energy bin (near-forward
    scatters). This is typically very small because the bin width >> kinematic
    energy shift for sub-TeV DM, but is included for consistency.

    Note
    ----
    This is NOT a full radiative-transfer solution. It neglects:
    - Multiple scatters (order tau^2 corrections)
    - Scattering of already-in-scattered photons
    Both are completely negligible at tau << 1.

    Parameters
    ----------
    Phi_0 : (nE,)   unattenuated source spectrum
    K : (nE, nE)    photon-number redistribution matrix
    tau : (nE,)     optical depth per energy bin
    E_bins : (nE,) optional
        If provided, apply the photon-number-to-energy-flux E_i/E_j correction.

    Returns
    -------
    Phi_obs : (nE,)  reshaped observed spectrum, same units as Phi_0
    """
    Phi_0 = np.asarray(Phi_0, dtype=float)
    tau = np.asarray(tau, dtype=float)

    survival = Phi_0 * np.exp(-tau)
    inscatter = inscatter_flux(Phi_0, K, tau, E_bins=E_bins)
    return survival + inscatter


# ---------------------------------------------------------------------------
# Redistribution kernel on exact per-bin u-intervals (log-u nodes)
# ---------------------------------------------------------------------------
# Compton kinematics give E'/E = 1/(1+u) monotonically, so the photons from
# input bin j landing in output bin i occupy a CONTIGUOUS interval in u with
# analytic endpoints:
#
#     E' in [E_lo_i, E_hi_i]   <=>   u in [E_j/E_hi_i - 1,  E_j/E_lo_i - 1]
#
# clipped to the physical range [0, 2r]. Each K[i,j] is integrated between those
# exact limits, with no angular grid, no masking and no bin-edge quantisation.
# Within each interval the integrand is smooth but strongly peaked near small u
# (the forward peak has width ~ m_chi/omega), so the interval is segmented
# geometrically and each segment integrated by Gauss-Legendre.

_GLK_X, _GLK_W = np.polynomial.legendre.leggauss(32)


def _u_segment_edges(u_a, u_b, n_seg):
    """Geometric segmentation of [u_a, u_b], handling u_a = 0.

    The integrand is peaked near small u, so segments must be geometric, not
    uniform. When u_a = 0 the first segment is linear from zero to a small
    fraction of u_b; the integrand vanishes there like a positive power of u,
    so that segment contributes negligibly and needs no resolution.
    """
    if u_b <= u_a:
        return None
    if u_a <= 0.0:
        return np.concatenate([[0.0], np.geomspace(u_b * 1e-13, u_b, n_seg)])
    return np.geomspace(u_a, u_b, n_seg + 1)


def _integrate_u(f, u_a, u_b, n_seg):
    """int_{u_a}^{u_b} f(u) du by segmented Gauss-Legendre."""
    edges = _u_segment_edges(u_a, u_b, n_seg)
    if edges is None:
        return 0.0
    lo, hi = edges[:-1], edges[1:]
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    u = mid[:, None] + half[:, None] * _GLK_X          # (nseg, ngl)
    return float(np.sum(half[:, None] * _GLK_W * f(u)))


def build_redistribution_matrix_exact(
    E_bins: np.ndarray,
    m_chi: float,
    dsigma_fn,
    *,
    w_fn=None,
    n_seg: int = 24,
    return_diagnostics: bool = False,
):
    """Redistribution kernel on exact per-bin u-intervals with log-u nodes.

    Parameters
    ----------
    E_bins : (nE,)
        Photon energy bin centres [GeV].
    m_chi : float
        DM mass [GeV].
    dsigma_fn : callable(E, theta) -> dsigma/dOmega [cm^2/sr]
        Must broadcast over an array of theta at fixed scalar E.
    w_fn : callable(theta_deg) -> weight in [0, 1], optional
        ROI recovery weight. None means unity (every scattered photon
        recovered). Applies to the NUMERATOR only; sigma_tot in the denominator
        stays unweighted, so column sums are <= 1 with equality iff w == 1 and
        nothing leaks below the lowest bin -- the same convention as
        build_redistribution_matrix.
    n_seg : int
        Geometric segments per integration interval.

    Returns
    -------
    K : (nE, nE) upper-triangular, column sums <= 1.
    diagnostics : dict, only if return_diagnostics
        'sigma_tot', 'column_sum', 'leak_below' (fraction falling under the
        lowest bin edge, which is a physical loss, not a normalisation error).
    """
    E_bins = np.asarray(E_bins, dtype=float)
    nE = len(E_bins)

    log_E = np.log(E_bins)
    log_edges = np.empty(nE + 1)
    log_edges[1:-1] = 0.5 * (log_E[:-1] + log_E[1:])
    log_edges[0] = log_E[0] - 0.5 * (log_E[1] - log_E[0])
    log_edges[-1] = log_E[-1] + 0.5 * (log_E[-1] - log_E[-2])
    E_lo, E_hi = np.exp(log_edges[:-1]), np.exp(log_edges[1:])

    K = np.zeros((nE, nE), dtype=float)
    sigma_tot = np.zeros(nE, dtype=float)
    leak = np.zeros(nE, dtype=float)

    for j in range(nE):
        E_j = float(E_bins[j])
        r = E_j / float(m_chi)
        u_max = 2.0 * r

        def integrand(u, r=r, E_j=E_j, weight=False):
            cos_t = np.clip(1.0 - u / r, -1.0, 1.0)
            theta = np.arccos(cos_t)
            d = np.asarray(dsigma_fn(E_j, theta), dtype=float)
            d = np.where(np.isfinite(d) & (d >= 0.0), d, 0.0)
            if weight and w_fn is not None:
                d = d * np.asarray(w_fn(np.rad2deg(theta)), dtype=float)
            # dx = du / r, and sigma = 2 pi int (dsigma/dOmega) dx
            return d * (2.0 * np.pi / r)

        sigma_tot[j] = _integrate_u(lambda u: integrand(u), 0.0, u_max, n_seg)
        if sigma_tot[j] <= 0.0:
            continue

        for i in range(j + 1):
            # Exact interval: E' in [E_lo_i, E_hi_i]  <=>  u in [...]
            u_a = max(0.0, E_j / E_hi[i] - 1.0)
            u_b = min(u_max, E_j / E_lo[i] - 1.0)
            if u_b <= u_a:
                continue
            K[i, j] = _integrate_u(
                lambda u: integrand(u, weight=True), u_a, u_b, n_seg
            ) / sigma_tot[j]

        # Flux scattered below the lowest bin edge is genuinely lost from the
        # observable, so it must NOT be folded back into the column sum.
        u_leak_a = max(0.0, E_j / E_lo[0] - 1.0)
        if u_max > u_leak_a:
            leak[j] = _integrate_u(
                lambda u: integrand(u, weight=True), u_leak_a, u_max, n_seg
            ) / sigma_tot[j]

    if return_diagnostics:
        return K, {"sigma_tot": sigma_tot, "column_sum": K.sum(axis=0),
                   "leak_below": leak}
    return K
