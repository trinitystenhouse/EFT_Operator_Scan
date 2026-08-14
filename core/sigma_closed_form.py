"""
Closed-form total cross sections for the photon-DM EFT operators.

sigma(u_min, u_max) = F_i(u_min) - F_i(u_max),  u = (omega/m_chi)(1 - cos theta)

The F_i are Eqs. (IV.12)-(IV.16) of the paper, in the u-form that retains
r = omega/m_chi. Verified against the amplitude by symbolic differentiation:
dF_i/du equals the exact dsigma/du (exact Compton kinematics,
t = -2 m omega u/(1+u), omega'/omega = 1/(1+u)) to 1e-7 or better at
r = 1e3, 1e5, 1e7, with unit constant of proportionality; the check runs in
tests/test_sigma_closed_form.py.

WHY THIS EXISTS
---------------
Integrating dsigma/dOmega on an angular grid does not work here. The forward
peak has angular width ~ m_chi/omega, so for omega/m_chi > ~100 it falls between
the nodes of any linear cos(theta) grid and the trapezoid rule returns the peak
height times the grid spacing. That estimate is not convergent: it scales as
1/n_theta and only stabilises once n_theta > 2*omega/m_chi, which is ~3e5 at
m_chi = 1e-3 GeV.

These closed forms remove the quadrature from the problem entirely. Resolved
log-grid quadrature is retained in the test suite as a CONSISTENCY CHECK
validated against these forms, not the reverse.

NUMERICAL NOTE
--------------
At high m_chi, r -> 0 and both integration limits approach zero, so
F_i(u_min) - F_i(u_max) differences two nearly equal numbers. Every term is
therefore evaluated through a cancellation-safe identity:

    1/(1+a)^k - 1/(1+b)^k = expm1(k * log1p((b-a)/(1+a))) / (1+b)^k
    log(1+a) - log(1+b)    = -log1p((b-a)/(1+a))

which is exact to machine precision for a -> b, rather than losing ~6 digits at
m_chi = 1e8 GeV (inside the plotted range).
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np

GEV2_TO_FB = 3.89379e11
FB_TO_CM2 = 1e-39
GEV2_TO_CM2 = GEV2_TO_FB * FB_TO_CM2


def _pow_diff(a, b, k):
    """1/(1+a)^k - 1/(1+b)^k, stable as a -> b.  Requires b >= a > -1."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.expm1(k * np.log1p((b - a) / (1.0 + a))) / (1.0 + b) ** k


def _log_diff(a, b):
    """log(1+a) - log(1+b), stable as a -> b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return -np.log1p((b - a) / (1.0 + a))


# --------------------------------------------------------------------------- #
# F_i(u_min) - F_i(u_max), assembled term by term through the stable helpers.
# Each returns GeV^-2; multiply by GEV2_TO_CM2 for cm^2.
# --------------------------------------------------------------------------- #

def _d_scalar(a, b, m, w, Lam, c_phi):
    """F_scal = 2 c_phi^2 m omega (1 + 3u + 3u^2) / (3 pi Lambda^4 (1+u)^3).

    Rewrite 1+3u+3u^2 = 3(1+u)^2 - 3(1+u) + 1 so the whole form collapses to a
    sum of inverse powers, each differenced stably.
    """
    pref = 2.0 * c_phi**2 * m * w / (3.0 * np.pi * Lam**4)
    return pref * (3.0 * _pow_diff(a, b, 1) - 3.0 * _pow_diff(a, b, 2) + _pow_diff(a, b, 3))


def _d_odd(a, b, m, w, Lam, c_p):
    """F_odd = c_p^2 m^2 omega^2 (1+2u)(1+2u+2u^2) / (64 pi Lambda^6 (1+u)^4).

    (1+2u)(1+2u+2u^2) = 4(1+u)^3 - 6(1+u)^2 + 4(1+u) - 1.
    """
    pref = c_p**2 * m**2 * w**2 / (64.0 * np.pi * Lam**6)
    return pref * (4.0 * _pow_diff(a, b, 1) - 6.0 * _pow_diff(a, b, 2)
                   + 4.0 * _pow_diff(a, b, 3) - _pow_diff(a, b, 4))


def _d_even(a, b, m, w, Lam, c_s, r):
    """F_even = c_s^2 m^3 omega P(u) / (192 pi Lambda^6 (1+u)^4),
    P(u) = 8(1+4u+6u^2+3u^3) + 3r(1+4u+6u^2+4u^3).

    In powers of v = 1+u:
      1+4u+6u^2+3u^3 = 3v^3 - 3v^2 + v      (check: expand)
      1+4u+6u^2+4u^3 = 4v^3 - 6v^2 + 4v - 1
    """
    pref = c_s**2 * m**3 * w / (192.0 * np.pi * Lam**6)
    t8 = 8.0 * (3.0 * _pow_diff(a, b, 1) - 3.0 * _pow_diff(a, b, 2) + _pow_diff(a, b, 3))
    t3r = 3.0 * r * (4.0 * _pow_diff(a, b, 1) - 6.0 * _pow_diff(a, b, 2)
                     + 4.0 * _pow_diff(a, b, 3) - _pow_diff(a, b, 4))
    return pref * (t8 + t3r)


def _d_dipole(a, b, m, w, Lam, c_s, r):
    """F_dip = (m^2 mu^4 / 8 pi r) * { [(3+4u) + 2r(1+2u) + 2r^2]/(1+u)^2 + 2 ln(1+u) },
    with mu = 2 c / Lambda.

    Numerator in v = 1+u:  (3+4u) = 4v - 1,  2r(1+2u) = 2r(2v - 1).
    """
    mu = 2.0 * c_s / Lam
    pref = m**2 * mu**4 / (8.0 * np.pi * r)
    rational = (4.0 * _pow_diff(a, b, 1) - _pow_diff(a, b, 2)
                + 2.0 * r * (2.0 * _pow_diff(a, b, 1) - _pow_diff(a, b, 2))
                + 2.0 * r**2 * _pow_diff(a, b, 2))
    return pref * (rational + 2.0 * _log_diff(a, b))


# --------------------------------------------------------------------------- #
# SINGLE SOURCE OF TRUTH FOR EACH F_i
# --------------------------------------------------------------------------- #
# Every F_i in the paper has the form
#
#     F(u) = pref * [ sum_k c_k (1+u)^-k  +  c_log * ln(1+u) ]
#
# so ONE coefficient list determines both the antiderivative difference and the
# integrand.  Deriving them separately is what produced the bug this replaces:
# the hand-expanded small-u series for the `even` and `dipole` shapes was
# expanded in small u while treating r = omega/m_chi as O(1), but in that branch
# u <= 2r, so r is small too and the 1/r coefficients never become subleading.
# The result was wrong by a factor 1/u_max -- 2.96e3 at m_chi = 1e6 GeV and
# 2.96e5 at 1e8, growing without bound -- while `scalar` and `odd`, whose
# shapes carry no r, were correct.  Both objects now come from _f_terms(), so
# they cannot disagree again.
# The c_k are POLYNOMIALS IN r, stored as exact rational coefficient lists in
# ascending powers of r and never evaluated numerically until the very end.
# That matters: reducing -dF/du to powers of u involves cancellations among the
# c_k whose surviving remainder is O(r^2). For the dipole the leading term is
# a_0 = 4r^2, obtained from (4+4r) + 2(-1-2r+2r^2) - 2, where the O(1) and O(r)
# parts cancel exactly. At m_chi = 1e8 GeV, r ~ 4e-8, so 4r^2 ~ 7e-15 is the
# same size as the roundoff in that sum, and evaluating it in floating point
# lost the whole coefficient (measured: sigma_tot off by 1.27% in the lowest
# bin). Carrying the c_k symbolically in r makes the cancellation exact.
_F_SHAPES = {
    #           c_k(r) as [r^0, r^1, ...]                     c_log
    "scalar": ({1: [3], 2: [-3], 3: [1]}, 0),
    "odd":    ({1: [4], 2: [-6], 3: [4], 4: [-1]}, 0),
    "even":   ({1: [24, 12], 2: [-24, -18], 3: [8, 12], 4: [0, -3]}, 0),
    "dipole": ({1: [4, 4], 2: [-1, -2, 2]}, 2),
}


def _padd(p, q):
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else Fraction(0))
            + (q[i] if i < len(q) else Fraction(0)) for i in range(n)]


@lru_cache(maxsize=None)
def _u_poly(kind):
    """Exact reduction of -dF/du to  sum_j a_j(r) u^j / (1+u)^(K+1).

    Returns (K, a) with a[j] the coefficient list of a_j as a polynomial in r.
    All arithmetic is in Fraction, so every cancellation is exact.
    """
    coeffs, c_log = _F_SHAPES[kind]
    K = max(coeffs)

    # N(v) = sum_k c_k k v^(K-k)  -  c_log v^K, as coefficients b[n] of v^n
    b = [[Fraction(0)] for _ in range(K + 1)]
    for k, cpoly in coeffs.items():
        b[K - k] = _padd(b[K - k], [Fraction(c) * k for c in cpoly])
    if c_log:
        b[K] = _padd(b[K], [-Fraction(c_log)])

    # v^n = sum_j C(n,j) u^j   ->   a[j] = sum_n b[n] C(n,j)
    a = [[Fraction(0)] for _ in range(K + 1)]
    for n in range(K + 1):
        for j in range(n + 1):
            a[j] = _padd(a[j], [c * math.comb(n, j) for c in b[n]])
    return K, tuple(tuple(float(c) for c in aj) for aj in a)


def _f_prefactor(kind, m, w, Lam, c_s, c_p, c_phi, r):
    """Constant multiplying the bracket in F_kind.  See Eqs. (IV.12)-(IV.16)."""
    if kind == "scalar":
        return 2.0 * c_phi**2 * m * w / (3.0 * np.pi * Lam**4)
    if kind == "odd":
        return c_p**2 * m**2 * w**2 / (64.0 * np.pi * Lam**6)
    if kind == "even":
        return c_s**2 * m**3 * w / (192.0 * np.pi * Lam**6)
    if kind == "dipole":
        # c_s for the magnetic dipole, c_p for the electric one. The two have
        # identical |M|^2, so they share a shape -- but NOT a coefficient, and
        # production resolves dipole_electric to (c_s, c_p) = (0, 1) via
        # operator_couplings(). Reading c_s unconditionally here returned
        # sigma = 0 for the electric dipole, silently.
        return m**2 * (2.0 * c_s / Lam)**4 / (8.0 * np.pi * r)
    raise KeyError(kind)


def _f_terms(kind, m, w, Lam, c_s, c_p, c_phi, r):
    """(pref, [(k, c_k)], c_log) with the c_k evaluated at this r."""
    coeffs, c_log = _F_SHAPES[kind]
    pref = _f_prefactor(kind, m, w, Lam, c_s, c_p, c_phi, r)
    ev = [(k, sum(c * r**i for i, c in enumerate(cpoly)))
          for k, cpoly in sorted(coeffs.items())]
    return pref, ev, float(c_log)


def _f_diff(kind, a, b, m, w, Lam, c_s, c_p, c_phi, r):
    """F(a) - F(b), each term differenced stably.  Exact for well-separated a, b."""
    pref, coeffs, c_log = _f_terms(kind, m, w, Lam, c_s, c_p, c_phi, r)
    tot = sum(c * _pow_diff(a, b, k) for k, c in coeffs)
    if c_log:
        tot = tot + c_log * _log_diff(a, b)
    return pref * tot


def _f_integrand(kind, u, m, w, Lam, c_s, c_p, c_phi, r):
    """-dF/du, i.e. the positive dsigma/du.

    Differentiating F term by term gives a sum of inverse powers of v = 1+u.
    Evaluating that directly reintroduces exactly the cancellation the closed
    form suffers from: for F_odd the coefficients are 4, -12, 12, -4, which sum
    to zero at v = 1, because the odd angular weight is u^3 (1+u)^-5 and really
    does vanish as u^3. Summing them at u ~ 1e-6 loses every significant digit
    and the result changes sign (measured: rayleigh_odd at m_chi = 1e8 GeV came
    out negative, -4.4x the true value).

    _u_poly does that reduction exactly, in rational arithmetic and symbolically
    in r, so each surviving a_j(r) is evaluated directly instead of being
    recovered from a cancelling sum.
    """
    K, a = _u_poly(kind)
    pref = _f_prefactor(kind, m, w, Lam, c_s, c_p, c_phi, r)
    num = 0.0
    for aj in reversed(a):                       # Horner in u
        cj = 0.0
        for c in reversed(aj):                   # Horner in r
            cj = cj * r + c
        num = num * u + cj
    return pref * num / (1.0 + u) ** (K + 1)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def _f_quad(kind, a, b, m, w, Lam, c_s, c_p, c_phi, r):
    """int_a^b (-dF/du) du by 64-node Gauss-Legendre.

    Used where b - a is small enough that F(a) - F(b) loses precision to
    cancellation BETWEEN terms (for F_odd the leading coefficients sum to
    4-12+12-4 = 0, so the O(b-a) parts vanish identically and the result is
    O((b-a)^4)).  The integrand is positive and analytic on [a, b], so the
    Gauss rule has no cancellation at all and is exact to machine precision
    for an interval this short.
    """
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float),
                               np.asarray(b, dtype=float))
    half, mid = 0.5 * (b - a), 0.5 * (b + a)
    u = mid[..., None] + half[..., None] * _GL_X

    # EVERY array-valued parameter needs the trailing node axis, not just some
    # of them: omega is per energy bin, so leaving it un-expanded broadcasts
    # (nbins,) against (nbins, 64) and raises.
    def _n(v):
        return np.asarray(v)[..., None] if np.ndim(v) else v

    g = _f_integrand(kind, u, _n(m), _n(w), _n(Lam),
                     _n(c_s), _n(c_p), _n(c_phi), _n(r))
    return half * np.sum(g * _GL_W, axis=-1)


# Operators whose effective Wilson coefficient is c_p rather than c_s. Mirrors
# attenuation_eft._effective_coefficient; `even`/`odd` already select correctly
# inside _f_prefactor, so only the shared dipole shape needs routing.
_COEFF_IS_C_P = {"dipole_electric"}

_DISPATCH = {
    "scalar_rayleigh": "scalar",
    "rayleigh_odd": "odd",
    "rayleigh_even": "even",
    "rayleigh_full": "full",
    "dipole_magnetic": "dipole",
    "dipole_electric": "dipole",
}


def sigma_removal_cm2(operator, m_chi, omega, Lambda, *,
                      c_s=1.0, c_p=1.0, c_phi=1.0,
                      f_bin=0.231, x_roi=None, u_min=None,
                      check=True, closed_channel="raise"):
    """Removal cross section [cm^2] between u_min and u_max = 2*omega/m_chi.

    u_min = min( f/(1-f),  r * x_roi )   -- energy migration OR angular escape.

    The min() is load bearing. f/(1-f) is constant, but r = omega/m_chi shrinks
    with mass, so above m_chi = omega*x_roi/(f/(1-f)) the ROI term takes over.
    Hard-coding the constant would give u_min > u_max for
    m_chi > 2*omega*(1-f)/f (= 6.658*omega, i.e. 1.12 TeV at omega = 169 GeV)
    and silently return a NEGATIVE cross section, since both endpoints are legal
    inputs to F_i. Taking the min makes that unreachable (x_roi <= 2 implies
    r*x_roi < 2r = u_max identically); the assertion below guards future edits.
    """
    m_chi = np.asarray(m_chi, dtype=float)
    omega = np.asarray(omega, dtype=float)
    r = omega / m_chi
    u_max = 2.0 * r

    if u_min is None:
        u_e = f_bin / (1.0 - f_bin)
        if x_roi is None:
            u_min = np.full_like(u_max, u_e)
        else:
            u_min = np.minimum(u_e, r * float(x_roi))
    u_min = np.asarray(u_min, dtype=float)

    # Where u_min >= u_max the removal channel is CLOSED: even a full
    # backscatter shifts the photon by less than the removal criterion, so
    # nothing is removed and sigma is exactly zero. That is a physical result
    # per energy bin, not an error -- but it is also the signature of a
    # misconfigured scalar call, so the caller must say which it expects.
    closed = ~(u_min < u_max)
    if np.any(closed):
        if closed_channel == "raise":
            u_e = f_bin / (1.0 - f_bin)
            which = "energy migration (f/(1-f))" if np.all(u_min[closed] >= u_e - 1e-12) \
                else "angular escape (r*x_roi)"
            raise ValueError(
                f"u_min >= u_max for {int(np.sum(closed))} point(s); binding criterion is "
                f"{which}. u_max = 2*omega/m_chi collapses at high m_chi: a full "
                f"backscatter then shifts the photon by less than the criterion, so "
                f"nothing is removed. Worst case u_min={np.max(u_min[closed]):.6g}, "
                f"u_max={np.min(u_max[closed]):.6g}. Pass closed_channel='zero' if "
                f"a zero removal cross section is the intended physical answer."
            )
        if closed_channel != "zero":
            raise ValueError(f"closed_channel must be 'raise' or 'zero', got {closed_channel!r}")
        # Keep the arithmetic finite; the result is masked to zero below.
        u_min = np.where(closed, 0.0, u_min)

    kind = _DISPATCH.get(str(operator))
    if str(operator) in _COEFF_IS_C_P:
        # See _f_prefactor: the dipole shape carries whichever coefficient the
        # operator actually couples through.
        c_s = c_p
    if kind is None:
        raise KeyError(f"no closed form for operator {operator!r}; "
                       f"known: {sorted(_DISPATCH)}")

    # rayleigh_full is additive: the even/odd interference vanishes in the
    # polarisation sum, verified to machine precision across four decades of
    # mass and the full band.
    kinds = ("even", "odd") if kind == "full" else (kind,)

    # Both evaluations come from the same _f_terms coefficients.  The closed-form
    # difference is exact for well-separated limits; below U_SERIES_MAX it loses
    # digits to cancellation between terms and the Gauss rule takes over.  Both
    # are computed and selected elementwise so mixed-mass arrays are handled.
    small = np.broadcast_to(np.asarray(u_max) < U_SERIES_MAX, np.shape(u_max * u_min))
    v = 0.0
    for kd in kinds:
        args = (m_chi, omega, Lambda, c_s, c_p, c_phi, r)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            exact = _f_diff(kd, u_min, u_max, *args)
            quad = _f_quad(kd, u_min, u_max, *args) if np.any(small) else exact
        v = v + np.where(small, quad, exact)
    v = np.where(closed, 0.0, v)
    return v * GEV2_TO_CM2

    if kind == "scalar":
        v = _d_scalar(u_min, u_max, m_chi, omega, Lambda, c_phi)
    elif kind == "odd":
        v = _d_odd(u_min, u_max, m_chi, omega, Lambda, c_p)
    elif kind == "even":
        v = _d_even(u_min, u_max, m_chi, omega, Lambda, c_s, r)
    elif kind == "full":
        # Verified additive: interference vanishes in the polarisation sum, so
        # sigma_full(c_s, c_p) = sigma_even(c_s) + sigma_odd(c_p) to machine
        # precision across four decades of mass and the full band.
        v = (_d_even(u_min, u_max, m_chi, omega, Lambda, c_s, r)
             + _d_odd(u_min, u_max, m_chi, omega, Lambda, c_p))
    else:
        v = _d_dipole(u_min, u_max, m_chi, omega, Lambda, c_s, r)
    v = np.where(closed, 0.0, v)
    return v * GEV2_TO_CM2


# Gauss-Legendre is exact for the analytic integrand well beyond this, so the
# switch is set where the closed-form difference starts losing digits, not
# where the quadrature starts working.
U_SERIES_MAX = 1.0e-1


# --------------------------------------------------------------------------- #
# SMOOTH REMOVAL WITH A COMPUTED RECOVERY WEIGHT
# --------------------------------------------------------------------------- #

def sigma_removal_smooth_w_cm2(operator, m_chi, omega, Lambda, w_fn, *,
                               c_s=1.0, c_p=1.0, c_phi=1.0, f_bin=0.231,
                               n_seg=24):
    r"""Removal cross section under a SMOOTH angular recovery weight w(theta).

    The closed forms alone cannot express this. sigma_removal_cm2 implements a
    HARD cut -- a photon is removed iff u > u_min -- which is exact for energy
    migration (a bin edge really is a step) but wrong for angular escape, where
    the probability of leaving the ROI rises continuously with deflection. The
    two channels therefore have to be combined, not chosen between:

        sigma_rem = [F(u_E) - F(u_max)]              energy migration, exact
                  + int_0^{u_E} (dsigma/du)(1 - w) du   angular escape, smooth

    The first term is every photon that leaves its energy bin, removed for sure.
    The second is photons that stay in their bin but are deflected out of the
    ROI, weighted by their probability (1 - w) of doing so. No photon is counted
    twice: the integral is cut off exactly where the hard term begins.

    This matters because the computed w is already 0.90 by 5 degrees (see
    core/roi_recovery), so treating angular escape as a step at some theta_ROI
    is not a good approximation to it. Measured against the hard cut at
    theta_ROI = 60 deg the difference reaches 19% in sigma for the dipole near
    m_chi = 100 GeV, and it changes sign with mass.

    Parameters
    ----------
    w_fn : callable(theta_deg) -> weight in [0, 1]
        Recovery fraction. Pass roi_recovery.halo_recovery_fraction for the
        template-computed curve, or a lambda returning 1.0 for the IGRB, where
        angular escape removes nothing and energy migration is the only channel.

    Returns
    -------
    sigma [cm^2], same shape as broadcast(m_chi, omega).
    """
    m_chi = np.asarray(m_chi, dtype=float)
    omega = np.asarray(omega, dtype=float)
    r = omega / m_chi
    u_max = 2.0 * r
    u_e = f_bin / (1.0 - f_bin)

    kind = _DISPATCH.get(str(operator))
    if kind is None:
        raise KeyError(f"no closed form for operator {operator!r}; "
                       f"known: {sorted(_DISPATCH)}")
    if str(operator) in _COEFF_IS_C_P:
        c_s = c_p
    kinds = ("even", "odd") if kind == "full" else (kind,)

    # --- hard term: everything that leaves its energy bin ---------------------
    u_lo = np.minimum(u_e, u_max)
    hard = np.where(
        u_lo < u_max,
        sigma_removal_cm2(operator, m_chi, omega, Lambda, c_s=c_s, c_p=c_p,
                          c_phi=c_phi, f_bin=f_bin, u_min=u_lo,
                          closed_channel="zero"),
        0.0,
    )

    # --- smooth term: deflected out of the ROI while staying in bin -----------
    # Segmented geometrically: the integrand is peaked at small u, and w varies
    # fastest there too (w falls 4.5% by 2 deg).
    def _fwd(rr, mm, ww, ulim):
        if not (ulim > 0.0):
            return 0.0
        edges = np.concatenate([[0.0], np.geomspace(ulim * 1e-13, ulim, n_seg)])
        lo, hi = edges[:-1], edges[1:]
        half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
        u = mid[:, None] + half[:, None] * _GL_X
        theta = np.rad2deg(np.arccos(np.clip(1.0 - u / rr, -1.0, 1.0)))
        one_minus_w = 1.0 - np.asarray(w_fn(theta), dtype=float)
        tot = 0.0
        for kd in kinds:
            g = _f_integrand(kd, u, mm, ww, Lambda, c_s, c_p, c_phi, rr)
            tot = tot + np.sum(half[:, None] * _GL_W * g * one_minus_w)
        return float(tot) * GEV2_TO_CM2

    rb, mb, wb, ub = np.broadcast_arrays(r, m_chi, omega, u_lo)
    fwd = np.array([_fwd(float(a), float(b), float(c), float(d))
                    for a, b, c, d in zip(rb.ravel(), mb.ravel(),
                                          wb.ravel(), ub.ravel())]).reshape(rb.shape)
    return np.asarray(hard) + fwd
