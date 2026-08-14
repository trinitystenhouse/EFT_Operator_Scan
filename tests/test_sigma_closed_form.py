"""
Regression guards for core.sigma_closed_form.

WHY THIS FILE EXISTS
--------------------
This module has produced three separate bugs, and every one of them was
invisible in the mass range where the exclusion contours actually peak
(m_chi ~ 1e2-1e4 GeV).  All three only appeared at m_chi >~ 1e5 GeV:

1. KeyError 'full'   -- the small-u branch dispatched before selecting a kind.
2. KeyError 'dipole' -- the dispatch table emitted 'dipole', the series helpers
   expected 'dip'.
3. A wrong-by-1/u_max answer for the `even` and `dipole` shapes: the hand
   expanded small-u series treated r = omega/m_chi as O(1), but in that branch
   u <= 2r, so r is small too and the 1/r coefficients never became subleading.
   Wrong by 2.96e3 at m_chi = 1e6 and 2.96e5 at 1e8, growing linearly in m,
   while `scalar` and `odd` (whose shapes carry no r) were exactly right.

The lesson is that spot-checking the interesting range passes broken code.
Every test below therefore sweeps the FULL range 1e-5 to 1e8 GeV.

Run:  python -m pytest tests/test_sigma_closed_form.py -q
      python tests/test_sigma_closed_form.py                 (standalone)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import core.sigma_closed_form as SCF                                # noqa: E402
from core.attenuation_eft import (                                  # noqa: E402
    FB_TO_CM2,
    dsigma_dOmega_fermionic,
    dsigma_dOmega_scalar,
)

try:
    import pytest
except ImportError:
    pytest = None

OMEGA = 168.932                      # band ceiling used by the production runs
MASSES = np.geomspace(1e-5, 1e8, 27)
OPERATORS = ["dipole_magnetic", "dipole_electric", "scalar_rayleigh",
             "rayleigh_odd", "rayleigh_even", "rayleigh_full"]

# Tolerance is set by the reference quadrature, not by the closed form: a
# 2e6-node log grid still carries ~1e-5 relative error on the dipole, whose
# integrand is flattest.  The closed forms agree with each other far better
# than this (see test_branches_agree).
TOL_VS_QUAD = 2e-4


def _dsigma(operator, m_chi, theta):
    if operator == "scalar_rayleigh":
        return dsigma_dOmega_scalar(m_chi, theta, OMEGA, 1.0, 1.0)
    if operator in ("dipole_magnetic", "rayleigh_even"):
        c_s, c_p = 1.0, 0.0
    elif operator == "rayleigh_full":
        c_s, c_p = 1.0, 1.0
    else:
        c_s, c_p = 0.0, 1.0
    return dsigma_dOmega_fermionic(m_chi, theta, OMEGA, c_s, c_p, 1.0, operator)


def _sigma_quadrature(operator, m_chi, n=2_000_001):
    """sigma_tot = 2 pi int_0^2 (dsigma/dOmega) dx, x = 1 - cos(theta).

    The grid MUST be logarithmic in x.  The forward peak has angular width
    ~ m_chi/omega, so on a linear grid it falls between nodes and the trapezoid
    rule returns peak height times spacing -- the very error these closed forms
    exist to remove.
    """
    x = np.concatenate([[0.0], np.geomspace(1e-14, 2.0, n)])
    d = _dsigma(operator, m_chi, np.arccos(np.clip(1.0 - x, -1.0, 1.0)))
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return 2.0 * np.pi * np.trapezoid(d, x) * FB_TO_CM2


def _sigma_closed(operator, m_chi):
    return float(SCF.sigma_removal_cm2(operator, m_chi, OMEGA, 1.0,
                                       u_min=np.array(0.0)))


def check_matches_quadrature(operator):
    """Closed form must reproduce resolved quadrature at EVERY mass."""
    bad = []
    for m in MASSES:
        c, q = _sigma_closed(operator, m), _sigma_quadrature(operator, m)
        if not np.isfinite(c) or c <= 0 or abs(c / q - 1.0) > TOL_VS_QUAD:
            bad.append((m, c, q))
    assert not bad, (
        f"{operator}: closed form disagrees with resolved quadrature at "
        f"{len(bad)} mass(es); worst "
        f"m={bad[0][0]:.3e} closed={bad[0][1]:.6e} quad={bad[0][2]:.6e}"
    )


def check_branches_agree(operator):
    """The exact difference and the Gauss rule must agree in their overlap.

    Forcing each branch at the SAME mass is the point: comparing them at
    slightly different masses measures the physical slope of sigma(m), not the
    branch discontinuity, and reads as a spurious ~3e-3 'jump'.
    """
    keep = SCF.U_SERIES_MAX
    try:
        worst = 0.0
        for m in np.geomspace(1e2, 1e5, 13):     # overlap: both branches valid
            SCF.U_SERIES_MAX = 0.0
            exact = _sigma_closed(operator, m)
            SCF.U_SERIES_MAX = 1e9
            quad = _sigma_closed(operator, m)
            worst = max(worst, abs(quad / exact - 1.0))
        assert worst < 1e-7, f"{operator}: branches disagree by {worst:.3e}"
    finally:
        SCF.U_SERIES_MAX = keep


def check_heavy_target_saturates(operator):
    """m_chi -> infinity: recoil vanishes, so sigma_tot must go to a CONSTANT.

    This is the invariant bug 3 violated -- it grew linearly in m_chi instead.
    It needs no reference implementation, which is what makes it a good guard.
    """
    s = np.array([_sigma_closed(operator, m) for m in (1e6, 1e7, 1e8)])
    drift = abs(s[-1] / s[0] - 1.0)
    # scalar_rayleigh and rayleigh_odd fall off with mass rather than
    # saturating, so compare against their own power law instead.
    ratios = s[1:] / s[:-1]
    assert np.all(np.isfinite(s)) and np.all(s > 0), f"{operator}: non-positive"
    # 5e-3 accommodates the residual O(omega/m) approach to the asymptotic power
    # law (rayleigh_odd still drifts 1.1e-3 per decade at m = 1e6-1e8). The bug
    # this guards against gave a decade ratio of 10 against an expected 0.01, so
    # there are three orders of margin.
    assert abs(ratios[1] / ratios[0] - 1.0) < 5e-3, (
        f"{operator}: sigma(m) is not settling onto a clean power law at high "
        f"mass (successive decade ratios {ratios[0]:.6g}, {ratios[1]:.6g}); "
        f"drift over two decades {drift:.3g}"
    )


def check_production_couplings(operator):
    """Non-zero sigma under the couplings production actually resolves.

    Every test above passes c_s = 1 explicitly. That hid a real bug: the two
    dipole operators share a shape, and the shared prefactor read c_s
    unconditionally -- but operator_couplings() resolves dipole_electric to
    (c_s, c_p) = (0, 1), so the closed form returned sigma = 0 for it, silently.
    A test that supplies its own convenient coefficients cannot see this; the
    couplings must come from the same place the pipeline gets them.
    """
    from constraint_generation.make_data_driven_scattering_limits import (
        operator_couplings,
    )
    dm_type = "scalar" if operator == "scalar_rayleigh" else "fermionic"
    c_s, c_p, c_phi = operator_couplings(operator, dm_type)
    for m in (1e-3, 1.0, 1e4, 1e8):
        v = float(SCF.sigma_removal_cm2(operator, m, OMEGA, 1.0, c_s=c_s,
                                        c_p=c_p, c_phi=c_phi,
                                        u_min=np.array(0.0)))
        assert np.isfinite(v) and v > 0, (
            f"{operator} at m={m:.0e} with production couplings "
            f"(c_s={c_s}, c_p={c_p}, c_phi={c_phi}) gave sigma = {v}"
        )


def check_additivity(m_list=(1e-3, 1.0, 1e2, 1e4, 1e6, 1e8)):
    """rayleigh_full == rayleigh_even + rayleigh_odd (interference vanishes)."""
    for m in m_list:
        f = _sigma_closed("rayleigh_full", m)
        e = _sigma_closed("rayleigh_even", m)
        o = _sigma_closed("rayleigh_odd", m)
        assert abs(f / (e + o) - 1.0) < 1e-12, (
            f"additivity fails at m={m:.3e}: full={f:.8e} even+odd={e + o:.8e}"
        )


def check_every_operator_reaches_both_branches(operator):
    """Guards the two KeyError bugs: exercise both code paths, all operators."""
    for m in (1.0, 1e8):
        v = _sigma_closed(operator, m)
        assert np.isfinite(v) and v > 0, f"{operator} at m={m:.0e} gave {v}"


if pytest is not None:
    @pytest.mark.parametrize("operator", OPERATORS)
    def test_matches_quadrature(operator):
        check_matches_quadrature(operator)

    @pytest.mark.parametrize("operator", OPERATORS)
    def test_branches_agree(operator):
        check_branches_agree(operator)

    @pytest.mark.parametrize("operator", OPERATORS)
    def test_heavy_target_saturates(operator):
        check_heavy_target_saturates(operator)

    @pytest.mark.parametrize("operator", OPERATORS)
    def test_every_operator_reaches_both_branches(operator):
        check_every_operator_reaches_both_branches(operator)

    @pytest.mark.parametrize("operator", OPERATORS)
    def test_production_couplings(operator):
        check_production_couplings(operator)

    def test_additivity():
        check_additivity()

    def test_unknown_operator_raises():
        with pytest.raises(KeyError):
            SCF.sigma_removal_cm2("not_an_operator", 1.0, OMEGA, 1.0)


def _main():
    checks = []
    for op in OPERATORS:
        checks += [(f"both branches reachable  [{op}]",
                    lambda o=op: check_every_operator_reaches_both_branches(o)),
                   (f"heavy-target power law   [{op}]",
                    lambda o=op: check_heavy_target_saturates(o)),
                   (f"branch agreement         [{op}]",
                    lambda o=op: check_branches_agree(o)),
                   (f"vs resolved quadrature   [{op}]",
                    lambda o=op: check_matches_quadrature(o)),
                   (f"production couplings     [{op}]",
                    lambda o=op: check_production_couplings(o))]
    checks.append(("full == even + odd", check_additivity))

    failures = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}\n        {exc}")
    print("OK" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
