"""
Null-limit guards on the tau / K normalisation of the single-scatter transfer.

THE LIMIT
---------
As m_chi -> infinity the DM is an infinitely heavy, static target. Recoil
vanishes (omega'/omega = 1/(1+u) with u = 2 omega/m_chi -> 0), so no photon can
change energy bin and the redistribution kernel must become the IDENTITY. The
transfer then collapses to something exactly calculable:

    Phi_obs = Phi e^-tau + K (tau Phi e^-tau)  ->  Phi e^-tau (1 + tau)

and, expanding, the surviving distortion is second order:

    1 - e^-tau (1 + tau) = tau^2/2 - tau^3/3 + ...

WHY THIS IS A GOOD GUARD
------------------------
It needs no reference implementation and no tolerance tuning, and it is
sensitive to BOTH factors of the transfer at once:

  * a mis-normalised K shows up immediately, since a kernel whose columns do
    not sum to one cannot reach the identity limit;
  * a wrong tau shows up in the e^-tau (1 + tau) comparison, which pins the
    relative normalisation of the two terms.

The first-order cancellation is the physical content: in the static limit every
photon removed from a bin is scattered straight back into it, so the O(tau)
terms cancel exactly and only O(tau^2) survives. Any error in the relative
normalisation of tau and K breaks that cancellation and shows up at first
order, i.e. amplified by 2/tau relative to the true signal.

NUMERICAL WINDOW -- DO NOT "TIGHTEN" THIS
-----------------------------------------
The tau^2/2 check must be run where tau^2 is comfortably above double
precision. It is computed as 1 - (e^-tau + tau e^-tau), so once tau^2/2 falls
below ~1e-16 the subtraction has no significant digits left and the agreement
DEGRADES as tau gets smaller -- the opposite of the usual convergence
intuition. Measured, at m_chi = 1e8:

    tau_max      3.6e-4    4.4e-6    3.6e-8    4.4e-10
    rel. error   1.3e-3    2.3e+0    1.0e+0    1.0e+0

So the window is chosen for tau ~ 1e-4 to 1e-3, where the leading neglected
term (tau^3/3, i.e. a relative error of about 2 tau/3) still dominates the
roundoff. Pushing to smaller tau tests floating point, not physics.

Run:  python -m pytest tests/test_transfer_null_limits.py -q
      python tests/test_transfer_null_limits.py            (standalone)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.spectral_reshaping import (                                # noqa: E402
    ReshapingConfig,
    build_kernel,
    compute_tau_spectrum,
)

try:
    import pytest
except ImportError:
    pytest = None

E_FIT = np.array([4.308, 7.276, 12.289, 20.757, 35.059, 59.216, 100.017, 168.932])
_ONES = np.ones_like(E_FIT)

# Masses at which recoil is negligible across the whole band.
HEAVY = (1e4, 1e5, 1e6, 1e8, 1e10)
OPERATORS = ["dipole_magnetic", "scalar_rayleigh", "rayleigh_full"]


def _coeffs(operator):
    if operator == "scalar_rayleigh":
        return dict(dm_type="scalar", c_s=0.0, c_p=0.0, c_phi=1.0)
    if operator == "rayleigh_full":
        return dict(dm_type="fermionic", c_s=1.0, c_p=1.0, c_phi=1.0)
    return dict(dm_type="fermionic", c_s=1.0, c_p=0.0, c_phi=1.0)


def _cfg(operator, m_chi, Lambda):
    # apply_roi_weight=False is essential: with a recovery weight the kernel
    # tends to <w> * identity, not identity, and the null limit would test the
    # ROI model rather than the tau/K normalisation this guard is for.
    return ReshapingConfig(
        m_chi=float(m_chi), Lambda=float(Lambda), operator=operator,
        E_bins=E_FIT.copy(), phi_0=_ONES.copy(), phi_data=_ONES.copy(),
        phi_err=_ONES.copy(), apply_roi_weight=False,
        **_coeffs(operator))


def check_kernel_becomes_identity(operator):
    for m in HEAVY:
        K = build_kernel(_cfg(operator, m, 1.0))
        off = K - np.diag(np.diag(K))
        assert np.max(np.abs(np.diag(K) - 1.0)) < 1e-9, (
            f"{operator} at m_chi={m:.0e}: kernel diagonal is "
            f"{np.min(np.diag(K)):.9f}, not 1 -- recoil has vanished, so every "
            f"photon must stay in its own bin."
        )
        assert np.max(np.abs(off)) < 1e-9, (
            f"{operator} at m_chi={m:.0e}: off-diagonal reaches "
            f"{np.max(np.abs(off)):.3e}; a static target cannot move photons "
            f"between energy bins."
        )


def check_transfer_collapses(operator):
    """With K = identity the transfer must equal Phi e^-tau (1 + tau) exactly."""
    for m in HEAVY:
        cfg = _cfg(operator, m, 0.1)
        K = build_kernel(cfg)
        tau = np.atleast_1d(compute_tau_spectrum(cfg, arm="reshaping"))
        obs = np.exp(-tau) + K @ (tau * np.exp(-tau))
        pred = np.exp(-tau) * (1.0 + tau)
        assert np.max(np.abs(obs - pred)) < 1e-12, (
            f"{operator} at m_chi={m:.0e}: transfer gives "
            f"{obs} but the identity-kernel limit requires {pred}. This pins "
            f"the RELATIVE normalisation of tau and K."
        )


def check_second_order_floor(operator):
    """Net distortion -> tau^2/2. See the module docstring on the window."""
    # Lambda must be SEARCHED, not guessed: tau scales as Lambda^-4 for the
    # dim-5/6 operators and Lambda^-6 for the dim-7 ones, and sigma itself
    # differs by twelve orders of magnitude between them at m_chi = 1e8. A
    # fixed list of Lambdas lands in the window for the dipole and misses it
    # entirely for the Rayleigh family.
    for m in (1e6, 1e8):
        for Lambda in np.geomspace(1e-4, 1e2, 61):
            cfg = _cfg(operator, m, Lambda)
            tau = np.atleast_1d(compute_tau_spectrum(cfg, arm="reshaping"))
            if not (1e-5 < np.max(tau) < 1e-2):
                continue
            K = build_kernel(cfg)
            net = 1.0 - (np.exp(-tau) + K @ (tau * np.exp(-tau)))
            floor = tau ** 2 / 2.0
            # The window is PER BIN, not on tau_max. tau(E) spans several
            # decades across the band for the Rayleigh operators, so the
            # low-tau bins underflow to net == 0 exactly while the max is still
            # comfortably resolvable -- and those bins would then report a
            # spurious 100% deviation.
            ok = floor > 1e-12
            if not ok.any():
                continue
            rel = np.max(np.abs(net[ok] / floor[ok] - 1.0))
            # leading neglected term is tau^3/3 -> relative error ~ 2 tau/3
            assert rel < 1e-2, (
                f"{operator} m={m:.0e} Lambda={Lambda}: net "
                f"distortion departs from tau^2/2 by {rel:.3e}; the O(tau) "
                f"terms are not cancelling, which means tau and K are not "
                f"consistently normalised."
            )
            return
    raise AssertionError(f"{operator}: no configuration landed in the window")


def check_tau_vanishes_with_mass(operator):
    """tau ~ sigma/m_chi, and sigma saturates, so tau must fall as 1/m_chi."""
    taus = []
    for m in (1e6, 1e7, 1e8):
        cfg = _cfg(operator, m, 1.0)
        taus.append(float(np.max(compute_tau_spectrum(cfg, arm="reshaping"))))
    for a, b in zip(taus[:-1], taus[1:]):
        assert b < a, f"{operator}: tau is not decreasing with mass: {taus}"


if pytest is not None:
    @pytest.mark.parametrize("operator", OPERATORS)
    def test_kernel_becomes_identity(operator):
        check_kernel_becomes_identity(operator)

    @pytest.mark.parametrize("operator", OPERATORS)
    def test_transfer_collapses(operator):
        check_transfer_collapses(operator)

    @pytest.mark.parametrize("operator", OPERATORS)
    def test_second_order_floor(operator):
        check_second_order_floor(operator)

    @pytest.mark.parametrize("operator", OPERATORS)
    def test_tau_vanishes_with_mass(operator):
        check_tau_vanishes_with_mass(operator)


def _main():
    checks = []
    for op in OPERATORS:
        checks += [
            (f"K -> identity           [{op}]", lambda o=op: check_kernel_becomes_identity(o)),
            (f"transfer -> e^-t(1+t)   [{op}]", lambda o=op: check_transfer_collapses(o)),
            (f"net distortion -> t^2/2 [{op}]", lambda o=op: check_second_order_floor(o)),
            (f"tau falls with mass     [{op}]", lambda o=op: check_tau_vanishes_with_mass(o)),
        ]
    failures = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}\n        {str(exc)[:160]}")
    print("OK" if not failures else f"{failures} unexpected failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
