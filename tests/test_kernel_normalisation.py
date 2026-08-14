"""
Normalisation invariants for the redistribution kernel.

THE INVARIANT
-------------
K[i, j] is the probability that a photon born in energy bin j is observed in
bin i after one scatter. Probabilities cannot exceed one, so

    sum_i K[i, j] <= 1   for every column j

with the deficit accounted for by flux scattered below the lowest bin edge or
lost from the ROI. This holds regardless of operator, mass, or angular model,
which is what makes it a good guard: it needs no reference implementation and
no tolerance tuning.

WHY IT BITES HERE
-----------------
The forward peak of the differential cross section has angular width
~ m_chi/omega, so at low mass essentially the whole cross section sits in a
region no coarse angular grid resolves. Any kernel whose numerator and
denominator are integrated with different quadrature rules therefore fails this
invariant badly at exactly the masses the scan cares about, while looking fine
at m_chi ~ 1 TeV. Two symptoms are checked: column sums above one, and a
diagonal that collapses to zero at low mass (photons that stay in their own
energy bin are precisely the near-forward ones).

The exact-interval kernel of core/kinematics.py satisfies both to machine
precision, and closure (column sum + leak below the band == 1) is checked
exactly.

Run:  python -m pytest tests/test_kernel_normalisation.py -q
      python tests/test_kernel_normalisation.py            (standalone)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.attenuation_eft import (                                  # noqa: E402
    FB_TO_CM2,
    dsigma_dOmega_fermionic,
    dsigma_dOmega_scalar,
)
from core.kinematics import build_redistribution_matrix_exact       # noqa: E402
from core.spectral_reshaping import ReshapingConfig, build_kernel   # noqa: E402

try:
    import pytest
except ImportError:
    pytest = None

E_FIT = np.array([4.308, 7.276, 12.289, 20.757, 35.059, 59.216, 100.017, 168.932])
MASSES = np.geomspace(1e-4, 1e4, 17)
OPERATORS = ["dipole_magnetic", "scalar_rayleigh", "rayleigh_full"]

# Column sums may sit fractionally above 1 only by quadrature noise.
COLSUM_TOL = 1e-9


def _coeffs(operator):
    if operator == "scalar_rayleigh":
        return dict(dm_type="scalar", c_s=1.0, c_p=0.0, c_phi=1.0)
    if operator == "rayleigh_full":
        return dict(dm_type="fermionic", c_s=1.0, c_p=1.0, c_phi=1.0)
    if operator in ("dipole_magnetic", "rayleigh_even"):
        return dict(dm_type="fermionic", c_s=1.0, c_p=0.0, c_phi=1.0)
    return dict(dm_type="fermionic", c_s=0.0, c_p=1.0, c_phi=1.0)


def _dsigma_fn(operator, m_chi):
    cc = _coeffs(operator)

    def f(E, theta):
        if operator == "scalar_rayleigh":
            d = dsigma_dOmega_scalar(m_chi, theta, E, cc["c_phi"], 1.0)
        else:
            d = dsigma_dOmega_fermionic(m_chi, theta, E, cc["c_s"], cc["c_p"],
                                        1.0, operator=operator)
        return np.asarray(d, dtype=float) * FB_TO_CM2
    return f


def _kernel(operator, m_chi, **kw):
    cfg_kw = dict(m_chi=float(m_chi), Lambda=1.0, operator=operator,
                  E_bins=E_FIT.copy(), phi_0=np.ones_like(E_FIT),
                  phi_data=np.ones_like(E_FIT), phi_err=np.ones_like(E_FIT))
    cfg_kw.update(_coeffs(operator))
    cfg_kw.update(kw)
    return build_kernel(ReshapingConfig(**cfg_kw))


def check_column_sums_le_one(operator):
    bad = []
    for m in MASSES:
        K = _kernel(operator, m)
        s = K.sum(axis=0)
        if np.max(s) > 1.0 + COLSUM_TOL:
            bad.append((m, float(np.max(s))))
    assert not bad, (
        f"{operator}: column sum exceeds 1 at {len(bad)} mass(es) -- K is not "
        f"a probability. Worst m={bad[0][0]:.3e} sum={bad[0][1]:.6f}."
    )


def check_closure_exact(operator):
    """sum_i K + leak_below == 1 exactly, with no ROI weight."""
    for m in MASSES:
        _, d = build_redistribution_matrix_exact(
            E_FIT, float(m), _dsigma_fn(operator, float(m)),
            return_diagnostics=True,
        )
        closure = d["column_sum"] + d["leak_below"]
        worst = float(np.max(np.abs(closure - 1.0)))
        assert worst < 1e-9, (
            f"{operator} at m={m:.3e}: closure violated by {worst:.3e}"
        )


def check_diagonal_nonzero_at_low_mass(operator):
    """Photons scattering through a small angle stay in their own bin.

    A kernel whose diagonal collapses to zero at low mass would impose a
    spurious systematic downshift of the whole spectrum, which a shape
    statistic cannot absorb.
    """
    K = _kernel(operator, 1e-3)
    assert np.max(np.diag(K)) > 0.0, (
        f"{operator}: kernel diagonal is identically zero at m_chi = 1e-3 GeV; "
        f"every photon is moved to a lower bin."
    )


if pytest is not None:
    @pytest.mark.parametrize("operator", OPERATORS)
    def test_column_sums_exact_u(operator):
        check_column_sums_le_one(operator)

    @pytest.mark.parametrize("operator", OPERATORS)
    def test_closure_exact_u(operator):
        check_closure_exact(operator)

    @pytest.mark.parametrize("operator", OPERATORS)
    def test_diagonal_exact_u(operator):
        check_diagonal_nonzero_at_low_mass(operator)


def _main():
    failures = 0
    print(f"{'operator':17s}{'check':28s}{'result':>12}")
    checks = (
        ("column sums <= 1", check_column_sums_le_one),
        ("diagonal nonzero @1e-3", check_diagonal_nonzero_at_low_mass),
        ("closure sum+leak == 1", check_closure_exact),
    )
    for op in OPERATORS:
        for label, fn in checks:
            try:
                fn(op)
                v = "pass"
            except AssertionError as exc:
                v, failures = f"FAIL {exc}", failures + 1
            print(f"{op:17s}{label:28s}{v:>12}")

    print("OK" if not failures else f"{failures} unexpected failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
