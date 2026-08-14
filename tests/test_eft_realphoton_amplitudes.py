"""
Regression tests for the gamma chi -> gamma chi real-photon amplitudes in
core/attenuation_eft.py.

Guards against reintroducing the three transcribed direct-detection cross
sections (the "4*ALPHA_EM" bug):
  1. anapole / charge radius must be exactly zero for real photons;
  2. dipole must scale as Lambda^-4 (two insertions), with NO factor of alpha;
  3. dipole low-energy limit must be sigma -> 4 mu^4 omega^2 / (3 pi), mu = 2c/Lambda;
  4. the lab prefactor + averaged-|M|^2 convention must reproduce Klein-Nishina
     for QED Compton (certifies the convention the dipole expression assumes);
  5. Rayleigh / scalar branches unchanged (frozen reference values).

Run:  python -m pytest tests/test_eft_realphoton_amplitudes.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# None of the amplitude physics tested here depends on the halo posterior, so
# on machines without it we point the loader at synthetic placeholder files via
# the TOTANI_MCMC_DIR override.
if not os.environ.get("TOTANI_MCMC_DIR"):
    _tmp = Path(tempfile.mkdtemp(prefix="totani_stub_"))
    for k in range(13):
        np.savez(_tmp / f"mcmc_results_k{k:02d}.npz",
                 Ectr_mev=np.array(1000.0 * (k + 1)),
                 iso_target_e2=np.array(1e-4),
                 labels=np.array(["nfw"]),
                 f_p50=np.array([1.0]), f_p16=np.array([0.9]),
                 f_p84=np.array([1.1]))
    os.environ["TOTANI_MCMC_DIR"] = str(_tmp)

from core.attenuation_eft import (
    dsigma_dOmega_fermionic,
    dsigma_dOmega_scalar,
    sigma_tot_fermionic,
    lab_dsigma_prefactor,
    OPERATOR_METADATA,
    GEV2_TO_FB,
    FB_TO_CM2,
)
from core.cross_sections import lab_recoil_ratio, get_t_lab_DMrest


THETAS = np.linspace(0.05, np.pi - 0.05, 40)


# ---------------------------------------------------------------------------
# 1. anapole & charge radius vanish for on-shell photons
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("operator", ["anapole", "charge_radius"])
def test_qsq_suppressed_operators_vanish(operator):
    """d^nu F_nu_mu -> -(q^2 eps_mu - q_mu q.eps) = 0 at q^2=0, q.eps=0."""
    val = dsigma_dOmega_fermionic(
        10.0, THETAS, 5.0, c_s=1.0, c_p=1.0, Lambda=100.0, operator=operator
    )
    assert np.all(val == 0.0), f"{operator} must vanish for real photons"


def test_transverse_projector_vanishes_onshell():
    """Direct vertex-level check: -(q^2 eps - q (q.eps)) == 0 for a real photon."""
    q = np.array([3.0, 0.0, 0.0, 3.0])          # on-shell: q^2 = 0
    eps1 = np.array([0.0, 1.0, 0.0, 0.0])
    eps2 = np.array([0.0, 0.0, 1.0, 0.0])
    metric = np.diag([1.0, -1.0, -1.0, -1.0])
    for eps in (eps1, eps2):
        q2 = q @ metric @ q
        qe = q @ metric @ eps
        vertex = -(q2 * eps - q * qe)
        assert np.allclose(vertex, 0.0)


# ---------------------------------------------------------------------------
# 2. dipole: Lambda^-4 scaling, no alpha
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("operator", ["dipole_magnetic", "dipole_electric"])
def test_dipole_lambda_scaling(operator):
    kw = dict(c_s=1.0, c_p=1.0)
    v1 = dsigma_dOmega_fermionic(10.0, THETAS, 5.0, Lambda=50.0, operator=operator, **kw)
    v2 = dsigma_dOmega_fermionic(10.0, THETAS, 5.0, Lambda=100.0, operator=operator, **kw)
    np.testing.assert_allclose(v1 / v2, (100.0 / 50.0) ** 4, rtol=1e-12)


@pytest.mark.parametrize("operator", ["dipole_magnetic", "dipole_electric"])
def test_dipole_coefficient_scaling(operator):
    """|M|^2 ~ c^4 (two insertions), not c^2 (one insertion)."""
    v1 = dsigma_dOmega_fermionic(10.0, THETAS, 5.0, 1.0, 1.0, 100.0, operator=operator)
    v2 = dsigma_dOmega_fermionic(10.0, THETAS, 5.0, 2.0, 2.0, 100.0, operator=operator)
    np.testing.assert_allclose(v2 / v1, 2.0 ** 4, rtol=1e-12)


def test_dipole_magnetic_equals_electric():
    """Spin-averaged real-photon Compton is identical for M and E dipoles (duality)."""
    vm = dsigma_dOmega_fermionic(3.0, THETAS, 1.0, 1.3, 0.0, 40.0, "dipole_magnetic")
    ve = dsigma_dOmega_fermionic(3.0, THETAS, 1.0, 0.0, 1.3, 40.0, "dipole_electric")
    np.testing.assert_allclose(vm, ve, rtol=1e-12)


def test_dipole_no_alpha():
    """The exact low-E angular form fixes the overall constant: any stray alpha
    (1/137) or missing O(1) factor fails this test.
    dsigma/dOmega -> mu^4 omega^2 (2 + sin^2 theta) / (8 pi^2), mu = 2 c/Lambda."""
    m, w, c, L = 100.0, 1e-3, 1.0, 10.0     # omega << m: deep low-energy limit
    mu = 2.0 * c / L
    got = dsigma_dOmega_fermionic(m, THETAS, w, c, 0.0, L, "dipole_magnetic")
    expected_gev2 = mu**4 * w**2 * (2.0 + np.sin(THETAS) ** 2) / (8.0 * np.pi**2)
    np.testing.assert_allclose(got, expected_gev2 * GEV2_TO_FB, rtol=1e-4)


def test_dipole_low_energy_total_cross_section():
    """sigma_tot -> 4 mu^4 omega^2 / (3 pi) as omega -> 0."""
    m, w, c, L = 100.0, 1e-3, 1.0, 10.0
    mu = 2.0 * c / L
    sig = sigma_tot_fermionic(w, m, c, 0.0, L, operator="dipole_magnetic")  # cm^2
    expected = 4.0 * mu**4 * w**2 / (3.0 * np.pi) * GEV2_TO_FB * FB_TO_CM2
    np.testing.assert_allclose(sig, expected, rtol=1e-3)


def test_dipole_positive_and_finite_relativistic():
    """omega ~ m and omega >> m must stay positive and finite."""
    for w in (0.5, 5.0, 500.0):
        v = dsigma_dOmega_fermionic(5.0, THETAS, w, 1.0, 0.0, 100.0, "dipole_magnetic")
        assert np.all(np.isfinite(v)) and np.all(v > 0.0)


# ---------------------------------------------------------------------------
# 3. convention check: prefactor + averaged |M|^2 == Klein-Nishina for QED
# ---------------------------------------------------------------------------
def test_prefactor_convention_reproduces_klein_nishina():
    """<|M|^2>_QED = 2 e^4 (w'/w + w/w' - sin^2 th);  KN dsigma/dOmega =
    (alpha^2/2 m^2)(w'/w)^2 (w'/w + w/w' - sin^2 th).  This certifies that
    lab_dsigma_prefactor expects the INITIAL-AVERAGED |M|^2 -- the convention
    used by the corrected dipole branch."""
    alpha = 1.0 / 137.035999084
    e2 = 4.0 * np.pi * alpha
    m, w = 0.000511, 0.001
    ratio = lab_recoil_ratio(m, w, THETAS)
    wp = w * ratio
    M2_avg = 2.0 * e2**2 * (wp / w + w / wp - np.sin(THETAS) ** 2)
    got = lab_dsigma_prefactor(m, w, THETAS) * M2_avg
    kn = (alpha**2 / (2.0 * m**2)) * ratio**2 * (wp / w + w / wp - np.sin(THETAS) ** 2)
    np.testing.assert_allclose(got, kn, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. dipole matches the exact invariant-form |M|^2 (independent implementation)
# ---------------------------------------------------------------------------
def test_dipole_matches_invariant_form():
    """<|M|^2> = 4 mu^4 [-ab - 2 m^2 t + 2 m^4 t^2/(ab)], a=2 m w, b=-2 m w'.
    Derived symbolically (exact rational kinematics, trace machinery checked
    against Klein-Nishina)."""
    m, w, c, L = 7.0, 3.0, 1.7, 55.0
    mu = 2.0 * c / L
    t = get_t_lab_DMrest(m, w, THETAS)
    wp = w * lab_recoil_ratio(m, w, THETAS)
    a, b = 2.0 * m * w, -2.0 * m * wp
    M2 = 4.0 * mu**4 * (-a * b - 2.0 * m**2 * t + 2.0 * m**4 * t**2 / (a * b))
    expected = lab_dsigma_prefactor(m, w, THETAS) * M2 * GEV2_TO_FB
    got = dsigma_dOmega_fermionic(m, THETAS, w, c, 0.0, L, "dipole_magnetic")
    np.testing.assert_allclose(got, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. Rayleigh / scalar branches frozen (unchanged by the correction)
# ---------------------------------------------------------------------------
def test_rayleigh_and_scalar_frozen():
    m, w, L = 10.0, 5.0, 100.0
    t = get_t_lab_DMrest(m, w, THETAS)
    phase = lab_dsigma_prefactor(m, w, THETAS)
    np.testing.assert_allclose(
        dsigma_dOmega_fermionic(m, THETAS, w, 1.0, 0.0, L, "rayleigh_even"),
        (4 * m**2 - t) * t**2 / (4.0 * L**6) * phase * GEV2_TO_FB, rtol=1e-12)
    np.testing.assert_allclose(
        dsigma_dOmega_fermionic(m, THETAS, w, 0.0, 1.0, L, "rayleigh_odd"),
        (-t) ** 3 / (4.0 * L**6) * phase * GEV2_TO_FB, rtol=1e-12)
    # Scalar Rayleigh: 16 c^2 t^2 / L^4 (Barducci et al. normalisation, no 1/4).
    # Corrected from t^2/(4 L^4), which was 64x too small; see
    # test_scalar_rayleigh_barducci_normalisation below.
    np.testing.assert_allclose(
        dsigma_dOmega_scalar(m, THETAS, w, 1.0, L),
        16.0 * t**2 / L**4 * phase * GEV2_TO_FB, rtol=1e-12)


# ---------------------------------------------------------------------------
# 6. metadata consistency
# ---------------------------------------------------------------------------
def test_metadata_lambda_powers():
    assert OPERATOR_METADATA["dipole_magnetic"]["lambda_power"] == 4
    assert OPERATOR_METADATA["dipole_electric"]["lambda_power"] == 4
    assert OPERATOR_METADATA["dipole_magnetic"]["coefficient_power"] == 1.0


def test_majorana_guard_still_raises():
    with pytest.raises(ValueError):
        dsigma_dOmega_fermionic(10.0, THETAS, 5.0, 1.0, 1.0, 100.0,
                                "dipole_magnetic", majorana=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# 7. Weiner-Yavin normalisation anchor (resolves the factor-64 question)
# ---------------------------------------------------------------------------
def test_weiner_yavin_annihilation_anchor():
    """Pin the Rayleigh operator normalisation to Weiner & Yavin 1206.2910,
    L = (c/4 Lambda^3)[chibar chi FF + chibar i g5 chi F Ftilde]:
    naive single-contraction rules in that convention reproduce their Eq. (15),
    sigma v(chi chi -> gamma gamma) = g^2 m^4/(4 pi Lambda^6), exactly.
    Under the same convention the coded elastic Rayleigh branches are exactly
    the initial-averaged |M|^2 (the historical 'factor 64' = 16 from (1/4)^2
    times 4 from averaging).  Numeric (float) version of the exact sympy check
    in the verification record."""
    m = 3.0
    # threshold kinematics: chi chi at rest -> back-to-back photons along z
    k1 = np.array([m, 0.0, 0.0, m]); k2 = np.array([m, 0.0, 0.0, -m])
    metric = np.diag([1.0, -1.0, -1.0, -1.0])
    id2 = np.eye(2)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    z2 = np.zeros((2, 2), dtype=complex)
    g0 = np.block([[id2, z2], [z2, -id2]])
    g5 = np.block([[z2, id2], [id2, z2]])
    s2m = np.sqrt(2 * m)
    us = [s2m * np.eye(4, dtype=complex)[:, i] for i in (0, 1)]
    vs = [s2m * np.eye(4, dtype=complex)[:, i] for i in (2, 3)]
    eps = (np.array([0, 1, 0, 0.0]), np.array([0, 0, 1, 0.0]))

    def fmn(k, e):
        return 1j * (np.outer(k, e) - np.outer(e, k))

    lev = np.zeros((4, 4, 4, 4))
    from itertools import permutations
    for p in permutations(range(4)):
        sgn, pl = 1, list(p)
        for i in range(4):
            for j in range(i + 1, 4):
                if pl[i] > pl[j]:
                    sgn = -sgn
        lev[p] = sgn

    def dual(F):
        Fl = metric @ F @ metric
        return 0.5 * np.einsum("abcd,cd->ab", lev, Fl)

    def ddot(F, G):
        return np.einsum("ab,ab->", metric @ F @ metric, G)

    tot = 0.0
    for e1 in eps:
        for e2 in eps:
            F1, F2 = fmn(k1, e1), fmn(k2, e2)
            X = ddot(F1, dual(F2)) + ddot(F2, dual(F1))
            for u in us:
                for v in vs:
                    bil = np.conj(v) @ g0 @ (1j * g5) @ u
                    tot += abs(0.25 * X * bil) ** 2
    sv = (tot / 4.0) / (64.0 * np.pi * m**2)   # avg spins; 1/2 identical photons
    np.testing.assert_allclose(sv, m**4 / (4.0 * np.pi), rtol=1e-12)


def test_scalar_rayleigh_barducci_normalisation():
    """Real-scalar Rayleigh |M|^2 must equal 16 c_phi^2 t^2 / Lambda^4.

    The operator is O = (c_phi/Lambda^2) phi^2 F_mu_nu F^mu_nu with NO factor
    of 1/4, following Barducci et al. arXiv:2501.09073 (Eqs. 2.1/2.3/2.5), and
    matching main.tex.  This differs from the fermionic Rayleigh branch, which
    keeps the 1/4 of Weiner & Yavin so that Lambda matches their Lambda_R.

    Guards against reverting to the appendix value c_phi^2 t^2 / (4 Lambda^4),
    which is 64x too small and weakens the scalar Lambda limit by 64^(1/4).
    """
    from core.attenuation_eft import (
        dsigma_dOmega_scalar, get_t_lab_DMrest, lab_dsigma_prefactor,
        GEV2_TO_FB,
    )
    mchi, E_gamma, Lambda, c_phi = 1.0, 0.5, 10.0, 1.0
    theta = np.linspace(0.15, np.pi - 0.15, 11)

    t = get_t_lab_DMrest(mchi, E_gamma, theta)
    phase = lab_dsigma_prefactor(mchi, E_gamma, theta)
    expected = 16.0 * c_phi**2 * t**2 / Lambda**4 * phase * GEV2_TO_FB

    got = dsigma_dOmega_scalar(mchi, theta, E_gamma, c_phi, Lambda)
    assert np.allclose(got, expected, rtol=1e-12), "scalar Rayleigh normalisation changed"

    # Quadratic in the Wilson coefficient, quartic in 1/Lambda (dimension 6).
    got_2c = dsigma_dOmega_scalar(mchi, theta, E_gamma, 2.0 * c_phi, Lambda)
    assert np.allclose(got_2c, 4.0 * got, rtol=1e-12)
    got_2L = dsigma_dOmega_scalar(mchi, theta, E_gamma, c_phi, 2.0 * Lambda)
    assert np.allclose(got_2L, got / 16.0, rtol=1e-12)
