"""
Tests for the EFT-validity wedge of Eq. (IV.18),

    Lambda^2 >= max(s_max, |t|_max),   s_max = m^2 + 2 m omega_max,
                                       |t|_max = 4 omega_max^2/(1 + 2 omega_max/m)

which is the cyan boundary drawn in Figs. 2-4 and the criterion behind every
"below the wedge" statement in the paper.

Two properties are pinned:

  * scale covariance. Lambda(m) is a physical scale, so expressing the same
    mass in a different unit and converting back must reproduce it:
    Lambda(m [GeV]) == Lambda(1000*m [MeV]) / 1000.
  * per-key dispatch. Operator lookup is an explicit table, not a substring
    match, so an unknown key raises instead of silently inheriting another
    operator's Lambda power.

Run:  python -m pytest tests/test_eft_validity.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.eft_validity import (  # noqa: E402
    EFT_OPERATOR_VALIDITY,
    eft_kinematic_lambda_curve,
    lambda_min_curve,
    normalise_operator_key,
)

try:
    import pytest
except ImportError:                                   # standalone fallback
    pytest = None

# Physical masses to probe, in GeV, and the paper's photon-energy ceiling.
M_GEV = np.array([1e-5, 1e-3, 1e-1, 1.0, 1e2])
OMEGA_MAX = 168.93225
TOL = 1e-12

OPERATORS = [k for k in EFT_OPERATOR_VALIDITY if k != "higgs_portal"]


def _scale_covariance_residual(m_gev, unit_factor=1.0e3):
    """Max relative violation of Lambda(m[GeV]) == Lambda(m[MeV])/unit_factor."""
    m = np.asarray(m_gev, dtype=float)
    a = eft_kinematic_lambda_curve(m, omega_max=OMEGA_MAX)
    b = eft_kinematic_lambda_curve(
        m * unit_factor, omega_max=OMEGA_MAX * unit_factor) / unit_factor
    good = np.isfinite(a) & np.isfinite(b) & (a != 0.0)
    return float(np.max(np.abs(b[good] - a[good]) / np.abs(a[good])))


def check_scale_covariance():
    resid = _scale_covariance_residual(M_GEV)
    assert resid < TOL, (
        f"kinematic wedge is not scale covariant (max relative deviation "
        f"{resid:.3g}): Lambda(m in GeV) != Lambda(m in MeV)/1000."
    )


def check_heavy_and_light_limits():
    """Lambda -> m for m >> omega_max, and -> sqrt(2 m omega_max) for m << omega_max."""
    heavy = np.array([1e6, 1e8])
    lam = eft_kinematic_lambda_curve(heavy, omega_max=OMEGA_MAX)
    assert np.allclose(lam / heavy, 1.0, rtol=1e-3), (
        f"for m >> omega_max the wedge must reduce to Lambda ~ m; got {lam / heavy}"
    )
    light = np.array([1e-5, 1e-3])
    lam = eft_kinematic_lambda_curve(light, omega_max=OMEGA_MAX)
    expect = np.sqrt(2.0 * light * OMEGA_MAX)
    assert np.allclose(lam, expect, rtol=1e-3), (
        f"for m << omega_max the wedge must reduce to sqrt(2 m omega_max); "
        f"got {lam} against {expect}"
    )


def check_wedge_scales_with_sqrt_omega_max():
    """Sec. IV B: the low-mass wedge moves as sqrt(omega_max)."""
    m = np.array([1e-3])
    a = eft_kinematic_lambda_curve(m, omega_max=814.0)
    b = eft_kinematic_lambda_curve(m, omega_max=168.93225)
    assert np.allclose(a / b, np.sqrt(814.0 / 168.93225), rtol=1e-3), (
        f"low-mass wedge ratio {float(a / b)} does not track sqrt(omega_max)"
    )


def check_operator_key_dispatch():
    """scalar_rayleigh must not be reached by substring match on 'rayleigh'."""
    assert normalise_operator_key("rayleigh", dm_type="scalar") == "scalar_rayleigh"
    assert normalise_operator_key("rayleigh_full") == "rayleigh_full"
    assert (EFT_OPERATOR_VALIDITY["scalar_rayleigh"]["lambda_power_in_cross_section"]
            == 4), "scalar Rayleigh is a Lambda^-2 operator: sigma ~ Lambda^-4"
    assert (EFT_OPERATOR_VALIDITY["rayleigh_full"]["lambda_power_in_cross_section"]
            == 6), "dimension-7 Rayleigh: sigma ~ Lambda^-6"


def check_lambda_min_includes_mass_floor():
    """require_lambda_gt_mdm puts Lambda > m into the floor."""
    m = np.array([1e6])
    with_floor = lambda_min_curve(
        "rayleigh_full", m, omega_max=OMEGA_MAX, require_lambda_gt_mdm=True)
    assert with_floor[0] >= m[0], (
        f"floor {with_floor[0]:.3e} does not enforce Lambda > m_chi = {m[0]:.3e}"
    )


if pytest is not None:
    def test_scale_covariance():
        check_scale_covariance()

    def test_heavy_and_light_limits():
        check_heavy_and_light_limits()

    def test_wedge_scales_with_sqrt_omega_max():
        check_wedge_scales_with_sqrt_omega_max()

    def test_operator_key_dispatch():
        check_operator_key_dispatch()

    def test_lambda_min_includes_mass_floor():
        check_lambda_min_includes_mass_floor()

    @pytest.mark.parametrize("key", OPERATORS)
    def test_every_operator_declares_a_lambda_power(key):
        entry = EFT_OPERATOR_VALIDITY[key]
        assert entry.get("lambda_power_in_cross_section") in (4, 6), (
            f"{key} must declare sigma ~ Lambda^-4 or Lambda^-6"
        )

    def test_unknown_operator_raises():
        with pytest.raises(KeyError):
            normalise_operator_key("not_an_operator")


def _main():
    """Standalone runner for environments without pytest."""
    checks = [
        ("scale covariance", check_scale_covariance),
        ("heavy/light limits", check_heavy_and_light_limits),
        ("sqrt(omega_max) scaling", check_wedge_scales_with_sqrt_omega_max),
        ("operator-key dispatch", check_operator_key_dispatch),
        ("mass floor", check_lambda_min_includes_mass_floor),
    ]
    failures = 0
    for name, fn in checks:
        try:
            fn()
            print(f"{name:28s} pass")
        except AssertionError as exc:
            failures += 1
            print(f"{name:28s} FAIL -- {exc}")
    print("OK" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
