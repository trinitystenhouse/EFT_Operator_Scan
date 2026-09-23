"""
Central EFT validity catalogue for Totani_Scattering scans.

The scan scripts use Lambda as the EFT scale for dipole, charge-radius,
anapole, and Rayleigh operators.  A sampled point is kept only when Lambda is
above all requested lower-bound curves:

  - Lambda > m_scat, the conservative EFT separation guide.
  - Lambda >= sqrt(kappa * max(s_max, |t|_max)), kinematic EFT validity.
  - Lambda >= Lambda_unitarity(operator, m_scat), the perturbative unitarity
    guide used in the existing overlay plots.

All masses, photon energies, and Lambda values are in GeV.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


# =============================================================================
# PERTURBATIVE-UNITARITY GUIDES
# =============================================================================
# Each operator owns its unitarity expression, as a callable stored in its
# EFT_OPERATOR_VALIDITY entry under "unitarity_lambda". This replaces the old
# if-chain that dispatched on `"rayleigh" in key`, which silently routed
# scalar_rayleigh (a Lambda^-2 operator, lambda_power_in_cross_section=4) into
# the dimension-7 fermionic-Rayleigh branch (Lambda^-3, power 6). See
# UNITARITY_DERIVATION.md for the full account.
#
# CONVENTION AUDIT (2026-08-06)
# -----------------------------
# The three legacy expressions below are NOT dimensionally consistent:
#
#     dipole    Lambda^2 = 16*pi*m           GeV^2 = GeV^1
#     dim-6     Lambda^4 = 16*pi*m^2         GeV^4 = GeV^2
#     Rayleigh  Lambda^6 = 128*pi^2*m^2      GeV^6 = GeV^2
#
# Each equates different powers of GeV, so each carries an undocumented
# reference scale. They are therefore unit-dependent: evaluating in MeV and
# converting back changes the dipole and dim-6 curves by 31.6x and the Rayleigh
# curve by 100x. Because of this there is no single self-consistent convention
# (|a_J| <= 1/2 vs 1, elastic vs crossed channel) that reproduces them, so the
# scalar-Rayleigh guide could not be derived "to match". They are retained here
# UNCHANGED so that no published curve silently moves; the dimensionally
# consistent alternative is provided alongside and is opt-in.
_LEGACY_UNITARITY_NOTE = "legacy, dimensionally inconsistent -- see UNITARITY_DERIVATION.md"


def _mark_legacy(fn):
    """Tag a guide as dimensionally inconsistent so tests can xfail it."""
    fn._unitarity_is_legacy = True
    fn._unitarity_note = _LEGACY_UNITARITY_NOTE
    return fn


def is_legacy_guide(fn) -> bool:
    """True if this guide is one of the retained, unit-dependent legacy forms."""
    return bool(getattr(fn, "_unitarity_is_legacy", False))


@_mark_legacy
def _legacy_dipole_unitarity(m):
    """Lambda >= sqrt(16 pi m).  [legacy form, retained for reproducibility]"""
    return np.sqrt(16.0 * np.pi * m)


@_mark_legacy
def _legacy_dim6_unitarity(m):
    """Lambda >= (16 pi m^2)^(1/4).  [legacy form; charge_radius / anapole]"""
    return (16.0 * np.pi * m**2) ** 0.25


@_mark_legacy
def _legacy_rayleigh_unitarity(m):
    """Lambda >= (128 pi^2 m^2)^(1/6).  [legacy form; dim-7 fermionic Rayleigh]"""
    return (128.0 * np.pi**2 * m**2) ** (1.0 / 6.0)


def scalar_rayleigh_unitarity_derived(m, *, omega_max: float, c_phi: float = 1.0):
    """Dimensionally consistent J=0 unitarity guide for O = (c_phi/Lambda^2) phi^2 F_mn F^mn.

    Derivation (see UNITARITY_DERIVATION.md for the long form).  Expanding
    F_mn F^mn for one incoming photon (k1, eps1) and one outgoing (k2, eps2*),
    and phi^2 for the two scalars, the contact amplitude is

        M = (8 c_phi / Lambda^2) [ (k1.k2)(eps1.eps2*) - (k1.eps2*)(k2.eps1) ]

    In the CM frame the photons are back to back with transverse polarisations,
    so k1.eps2* = k2.eps1 = 0 and k1.k2 = s/2, giving |M| = 4 c_phi s / Lambda^2.
    The amplitude is angle independent, hence pure J=0, and

        a_0 = M / (16 pi).

    CONVENTION: |a_0| <= 1/2 (the conservative elastic choice), and NO
    identical-particle 1/sqrt(2) factors for the two photons or the two
    scalars. Including those factors would loosen the bound by 2 in Lambda^2.
    Saturating gives |M| <= 8 pi and therefore

        Lambda^2 >= c_phi * s / (2 pi),      s = m^2 + 2 m omega_max

    which is dimensionally consistent (GeV^2 = GeV^2) and unit-covariant.

    For m << omega_max this is Lambda >= sqrt(c_phi m omega_max / pi), i.e. it
    scales as m^(1/2) -- the SAME power as the kinematic curve sqrt(s_max) --
    so the two run parallel, separated by the constant factor sqrt(2 pi / c_phi)
    = 2.507 for c_phi = 1. The guide therefore never overtakes the kinematic
    wedge at any mass, in contrast to the m^(1/3) Lambda^6 form that the buggy
    dispatch applied, which crossed it near m ~ 2e-5 GeV.
    """
    m = np.asarray(m, dtype=float)
    s = m**2 + 2.0 * m * float(omega_max)
    return np.sqrt(float(c_phi) * s / (2.0 * np.pi))


def scale_covariance_residual(operator: str, m_gev, unit_factor: float = 1.0e3) -> float:
    """Max relative violation of Lambda(m[GeV]) == Lambda(m[MeV])/1000.

    A guide that is a genuine physical scale is unit covariant: expressing the
    same mass in a different unit and converting the answer back reproduces it.
    Returns 0.0 for a dimensionally consistent guide.
    """
    m = np.asarray(m_gev, dtype=float)
    a = np.asarray(unitarity_lambda_curve(operator, m), dtype=float)
    b = np.asarray(unitarity_lambda_curve(operator, m * unit_factor), dtype=float) / unit_factor
    good = np.isfinite(a) & np.isfinite(b) & (a != 0.0)
    if not np.any(good):
        return 0.0
    return float(np.max(np.abs(b[good] - a[good]) / np.abs(a[good])))


EFT_OPERATOR_VALIDITY: dict[str, dict[str, Any]] = {
    "dipole_magnetic": {
        "dm_type": "fermionic",
        "operator_dimension": 5,
        "lambda_power_in_cross_section": 4,  # CORRECTED 2026-07-17: two-insertion real-photon Compton
        "wilson_coefficients": {"c_s": 1.0, "c_p": 0.0},
        "majorana_allowed": False,
        "unitarity_lambda": _legacy_dipole_unitarity,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
            "unitarity": "Lambda >= sqrt(16*pi*m_scat)",
        },
    },
    "dipole_electric": {
        "dm_type": "fermionic",
        "operator_dimension": 5,
        "lambda_power_in_cross_section": 4,  # CORRECTED 2026-07-17: two-insertion real-photon Compton
        "wilson_coefficients": {"c_s": 0.0, "c_p": 1.0},
        "majorana_allowed": False,
        "unitarity_lambda": _legacy_dipole_unitarity,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
            "unitarity": "Lambda >= sqrt(16*pi*m_scat)",
        },
    },
    "charge_radius": {
        "dm_type": "fermionic",
        "operator_dimension": 6,
        "lambda_power_in_cross_section": 4,
        "wilson_coefficients": {"c_s": 1.0, "c_p": 0.0},
        "majorana_allowed": False,
        "unitarity_lambda": _legacy_dim6_unitarity,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
            "unitarity": "Lambda >= (16*pi*m_scat^2)^(1/4)",
        },
    },
    "anapole": {
        "dm_type": "fermionic",
        "operator_dimension": 6,
        "lambda_power_in_cross_section": 4,
        "wilson_coefficients": {"c_s": 0.0, "c_p": 1.0},
        "majorana_allowed": True,
        "unitarity_lambda": _legacy_dim6_unitarity,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
            "unitarity": "Lambda >= (16*pi*m_scat^2)^(1/4)",
        },
    },
    "rayleigh_even": {
        "dm_type": "fermionic",
        "operator_dimension": 7,
        "lambda_power_in_cross_section": 6,
        "wilson_coefficients": {"c_s": 1.0, "c_p": 0.0},
        "majorana_allowed": True,
        "unitarity_lambda": _legacy_rayleigh_unitarity,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
            "unitarity": "Lambda >= (128*pi^2*m_scat^2)^(1/6)",
        },
    },
    "rayleigh_odd": {
        "dm_type": "fermionic",
        "operator_dimension": 7,
        "lambda_power_in_cross_section": 6,
        "wilson_coefficients": {"c_s": 0.0, "c_p": 1.0},
        "majorana_allowed": True,
        "unitarity_lambda": _legacy_rayleigh_unitarity,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
            "unitarity": "Lambda >= (128*pi^2*m_scat^2)^(1/6)",
        },
    },
    "rayleigh_full": {
        "dm_type": "fermionic",
        "operator_dimension": 7,
        "lambda_power_in_cross_section": 6,
        "wilson_coefficients": {"c_s": 1.0, "c_p": 1.0},
        "majorana_allowed": True,
        "unitarity_lambda": _legacy_rayleigh_unitarity,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
            "unitarity": "Lambda >= (128*pi^2*m_scat^2)^(1/6)",
        },
    },
    "scalar_rayleigh": {
        "dm_type": "scalar",
        "operator_dimension": 6,
        "lambda_power_in_cross_section": 4,
        "wilson_coefficients": {"c_phi": 1.0},
        "majorana_allowed": None,
        "aliases": ("rayleigh",),
        "unitarity_lambda": _legacy_dim6_unitarity,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
            # CORRECTED 2026-08-06: was the dimension-7 (128*pi^2*m^2)^(1/6)
            # form, inherited from the substring dispatch bug. scalar_rayleigh
            # is Lambda^-2 (lambda_power_in_cross_section=4), so it takes the
            # same form as charge_radius / anapole. See UNITARITY_DERIVATION.md.
            "unitarity": "Lambda >= (16*pi*m_scat^2)^(1/4)",
        },
    },
    "higgs_portal": {
        "dm_type": "fermionic",
        "operator_dimension": None,
        "lambda_power_in_cross_section": None,
        "wilson_coefficients": {"y_eff": "scan axis"},
        "majorana_allowed": None,
        "unitarity_lambda": None,
        "validity_lines": {
            "note": "UV-complete Higgs-portal calculation; EFT Lambda cuts are not applied.",
        },
    },
}


def normalise_operator_key(operator: str, dm_type: str | None = None) -> str:
    """Return the canonical key used by EFT_OPERATOR_VALIDITY."""
    op = str(operator).strip()
    if op == "rayleigh" and dm_type == "scalar":
        return "scalar_rayleigh"
    if op == "scalar_rayleigh":
        return "scalar_rayleigh"
    return op


def operator_validity_catalogue() -> dict[str, dict[str, Any]]:
    """Return a copy of the operator validity dictionary."""
    return deepcopy(EFT_OPERATOR_VALIDITY)


def get_s_max_lab_dmrest(mchi: np.ndarray | float, omega_max: float) -> np.ndarray:
    mchi = np.asarray(mchi, dtype=float)
    return mchi**2 + 2.0 * mchi * float(omega_max)


def get_t_abs_max_lab_dmrest(mchi: np.ndarray | float, omega_max: float) -> np.ndarray:
    mchi = np.asarray(mchi, dtype=float)
    ratio = np.full_like(mchi, np.nan, dtype=float)
    good = mchi > 0.0
    np.divide(2.0 * float(omega_max), mchi, out=ratio, where=good)
    denom = 1.0 + ratio
    return 4.0 * float(omega_max) ** 2 / denom


def eft_kinematic_lambda_curve(
    m_chi_arr: np.ndarray | float,
    *,
    omega_max: float,
    eft_kinematic_factor: float = 1.0,
) -> np.ndarray:
    """Lower-bound Lambda curve from the EFT kinematic expansion."""
    m = np.asarray(m_chi_arr, dtype=float)
    q2_max = np.maximum(
        get_s_max_lab_dmrest(m, omega_max),
        get_t_abs_max_lab_dmrest(m, omega_max),
    )
    return np.sqrt(float(eft_kinematic_factor) * q2_max)


def unitarity_lambda_curve(operator: str, m_chi_arr: np.ndarray | float) -> np.ndarray:
    """Perturbative-unitarity guide curve for the operator family."""
    key = normalise_operator_key(operator)
    m = np.asarray(m_chi_arr, dtype=float)
    x = np.where(m > 0.0, m, np.nan)

    # Explicit per-key lookup. Previously this was an if-chain ending in
    # `if "rayleigh" in key`, which matched scalar_rayleigh by substring and
    # gave a Lambda^-2 operator the dimension-7 Lambda^6 unitarity floor.
    # An unknown key is now a hard error rather than a silent NaN column.
    if key not in EFT_OPERATOR_VALIDITY:
        raise KeyError(
            f"unitarity_lambda_curve: unknown operator {operator!r} "
            f"(normalised to {key!r}). Known keys: "
            f"{sorted(EFT_OPERATOR_VALIDITY)}"
        )
    fn = EFT_OPERATOR_VALIDITY[key].get("unitarity_lambda")
    if fn is None:                      # e.g. higgs_portal: no EFT Lambda cut
        return np.full_like(x, np.nan, dtype=float)
    return np.asarray(fn(x), dtype=float)


def lambda_min_curve(
    operator: str,
    m_chi_arr: np.ndarray | float,
    *,
    omega_max: float,
    dm_type: str | None = None,
    eft_kinematic_factor: float = 1.0,
    require_lambda_gt_mdm: bool = True,
    include_kinematic: bool = True,
    include_unitarity: bool = True,
) -> np.ndarray:
    """Combined lower-bound Lambda curve for valid EFT scan points."""
    key = normalise_operator_key(operator, dm_type=dm_type)
    m = np.asarray(m_chi_arr, dtype=float)
    if key == "higgs_portal":
        return np.full_like(m, np.nan, dtype=float)

    pieces = []
    if require_lambda_gt_mdm:
        pieces.append(np.where(m > 0.0, m, np.nan))
    if include_kinematic:
        pieces.append(
            eft_kinematic_lambda_curve(
                m,
                omega_max=omega_max,
                eft_kinematic_factor=eft_kinematic_factor,
            )
        )
    if include_unitarity:
        pieces.append(unitarity_lambda_curve(key, m))

    if not pieces:
        return np.zeros_like(m, dtype=float)
    return np.nanmax(np.stack(pieces, axis=0), axis=0)


def validity_mask(
    operator: str,
    m_chi_grid: np.ndarray,
    lambda_grid: np.ndarray,
    *,
    omega_max: float,
    dm_type: str | None = None,
    eft_kinematic_factor: float = 1.0,
    require_lambda_gt_mdm: bool = True,
    include_kinematic: bool = True,
    include_unitarity: bool = True,
) -> np.ndarray:
    """Return mask[i,j] for valid (m_chi_grid[i], lambda_grid[j]) pairs."""
    key = normalise_operator_key(operator, dm_type=dm_type)
    m = np.asarray(m_chi_grid, dtype=float)
    lam = np.asarray(lambda_grid, dtype=float)
    if key == "higgs_portal":
        return np.ones((m.size, lam.size), dtype=bool)
    lam_min = lambda_min_curve(
        key,
        m,
        omega_max=omega_max,
        dm_type=dm_type,
        eft_kinematic_factor=eft_kinematic_factor,
        require_lambda_gt_mdm=require_lambda_gt_mdm,
        include_kinematic=include_kinematic,
        include_unitarity=include_unitarity,
    )
    return (
        np.isfinite(m[:, None])
        & np.isfinite(lam[None, :])
        & np.isfinite(lam_min[:, None])
        & (lam[None, :] >= lam_min[:, None])
    )


def is_eft_point_valid(
    operator: str,
    m_chi: float,
    Lambda: float,
    *,
    omega_max: float,
    dm_type: str | None = None,
    eft_kinematic_factor: float = 1.0,
    require_lambda_gt_mdm: bool = True,
    include_kinematic: bool = True,
    include_unitarity: bool = True,
) -> bool:
    """Scalar convenience wrapper around validity_mask."""
    mask = validity_mask(
        operator,
        np.asarray([m_chi], dtype=float),
        np.asarray([Lambda], dtype=float),
        omega_max=omega_max,
        dm_type=dm_type,
        eft_kinematic_factor=eft_kinematic_factor,
        require_lambda_gt_mdm=require_lambda_gt_mdm,
        include_kinematic=include_kinematic,
        include_unitarity=include_unitarity,
    )
    return bool(mask[0, 0])


def sample_valid_lambda_grid(
    operator: str,
    m_chi: float,
    *,
    omega_max: float,
    lambda_min: float,
    lambda_max: float,
    n_lambda: int,
    dm_type: str | None = None,
    eft_kinematic_factor: float = 1.0,
    require_lambda_gt_mdm: bool = True,
    include_kinematic: bool = True,
    include_unitarity: bool = True,
) -> np.ndarray:
    """Log-spaced Lambda grid restricted to the valid range for one m_chi."""
    lower = float(
        lambda_min_curve(
            operator,
            np.asarray([m_chi], dtype=float),
            omega_max=omega_max,
            dm_type=dm_type,
            eft_kinematic_factor=eft_kinematic_factor,
            require_lambda_gt_mdm=require_lambda_gt_mdm,
            include_kinematic=include_kinematic,
            include_unitarity=include_unitarity,
        )[0]
    )
    lo = max(float(lambda_min), lower)
    hi = float(lambda_max)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi >= lo and lo > 0.0):
        return np.array([], dtype=float)
    if int(n_lambda) <= 1 or np.isclose(lo, hi):
        return np.asarray([lo], dtype=float)
    return np.logspace(np.log10(lo), np.log10(hi), int(n_lambda))


# =============================================================================
# CATALOGUE CONSISTENCY CHECKS
# =============================================================================
# Run at import so a new operator cannot be added without a unitarity guide,
# and so the guide can never again disagree with the operator's own stated
# lambda_power_in_cross_section.
_EXPECTED_UNITARITY_BY_POWER = {
    4: (_legacy_dipole_unitarity, _legacy_dim6_unitarity),
    6: (_legacy_rayleigh_unitarity,),
}


def _validate_catalogue() -> None:
    for key, entry in EFT_OPERATOR_VALIDITY.items():
        if "unitarity_lambda" not in entry:
            raise AssertionError(
                f"EFT_OPERATOR_VALIDITY[{key!r}] has no 'unitarity_lambda' entry. "
                "Every operator must declare one explicitly (use None if the "
                "EFT Lambda cuts do not apply, as for higgs_portal)."
            )
        fn = entry["unitarity_lambda"]
        power = entry.get("lambda_power_in_cross_section")
        if fn is None or power is None:
            continue
        # Scale covariance. Enforced only for guides not marked legacy, so the
        # check activates automatically as each legacy form is replaced rather
        # than needing to be switched on by hand.
        if not is_legacy_guide(fn):
            resid = scale_covariance_residual(key, np.array([1e-3, 1e-1, 1.0]))
            if resid > 1e-12:
                raise AssertionError(
                    f"EFT_OPERATOR_VALIDITY[{key!r}] unitarity guide is not scale "
                    f"covariant (max relative deviation {resid:.3g}): "
                    "Lambda(m in GeV) != Lambda(m in MeV)/1000. The expression "
                    "equates different powers of GeV and so carries a hidden "
                    "reference scale. See UNITARITY_DERIVATION.md."
                )

        allowed = _EXPECTED_UNITARITY_BY_POWER.get(power)
        if allowed is not None and fn not in allowed:
            raise AssertionError(
                f"EFT_OPERATOR_VALIDITY[{key!r}] declares "
                f"lambda_power_in_cross_section={power} but its unitarity guide "
                f"is {getattr(fn, '__name__', fn)!r}, which belongs to a "
                "different Lambda power. This is the scalar_rayleigh class of "
                "bug -- see UNITARITY_DERIVATION.md."
            )


_validate_catalogue()
