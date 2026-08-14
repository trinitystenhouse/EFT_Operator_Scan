"""
Central EFT validity catalogue for the photon-DM scattering scans.

The scan scripts use Lambda as the EFT scale for dipole, charge-radius,
anapole, and Rayleigh operators.  A sampled point is kept only when Lambda is
above all requested lower-bound curves:

  - Lambda > m_scat, the conservative EFT separation guide.
  - Lambda >= sqrt(kappa * max(s_max, |t|_max)), the kinematic EFT-validity
    condition of Eq. (IV.18). This is the wedge drawn in the paper.

All masses, photon energies, and Lambda values are in GeV.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


EFT_OPERATOR_VALIDITY: dict[str, dict[str, Any]] = {
    "dipole_magnetic": {
        "dm_type": "fermionic",
        "operator_dimension": 5,
        "lambda_power_in_cross_section": 4,  # two-insertion real-photon Compton
        "wilson_coefficients": {"c_s": 1.0, "c_p": 0.0},
        "majorana_allowed": False,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
        },
    },
    "dipole_electric": {
        "dm_type": "fermionic",
        "operator_dimension": 5,
        "lambda_power_in_cross_section": 4,  # two-insertion real-photon Compton
        "wilson_coefficients": {"c_s": 0.0, "c_p": 1.0},
        "majorana_allowed": False,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
        },
    },
    "charge_radius": {
        "dm_type": "fermionic",
        "operator_dimension": 6,
        "lambda_power_in_cross_section": 4,
        "wilson_coefficients": {"c_s": 1.0, "c_p": 0.0},
        "majorana_allowed": False,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
        },
    },
    "anapole": {
        "dm_type": "fermionic",
        "operator_dimension": 6,
        "lambda_power_in_cross_section": 4,
        "wilson_coefficients": {"c_s": 0.0, "c_p": 1.0},
        "majorana_allowed": True,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
        },
    },
    "rayleigh_even": {
        "dm_type": "fermionic",
        "operator_dimension": 7,
        "lambda_power_in_cross_section": 6,
        "wilson_coefficients": {"c_s": 1.0, "c_p": 0.0},
        "majorana_allowed": True,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
        },
    },
    "rayleigh_odd": {
        "dm_type": "fermionic",
        "operator_dimension": 7,
        "lambda_power_in_cross_section": 6,
        "wilson_coefficients": {"c_s": 0.0, "c_p": 1.0},
        "majorana_allowed": True,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
        },
    },
    "rayleigh_full": {
        "dm_type": "fermionic",
        "operator_dimension": 7,
        "lambda_power_in_cross_section": 6,
        "wilson_coefficients": {"c_s": 1.0, "c_p": 1.0},
        "majorana_allowed": True,
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
        },
    },
    "scalar_rayleigh": {
        "dm_type": "scalar",
        "operator_dimension": 6,
        "lambda_power_in_cross_section": 4,
        "wilson_coefficients": {"c_phi": 1.0},
        "majorana_allowed": None,
        "aliases": ("rayleigh",),
        "validity_lines": {
            "eft_separation": "Lambda > m_scat",
            "kinematic_eft": "Lambda^2 >= kappa * max(s_max, |t|_max)",
        },
    },
    "higgs_portal": {
        "dm_type": "fermionic",
        "operator_dimension": None,
        "lambda_power_in_cross_section": None,
        "wilson_coefficients": {"y_eff": "scan axis"},
        "majorana_allowed": None,
        "validity_lines": {
            "note": "UV-complete Higgs-portal calculation; EFT Lambda cuts are not applied.",
        },
    },
}


def normalise_operator_key(operator: str, dm_type: str | None = None) -> str:
    """Return the canonical key used by EFT_OPERATOR_VALIDITY.

    Lookup is by exact key, never by substring: ``"rayleigh" in key`` would
    match ``scalar_rayleigh`` and hand a Lambda^-2 operator the dimension-7
    Lambda^-6 treatment. An unknown operator is a hard error rather than a
    silent fall-through.
    """
    op = str(operator).strip()
    if op == "rayleigh" and dm_type == "scalar":
        op = "scalar_rayleigh"
    if op not in EFT_OPERATOR_VALIDITY:
        raise KeyError(
            f"unknown operator {operator!r} (normalised to {op!r}). "
            f"Known keys: {sorted(EFT_OPERATOR_VALIDITY)}"
        )
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


def lambda_min_curve(
    operator: str,
    m_chi_arr: np.ndarray | float,
    *,
    omega_max: float,
    dm_type: str | None = None,
    eft_kinematic_factor: float = 1.0,
    require_lambda_gt_mdm: bool = True,
    include_kinematic: bool = True,
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
        )[0]
    )
    lo = max(float(lambda_min), lower)
    hi = float(lambda_max)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi >= lo and lo > 0.0):
        return np.array([], dtype=float)
    if int(n_lambda) <= 1 or np.isclose(lo, hi):
        return np.asarray([lo], dtype=float)
    return np.logspace(np.log10(lo), np.log10(hi), int(n_lambda))
