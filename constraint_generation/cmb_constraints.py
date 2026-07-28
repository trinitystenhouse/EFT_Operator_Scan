"""
CMB power spectrum constraints on DM-photon EFT operators.

Physical basis
--------------
DM-photon scattering modifies the CMB TT power spectrum by dragging the DM
component and suppressing small-scale structure. The constraint is parameterised
using the Planck 2018 bounds reported by Boddy & Gluscevic (2018),
arXiv:1801.08609.

Cross-section parameterisation:  sigma = sigma_0 * (T / T_CMB_0)^n
where T_CMB_0 = 2.726 K = 2.349e-4 eV is the present CMB temperature.

The Planck bound is on sigma_0 / m_chi [cm^2/GeV]:
  n=2 (dipole operators):                sigma_0/m_chi < 1.5e-23  cm^2/GeV
  n=4 (charge radius, anapole, Rayleigh): sigma_0/m_chi < 3.0e-14  cm^2/GeV
  n=6 (Rayleigh-odd):                    no meaningful published bound

Energy-scaling indices per operator (sigma ~ E^n for E << m_chi):
  dipole_magnetic, dipole_electric : n=2
  charge_radius, anapole           : n=4
  rayleigh_even, scalar_rayleigh   : n=4
  rayleigh_odd                     : n=6

To apply the constraint:
  1.  Evaluate sigma_0 = sigma(E = T_CMB_0) using the tree-level EFT amplitude.
  2.  If sigma_0 / m_chi > u_max(n), the point is CMB-excluded.

Note on expected reach:
  For n >= 2 operators the cross section at T_CMB_0 (~2e-13 GeV) is
  enormous suppressed relative to Fermi-LAT energies (~10-100 GeV).
  The CMB constraint on Lambda is therefore only competitive for
  very light DM (m_chi < few MeV for dipole; even lighter for n=4).
  The boundary is saved regardless; it will simply lie at very small
  Lambda for heavy DM and should be compared against the EFT validity
  line Lambda = m_chi.
"""

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import EFT cross-section and helper functions from the project root.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_WORKSPACE = _ROOT.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_WORKSPACE))

from core.attenuation_eft import (
    sigma_tot_fermionic,
    sigma_tot_scalar,
    _paper_y_axis_values,
    _operator_metadata,
    OPERATOR_METADATA,
    FB_TO_CM2,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T_CMB_0_EV  = 2.349e-4          # present CMB temperature in eV
T_CMB_0_GEV = T_CMB_0_EV * 1e-9 # same in GeV  (= 2.349e-13 GeV)

# Energy-scaling power n for sigma ~ E^n (valid in the E << m_chi regime):
OPERATOR_N = {
    "dipole_magnetic":  2,
    "dipole_electric":  2,
    "charge_radius":    4,
    "anapole":          4,
    "rayleigh_even":    4,
    "rayleigh_odd":     6,   # no Planck bound; stored so boundary returns empty
    "scalar_rayleigh":  4,
}

# Planck 2018 bounds on sigma_0 / m_chi [cm^2/GeV]
# Source: Boddy & Gluscevic 2018, arXiv:1801.08609, Table 1.
#   n=2: 1.5e-23 cm^2/GeV
#   n=4: 3.0e-14 cm^2/GeV
#   n=6: no published bound (interaction at T_CMB_0 is negligibly small)
CMB_U_MAX = {
    2: 1.5e-23,
    4: 3.0e-14,
    6: None,
}


# ---------------------------------------------------------------------------
# Core function: evaluate sigma at T_CMB_0
# ---------------------------------------------------------------------------

def sigma_at_cmb(m_chi_GeV, Lambda_GeV, dm_type="fermionic",
                 operator="dipole_magnetic",
                 c_s=1.0, c_p=1.0, c_phi=1.0):
    """
    Return sigma(E = T_CMB_0) in cm^2 for a single (m_chi, Lambda) point.

    We always use the lab frame with DM at rest, which is valid since
    m_chi >> T_CMB_0 for any DM mass above ~ eV.
    """
    E_cmb = T_CMB_0_GEV
    if dm_type == "fermionic":
        sig = sigma_tot_fermionic(
            E_cmb, m_chi_GeV, c_s, c_p, Lambda_GeV, operator
        )
    else:  # scalar
        sig = sigma_tot_scalar(E_cmb, m_chi_GeV, c_phi, Lambda_GeV)
    return float(sig)


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------

def cmb_exclusion_boundary_analytic(m_chi_arr, *, dm_type="fermionic",
                                    operator="dipole_magnetic",
                                    c_s=1.0, c_p=1.0, c_phi=1.0,
                                    majorana=False):
    """
    Invert the Planck bound sigma_0/m_chi < u_max at E = T_CMB_0.

    Inverts numerically off sigma_at_cmb() (which returns cm^2 from the
    tree-level EFT cross sections) using the exact power-law
    sigma ~ Lambda^-lambda_power:
        Lambda_crit = Lambda_ref * [ sigma(Lambda_ref) / (u_max * m_chi) ]^(1/lambda_power)
    Anapole and charge radius scatter real photons with exactly zero
    tree-level cross section, so they return an empty boundary.

    Returns
    -------
    boundary : array of shape (N, 2), columns [m_chi_GeV, Lambda_crit_GeV].
               Empty array if no constraint exists for this operator.
    """
    key = operator if dm_type == "fermionic" else "scalar_rayleigh"
    n = OPERATOR_N.get(key, None)
    u_max = CMB_U_MAX.get(n) if n is not None else None
    if u_max is None:
        return np.empty((0, 2))

    m_chi_arr = np.asarray(m_chi_arr, dtype=float)
    m_chi_arr = m_chi_arr[np.isfinite(m_chi_arr) & (m_chi_arr > 0.0)]
    if m_chi_arr.size == 0:
        return np.empty((0, 2))

    if dm_type == "scalar":
        coeff = float(c_phi)
    elif operator in ("dipole_magnetic", "charge_radius", "rayleigh_even"):
        coeff = float(c_s)
    else:
        coeff = float(c_p)

    if coeff <= 0.0:
        return np.empty((0, 2))

    # q^2-suppressed operators: exactly zero for on-shell photons -> no bound.
    if key in ("charge_radius", "anapole", "rayleigh_odd"):
        return np.empty((0, 2))

    # Numeric inversion off the corrected cross sections (cm^2), exact because
    # sigma is a pure power law in Lambda.
    meta = _operator_metadata(
        "scalar" if key == "scalar_rayleigh" else dm_type,
        "rayleigh" if key == "scalar_rayleigh" and dm_type != "scalar" else operator,
    )
    lam_power = meta.get("lambda_power", None)
    if lam_power is None:
        raise ValueError(f"Unsupported operator for CMB boundary: {operator}")

    LAMBDA_REF = 1.0  # GeV
    lam_crit = np.empty_like(m_chi_arr)
    for i, m in enumerate(m_chi_arr):
        sig_ref = sigma_at_cmb(
            m, LAMBDA_REF,
            dm_type=("scalar" if key == "scalar_rayleigh" else dm_type),
            operator=operator, c_s=c_s, c_p=c_p, c_phi=c_phi,
        )  # cm^2 at Lambda = LAMBDA_REF
        with np.errstate(divide="ignore", invalid="ignore"):
            lam_crit[i] = LAMBDA_REF * (sig_ref / (u_max * m)) ** (1.0 / lam_power)
    if majorana and "rayleigh" in key:
        lam_crit = lam_crit / (2.0 ** (1.0 / 12.0))  # unchanged legacy Majorana convention

    mask = np.isfinite(lam_crit) & (lam_crit > 0.0)
    if not np.any(mask):
        return np.empty((0, 2))

    return np.column_stack((m_chi_arr[mask], lam_crit[mask]))


# ---------------------------------------------------------------------------
# Save boundary in the same .npz format as attenuation_eft.py
# ---------------------------------------------------------------------------

def save_cmb_boundary_npz(boundary, dm_type, operator, *,
                           c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False):
    """
    Save the CMB exclusion boundary in constraint_boundaries/ as
    cmb_{dm_type}_{operator}_planck2018.npz, using the same format as
    the Totani boundaries (lambda_GeV and lambda_plot_GeV columns).
    """
    outdir = _ROOT / "constraint_boundaries"
    outdir.mkdir(exist_ok=True)

    if boundary.size == 0:
        print(f"  [CMB] No boundary for {dm_type}/{operator} — skipping save.")
        return None

    meta        = _operator_metadata(dm_type, operator)
    lambda_raw  = boundary[:, 1]
    lambda_plot = _paper_y_axis_values(
        lambda_raw, dm_type, operator, c_s=c_s, c_p=c_p, c_phi=c_phi
    )

    suffix = "_majorana" if majorana else ""
    outpath = outdir / f"cmb_{dm_type}_{operator}{suffix}_planck2018.npz"
    np.savez(
        outpath,
        mchi_GeV        = boundary[:, 0],
        lambda_GeV      = lambda_raw,
        lambda_plot_GeV = lambda_plot,
        paper_label     = meta["paper_label"],
        dm_type         = dm_type,
        operator        = operator,
        coefficient_power = meta["coefficient_power"],
        c_s   = float(c_s),
        c_p   = float(c_p),
        c_phi = float(c_phi),
        n_energy_scaling = OPERATOR_N.get(operator, -1),
        u_max_cm2_per_GeV = CMB_U_MAX.get(OPERATOR_N.get(operator, -1), np.nan),
        reference = "Boddy & Gluscevic 2018, arXiv:1801.08609, Planck 2018",
    )
    print(f"  [CMB] Saved: {outpath.name}")
    return outpath


# ---------------------------------------------------------------------------
# Driver: run all operators
# ---------------------------------------------------------------------------

def run_cmb_constraints(m_chi_arr, Lambda_arr, *,
                         operators=None, dm_type="fermionic",
                         c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False):
    """
    Compute and save CMB boundaries for a list of operators.

    Parameters
    ----------
    m_chi_arr   : 1D array of DM masses [GeV]
    Lambda_arr  : 1D array of EFT cutoff scales [GeV]
    operators   : list of operator keys (default: all fermionic + scalar_rayleigh)
    dm_type     : "fermionic" or "scalar"
    """
    if operators is None:
        if dm_type == "fermionic":
            operators = [
                "dipole_magnetic", "dipole_electric",
                "charge_radius", "anapole",
                "rayleigh_even", "rayleigh_odd",
            ]
        else:
            operators = ["scalar_rayleigh"]

    for op in operators:
        actual_dm_type = "scalar" if op == "scalar_rayleigh" else dm_type
        cs, cp, cphi = (0.0, 0.0, c_phi) if actual_dm_type == "scalar" else (c_s, c_p, 1.0)
        if op == "dipole_magnetic":
            cs, cp = 1.0, 0.0
        elif op in ("dipole_electric", "anapole", "rayleigh_odd"):
            cs, cp = 0.0, 1.0
        elif op in ("charge_radius", "rayleigh_even"):
            cs, cp = 1.0, 0.0

        print(f"\n  [CMB] {actual_dm_type} / {op}  (n={OPERATOR_N.get(op, '?')})")
        boundary = cmb_exclusion_boundary_analytic(
            m_chi_arr,
            dm_type=actual_dm_type, operator=op,
            c_s=cs, c_p=cp, c_phi=cphi, majorana=majorana,
        )
        print(f"         boundary points found: {len(boundary)}")
        save_cmb_boundary_npz(
            boundary, actual_dm_type, op,
            c_s=cs, c_p=cp, c_phi=cphi, majorana=majorana,
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CMB POWER SPECTRUM CONSTRAINTS (Planck 2018 estimate)")
    print("=" * 60)

    m_chi_arr  = np.logspace(-10, 20, 40)
    Lambda_arr = np.logspace(-10,  7, 40)

    # Fermionic operators
    run_cmb_constraints(
        m_chi_arr, Lambda_arr,
        operators=["dipole_magnetic", "dipole_electric",
                   "charge_radius", "anapole",
                   "rayleigh_even", "rayleigh_odd"],
        dm_type="fermionic",
        majorana=False,
    )

    # Scalar Rayleigh
    run_cmb_constraints(
        m_chi_arr, Lambda_arr,
        operators=["scalar_rayleigh"],
        dm_type="scalar",
        c_phi=1.0,
        majorana=False,
    )

    # Majorana Rayleigh rates are 2x larger than Dirac, so the CMB Lambda bound is tighter.
    run_cmb_constraints(
        m_chi_arr, Lambda_arr,
        operators=["rayleigh_even", "rayleigh_odd"],
        dm_type="fermionic",
        majorana=True,
    )

    # Majorana anapole uses the same CMB rate as Dirac for this purpose.
    run_cmb_constraints(
        m_chi_arr, Lambda_arr,
        operators=["anapole"],
        dm_type="fermionic",
        majorana=True,
    )

    print("\nDone. Boundaries saved to constraint_boundaries/cmb_*.npz")
