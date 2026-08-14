import numpy as np
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.eft_validity import (
    eft_kinematic_lambda_curve as _shared_eft_kinematic_lambda_curve,
)
# Canonical lab-frame kinematics live in cross_sections.py; imported rather
# than duplicated here.
from core.cross_sections import (
    get_s_lab_DMrest,
    get_t_lab_DMrest,
    lab_recoil_ratio,
)
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# =============================================================================
# CONSTANTS & NFW PARAMETERS (Totani / Via Lactea II)
# =============================================================================

# NFW parameters from Totani (Via Lactea II)
rho_s_tot   = 8.1e6          # M_sun / kpc^3  — Totani scale density
r_s_tot     = 21.0           # kpc             — Totani scale radius
r_sun       = 8.0            # kpc             — Sun–GC distance (Totani uses 8 kpc)
r_vir       = 402.0          # kpc             — virial radius

# Conversion factors
MSUN_KPC3_TO_GEV_CM3 = 3.817e-8   # 1 M_sun/kpc^3 = 3.817e-8 GeV/cm^3
KPC_TO_CM             = 3.0857e21  # 1 kpc in cm

rho_s_GeV = rho_s_tot * MSUN_KPC3_TO_GEV_CM3   # GeV/cm^3
r_s_cm    = r_s_tot   * KPC_TO_CM               # cm
r_sun_cm  = r_sun     * KPC_TO_CM               # cm
r_vir_cm  = r_vir     * KPC_TO_CM               # cm

# ---------------------------------------------------------------------------
# Halo energy bins and spectrum, read from the pixel-level MCMC posterior
# (pixelwise_mcmc_results_fig6, NFW rho^2, disk excluded).
#
# E_BINS_GEV  : 13 bin centres from Ectr_mev stored in mcmc_results_k*.npz
# PHI_TOTANI  : f_nfw_p50 * iso_target_e2  [MeV cm^-2 s^-1 sr^-1]
# SIGMA_TOTANI: 0.5*(f_p84 - f_p16) * iso_target_e2  (symmetric 1-sigma)
#
# Peak bin: k=5, E=20.757 GeV, phi=3.360e-5, SNR=14.14
# ---------------------------------------------------------------------------

def _load_totani_mcmc_arrays(mcmc_dir=None):
    """Load E_bins, phi, sigma from the pixel-level halo posterior.

    Parameters
    ----------
    mcmc_dir : Path or str, optional
        Directory containing mcmc_results_k*.npz files. If None, the rho^2
        entry of totani_data_loader._MCMC_DIRS is used, which resolves through
        HALO_POSTERIOR_ROOT / REPO_PATH; see README.md.
    """
    import os
    if mcmc_dir is None:
        # Environment override: lets tests point at (possibly synthetic)
        # posteriors without editing this file.
        mcmc_dir = os.environ.get("TOTANI_MCMC_DIR") or None
    if mcmc_dir is not None:
        _pc_dir = Path(mcmc_dir)
    else:
        from core.totani_data_loader import _MCMC_DIRS
        _pc_dir = _MCMC_DIRS["pixelwise_global_rho2"]
    if not _pc_dir.exists() or not any(_pc_dir.glob("mcmc_results_k*.npz")):
        raise FileNotFoundError(
            f"Halo posterior not found at {_pc_dir} (expected "
            "mcmc_results_k*.npz). It ships with the companion release, "
            "doi:10.5281/zenodo.21280725; point HALO_POSTERIOR_ROOT at it, or "
            "pass an explicit mcmc_dir. Only the exclusion-grid scan needs it "
            "-- the figure scripts read the committed grids instead."
        )

    iso_e2 = 1e-4   # MeV cm^-2 s^-1 sr^-1 (default; overwritten per bin)
    nE = 13
    E_mev  = np.full(nE, np.nan)
    f_p50  = np.full(nE, np.nan)
    f_p16  = np.full(nE, np.nan)
    f_p84  = np.full(nE, np.nan)
    iso_arr = np.full(nE, iso_e2)

    for k in range(nE):
        npz_path = _pc_dir / f"mcmc_results_k{k:02d}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        if "Ectr_mev" in npz.files:
            E_mev[k] = float(npz["Ectr_mev"])
        if "iso_target_e2" in npz.files:
            v = float(np.atleast_1d(npz["iso_target_e2"]).flat[0])
            if np.isfinite(v) and v > 0:
                iso_arr[k] = v
        labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]
        nfw_idx = next((i for i, l in enumerate(labels) if "nfw" in l.lower()), None)
        if nfw_idx is None:
            continue
        if "f_p50" in npz.files:
            f_p50[k] = float(np.atleast_1d(npz["f_p50"])[nfw_idx])
        if "f_p16" in npz.files:
            f_p16[k] = float(np.atleast_1d(npz["f_p16"])[nfw_idx])
        if "f_p84" in npz.files:
            f_p84[k] = float(np.atleast_1d(npz["f_p84"])[nfw_idx])

    phi   = f_p50 * iso_arr
    sigma = 0.5 * np.abs(f_p84 - f_p16) * iso_arr
    E_GeV = E_mev / 1000.0
    return E_GeV, phi, sigma


_LAZY_ARRAYS = ("E_BINS_GEV", "PHI_TOTANI", "SIGMA_TOTANI")


def _halo_arrays():
    """(E_BINS_GEV, PHI_TOTANI, SIGMA_TOTANI), loading the posterior on demand.

    These three arrays are only ever the *defaults* of the scan machinery --
    every production call passes its spectrum explicitly. Loading them eagerly
    would make importing this module require the companion posterior, which the
    figure scripts do not need: they read the committed exclusion grids.
    """
    if "PHI_TOTANI" not in globals():
        globals().update(zip(_LAZY_ARRAYS, _load_totani_mcmc_arrays()))
    return tuple(globals()[n] for n in _LAZY_ARRAYS)


def __getattr__(name):
    """Resolve the lazy halo arrays for importers (PEP 562)."""
    if name in _LAZY_ARRAYS:
        return _halo_arrays()[_LAZY_ARRAYS.index(name)]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def configure_totani_arrays(mcmc_dir) -> None:
    """Reconfigure the module-level E_BINS_GEV / PHI_TOTANI / SIGMA_TOTANI
    to use data from *mcmc_dir* (must contain mcmc_results_k*.npz files).

    Call this once at the start of any script that selects a non-default
    halo profile, e.g. global_rho2.5 or global_rho1, so that all
    downstream defaults derived from these arrays are consistent.
    """
    globals().update(zip(_LAZY_ARRAYS, _load_totani_mcmc_arrays(Path(mcmc_dir))))


# Forward-scattering angular cutoff. PHYSICAL, not merely a regulator.
#
# 1 - cos(theta) < 1e-4 corresponds to a deflection of 0.81 deg, comparable to
# the Fermi-LAT PSF (68% containment ~0.8 deg at 1 GeV; Atwood+ 2009). A photon
# scattered inside this cone is not resolved as displaced from its original
# direction, and by the Compton relation it also retains essentially all of its
# energy, so it has not been removed from the measured spectrum in any
# observable sense. sigma_tot below is therefore the cross section for
# OBSERVABLE REMOVAL from the line of sight, not the full elastic cross
# section. See Sec. IV A of the paper.
#
# It originated as a regulator for the gravitational case (t -> 0 Rutherford
# divergence) and was long assumed redundant for the EFT operators, on the
# grounds that dsigma/dOmega ~ (-t)^n with n >= 1 is regular at theta = 0.
# That assumption is FALSE for omega >> m_chi: the Compton kinematics compress
# most of the momentum-transfer range into 1 - cos(theta) <~ m_chi/omega, so
# once m_chi/omega < 1e-4 the cut removes the bulk of the t-integral. Fraction
# of the full elastic sigma_tot retained:
#
#   omega/m_chi     1e2     1e3     1e4     1e5     1e7
#   scalar         1.000   0.999   0.875   0.249   0.003
#   Rayleigh-odd   1.000   1.000   0.938   0.317   0.004
#   dipole         0.990   0.909   0.500   0.091   0.001
#
# Do not raise this to 1.0 without also changing what the paper claims
# sigma_tot means.
COS_THETA_MAX = 1.0 - 1e-4

# Unit conversions
HC2_GEV2_TO_M2 = 3.89379e-32   # 1 GeV^-2 = 3.89379e-32 m^2
GEV2_TO_FB     = 3.89379e11    # 1 GeV^-2 = 3.89379e11 fb
FB_TO_CM2      = 1e-39         # 1 fb = 1e-39 cm^2
ALPHA_EM       = 1.0 / 137.035999084


_ROI_PREFAC_CACHE = {}


OPERATOR_METADATA = {
    "dipole_magnetic": {
        # Real-photon Compton via a dipole coupling is a two-insertion process,
        # so sigma ~ Lambda^-4 (not Lambda^-2, the DM--charged-matter
        # single-insertion scaling).
        "paper_label": r"$O_{\psi\psi F}$",
        "lambda_power": 4,
        "coefficient_power": 1.0,
    },
    "dipole_electric": {
        # Same two-insertion scaling as dipole_magnetic.
        "paper_label": r"$O_{\psi5\psi F}$",
        "lambda_power": 4,
        "coefficient_power": 1.0,
    },
    "charge_radius": {
        "paper_label": r"$O_{\psi\psi \partial F}$",
        "lambda_power": 4,
        "coefficient_power": 0.5,
    },
    "anapole": {
        "paper_label": r"$O_{\chi5\chi \partial F}$",
        "lambda_power": 4,
        "coefficient_power": 0.5,
    },
    "rayleigh_even": {
        "paper_label": r"$O_{\chi\chi FF}$ / $O_{\psi\psi FF}$",
        "lambda_power": 6,
        "coefficient_power": 1.0 / 3.0,
    },
    "rayleigh_odd": {
        "paper_label": r"$O_{\chi5\chi FF}$ / $O_{\psi5\psi FF}$",
        "lambda_power": 6,
        "coefficient_power": 1.0 / 3.0,
    },
    "rayleigh_full": {
        # The combined dim-7 case, c_s and c_p both non-zero. It had NO entry,
        # so _paper_y_axis_values took its early return and handed back bare
        # Lambda: the dim-7 combined panel's axis was labelled as rescaled
        # while performing no rescaling. Invisible at the c = 1 benchmark,
        # since 1^(1/3) = 1, which is why it survived unnoticed. Same powers as
        # rayleigh_even/odd, which it is the incoherent sum of.
        "paper_label": r"$O_{\chi\chi FF} + O_{\chi5\chi FF}$",
        "lambda_power": 6,
        "coefficient_power": 1.0 / 3.0,
    },
    "scalar_rayleigh": {
        "paper_label": r"$O_{RRFF}$",
        "lambda_power": 4,
        "coefficient_power": 0.5,
    },
}


# =============================================================================
# NFW PROFILE AND GEOMETRY
# =============================================================================

def nfw_density_GeV_cm3(r_cm):
    """NFW density in GeV/cm^3 using Totani / Via Lactea II parameters."""
    x = r_cm / r_s_cm
    return rho_s_GeV / (x * (1.0 + x)**2)


def galactocentric_radius_cm(ell_cm, l_deg, b_deg):
    """Galactocentric radius for a point at distance ell_cm along line of sight (l, b)."""
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    r2 = (ell_cm**2
          + r_sun_cm**2
          - 2.0 * ell_cm * r_sun_cm * np.cos(b) * np.cos(l))
    return np.sqrt(np.maximum(r2, 0.0))


def los_length_cm(l_deg, b_deg, r_vir_cm=r_vir_cm):
    """Maximum integration length along line of sight (l, b) out to virial radius."""
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    A = 1.0
    B = -2.0 * r_sun_cm * np.cos(b) * np.cos(l)
    C = r_sun_cm**2 - r_vir_cm**2
    disc = np.maximum(B**2 - 4.0 * A * C, 0.0)
    L = (-B + np.sqrt(disc)) / 2.0
    return L


def compute_J_los(l_deg, b_deg, power=2, n_points=500):
    """J-factor integral  J = int rho^power dl  along line of sight (l, b)."""
    L = los_length_cm(l_deg, b_deg)
    ell = np.linspace(0.0, L, n_points)
    r   = galactocentric_radius_cm(ell, l_deg, b_deg)
    rho = nfw_density_GeV_cm3(r)
    integrand = rho**power
    return np.trapezoid(integrand, ell)


# =============================================================================
# EFFECTIVE OPERATOR CROSS SECTIONS
# =============================================================================

def get_s_max_lab_DMrest(mchi, omega_max):
    return float(mchi)**2 + 2.0 * float(mchi) * float(omega_max)


def get_t_abs_max_lab_DMrest(mchi, omega_max):
    denom = 1.0 + (2.0 * float(omega_max) / float(mchi))
    return 4.0 * float(omega_max)**2 / denom


def eft_validity_lambda_curve(m_chi_arr, *, omega_max, eft_kinematic_factor=1.0):
    return _shared_eft_kinematic_lambda_curve(
        m_chi_arr,
        omega_max=omega_max,
        eft_kinematic_factor=eft_kinematic_factor,
    )


def lab_dsigma_prefactor(mchi, omega, theta):
    """
    Lab-frame two-body phase-space prefactor for gamma + chi(rest) -> gamma + chi.

    For target-at-rest Compton kinematics, dσ/dΩ_lab carries an extra
    (omega'/omega)^2 recoil Jacobian and m_chi^2 in the denominator.  Using
    the CM expression 1/(64π²s) directly with the lab scattering angle gives
    the wrong angular/energy weighting once omega/m_chi is not tiny.
    """
    ratio = lab_recoil_ratio(mchi, omega, theta)
    return ratio**2 / (64.0 * np.pi**2 * mchi**2)


# ---------- FERMIONIC DM OPERATORS ----------

def dsigma_dOmega_fermionic(mchi, theta, E_gamma, c_s, c_p, Lambda, operator="rayleigh_full", majorana=False):
    """
    Differential cross section for fermionic DM effective operators.
    
    Parameters
    ----------
    mchi : float
        DM mass [GeV]
    theta : array
        Scattering angle [radians]
    E_gamma : float
        Photon energy [GeV]
    c_s, c_p : float
        Wilson coefficients (scalar/pseudoscalar)
    Lambda : float
        EFT cutoff scale [GeV]
    operator : str
        Operator type: 'rayleigh_full', 'rayleigh_even', 'rayleigh_odd',
        'dipole_magnetic', 'dipole_electric', 'charge_radius', 'anapole'
    majorana : bool
        If True, enforces that dipole operators are zero (they vanish
        for Majorana fermions by charge-conjugation symmetry).
        Only anapole, rayleigh_even, rayleigh_odd are non-zero for Majorana DM.

    Returns
    -------
    dsigma/dOmega [fb/sr]

    Amplitude-squared conventions
    -----------------------------
    All |M|^2 expressions are summed (not averaged) over initial DM spins
    and summed over final spins, then divided by the flux factor 2s in the
    lab frame. The phase-space denominator is therefore 64 pi^2 s (not
    128 pi^2 s).

    References
    ----------
      - Operator |M|^2 framework: Arina et al. (2020), "Light and Darkness:
        consistently coupling dark matter to photons via effective operators",
        arXiv:2005.12789 (EPJC 81 (2021) 333) — dim 5-7 operator basis and
        cross sections used here.
      - Direct-detection cross-check: Ibarra, Reichard & Tomar (2024), "Probing
        Dark Matter Electromagnetic Properties in Direct Detection Experiments",
        arXiv:2408.15760 (JCAP 2025) — all five EM operators incl. anapole and
        charge radius, from LZ + XENON1T + Migdal.
      - Dipole (MDM/EDM): Sigurdson et al. (2004), "Dark-Matter Electric and
        Magnetic Dipole Moments"; cf. Banks et al. (2010).
      - Rayleigh (scalar & fermionic): Weiner & Yavin (2012 MiDM, 2013 UV
        completions); "Scalar Rayleigh Dark Matter" review.

    Spin convention
    ---------------
      * lab_dsigma_prefactor = (omega'/omega)^2 / (64 pi^2 m^2) is the
        INITIAL-AVERAGED convention: multiplying it by <|M|^2> averaged over
        initial DM spin (1/2) and photon polarisation (1/2), summed over final,
        gives dsigma/dOmega_lab. This machinery reproduces the Klein-Nishina
        formula for QED Compton (tests/test_eft_realphoton_amplitudes.py).
      * The dipole branch below uses this averaged convention.
      * Rayleigh normalisation (factor 64): the operator source is Weiner &
        Yavin, arXiv:1206.2910, who define
        L = (c/4 Lambda^3) [chibar chi F F + chibar i g5 chi F Ftilde]  -- WITH
        the 1/4.  The 64 = 16 [(1/4)^2 in |M|^2] x 4 [initial-state averaging].
        Naive single-contraction Feynman rules in this convention reproduce WY
        Eq. (15), sigma v(chi chi -> gamma gamma) = g^2 m^4/(4 pi
        Lambda^6), exactly (tests/test_eft_realphoton_amplitudes.py::
        test_weiner_yavin_annihilation_anchor).  The coded even/odd/full
        branches are therefore exactly the initial-averaged |M|^2 in the WY
        convention -- main.tex Eq. (O_rayleigh) carries the 1/4.
      * The SCALAR Rayleigh branch does NOT carry the 1/4.  Its operator source
        is Barducci et al., arXiv:2501.09073 (Eqs. 2.1/2.3/2.5), who write
        L = C phi^2 F F with no 1/4; main.tex Eq. (O_scalar_rayleigh) matches.
        Its combinatorics are given in dsigma_dOmega_scalar.
        Residual O(1) normalisation ambiguity, noted but not adopted here: if
        the doubled Majorana bilinear contraction <chi'|chibar Gamma chi|chi>
        = 2 ubar Gamma u is used instead of WY's effective naive rule, the
        Majorana Rayleigh sigma rises x4 (Lambda limit x4^(1/6) ~ 1.26).
    """

    t = get_t_lab_DMrest(mchi, E_gamma, theta)
    phase = lab_dsigma_prefactor(mchi, E_gamma, theta)
    
    MAJORANA_FORBIDDEN = {"dipole_magnetic", "dipole_electric", "charge_radius"}
    if majorana and operator in MAJORANA_FORBIDDEN:
        raise ValueError(
            f"Operator '{operator}' vanishes for Majorana DM. "
            f"Use anapole, rayleigh_even, or rayleigh_odd instead."
        )

    if operator == "rayleigh_even":
        amp2 = c_s**2 * (4 * mchi**2 - t) * t**2 / (4.0 * Lambda**6)
        val = amp2 * phase
    elif operator == "rayleigh_odd":
        amp2 = c_p**2 * (-t)**3 / (4.0 * Lambda**6)
        val = amp2 * phase
    elif operator == "rayleigh_full":
        amp2 = (
            c_s**2 * (4 * mchi**2 - t) * t**2 / (4.0 * Lambda**6)
            + c_p**2 * (-t)**3 / (4.0 * Lambda**6)
        )
        val = amp2 * phase
    elif operator in ("dipole_magnetic", "dipole_electric"):
        # gamma chi -> gamma chi through a dipole coupling is a TWO-insertion
        # Compton process (s+u channel chi exchange): |M|^2 ~ (c/Lambda)^4, no
        # explicit alpha_EM factor. This differs from the DM--charged-matter
        # (direct-detection) single-insertion cross section used in the
        # Ibarra et al. framework, which applies to a different process
        # [4*ALPHA_EM*(c^2/Lambda^2)*(-t)].
        #
        # Exact tree-level result (derived and verified symbolically at exact
        # rational kinematics; trace machinery checked against Klein-Nishina):
        #   <|M|^2> = 4 mu^4 [ -ab - 2 m^2 t + 2 m^4 t^2 / (ab) ],
        # with mu = 2c/Lambda (operator (c/Lambda) chibar sigma^{mu nu} chi F_mu_nu),
        # a = s - m^2 = 2 m omega, b = u - m^2 = -2 m omega', ab = -4 m^2 omega omega'.
        # Averaged over initial DM spin (1/2) and photon polarisation (1/2),
        # summed over final -- the convention lab_dsigma_prefactor expects.
        # Magnetic and electric dipole give the identical spin-averaged real-photon
        # Compton |M|^2 (electromagnetic duality); they differ only in DM--matter
        # scattering. Low-energy limit: sigma -> 4 mu^4 omega^2 / (3 pi)  [~ mu^4 E^2,
        # magnetic-Thomson analogue; parametric form cf. Sigurdson et al. 2004].
        c = c_s if operator == "dipole_magnetic" else c_p
        omega_p = E_gamma * lab_recoil_ratio(mchi, E_gamma, theta)
        amp2 = (64.0 * c**4 / Lambda**4) * mchi**2 * (
            4.0 * E_gamma * omega_p - 2.0 * t - t**2 / (2.0 * E_gamma * omega_p)
        )
        val = amp2 * phase
    elif operator in ("charge_radius", "anapole"):
        # Both operators couple through d^nu F_nu_mu, whose momentum-space
        # vertex -(q^2 eps_mu - q_mu q.eps) carries the transverse projector
        # and vanishes identically for on-shell external photons (q^2 = 0,
        # q.eps = 0).  The tree-level gamma chi -> gamma chi amplitude is
        # therefore exactly zero: these are q^2-dependent form factors that
        # couple DM to *virtual* photons (charged matter), not to free
        # radiation [Latimer 2017, arXiv:1706.08029; Kavanagh, Panci &
        # Ziegler]. The DM--charged-matter form-factor cross section
        # [4*ALPHA_EM*(c^2/Lambda^4)*t^2] applies to that different observable.
        # Kept in the catalogue so the paper can state the vanishing explicitly.
        val = np.zeros_like(np.asarray(t, dtype=float) * np.asarray(theta, dtype=float))
    else:
        raise ValueError(f"Unknown operator: {operator}")
    
    return val * GEV2_TO_FB


# ---------- SCALAR DM OPERATORS ----------

def dsigma_dOmega_scalar(mchi, theta, E_gamma, c_phi, Lambda):
    """
    Differential cross section for scalar DM effective operator.
    
    Parameters
    ----------
    mchi : float
        DM mass [GeV]
    theta : array
        Scattering angle [radians]
    E_gamma : float
        Photon energy [GeV]
    c_phi : float
        Wilson coefficient
    Lambda : float
        EFT cutoff scale [GeV]
    
    Returns
    -------
    dsigma/dOmega [fb/sr]
    """
    t = get_t_lab_DMrest(mchi, E_gamma, theta)
    phase = lab_dsigma_prefactor(mchi, E_gamma, theta)

    # Initial-photon-averaged |M|^2 for the real-scalar Rayleigh operator
    #   O = (c_phi / Lambda^2) phi^2 F_mu_nu F^mu_nu,
    # the normalisation of Barducci et al. (arXiv:2501.09073, Eqs. 2.1/2.3/2.5),
    # which carries NO factor of 1/4 -- unlike the fermionic Rayleigh operator,
    # where the 1/4 of Weiner & Yavin is retained so that Lambda matches their
    # Lambda_R.  The combinatorial factors are: 2 from the phi^2 contraction,
    # 2 from assigning the two photons to the two F's, and 2 from the
    # F^(1).F^(2) contraction, giving M = (8 c_phi / Lambda^2) x
    # [(k1.k2)(e1.e2) - (k1.e2)(k2.e1)].  Summing over final and averaging over
    # the two initial photon polarisations with
    # sum_pol |...|^2 = 2 (k1.k2)^2 and k1.k2 = -t/2 gives 16 c_phi^2 t^2 / Lambda^4.
    amp2 = 16.0 * c_phi**2 * t**2 / Lambda**4
    val = amp2 * phase
    return val * GEV2_TO_FB


# ---------- TOTAL CROSS SECTIONS ----------

def sigma_tot_fermionic(E_gamma, mchi, c_s, c_p, Lambda, operator="rayleigh_full", majorana=False, n_theta=200):
    """Total cross section for fermionic DM [cm^2]."""
    cos_vals = np.linspace(-1.0, COS_THETA_MAX, n_theta)
    theta_vals = np.arccos(cos_vals)
    
    dsig = dsigma_dOmega_fermionic(mchi, theta_vals, E_gamma, c_s, c_p, Lambda, operator, majorana=majorana)
    dsig = np.nan_to_num(dsig, nan=0.0, posinf=0.0, neginf=0.0)
    
    sig = 2.0 * np.pi * np.trapezoid(dsig, cos_vals) * FB_TO_CM2
    return sig


def sigma_tot_fermionic_array(E_gamma_arr, mchi, c_s, c_p, Lambda, operator="rayleigh_full", majorana=False, n_theta=200):
    """Vectorized total cross section for fermionic DM [cm^2] over an energy array."""
    E_gamma_arr = np.asarray(E_gamma_arr, dtype=float)
    cos_vals = np.linspace(-1.0, COS_THETA_MAX, n_theta)
    theta_vals = np.arccos(cos_vals)

    # Broadcast to (nE, nTheta)
    E2 = E_gamma_arr[:, None]
    th2 = theta_vals[None, :]
    dsig = dsigma_dOmega_fermionic(mchi, th2, E2, c_s, c_p, Lambda, operator, majorana=majorana)
    dsig = np.nan_to_num(dsig, nan=0.0, posinf=0.0, neginf=0.0)
    dsig = np.where(dsig > 0.0, dsig, 0.0)

    sig_fb = 2.0 * np.pi * np.trapezoid(dsig, cos_vals, axis=1)
    return sig_fb * FB_TO_CM2


def sigma_tot_scalar(E_gamma, mchi, c_phi, Lambda, n_theta=200):
    """Total cross section for scalar DM [cm^2]."""
    cos_vals = np.linspace(-1.0, COS_THETA_MAX, n_theta)
    theta_vals = np.arccos(cos_vals)
    
    dsig = dsigma_dOmega_scalar(mchi, theta_vals, E_gamma, c_phi, Lambda)
    dsig = np.nan_to_num(dsig, nan=0.0, posinf=0.0, neginf=0.0)
    
    sig = 2.0 * np.pi * np.trapezoid(dsig, cos_vals) * FB_TO_CM2
    return sig


def sigma_tot_scalar_array(E_gamma_arr, mchi, c_phi, Lambda, n_theta=200):
    """Vectorized total cross section for scalar DM [cm^2] over an energy array."""
    E_gamma_arr = np.asarray(E_gamma_arr, dtype=float)
    cos_vals = np.linspace(-1.0, COS_THETA_MAX, n_theta)
    theta_vals = np.arccos(cos_vals)

    E2 = E_gamma_arr[:, None]
    th2 = theta_vals[None, :]
    dsig = dsigma_dOmega_scalar(mchi, th2, E2, c_phi, Lambda)
    dsig = np.nan_to_num(dsig, nan=0.0, posinf=0.0, neginf=0.0)
    dsig = np.where(dsig > 0.0, dsig, 0.0)

    sig_fb = 2.0 * np.pi * np.trapezoid(dsig, cos_vals, axis=1)
    return sig_fb * FB_TO_CM2


def _roi_prefactor_key(l_grid, b_grid):
    l_grid = np.asarray(l_grid, dtype=float)
    b_grid = np.asarray(b_grid, dtype=float)
    return (
        tuple(np.round(l_grid, 8).tolist()),
        tuple(np.round(b_grid, 8).tolist()),
    )


def roi_tau_prefactor(l_grid, b_grid, n_points_J=500):
    """
    Return K = sum_{ROI} (J1 * J2) / sum_{ROI} J2  [GeV/cm^2]

    where J1 = int rho dl  (column density, linear in rho — enters tau)
          J2 = int rho^2 dl (emissivity — enters the NFW-rho^2 halo flux)

    The rho^2 weighting is intentional: Totani's halo excess photons are
    produced by DM annihilation (rho^2 emissivity), so high-emissivity
    lines of sight contribute more to the observed flux and should receive
    proportionally more weight in the mean optical depth.

    tau_bar(E) = K * sigma(E) / m_chi
    """
    key = _roi_prefactor_key(l_grid, b_grid)
    cached = _ROI_PREFAC_CACHE.get(key)
    if cached is not None:
        return float(cached)

    l_grid = np.asarray(l_grid, dtype=float)
    b_grid = np.asarray(b_grid, dtype=float)
    sum_J2 = 0.0
    sum_J1J2 = 0.0

    for b in b_grid:
        for l in l_grid:
            J2 = float(compute_J_los(float(l), float(b), power=2, n_points=int(n_points_J)))
            if not np.isfinite(J2) or J2 <= 0.0:
                continue

            L = los_length_cm(float(l), float(b))
            ell = np.linspace(0.0, L, int(n_points_J))
            r = galactocentric_radius_cm(ell, float(l), float(b))
            rho = nfw_density_GeV_cm3(r)
            J1 = float(np.trapezoid(rho, ell))

            if not np.isfinite(J1) or J1 <= 0.0:
                continue

            sum_J2 += J2
            sum_J1J2 += (J1 * J2)

    if sum_J2 <= 0.0:
        raise ValueError("ROI prefactor failed: sum_J2 <= 0.")

    K = sum_J1J2 / sum_J2
    _ROI_PREFAC_CACHE[key] = float(K)
    return float(K)


# =============================================================================
# OPTICAL DEPTH CALCULATIONS
# =============================================================================

def compute_tau_los_eft(l_deg, b_deg, E_GeV, m_chi_GeV, Lambda_GeV, 
                        dm_type="fermionic", operator="rayleigh_full",
                        c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False, n_los=300):
    """
    Optical depth along a single line of sight for EFT operators.
    
    tau = (1/m_chi) * int rho(r(ell)) * sigma dl
    """
    if dm_type == "fermionic":
        sig = sigma_tot_fermionic(E_GeV, m_chi_GeV, c_s, c_p, Lambda_GeV, operator, majorana=majorana)
    else:  # scalar
        sig = sigma_tot_scalar(E_GeV, m_chi_GeV, c_phi, Lambda_GeV)
    
    L   = los_length_cm(l_deg, b_deg)
    ell = np.linspace(0.0, L, n_los)
    r   = galactocentric_radius_cm(ell, l_deg, b_deg)
    rho = nfw_density_GeV_cm3(r)
    
    J_los = np.trapezoid(rho, ell)  # GeV/cm^2
    tau = J_los * sig / m_chi_GeV
    
    return tau


def compute_tau_bar_eft(E_GeV, m_chi_GeV, Lambda_GeV, l_grid, b_grid,
                        dm_type="fermionic", operator="rayleigh_full",
                        c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False):
    """
    Emissivity-weighted average optical depth over the ROI.
    
    tau_bar(E) = sum_{l,b} [tau(l,b) * J2(l,b)] / sum_{l,b} J2(l,b)
    """
    if dm_type == "fermionic":
        sig = sigma_tot_fermionic(E_GeV, m_chi_GeV, c_s, c_p, Lambda_GeV, operator, majorana=majorana)
    else:
        sig = sigma_tot_scalar(E_GeV, m_chi_GeV, c_phi, Lambda_GeV)

    K = roi_tau_prefactor(l_grid, b_grid)
    return float(K) * float(sig) / float(m_chi_GeV)


def compute_tau_bar_spectrum_eft(E_bins, m_chi_GeV, Lambda_GeV, l_grid, b_grid,
                                  dm_type="fermionic", operator="rayleigh_full",
                                  c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False,
                                  K_override=None):
    """Compute tau_bar at each energy bin.

    Parameters
    ----------
    K_override : float or None, optional
        Scalar column-density prefactor K [GeV/cm^2] to use in place of the
        emissivity-weighted ROI average computed from ``l_grid``/``b_grid``.
        When set, the ROI integration is bypassed entirely. Use this when the
        line-of-sight column is a well-defined single scalar (dwarf spheroidals
        with a catalog J-factor; the cosmological IGRB baseline). The default
        ``None`` preserves the original halo-ROI behaviour via
        :func:`roi_tau_prefactor`.
    """
    E_bins = np.asarray(E_bins, dtype=float)
    if dm_type == "fermionic":
        sig_arr = sigma_tot_fermionic_array(E_bins, m_chi_GeV, c_s, c_p, Lambda_GeV, operator, majorana=majorana)
    else:
        sig_arr = sigma_tot_scalar_array(E_bins, m_chi_GeV, c_phi, Lambda_GeV)

    if K_override is not None:
        K = float(K_override)
    else:
        K = roi_tau_prefactor(l_grid, b_grid)
    tau_bar = (float(K) / float(m_chi_GeV)) * np.asarray(sig_arr, dtype=float)
    tau_bar = np.nan_to_num(tau_bar, nan=0.0, posinf=0.0, neginf=0.0)
    tau_bar = np.where(tau_bar > 0.0, tau_bar, 0.0)
    return tau_bar


# =============================================================================
# OPERATOR METADATA AND BOUNDARY EXTRACTION
# =============================================================================

def _operator_metadata(dm_type, operator):
    if dm_type == "scalar" and operator == "rayleigh":
        key = "scalar_rayleigh"
    else:
        key = operator
    return OPERATOR_METADATA.get(key, {
        "paper_label": operator,
        "lambda_power": None,
        "coefficient_power": None,
    })


def _effective_coefficient(dm_type, operator, c_s=1.0, c_p=1.0, c_phi=1.0):
    if dm_type == "scalar":
        return float(c_phi)
    if operator in ("dipole_magnetic", "charge_radius", "rayleigh_even"):
        return float(c_s)
    if operator in ("dipole_electric", "anapole", "rayleigh_odd"):
        return float(c_p)
    if operator == "rayleigh_full":
        # Incoherent sum of the even (c_s) and odd (c_p) amplitudes. The scan
        # sets c_s = c_p, so the rescaling is well defined; if they differ there
        # is no single c and the rescaled axis is not meaningful, so we say so
        # rather than silently returning 1.0 as the old fallback did.
        if float(c_s) != float(c_p):
            raise ValueError(
                "rayleigh_full rescaling needs c_s == c_p; got "
                f"c_s={c_s!r}, c_p={c_p!r}. The combined operator has no single "
                "Wilson coefficient when the two differ."
            )
        return float(c_s)
    return 1.0


# Fraction of the halo that scatters. tau ~ f_scat * c^2 / Lambda^p, so f_scat
# is EXACTLY degenerate with the Wilson coefficient and enters the rescaled axis
# through the same invariant combination:
#
#     Lambda / (c^2 f_scat)^(1/p),   p = 4 (dim-5 dipoles, dim-6 scalar Rayleigh)
#                                    p = 6 (dim-7 Rayleigh)
#
# (c^2)^(1/4) = c^(1/2) and (c^2)^(1/6) = c^(1/3), so this reproduces the
# existing c^(1/2) and c^(1/3) axes exactly at the default f_scat = 1: a limit
# quoted here assumes the WHOLE halo scatters, and a sub-component f_scat < 1
# weakens it by f_scat^(-1/p).
F_SCAT_DEFAULT = 1.0


# Lambda power p in sigma ~ Lambda^-p, used ONLY for the f_scat exponent. Some
# combined keys (notably rayleigh_full) carry no metadata entry, so
# coefficient_power and lambda_power are both None and the existing c-rescaling
# silently returns Lambda unchanged for them. That is pre-existing behaviour and
# is left alone here; this table exists so f_scat still gets the right exponent
# in those cases rather than being dropped along with it.
_FSCAT_POWER_FALLBACK = {
    "rayleigh_full": 6, "rayleigh_even": 6, "rayleigh_odd": 6,
    "scalar_rayleigh": 4, "dipole_magnetic": 4, "dipole_electric": 4,
}


def _paper_y_axis_values(Lambda_arr, dm_type, operator, c_s=1.0, c_p=1.0, c_phi=1.0,
                         f_scat=F_SCAT_DEFAULT):
    meta = _operator_metadata(dm_type, operator)
    coeff_power = meta.get("coefficient_power", None)
    if coeff_power is None:
        out = np.asarray(Lambda_arr, dtype=float)
        p = _FSCAT_POWER_FALLBACK.get(str(operator))
        if f_scat != 1.0 and p is not None:
            if not (f_scat > 0):
                raise ValueError(f"f_scat must be positive, got {f_scat!r}")
            out = out / (float(f_scat) ** (1.0 / p))
        return out
    coeff = _effective_coefficient(dm_type, operator, c_s=c_s, c_p=c_p, c_phi=c_phi)
    if coeff <= 0:
        raise ValueError("Operator coefficient must be positive for paper-style rescaling.")
    if not (f_scat > 0):
        raise ValueError(f"f_scat must be positive, got {f_scat!r}")
    p = float(meta.get("lambda_power", None) or _FSCAT_POWER_FALLBACK.get(str(operator), 4))
    out = np.asarray(Lambda_arr, dtype=float) / (coeff ** coeff_power)
    if f_scat != 1.0:                                # exact no-op at the default
        out = out / (f_scat ** (1.0 / p))
    return out


def extract_90cl_boundary(m_chi_arr, Lambda_arr, chi2_grid):
    """
    Extract the 90% CL lower-limit boundary from the actual
    matplotlib contour used in the chi2 plot.

    Points with Delta-chi^2 >= 4.61 are taken to be excluded. Since the
    EFT cross sections grow as Lambda decreases, the excluded region is
    the area below the 90% CL contour in the (m_chi, Lambda) plane.
    We therefore extract the upper envelope of the Delta-chi^2 = 4.61
    contour directly, so the saved boundary matches the plotted contour.
    """
    threshold = 4.61
    # nanmin, not min: a single NaN anywhere in the grid would otherwise make
    # chi2_min NaN, dchi2 all-NaN, and contour() return no segments at all --
    # silently reporting "no 90% CL contour" for a grid that has one. Grids
    # computed with the source normalisation held fixed contain no NaN, so this
    # is a no-op for them; profiled grids do, because best_fit_normalization()
    # returns NaN wherever the attenuated model underflows at large tau.
    chi2_min = np.nanmin(chi2_grid)
    dchi2 = chi2_grid - chi2_min

    M, L = np.meshgrid(m_chi_arr, Lambda_arr, indexing='ij')
    fig, ax = plt.subplots()
    cs = ax.contour(np.log10(M), np.log10(L), dchi2, levels=[threshold])
    segments = cs.allsegs[0] if cs.allsegs else []
    plt.close(fig)

    if not segments:
        return np.empty((0, 2))

    x_eval = np.linspace(np.log10(np.min(m_chi_arr)), np.log10(np.max(m_chi_arr)), 800)
    y_best = np.full_like(x_eval, np.nan, dtype=float)

    for seg in segments:
        seg = np.asarray(seg, dtype=float)
        if seg.ndim != 2 or seg.shape[0] < 2:
            continue

        x = seg[:, 0]
        y = seg[:, 1]
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]
        if x_unique.size < 2:
            continue

        mask = (x_eval >= x_unique.min()) & (x_eval <= x_unique.max())
        if not np.any(mask):
            continue

        y_interp = np.interp(x_eval[mask], x_unique, y_unique)
        current = y_best[mask]
        y_best[mask] = np.where(np.isnan(current), y_interp, np.maximum(current, y_interp))

    good = np.isfinite(y_best)
    if not np.any(good):
        return np.empty((0, 2))

    return np.column_stack((10 ** x_eval[good], 10 ** y_best[good]))

