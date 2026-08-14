"""
spectral_reshaping.py
=====================
Spectral transport for photon-DM scattering, Sec. II D of the paper.

The calculation is a halo-component transfer: scattering is applied to the
extracted NFW/PPPC halo component and compared against the halo-template
posterior. It is not a full sky-level forward model.

This module replaces the simple attenuation model
    Phi_att = Phi_0 * exp(-tau)
with the physically complete single-scatter redistribution:
    Phi_obs[i] = Phi_0[i]*exp(-tau[i]) + sum_j K[i,j]*tau[j]*Phi_0[j]*exp(-tau[j])

where K is the energy redistribution matrix built from the differential
cross section and DM-frame kinematics.

Public API
----------
ReshapingConfig                  : dataclass holding all run parameters
build_kernel(...)                : assemble redistribution matrix K[i,j], Eq. (II.13)
compute_tau_spectrum(...)        : tau(E), Eq. (II.7)
halo_component_transfer_spectrum(...): transferred single-component Phi_obs, Eq. (II.10)
reshaped_halo_spectrum(...)      : alias for the above
chi2_reshaping(...)              : chi^2 vs the measured spectrum at one (m_chi, Lambda)
scan_reshaping_chi2(...)         : 2D grid scan over (m_chi, Lambda)
save_reshaping_scan(...)         : save scan results to compressed .npz
load_reshaping_scan(...)         : reload scan results
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import os

# Local imports
from core.cross_sections import (
    get_flat_weak_cross_sections,
    get_t_lab_DMrest,
    get_s_lab_DMrest,
)
from core.attenuation_eft import (
    compute_tau_bar_spectrum_eft,
    sigma_tot_fermionic_array,
    sigma_tot_scalar_array,
    dsigma_dOmega_fermionic,
    dsigma_dOmega_scalar,
    roi_tau_prefactor,
    COS_THETA_MAX,
    FB_TO_CM2,
    OPERATOR_METADATA,
)
import core.attenuation_eft as _aeft
from core.kinematics import (
    reshaped_spectrum,
    scattered_energy_grid,
    max_energy_loss_fraction,
)

# ---------------------------------------------------------------------------
# Constants (replicated here for clarity; sourced from attenuation_eft.py)
# ---------------------------------------------------------------------------

_N_THETA_DEFAULT = 1000   # integration nodes in cos(theta); 1000 is sufficient
                           # for the dipole/Rayleigh operators at these energies


def configure_totani_arrays(mcmc_dir) -> None:
    """Reconfigure the default halo spectrum for the chosen profile.

    Call this once at the start of any pipeline script after selecting a halo
    profile, passing the corresponding entry from totani_data_loader._MCMC_DIRS.
    """
    _aeft.configure_totani_arrays(mcmc_dir)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReshapingConfig:
    """
    All parameters that define one reshaping calculation.

    Physical parameters
    -------------------
    m_chi : float
        DM mass [GeV].
    Lambda : float
        EFT cutoff scale [GeV]. Ignored for Higgs-portal mode.
    dm_type : str
        'fermionic' or 'scalar'.
    operator : str
        EFT operator key (must match attenuation_eft.OPERATOR_METADATA) or
        'higgs_portal' for the UV-complete Higgs-mediated calculation.
    c_s, c_p, c_phi : float
        Wilson coefficients (dimensionless). Default 1.0 each.
    majorana : bool
        If True, enforce Majorana constraints on operator selection.
    y_eff : float
        Effective Higgs-portal coupling [GeV] (only used when
        operator='higgs_portal').

    ROI / integration parameters
    -----------------------------
    l_grid, b_grid : array
        Galactic longitude/latitude grids for the emissivity-weighted
        optical depth average. Defaults match attenuation_eft.py driver.
    n_theta : int
        Number of cos(theta) integration nodes.
    cos_theta_max : float
        Upper limit on cos(theta) to avoid the forward-scattering divergence
        in gravitational cross sections. Set to 1.0 for EFT operators
        (which are regular at theta=0). Defaults to COS_THETA_MAX from
        attenuation_eft.py.
    apply_roi_weight : bool
        Whether to apply the ROI recovery weight w(theta) to the kernel.

    Energy axis
    -----------
    E_bins : array
        Photon energy bin centres [GeV]. Defaults to the halo posterior's
        13 native bins.
    phi_0 : array
        Intrinsic source spectrum to reshape. Defaults to the halo posterior;
        for the two-component annihilation variants pass a PPPC template from
        pppc_energy_flux_template.
    phi_data : array
        Observed target spectrum. Defaults to the halo posterior.
    phi_err : array
        1-sigma errors on phi_data. Defaults to the halo posterior's.
    fit_normalization : bool
        If True, chi-squared calculations fit the overall intrinsic template
        normalization analytically. This is the quantity that maps onto the
        required annihilation cross section for a fixed annihilation spectrum.
    max_tau_single_scatter : float or None
        Reject chi-squared points whose maximum optical depth is above this
        value. The transfer model keeps only zero- and one-scatter terms, so
        opaque points require a full cascade/radiative-transfer calculation.
    require_lambda_gt_mdm : bool
        If True, points with Lambda <= m_chi are treated as outside the EFT
        regime and return NaN chi-squared values.
    """

    # Physical
    m_chi: float = 600.0          # GeV
    Lambda: float = 1e3           # GeV
    dm_type: str = "fermionic"
    operator: str = "dipole_magnetic"
    c_s: float = 1.0
    c_p: float = 0.0
    c_phi: float = 1.0
    majorana: bool = False
    y_eff: float = 1.0            # Higgs portal only [GeV]

    # ROI
    l_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(-60.0, 60.0, 15)
    )
    b_grid: np.ndarray = field(
        default_factory=lambda: np.concatenate([
            np.linspace(-60.0, -10.0, 8),
            np.linspace(10.0, 60.0, 8),
        ])
    )
    # Optional scalar column-density prefactor K [GeV/cm^2]. When set, the ROI
    # integration in ``roi_tau_prefactor`` is bypassed and ``tau = K * sigma / m_chi``
    # is used directly. Populate this for datasets with a well-defined single
    # J-factor, i.e. the cosmological IGRB baseline. Leave as ``None`` for the
    # halo ROI-average path.
    tau_prefactor_override: Optional[float] = None

    # Integration
    #
    # sigma_tot is evaluated in closed form: F_i(u_min) - F_i(u_max) from
    # Eqs. (IV.12)-(IV.16) of the paper, with no angular grid anywhere. Verified
    # against the amplitude by symbolic differentiation and against resolved
    # quadrature to 1e-6 from m_chi = 1e-5 to 1e9 GeV
    # (tests/test_sigma_closed_form.py).

    # Fractional energy loss defining bin migration (native grid ratio 1.69,
    # half-width sqrt(1.69) -> f = 0.231). Eq. (IV.11).
    f_bin: float = 0.231
    # 1 - cos(theta_ROI) for the hard-cut removal model. None -> energy
    # migration only, which is the roi_recovery_model="unity" (IGRB) case; the
    # halo uses the computed w(theta) instead (see core/roi_recovery.py).
    x_roi: Optional[float] = None

    n_theta: int = _N_THETA_DEFAULT
    cos_theta_max: float = COS_THETA_MAX
    apply_roi_weight: bool = True

    # ROI recovery weight applied inside the kernel.
    #   "template" : roi_recovery.halo_recovery_fraction, computed from the
    #                rho^2 template over the actual box ROI. Not a cone at any
    #                radius: 0.895 at 5 deg, 0.403 at 60 deg, 0.019 at 120 deg.
    #                This is the halo arm.
    #   "unity"    : w == 1 identically -- the IGRB arm. An isotropic source
    #                seen against a scatterer of the cosmological mean density
    #                has no boundary for a photon to be deflected across: every
    #                photon scattered out of a line of sight is replaced by a
    #                statistically identical one scattered in. Angular escape
    #                removes nothing, so ENERGY MIGRATION is the only channel.
    #                See roi_recovery.igrb_recovery_fraction for the derivation.
    roi_recovery_model: str = "template"

    # Energy / spectrum
    E_bins: np.ndarray = field(default_factory=lambda: _aeft.E_BINS_GEV.copy())
    phi_0: np.ndarray = field(default_factory=lambda: _aeft.PHI_TOTANI.copy())
    phi_data: np.ndarray = field(default_factory=lambda: _aeft.PHI_TOTANI.copy())
    phi_err: np.ndarray = field(default_factory=lambda: _aeft.SIGMA_TOTANI.copy())
    fit_normalization: bool = True
    max_tau_single_scatter: Optional[float] = 0.3
    require_lambda_gt_mdm: bool = True

    def __post_init__(self):
        self.E_bins = np.asarray(self.E_bins, dtype=float)
        self.phi_0 = np.asarray(self.phi_0, dtype=float)
        self.phi_data = np.asarray(self.phi_data, dtype=float)
        self.phi_err = np.asarray(self.phi_err, dtype=float)
        self.l_grid = np.asarray(self.l_grid, dtype=float)
        self.b_grid = np.asarray(self.b_grid, dtype=float)

        if self.E_bins.shape != self.phi_0.shape:
            raise ValueError(
                f"E_bins ({self.E_bins.shape}) and phi_0 ({self.phi_0.shape}) "
                f"must have the same length."
            )
        if self.E_bins.shape != self.phi_data.shape:
            raise ValueError(
                f"E_bins ({self.E_bins.shape}) and phi_data ({self.phi_data.shape}) "
                f"must have the same length."
            )
        if self.phi_err.shape != self.phi_data.shape:
            raise ValueError(
                f"phi_err ({self.phi_err.shape}) must match phi_data ({self.phi_data.shape})."
            )
def default_pppc_gamma_table_path() -> Path:
    """
    Locate the PPPC Release 6.0 gamma-ray production table if it is available.

    Preferred locations:
      1. $PPPC4DMID_GAMMAS
      2. <repo>/data/AtProduction_gammas.dat
      3. $GAMMAPY_DATA/dark_matter_spectra/PPPC4DMID/AtProduction_gammas.dat
    """
    candidates = []
    if os.environ.get("PPPC4DMID_GAMMAS"):
        candidates.append(Path(os.environ["PPPC4DMID_GAMMAS"]))
    candidates.append(Path(__file__).resolve().parent.parent / "data" / "AtProduction_gammas.dat")
    if os.environ.get("GAMMAPY_DATA"):
        candidates.append(
            Path(os.environ["GAMMAPY_DATA"])
            / "dark_matter_spectra"
            / "PPPC4DMID"
            / "AtProduction_gammas.dat"
        )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else Path("AtProduction_gammas.dat")


def _pppc_column_name(channel: str) -> str:
    key = str(channel).strip()
    if key in _PPPC_CHANNEL_ALIASES.values():
        return key
    norm = key.lower().replace(" ", "").replace("_", "")
    return _PPPC_CHANNEL_ALIASES.get(norm, key)


def pppc_dnde_from_table(
    E_bins: np.ndarray,
    m_ann_GeV: float,
    *,
    channel: str = "WW",
    table_path: str | Path | None = None,
) -> np.ndarray:
    """
    Interpolate the PPPC4DMID gamma-ray production table.

    The table columns are mDM, Log[10,x], and dN/dLog10x for each primary
    annihilation channel, with x = E_gamma / mDM.  Totani cites PPPC 4 DM ID
    Release 6.0 and uses these production spectra for bb, W+W-, and tau+tau-.
    """
    try:
        from astropy.table import Table
    except ImportError as exc:
        raise ImportError(
            "Reading PPPC tables requires astropy. Install requirements_analysis.txt "
            "or use a Python environment with astropy available."
        ) from exc

    E_bins = np.asarray(E_bins, dtype=float)
    path = Path(table_path) if table_path is not None else default_pppc_gamma_table_path()
    if not path.exists():
        raise FileNotFoundError(
            f"PPPC gamma table not found: {path}\n"
            "Download the PPPC4DMID Release 6.0 numerical gamma table "
            "'AtProduction_gammas.dat' from https://www.marcocirelli.net/PPPC4DMID.html "
            "and place it at <repo>/data/AtProduction_gammas.dat, or set "
            "PPPC4DMID_GAMMAS=/path/to/AtProduction_gammas.dat."
        )

    table = Table.read(str(path), format="ascii.fast_basic", guess=False, delimiter=" ")
    channel_col = _pppc_column_name(channel)
    if channel_col not in table.colnames:
        raise ValueError(
            f"Channel {channel!r} maps to PPPC column {channel_col!r}, "
            f"but that column is not in {path}. Available columns include: "
            f"{', '.join(table.colnames[:10])}, ..."
        )

    masses = np.asarray(table["mDM"], dtype=float)
    log10x = np.asarray(table["Log[10,x]"], dtype=float)
    dndlogx = np.asarray(table[channel_col], dtype=float)

    target_logm = np.log10(float(m_ann_GeV))
    target_logx = np.log10(E_bins / float(m_ann_GeV))
    unique_masses = np.unique(masses)
    unique_logm = np.log10(unique_masses)

    if target_logm < unique_logm.min() or target_logm > unique_logm.max():
        raise ValueError(
            f"m_ann={m_ann_GeV:g} GeV is outside the PPPC table mass range "
            f"[{unique_masses.min():g}, {unique_masses.max():g}] GeV."
        )

    spectra_at_mass = []
    for m in unique_masses:
        mask = masses == m
        order = np.argsort(log10x[mask])
        lx = log10x[mask][order]
        vals = dndlogx[mask][order]
        interp_vals = np.interp(target_logx, lx, vals, left=0.0, right=0.0)
        spectra_at_mass.append(interp_vals)
    spectra_at_mass = np.asarray(spectra_at_mass, dtype=float)

    dndlogx_target = np.array([
        np.interp(target_logm, unique_logm, spectra_at_mass[:, i])
        for i in range(len(E_bins))
    ])
    return dndlogx_target / (E_bins * np.log(10.0))


def pppc_energy_flux_template(
    E_bins: np.ndarray,
    m_ann_GeV: float,
    *,
    channel: str = "WW",
    primary: str = "gamma",
    table_path: str | Path | None = None,
    normalise: bool = False,
) -> np.ndarray:
    """
    Build an intrinsic annihilation photon template from the PPPC4DMID table.

    The package returns dN/dE in GeV^-1.  Totani's spectrum is plotted as an
    energy-flux-like quantity, so the shape used here is E^2 dN/dE.  The
    The default keeps the physical PPPC yield normalization.  With this
    convention, the fitted template amplitude can be converted directly to
    <sigma v> using smooth_nfw_sigma_v_from_norm.
    """
    E_bins = np.asarray(E_bins, dtype=float)
    if primary != "gamma":
        raise ValueError("Only primary='gamma' is supported by the PPPC gamma table loader.")
    dnde = pppc_dnde_from_table(
        E_bins,
        float(m_ann_GeV),
        channel=channel,
        table_path=table_path,
    )
    dnde = np.where(np.isfinite(dnde) & (dnde > 0.0), dnde, 0.0)

    template = E_bins**2 * dnde
    if normalise:
        positive_data = _aeft.PHI_TOTANI > 0.0
        template_peak = float(np.nanmax(template[positive_data])) if np.any(positive_data) else 0.0
        data_peak = float(np.nanmax(_aeft.PHI_TOTANI[positive_data])) if np.any(positive_data) else 1.0
        if template_peak > 0.0 and np.isfinite(template_peak):
            template = template * (data_peak / template_peak)

    return template


def smooth_nfw_sigma_v_from_norm(
    fitted_norm: float,
    m_ann_GeV: float,
    *,
    J_pole_Msun2_kpc5: float = 8.93e14,
) -> float:
    """
    Convert the fitted E^2 dN/dE template amplitude into <sigma v>.

    For the smooth NFW-rho^2 case, Totani uses

        dPhi/dE = <sigma v> / (8 pi m_chi^2) * dN/dE * J_pole.

    The fitting template is E_GeV^2 dN/dE in GeV per annihilation, while
    Totani's plotted spectrum is in MeV cm^-2 s^-1 sr^-1. Therefore

        y_MeV = fitted_norm * E_GeV^2 dN/dE

    implies

        fitted_norm = 1000 * <sigma v> * J_pole / (8 pi m_chi^2).

    Returns <sigma v> in cm^3 s^-1.
    """
    MSUN_KPC3_TO_GEV_CM3 = 3.817e-8
    KPC_TO_CM = 3.0857e21
    J_pole_GeV2_cm5 = (
        float(J_pole_Msun2_kpc5)
        * MSUN_KPC3_TO_GEV_CM3**2
        * KPC_TO_CM
    )
    return float(fitted_norm) * 8.0 * np.pi * float(m_ann_GeV)**2 / (
        1000.0 * J_pole_GeV2_cm5
    )


# ---------------------------------------------------------------------------
# Optical depth spectrum
# ---------------------------------------------------------------------------

def compute_tau_spectrum(cfg: ReshapingConfig, *, arm: str = "attenuation") -> np.ndarray:
    """Optical depth for the requested ARM.  See _tau_closed_form for why the
    two arms must not share a definition."""
    return _tau_closed_form(cfg, arm=arm)


def _tau_closed_form(cfg: ReshapingConfig, *, arm: str) -> np.ndarray:
    """tau(E) from the closed-form cross sections, with ARM-DEPENDENT sigma.

    attenuation : sigma = REMOVAL cross section, integrated from
                  u_min = min(f/(1-f), r*x_roi) to u_max = 2r. A photon counts
                  as removed when it leaves its energy bin OR leaves the ROI.

    reshaping   : sigma = FULL ELASTIC cross section, u_min = 0. The kernel K
                  does the removal bookkeeping in this arm -- it returns a
                  near-forward-scattered photon to its own energy bin and
                  applies the ROI recovery weight. Using the removal sigma here
                  would count the same loss twice, once in tau and again in K.

    This is a real physical distinction between the arms, not a convention, so
    it is exposed through the `arm` argument rather than silently folded into
    sigma_tot.
    """
    from core.sigma_closed_form import sigma_removal_cm2

    E_bins = np.asarray(cfg.E_bins, dtype=float)
    if arm not in ("attenuation", "reshaping"):
        raise ValueError(f"arm must be 'attenuation' or 'reshaping', got {arm!r}")

    kwargs = dict(c_s=cfg.c_s, c_p=cfg.c_p, c_phi=cfg.c_phi, f_bin=cfg.f_bin)
    if arm == "reshaping":
        # full elastic: integrate the whole angular range
        sigma = sigma_removal_cm2(cfg.operator, cfg.m_chi, E_bins, cfg.Lambda,
                                  u_min=np.zeros_like(E_bins), **kwargs)
    else:
        x_roi = cfg.x_roi
        if cfg.roi_recovery_model == "template":
            # Energy migration (exact step at the bin edge) PLUS angular escape
            # weighted by the computed w(theta). A hard theta_ROI cut cannot
            # represent the latter: the template w is already 0.90 by 5 deg.
            from core.sigma_closed_form import sigma_removal_smooth_w_cm2
            from core.roi_recovery import halo_recovery_fraction
            sigma = sigma_removal_smooth_w_cm2(
                cfg.operator, cfg.m_chi, E_bins, cfg.Lambda,
                halo_recovery_fraction, **kwargs)
        else:
            # Per bin, a closed removal channel (u_min >= u_max, i.e. even a
            # full backscatter shifts the photon by less than the removal
            # criterion) means sigma = 0. That is the physical answer for that
            # bin, not a configuration error. With x_roi=None this is energy
            # migration alone, i.e. w == 1 -- the correct IGRB treatment, where
            # an isotropic source against an isotropic scatterer has no
            # boundary to be deflected across.
            sigma = sigma_removal_cm2(cfg.operator, cfg.m_chi, E_bins,
                                      cfg.Lambda, x_roi=x_roi,
                                      closed_channel="zero", **kwargs)

    if cfg.tau_prefactor_override is not None:
        K = float(cfg.tau_prefactor_override)
    else:
        K = roi_tau_prefactor(cfg.l_grid, cfg.b_grid)
    return np.asarray((K / cfg.m_chi) * sigma, dtype=float)


# ---------------------------------------------------------------------------
# Redistribution kernel
# ---------------------------------------------------------------------------

def _dsigma_callable(cfg):
    """dsigma/dOmega [cm^2/sr] as a function of (E, theta), for this config."""
    def f(E, theta):
        if cfg.operator == "higgs_portal":
            d = get_flat_weak_cross_sections(
                cfg.m_chi, cfg.y_eff, theta, E,
                frame="lab", in_SI=False, which="full",
            )
        elif cfg.dm_type == "fermionic":
            d = dsigma_dOmega_fermionic(
                cfg.m_chi, theta, E, cfg.c_s, cfg.c_p, cfg.Lambda,
                operator=cfg.operator, majorana=cfg.majorana,
            )
        elif cfg.dm_type == "scalar":
            d = dsigma_dOmega_scalar(cfg.m_chi, theta, E, cfg.c_phi, cfg.Lambda)
        else:
            raise ValueError(f"Unknown dm_type: {cfg.dm_type!r}")
        return np.asarray(d, dtype=float) * FB_TO_CM2
    return f


def _recovery_callable(cfg):
    """w(theta_deg) for the kernel numerator, or None for unity."""
    if not cfg.apply_roi_weight:
        return None
    model = cfg.roi_recovery_model
    if model == "unity":
        return None                      # unity weights; see ReshapingConfig
    if model == "template":
        from core.roi_recovery import halo_recovery_fraction
        return halo_recovery_fraction
    raise ValueError(
        f"roi_recovery_model must be 'template' or 'unity', got {model!r}")


def build_kernel(cfg: ReshapingConfig) -> np.ndarray:
    """
    Build the (nE, nE) redistribution kernel K for this config.

    The kernel is evaluated on analytic per-bin u-intervals with log-u
    segmented Gauss-Legendre quadrature: column-sum closure then holds to
    machine precision and sigma_tot matches the closed forms of
    core/sigma_closed_form.py to 1.8e-10 across m_chi = 1e-5 to 1e8 GeV
    (tests/test_kernel_normalisation.py).

    Returns
    -------
    K : (nE, nE)   redistribution matrix (upper-triangular, column-sum <= 1)
    """
    from core.kinematics import build_redistribution_matrix_exact
    return build_redistribution_matrix_exact(
        cfg.E_bins, cfg.m_chi, _dsigma_callable(cfg),
        w_fn=_recovery_callable(cfg),
    )


def energy_flux_transfer_matrix(
    K_photon: np.ndarray,
    E_bins: np.ndarray,
) -> np.ndarray:
    """
    Convert a photon-number redistribution kernel into an energy-flux kernel.

    K_photon[i,j] is a probability for photons injected in bin j to be observed
    in bin i. Totani's plotted halo spectrum is treated here as an energy-flux
    per logarithmic bin, approximately E^2 dN/dE.  A photon moved from E_j to
    E_i contributes less energy flux by E_i/E_j, so the energy-flux transfer is

        K_energy[i,j] = (E_i / E_j) K_photon[i,j].

    For non-log-spaced data or a different spectral convention, replace this
    with a bin-integrated number-flux transfer.
    """
    K_photon = np.asarray(K_photon, dtype=float)
    E_bins = np.asarray(E_bins, dtype=float)
    return K_photon * (E_bins[:, None] / E_bins[None, :])


def apply_single_scatter_transfer(
    phi_source: np.ndarray,
    tau: np.ndarray,
    K_photon: np.ndarray,
    E_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply zero- and one-scatter transfer to an energy-flux-like spectrum.

    Returns (phi_obs, phi_survival, phi_inscatter), all in the same units as
    phi_source.
    """
    phi_source = np.asarray(phi_source, dtype=float)
    tau = np.asarray(tau, dtype=float)
    K_energy = energy_flux_transfer_matrix(K_photon, E_bins)

    phi_survival = phi_source * np.exp(-tau)
    phi_inscatter = K_energy @ (tau * phi_source * np.exp(-tau))
    return phi_survival + phi_inscatter, phi_survival, phi_inscatter


# ---------------------------------------------------------------------------
# Full reshaping calculation
# ---------------------------------------------------------------------------

def halo_component_transfer_spectrum(
    cfg: ReshapingConfig,
    *,
    return_components: bool = False,
) -> np.ndarray | dict:
    """
    Compute the transferred spectrum in the halo-component approximation.

    This applies the scattering transfer only to cfg.phi_0 and compares the
    result to cfg.phi_data. In the Totani scripts cfg.phi_data is usually the
    MCMC-extracted NFW halo template coefficient spectrum, not the total LAT
    sky intensity.

    Steps:
      1. Compute tau(E) via the emissivity-weighted ROI average, Eq. (II.7).
      2. Build the redistribution kernel K[i,j], Eq. (II.13).
      3. Apply: Phi_obs = survival + in-scatter, Eq. (II.10).

    Parameters
    ----------
    cfg : ReshapingConfig
    return_components : bool
        If True, return a dict with keys:
        'phi_obs', 'phi_survival', 'phi_inscatter', 'tau', 'K', 'sigma_tot'

    Returns
    -------
    phi_obs : (nE,) or dict if return_components=True
    """
    tau = compute_tau_spectrum(cfg, arm="reshaping")
    K = build_kernel(cfg)

    phi_obs, phi_survival, phi_inscatter = apply_single_scatter_transfer(
        cfg.phi_0,
        tau,
        K,
        cfg.E_bins,
    )

    if return_components:
        return {
            "phi_obs": phi_obs,
            "phi_survival": phi_survival,
            "phi_inscatter": phi_inscatter,
            "tau": tau,
            "K": K,
            "K_energy_flux": energy_flux_transfer_matrix(K, cfg.E_bins),
        }
    return phi_obs


def reshaped_halo_spectrum(
    cfg: ReshapingConfig,
    *,
    return_components: bool = False,
) -> np.ndarray | dict:
    """
    Backward-compatible alias for halo_component_transfer_spectrum.

    The old name is retained because several plotting/scan scripts import it,
    but the calculation is specifically a halo-component transfer
    approximation when cfg.phi_0/cfg.phi_data are the extracted NFW spectrum.
    """
    return halo_component_transfer_spectrum(cfg, return_components=return_components)


def best_fit_normalization(
    model_shape: np.ndarray,
    data: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Analytic weighted least-squares normalization for data ~= A * model."""
    model_shape = np.asarray(model_shape, dtype=float)
    data = np.asarray(data, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    denom = np.sum((model_shape[mask] / err[mask])**2)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.nan
    numer = np.sum(data[mask] * model_shape[mask] / err[mask]**2)
    return float(numer / denom)


def _tau_is_valid_for_single_scatter(cfg: ReshapingConfig, tau: np.ndarray) -> bool:
    if cfg.require_lambda_gt_mdm and cfg.operator != "higgs_portal" and cfg.Lambda <= cfg.m_chi:
        return False
    if cfg.max_tau_single_scatter is None or cfg.max_tau_single_scatter < 0.0:
        return True
    tau_max = float(np.nanmax(np.asarray(tau, dtype=float)))
    return np.isfinite(tau_max) and tau_max <= float(cfg.max_tau_single_scatter)


def _tau_is_valid_for_attenuation(cfg: ReshapingConfig, tau: np.ndarray) -> bool:
    """Validity guard for raw attenuation, which does not use a one-scatter expansion."""
    if cfg.require_lambda_gt_mdm and cfg.operator != "higgs_portal" and cfg.Lambda <= cfg.m_chi:
        return False
    tau = np.asarray(tau, dtype=float)
    return bool(np.all(np.isfinite(tau)) and np.all(tau >= 0.0))


# ---------------------------------------------------------------------------
# Chi-squared
# ---------------------------------------------------------------------------

def chi2_reshaping(
    cfg: ReshapingConfig,
    *,
    positive_bins_only: bool = True,
) -> float:
    """
    Chi-squared between the reshaped model spectrum and Totani data.

        chi2 = sum_i [(Phi_obs_i - Phi_data_i)^2 / sigma_i^2]

    where the sum runs over bins with phi_data > 0 if positive_bins_only=True
    (matches the convention in attenuation_eft.compute_chi2).

    Parameters
    ----------
    cfg : ReshapingConfig
    positive_bins_only : bool
        Skip bins where cfg.phi_data <= 0 (avoids fitting noise-dominated bins).

    Returns
    -------
    chi2 : float
    """
    result = reshaped_halo_spectrum(cfg, return_components=True)
    phi_obs = result["phi_obs"]
    tau = result["tau"]
    if not _tau_is_valid_for_single_scatter(cfg, tau):
        return np.nan

    mask = cfg.phi_data > 0.0 if positive_bins_only else np.ones(len(cfg.E_bins), dtype=bool)
    norm = best_fit_normalization(phi_obs, cfg.phi_data, cfg.phi_err, mask) if cfg.fit_normalization else 1.0
    if not np.isfinite(norm):
        # The analytic profile is degenerate: the attenuated model has
        # underflowed, so no A fits the data. That is not "no information" --
        # it is the most strongly excluded corner of the plane. Falling back to
        # A=1 keeps chi2 large and FINITE there. Returning NaN instead used to
        # punch a hole through every downstream consumer: the sensitivity-map
        # heatmap rendered it as blank (masked) rather than saturated, and it
        # poisoned the contour extractor's grid minimum.
        norm = 1.0
    residuals = (norm * phi_obs[mask] - cfg.phi_data[mask]) / cfg.phi_err[mask]
    return float(np.sum(residuals**2))


def chi2_attenuation_only(cfg: ReshapingConfig) -> float:
    """
    Reference chi-squared using simple attenuation (no redistribution).

    Useful for quantifying how much the redistribution term changes the fit.
    """
    tau = compute_tau_spectrum(cfg, arm="attenuation")
    if not _tau_is_valid_for_attenuation(cfg, tau):
        return np.nan
    phi_att = cfg.phi_0 * np.exp(-tau)
    mask = cfg.phi_data > 0.0
    norm = best_fit_normalization(phi_att, cfg.phi_data, cfg.phi_err, mask) if cfg.fit_normalization else 1.0
    if not np.isfinite(norm):
        # The analytic profile is degenerate: the attenuated model has
        # underflowed, so no A fits the data. That is not "no information" --
        # it is the most strongly excluded corner of the plane. Falling back to
        # A=1 keeps chi2 large and FINITE there. Returning NaN instead used to
        # punch a hole through every downstream consumer: the sensitivity-map
        # heatmap rendered it as blank (masked) rather than saturated, and it
        # poisoned the contour extractor's grid minimum.
        norm = 1.0
    residuals = (norm * phi_att[mask] - cfg.phi_data[mask]) / cfg.phi_err[mask]
    return float(np.sum(residuals**2))


def fitted_norm_reshaping(cfg: ReshapingConfig, *, positive_bins_only: bool = True) -> float:
    """Best-fit intrinsic normalization for the reshaped template."""
    result = reshaped_halo_spectrum(cfg, return_components=True)
    if not _tau_is_valid_for_single_scatter(cfg, result["tau"]):
        return np.nan
    mask = cfg.phi_data > 0.0 if positive_bins_only else np.ones(len(cfg.E_bins), dtype=bool)
    return best_fit_normalization(result["phi_obs"], cfg.phi_data, cfg.phi_err, mask)


def fitted_norm_attenuation_only(cfg: ReshapingConfig) -> float:
    """Best-fit intrinsic normalization for the attenuation-only template."""
    tau = compute_tau_spectrum(cfg, arm="attenuation")
    if not _tau_is_valid_for_attenuation(cfg, tau):
        return np.nan
    mask = cfg.phi_data > 0.0
    return best_fit_normalization(cfg.phi_0 * np.exp(-tau), cfg.phi_data, cfg.phi_err, mask)


# ---------------------------------------------------------------------------
# 2D parameter scan
# ---------------------------------------------------------------------------

def scan_reshaping_chi2(
    m_chi_arr: np.ndarray,
    Lambda_arr: np.ndarray,
    *,
    dm_type: str = "fermionic",
    operator: str = "dipole_magnetic",
    c_s: float = 1.0,
    c_p: float = 0.0,
    c_phi: float = 1.0,
    majorana: bool = False,
    E_bins: np.ndarray | None = None,
    phi_0: np.ndarray | None = None,
    phi_data: np.ndarray | None = None,
    phi_err: np.ndarray | None = None,
    l_grid: np.ndarray | None = None,
    b_grid: np.ndarray | None = None,
    n_theta: int = _N_THETA_DEFAULT,
    apply_roi_weight: bool = True,
    also_compute_attenuation: bool = True,
    fit_normalization: bool = True,
    max_tau_single_scatter: float | None = 0.3,
    require_lambda_gt_mdm: bool = True,
    tau_prefactor_override: float | None = None,
    roi_recovery_model: str = "template",
    x_roi: float | None = None,
    verbose: bool = True,
) -> dict:
    """
    Scan (m_chi, Lambda) and compute chi^2 under both reshaping and
    simple attenuation models.

    Parameters
    ----------
    m_chi_arr : (nM,)
    Lambda_arr : (nL,)
    dm_type, operator, c_s, c_p, c_phi, majorana : operator specification
    E_bins, phi_0, phi_data, phi_err : energy axis / source / data
    l_grid, b_grid : ROI grids for emissivity weighting
    n_theta : angular integration resolution
    apply_roi_weight : apply the ROI recovery weight w(theta) to the kernel
    roi_recovery_model : 'template' (halo ROI) or 'unity' (IGRB); see ReshapingConfig
    also_compute_attenuation : also run the simple attenuation chi2 for comparison
    verbose : print progress

    Returns
    -------
    result : dict with keys:
        'm_chi_arr', 'Lambda_arr',
        'chi2_reshaping'     : (nM, nL)
        'chi2_attenuation'   : (nM, nL)  (if also_compute_attenuation)
        'tau_grid'           : (nM, nL, nE)  optical depths
        'norm_reshaping'     : (nM, nL) best-fit intrinsic normalization
        'norm_attenuation'   : (nM, nL) best-fit intrinsic normalization
        'delta_chi2'         : chi2_reshaping - chi2_attenuation  (nM, nL)
    """
    m_chi_arr = np.asarray(m_chi_arr, dtype=float)
    Lambda_arr = np.asarray(Lambda_arr, dtype=float)
    nM = len(m_chi_arr)
    nL = len(Lambda_arr)

    _E = _aeft.E_BINS_GEV if E_bins is None else np.asarray(E_bins, dtype=float)
    _phi0 = _aeft.PHI_TOTANI if phi_0 is None else np.asarray(phi_0, dtype=float)
    _phi_data = _aeft.PHI_TOTANI if phi_data is None else np.asarray(phi_data, dtype=float)
    _phi_err = _aeft.SIGMA_TOTANI if phi_err is None else np.asarray(phi_err, dtype=float)
    _l = np.linspace(-60.0, 60.0, 15) if l_grid is None else np.asarray(l_grid, dtype=float)
    _b = (np.concatenate([np.linspace(-60, -10, 8), np.linspace(10, 60, 8)])
          if b_grid is None else np.asarray(b_grid, dtype=float))

    nE = len(_E)
    chi2_r = np.full((nM, nL), np.nan, dtype=float)
    chi2_a = np.full((nM, nL), np.nan, dtype=float)
    norm_r = np.full((nM, nL), np.nan, dtype=float)
    norm_a = np.full((nM, nL), np.nan, dtype=float)
    tau_grid = np.full((nM, nL, nE), np.nan, dtype=float)

    n_total = nM * nL
    n_done = 0

    for i, m_chi in enumerate(m_chi_arr):
        for j, Lambda in enumerate(Lambda_arr):

            cfg = ReshapingConfig(
                m_chi=float(m_chi),
                Lambda=float(Lambda),
                dm_type=dm_type,
                operator=operator,
                c_s=c_s,
                c_p=c_p,
                c_phi=c_phi,
                majorana=majorana,
                l_grid=_l,
                b_grid=_b,
                n_theta=n_theta,
                apply_roi_weight=apply_roi_weight,
                E_bins=_E,
                phi_0=_phi0,
                phi_data=_phi_data,
                phi_err=_phi_err,
                fit_normalization=fit_normalization,
                max_tau_single_scatter=max_tau_single_scatter,
                require_lambda_gt_mdm=require_lambda_gt_mdm,
                tau_prefactor_override=tau_prefactor_override,
                roi_recovery_model=roi_recovery_model,
                x_roi=x_roi,
            )

            tau = compute_tau_spectrum(cfg, arm="reshaping")
            tau_grid[i, j] = tau

            if also_compute_attenuation and _tau_is_valid_for_attenuation(cfg, tau):
                chi2_a[i, j] = chi2_attenuation_only(cfg)
                norm_a[i, j] = fitted_norm_attenuation_only(cfg)

            if _tau_is_valid_for_single_scatter(cfg, tau):
                chi2_r[i, j] = chi2_reshaping(cfg)
                norm_r[i, j] = fitted_norm_reshaping(cfg)

            n_done += 1
            if verbose and (n_done % max(1, n_total // 20) == 0):
                pct = 100.0 * n_done / n_total
                print(f"  scan progress: {pct:.0f}%  "
                      f"(m_chi={m_chi:.2e}, Lambda={Lambda:.2e}, "
                      f"chi2_r={chi2_r[i,j]:.3f})")

    result = {
        "m_chi_arr": m_chi_arr,
        "Lambda_arr": Lambda_arr,
        "chi2_reshaping": chi2_r,
        "chi2_attenuation": chi2_a,
        "norm_reshaping": norm_r,
        "norm_attenuation": norm_a,
        "tau_grid": tau_grid,
        "delta_chi2": chi2_r - chi2_a,
        "operator": operator,
        "dm_type": dm_type,
        "c_s": c_s,
        "c_p": c_p,
        "c_phi": c_phi,
        "majorana": majorana,
        "fit_normalization": fit_normalization,
        "max_tau_single_scatter": -1.0 if max_tau_single_scatter is None else max_tau_single_scatter,
        "require_lambda_gt_mdm": require_lambda_gt_mdm,
        "roi_recovery_model": roi_recovery_model,
    }
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_reshaping_scan(result: dict, path: str | Path) -> Path:
    """
    Save a scan result dict to a compressed .npz file.

    All ndarray values are stored as float32 to minimise disk usage.
    String / scalar metadata are preserved.

    Parameters
    ----------
    result : dict   output of scan_reshaping_chi2
    path : str or Path   output file path (will add .npz if missing)

    Returns
    -------
    path : Path   resolved output path
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v.astype(np.float32)
        elif isinstance(v, (int, float, bool, str)):
            arrays[k] = np.array(v)
        else:
            # Convert iterables
            try:
                arrays[k] = np.array(v)
            except Exception:
                pass  # skip non-serialisable metadata

    np.savez_compressed(str(path), **arrays)
    return path


def load_reshaping_scan(path: str | Path) -> dict:
    """
    Load a scan result previously saved by save_reshaping_scan.

    Arrays are returned as float64. Scalar metadata is unwrapped from
    0-d arrays.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    result : dict
    """
    data = np.load(str(path), allow_pickle=False)
    result = {}
    for k in data.files:
        arr = data[k]
        if arr.ndim == 0:
            result[k] = arr.item()
        else:
            result[k] = arr.astype(float) if np.issubdtype(arr.dtype, np.floating) else arr
    return result


# ---------------------------------------------------------------------------
# Diagnostics: kinematic validity
# ---------------------------------------------------------------------------

def print_kinematics_summary(cfg: ReshapingConfig) -> None:
    """
    Print a summary of the scattering kinematics for this configuration:
    maximum energy loss per bin, E'/E at theta=pi, and tau values.
    Useful as a sanity check before running a scan.
    """
    from core.kinematics import max_energy_loss_fraction

    tau = compute_tau_spectrum(cfg, arm="reshaping")
    delta_max = max_energy_loss_fraction(cfg.E_bins, cfg.m_chi)

    print(f"\n{'='*60}")
    print(f"Kinematics summary: {cfg.operator}, m_chi={cfg.m_chi:.2e} GeV, "
          f"Lambda={cfg.Lambda:.2e} GeV")
    print(f"{'='*60}")
    print(f"{'E [GeV]':>10}  {'tau':>10}  {'Delta_max/E':>12}  {'E_min [GeV]':>12}")
    print(f"{'-'*50}")
    for k in range(len(cfg.E_bins)):
        E_min = cfg.E_bins[k] * (1.0 - delta_max[k])
        print(f"{cfg.E_bins[k]:10.2f}  {tau[k]:10.2e}  {delta_max[k]:12.4f}  {E_min:12.2f}")
    print()
