import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from helpers.trinity_plotting import current_savefig_kwargs, set_plot_style
from core.cross_sections import get_flat_grav_cross_sections, get_flat_weak_cross_sections

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

# Totani energy bins (13 log-spaced, GeV)
E_BINS_GEV = np.array([1.51, 2.48, 4.07, 6.68, 11.0, 18.0,
                        29.5, 48.5, 79.6, 131., 215., 352., 814.])

# Totani Figure 8 NFW-rho^2 flux at Galactic poles [MeV cm^-2 s^-1 sr^-1]
# (x 10^-5), middle panel — read off graph
PHI_TOTANI = np.array([-2.0, 0.0, 1.5, 2.2, 3.0, 3.8,
                         4.5, 3.3, 2.0, 1.8, 1.7, 0.8, 1.0]) * 1e-5

# Statistical 1-sigma errors [MeV cm^-2 s^-1 sr^-1]
SIGMA_TOTANI = np.array([0.8, 0.5, 0.5, 0.5, 0.6, 0.5,
                          0.5, 0.6, 0.6, 0.6, 0.7, 0.7, 0.8]) * 1e-5

# Forward-scattering angular cutoff (your thesis convention)
COS_THETA_MAX = 1.0 - 1e-4

# Gravitational coupling lambda = sqrt(8 pi G_N)
# G_N = 6.708e-39 GeV^-2  =>  lambda = sqrt(8 pi * 6.708e-39)
G_NEWTON_GEV2 = 6.708e-39   # GeV^-2
LAM = np.sqrt(8.0 * np.pi * G_NEWTON_GEV2)   # GeV^-1

# Higgs-portal effective-coupling ceiling used in York_paper_check.
# y_eff = (alpha / (pi v_EW)) y_chi sin(theta), with perturbative
# y_chi <= sqrt(4 pi) and LHC Higgs-mixing bound sin(theta) <= 0.33.
ALPHA_EM = 1.0 / 137.036
V_EW_GEV = 246.22
SIN_THETA_MAX = 0.33
Y_CHI_PERT_MAX = np.sqrt(4.0 * np.pi)
Y_EFF_PERT_MAX = (ALPHA_EM / (np.pi * V_EW_GEV)) * Y_CHI_PERT_MAX * SIN_THETA_MAX


def is_perturbative_y_eff(yeff):
    """Return True where y_eff satisfies the York perturbativity/mixing ceiling."""
    yeff = np.asarray(yeff, dtype=float)
    return np.isfinite(yeff) & (yeff >= 0.0) & (yeff <= Y_EFF_PERT_MAX)


# =============================================================================
# NFW PROFILE AND GEOMETRY
# =============================================================================

def nfw_density_GeV_cm3(r_cm):
    """
    NFW density in GeV/cm^3 using Totani / Via Lactea II parameters.
    r_cm : Galactocentric radius in cm.
    """
    x = r_cm / r_s_cm
    return rho_s_GeV / (x * (1.0 + x)**2)


def galactocentric_radius_cm(ell_cm, l_deg, b_deg):
    """
    Galactocentric radius for a point at distance ell_cm along
    line of sight (l, b) from the Sun.
    """
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    # Standard coordinate transform
    r2 = (ell_cm**2
          + r_sun_cm**2
          - 2.0 * ell_cm * r_sun_cm * np.cos(b) * np.cos(l))
    return np.sqrt(np.maximum(r2, 0.0))


def los_length_cm(l_deg, b_deg, r_vir_cm=r_vir_cm):
    """
    Maximum integration length along line of sight (l, b)
    out to the virial radius. Solved from quadratic:
        ell^2 - 2*r_sun*cos(b)cos(l)*ell + (r_sun^2 - r_vir^2) = 0
    Take the positive root.
    """
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    A = 1.0
    B = -2.0 * r_sun_cm * np.cos(b) * np.cos(l)
    C = r_sun_cm**2 - r_vir_cm**2
    disc = np.maximum(B**2 - 4.0 * A * C, 0.0)
    L = (-B + np.sqrt(disc)) / 2.0
    return L


def compute_J_los(l_deg, b_deg, power=2, n_points=500):
    """
    J-factor integral  J = int rho^power dl  along line of sight (l, b).
    power=2 for NFW-rho^2 (annihilation), power=1 for NFW-rho^1.
    Returns J in (GeV/cm^3)^power * cm  =  GeV^power / cm^(3*power - 1).
    """
    L = los_length_cm(l_deg, b_deg)
    ell = np.linspace(0.0, L, n_points)
    r   = galactocentric_radius_cm(ell, l_deg, b_deg)
    rho = nfw_density_GeV_cm3(r)
    integrand = rho**power
    return np.trapezoid(integrand, ell)


# =============================================================================
# CALCULATION 1: POLARISATION-RESOLVED OPTICAL DEPTHS
# =============================================================================

def sigma_tot_polarised(E_GeV, m_chi_GeV, mode="grav", yeff=1.0,
                        n_theta=2000):
    """
    Total polarisation-resolved cross sections [cm^2], integrated over
    solid angle with forward cutoff.

    Parameters
    ----------
    E_GeV     : photon energy in GeV
    m_chi_GeV : DM mass in GeV
    mode      : "grav" for gravitational, "weak" for Higgs-portal
    yeff      : effective coupling (only used for mode="weak")
    n_theta   : number of integration points in cos(theta)

    Returns
    -------
    sig_perp, sig_par : total cross sections [cm^2]
    """
    FB_TO_CM2 = 1e-39  # 1 fb = 1e-39 cm^2

    cos_vals = np.linspace(-1.0, COS_THETA_MAX, n_theta)
    theta_vals = np.arccos(cos_vals)

    if mode == "grav":
        dsig_perp, dsig_par, _ = get_flat_grav_cross_sections(
            E_GeV, m_chi_GeV, theta_vals, LAM
        )
    else:  # weak / Higgs-portal
        if not bool(is_perturbative_y_eff(yeff)):
            raise ValueError(
                f"Non-perturbative Higgs-portal y_eff={yeff:.3e}; "
                f"York ceiling is {Y_EFF_PERT_MAX:.3e} GeV^-1."
            )
        # Higgs portal has no polarisation splitting — perp = par = unpol
        dsig = get_flat_weak_cross_sections(
            m_chi_GeV, yeff, theta_vals, E_GeV, frame="lab"
        )
        dsig_perp = dsig
        dsig_par  = dsig

    # Integrate: sigma = 2 pi * int d(cos theta) * dsig/dOmega
    sig_perp = 2.0 * np.pi * np.trapezoid(dsig_perp, cos_vals) * FB_TO_CM2
    sig_par  = 2.0 * np.pi * np.trapezoid(dsig_par,  cos_vals) * FB_TO_CM2

    return sig_perp, sig_par


def compute_tau_los(l_deg, b_deg, E_GeV, m_chi_GeV,
                    mode="grav", yeff=1.0, n_los=300):
    """
    Optical depths tau_perp and tau_par along a single line of sight (l, b)
    at photon energy E_GeV.

    tau_s = (1/m_chi) * int rho(r(ell)) * sigma_s dl

    Returns tau_perp, tau_par  (dimensionless)
    """
    sig_perp, sig_par = sigma_tot_polarised(E_GeV, m_chi_GeV,
                                             mode=mode, yeff=yeff)

    L   = los_length_cm(l_deg, b_deg)
    ell = np.linspace(0.0, L, n_los)
    r   = galactocentric_radius_cm(ell, l_deg, b_deg)
    rho = nfw_density_GeV_cm3(r)   # GeV / cm^3

    # Convert m_chi to GeV, rho in GeV/cm^3, ell in cm, sig in cm^2
    # tau = (1/m_chi[GeV]) * int rho[GeV/cm^3] * sigma[cm^2] d_ell[cm]
    # dimensions: GeV^-1 * GeV/cm^3 * cm^2 * cm = dimensionless
    GeV_TO_ERG    = 1.602e-3          # not needed — everything in GeV & cm
    J_los = np.trapezoid(rho, ell)       # GeV/cm^2

    tau_perp = J_los * sig_perp / m_chi_GeV
    tau_par  = J_los * sig_par  / m_chi_GeV

    return tau_perp, tau_par


def build_tau_grid(E_GeV, m_chi_GeV, l_grid, b_grid,
                   mode="grav", yeff=1.0):
    """
    Build 2D arrays of tau_perp and tau_par over the (l, b) ROI
    at a single photon energy.

    l_grid, b_grid : 1D arrays of Galactic longitude / latitude in degrees.

    Returns tau_perp_2d, tau_par_2d with shape (len(b_grid), len(l_grid)).
    """
    tau_perp_2d = np.zeros((len(b_grid), len(l_grid)))
    tau_par_2d  = np.zeros((len(b_grid), len(l_grid)))

    for j, b in enumerate(b_grid):
        for i, l in enumerate(l_grid):
            tp, tpa = compute_tau_los(l, b, E_GeV, m_chi_GeV,
                                       mode=mode, yeff=yeff)
            tau_perp_2d[j, i] = tp
            tau_par_2d[j, i]  = tpa

    return tau_perp_2d, tau_par_2d


# =============================================================================
# CALCULATION 2: EMISSIVITY-WEIGHTED AVERAGE OPTICAL DEPTH
# =============================================================================

def compute_J_nfw_rho2_pole():
    """
    J-factor at the Galactic pole (Totani Eq. 4.2):
    int_GP rho^2 dl = 8.93e14 M_sun^2 kpc^-5
    Convert to GeV^2 / cm^5.
    """
    J_pole_Msun2_kpc5 = 8.93e14
    # 1 M_sun^2/kpc^5 = (1 M_sun/kpc^3)^2 * kpc
    MSUN2_KPC5_TO_GEV2_CM5 = (MSUN_KPC3_TO_GEV_CM3)**2 * KPC_TO_CM
    return J_pole_Msun2_kpc5 * MSUN2_KPC5_TO_GEV2_CM5


def compute_tau_bar(E_GeV, m_chi_GeV, l_grid, b_grid,
                    mode="grav", yeff=1.0):
    """
    Emissivity-weighted average optical depth over the ROI:

        tau_bar(E) = sum_{l,b} [tau_unpol(l,b) * J2(l,b)] / sum_{l,b} J2(l,b)

    where tau_unpol = 0.5*(tau_perp + tau_par)
    and J2(l,b) = int rho^2 dl  (the NFW-rho^2 emissivity weight).

    Returns tau_bar (scalar, dimensionless).
    """
    tau_perp_2d, tau_par_2d = build_tau_grid(
        E_GeV, m_chi_GeV, l_grid, b_grid, mode=mode, yeff=yeff
    )
    tau_unpol_2d = 0.5 * (tau_perp_2d + tau_par_2d)

    # Compute J2 weights over the same grid
    J2 = np.zeros((len(b_grid), len(l_grid)))
    for j, b in enumerate(b_grid):
        for i, l in enumerate(l_grid):
            J2[j, i] = compute_J_los(l, b, power=2)

    numerator   = np.sum(tau_unpol_2d * J2)
    denominator = np.sum(J2)

    return numerator / denominator


def compute_tau_bar_spectrum(E_bins, m_chi_GeV, l_grid, b_grid,
                              mode="grav", yeff=1.0):
    """
    Compute tau_bar at each energy bin.
    Returns array of shape (len(E_bins),).
    """
    # Pre-compute J2 grid once (energy-independent)
    J2 = np.zeros((len(b_grid), len(l_grid)))
    for j, b in enumerate(b_grid):
        for i, l in enumerate(l_grid):
            J2[j, i] = compute_J_los(l, b, power=2)

    tau_bar = np.zeros(len(E_bins))

    for k, E in enumerate(E_bins):
        sig_perp, sig_par = sigma_tot_polarised(E, m_chi_GeV,
                                                 mode=mode, yeff=yeff)
        # For each (l,b), tau = J_los * sigma / m_chi
        # We already have J2; for tau we need J1 = int rho dl
        # But sigma is position-independent in flat space, so:
        # tau_unpol(l,b) = J1(l,b) * sigma_unpol / m_chi
        # This means tau_bar = sigma_unpol/m_chi * sum(J1*J2)/sum(J2)

        sig_unpol = 0.5 * (sig_perp + sig_par)

        J1_times_J2 = np.zeros_like(J2)
        for j, b in enumerate(b_grid):
            for i, l in enumerate(l_grid):
                J1 = compute_J_los(l, b, power=1)
                J1_times_J2[j, i] = J1 * J2[j, i]

        tau_bar[k] = sig_unpol / m_chi_GeV * np.sum(J1_times_J2) / np.sum(J2)
        #print(f"  E = {E:.1f} GeV  tau_bar = {tau_bar[k]:.3e}")

    return tau_bar


# =============================================================================
# CALCULATION 3: ATTENUATED HALO SPECTRUM
# =============================================================================

def compute_attenuated_spectrum(tau_bar, phi_totani=PHI_TOTANI):
    """
    Apply scattering attenuation to Totani's halo flux.

    Phi_att(E) = Phi_Totani(E) * exp(-tau_bar(E))

    Parameters
    ----------
    tau_bar   : array of shape (13,) — weighted optical depths
    phi_totani: array of shape (13,) — Totani Figure 8 NFW-rho^2 flux

    Returns
    -------
    phi_att : attenuated flux, same units as phi_totani
    """
    return phi_totani * np.exp(-tau_bar)


# =============================================================================
# CALCULATION 4: CHI-SQUARED CONSTRAINT ON (m_chi, y_eff)
# =============================================================================

def compute_chi2(tau_bar, phi_totani=PHI_TOTANI, sigma_totani=SIGMA_TOTANI):
    """
    Chi-squared between attenuated and observed halo flux.
    Only use bins where phi_totani > 0 (avoid unphysical negative bins).
    """
    phi_att = compute_attenuated_spectrum(tau_bar, phi_totani)
    mask = phi_totani > 0   # skip bins consistent with zero / negative
    residuals = (phi_att[mask] - phi_totani[mask]) / sigma_totani[mask]
    return np.sum(residuals**2)


def chi2_grid_scan(m_chi_arr, yeff_arr, E_bins, l_grid, b_grid, mode="weak"):
    """
    Scan (m_chi, y_eff) parameter space and compute chi^2 at each point.
    For mode="grav", y_eff is ignored (lambda fixed by G_N).

    Returns chi2_grid of shape (len(m_chi_arr), len(yeff_arr)).
    """
    chi2_grid = np.full((len(m_chi_arr), len(yeff_arr)), np.nan)

    for i, m_chi in enumerate(m_chi_arr):
        for j, yeff in enumerate(yeff_arr):
            if mode == "weak" and not bool(is_perturbative_y_eff(yeff)):
                continue
            #print(f"m_chi={m_chi:.2e} GeV  yeff={yeff:.2e}")
            tau_bar = compute_tau_bar_spectrum(
                E_bins, m_chi, l_grid, b_grid, mode=mode, yeff=yeff
            )
            chi2_grid[i, j] = compute_chi2(tau_bar)

    return chi2_grid


def plot_chi2_contours(m_chi_arr, yeff_arr, chi2_grid, mode="weak"):
    """
    Plot Delta-chi^2 contours in (m_chi, y_eff) space.
    Contours at Delta chi^2 = 4.61 (90% CL) and 9.21 (99% CL).
    """
    finite = np.isfinite(chi2_grid)
    if not np.any(finite):
        print("Warning: no perturbative/finite Higgs-portal scan points to plot.")
        return

    chi2_min  = np.nanmin(chi2_grid)
    dchi2     = chi2_grid - chi2_min

    set_plot_style(style="paper", cmap_name="plasma", base_fontsize=12, linewidth=1.8, n_colors=10)
    fig, ax = plt.subplots(figsize=(9, 6))
    M, Y = np.meshgrid(m_chi_arr, yeff_arr, indexing='ij')

    # Filled contours with smooth colormap
    max_dchi2 = np.nanmax(dchi2)
    if not np.isfinite(max_dchi2) or max_dchi2 <= 0.0:
        print("Warning: perturbative Higgs-portal scan has no chi2 variation to plot.")
        return
    levels_fill = np.linspace(0, min(50, max_dchi2), 50)
    cf = ax.contourf(np.log10(M), np.log10(Y), dchi2,
                     levels=levels_fill, cmap='plasma', alpha=0.8)
    
    # Confidence level contours
    contour_levels = [level for level in [4.61, 9.21] if level < max_dchi2]
    if contour_levels:
        cs = ax.contour(np.log10(M), np.log10(Y), dchi2,
                        levels=contour_levels, colors='red', linewidths=2.5,
                        linestyles='solid')

        # Label contours
        ax.clabel(cs, contour_levels, inline=True, fontsize=11,
                  fmt={4.61: '90% CL', 9.21: '99% CL'})
    ax.axhline(np.log10(Y_EFF_PERT_MAX), color='white', lw=1.5, ls=':',
               label=r'York perturbativity/LHC ceiling')
    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, label=r'$\Delta \chi^2$')
    cbar.ax.tick_params(labelsize=11)
    
    # Labels and formatting
    ax.set_xlabel(r'$\log_{10}(m_\chi / \mathrm{GeV})$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(y_{\rm eff})$', fontsize=14)
    ax.set_title('Higgs-portal DM scattering constraint from Fermi halo excess', 
                 fontsize=13, pad=10)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.85)

    plt.tight_layout()
    plt.savefig('chi2_contours.png', dpi=300, bbox_inches='tight', **current_savefig_kwargs())
    plt.close()


# =============================================================================
# CALCULATION 5: POLARISATION FRACTION MAP
# =============================================================================

def compute_polarisation_fraction_map(E_GeV, m_chi_GeV, l_grid, b_grid,
                                       mode="grav", yeff=1.0):
    """
    Compute polarisation fraction Pi(l, b) at a single photon energy.

        Pi(l, b) = [exp(-tau_par) - exp(-tau_perp)]
                 / [exp(-tau_par) + exp(-tau_perp)]

    For small tau this linearises to Pi ~ (tau_perp - tau_par) / 2.

    Returns Pi_2d of shape (len(b_grid), len(l_grid)).
    """
    tau_perp_2d, tau_par_2d = build_tau_grid(
        E_GeV, m_chi_GeV, l_grid, b_grid, mode=mode, yeff=yeff
    )

    exp_par  = np.exp(-tau_par_2d)
    exp_perp = np.exp(-tau_perp_2d)

    Pi = (exp_par - exp_perp) / (exp_par + exp_perp)
    return Pi


def plot_polarisation_map(Pi_2d, l_grid, b_grid, E_GeV, m_chi_GeV):
    """
    2D map of polarisation fraction Pi(l, b) over Totani's ROI.
    """
    L, B = np.meshgrid(l_grid, b_grid)

    # Diagnostic output
    print(f"Polarisation map statistics:")
    print(f"  Min: {np.min(Pi_2d):.3e}")
    print(f"  Max: {np.max(Pi_2d):.3e}")
    print(f"  Mean: {np.mean(Pi_2d):.3e}")
    print(f"  Std: {np.std(Pi_2d):.3e}")
    
    vmax = np.max(np.abs(Pi_2d))
    
    # Check if data is effectively zero (gravitational case)
    if vmax < 1e-10:
        print(f"  Warning: Polarisation fraction is extremely small (max={vmax:.3e})")
        print(f"  This is expected for gravitational scattering with tau ~ 1e-67")
        # Use a symmetric range around the actual data
        vmin_data = np.min(Pi_2d)
        vmax_data = np.max(Pi_2d)
        if vmax_data - vmin_data < 1e-15:
            # Data is essentially constant - use arbitrary small range
            vmax = 1e-10
        else:
            vmax = max(abs(vmin_data), abs(vmax_data))
    elif vmax == 0:
        vmax = 1.0  # fallback for exactly zero data
    
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    set_plot_style(style="paper", cmap_name="plasma", base_fontsize=12, linewidth=1.8, n_colors=10)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(L, B, Pi_2d, cmap='RdBu_r', norm=norm, shading='auto')
    cb = plt.colorbar(im, ax=ax, label=r'Polarisation fraction $\Pi(l,b)$')
    cb.formatter.set_powerlimits((-2, 2))
    cb.update_ticks()
    
    ax.set_xlabel('Galactic longitude $l$ [deg]', fontsize=12)
    ax.set_ylabel('Galactic latitude $b$ [deg]', fontsize=12)
    ax.set_title(
        rf'$\Pi(l,b)$ at $E_\gamma={E_GeV:.0f}$ GeV, '
        rf'$m_\chi={m_chi_GeV:.2e}$ GeV',
        fontsize=11
    )
    
    # Add text annotation if values are extremely small
    if vmax < 1e-10:
        ax.text(0.02, 0.98, f'Max |Π| = {vmax:.2e}\n(negligible for grav.)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('polarisation_map.png', dpi=150, **current_savefig_kwargs())
    plt.close()


# =============================================================================
# DRIVER: RUN ALL CALCULATIONS
# =============================================================================

if __name__ == "__main__":

    # --- Grid over Totani's ROI (coarse for speed; refine as needed) ---
    l_grid = np.linspace(-60, 60, 25)    # deg, 25 points
    b_grid = np.concatenate([
        np.linspace(-60, -10, 13),        # southern halo
        np.linspace( 10,  60, 13)         # northern halo  (skip |b|<10)
    ])

    # --- Benchmark parameters (Totani preferred range for bb-bar) ---
    m_chi_benchmark = 5e19   # GeV  (midpoint of 500-800 GeV)
    E_peak          = 21.0    # GeV  (peak of halo excess)

    print("=" * 60)
    print("CALCULATION 1+2: tau_bar spectrum (gravitational)")
    print("=" * 60)
    tau_bar_grav = compute_tau_bar_spectrum(
        E_BINS_GEV, m_chi_benchmark, l_grid, b_grid, mode="grav"
    )

    print("\n" + "=" * 60)
    print("CALCULATION 1+2: tau_bar spectrum (Higgs-portal)")
    print("=" * 60)
    yeff_weak = Y_EFF_PERT_MAX
    print("m_chi = ", m_chi_benchmark, " GeV")
    print("y_eff = ", yeff_weak, " GeV^-1 (York perturbative/LHC ceiling)")
    tau_bar_weak = compute_tau_bar_spectrum(
        E_BINS_GEV, m_chi_benchmark, l_grid, b_grid, mode="weak", yeff=yeff_weak
    )

    print("\n" + "=" * 60)
    print("CALCULATION 3: Attenuated spectrum")
    print("=" * 60)
    phi_att_grav = compute_attenuated_spectrum(tau_bar_grav)
    phi_att_weak = compute_attenuated_spectrum(tau_bar_weak)

    print("\n" + "=" * 60)
    print("CALCULATION 4: Chi-squared")
    print("=" * 60)
    chi2_grav = compute_chi2(tau_bar_grav)
    chi2_weak = compute_chi2(tau_bar_weak)
    print(f"chi2 (grav, benchmark)      = {chi2_grav:.3f}")
    print(f"chi2 (weak, perturbative y_eff max) = {chi2_weak:.3f}")

    # --- Plot attenuated vs observed spectrum ---
    set_plot_style(style="paper", cmap_name="plasma", base_fontsize=12, linewidth=1.8, n_colors=10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(E_BINS_GEV, PHI_TOTANI, yerr=SIGMA_TOTANI,
                fmt='o', color='k', label='Totani NFW-ρ² (Fig. 8)', zorder=5)
    ax.plot(E_BINS_GEV, phi_att_grav, 's--', color='steelblue',
            label=rf'Grav. attenuated ($m_\chi$={m_chi_benchmark} GeV)')
    ax.plot(E_BINS_GEV, phi_att_weak, '^--', color='tomato',
            label=rf'Higgs attenuated ($y_{{eff}}$={yeff_weak:.2e} GeV$^{{-1}}$)')
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('Photon energy [GeV]', fontsize=12)
    ax.set_ylabel(r'$E^2 dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=12)
    ax.set_title('Halo excess: Totani vs scattering-attenuated', fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('attenuated_spectrum.png', dpi=150, **current_savefig_kwargs())
    plt.close()

    print("\n" + "=" * 60)
    print("CALCULATION 5: Polarisation fraction map at 21 GeV")
    print("=" * 60)
    Pi_map = compute_polarisation_fraction_map(
        E_peak, m_chi_benchmark, l_grid, b_grid, mode="grav"
    )
    plot_polarisation_map(Pi_map, l_grid, b_grid, E_peak, m_chi_benchmark)

    # --- Optional: chi2 grid scan (slow — comment out for quick runs) ---
    m_chi_arr = np.logspace(2, 20, 20)  
    yeff_arr  = np.logspace(-12, np.log10(Y_EFF_PERT_MAX), 15)
    chi2_grd  = chi2_grid_scan(m_chi_arr, yeff_arr, E_BINS_GEV,
                                l_grid, b_grid, mode="weak")
    plot_chi2_contours(m_chi_arr, yeff_arr, chi2_grd, mode="weak")

    # Extract 90% CL exclusion boundary
    if not np.any(np.isfinite(chi2_grd)):
        raise RuntimeError("No perturbative Higgs-portal scan points survived.")

    chi2_min = np.nanmin(chi2_grd)
    dchi2 = chi2_grd - chi2_min
    
    yeff_upper_90 = []
    for i, m_chi in enumerate(m_chi_arr):
        row = dchi2[i, :]
        finite = np.isfinite(row)
        if np.count_nonzero(finite) < 2:
            continue
        # Find where Delta chi2 crosses 4.61 (interpolate for smoother curve)
        if np.nanmin(row) < 4.61 < np.nanmax(row):
            # Interpolate to find exact crossing point
            kind = 'cubic' if np.count_nonzero(finite) >= 4 else 'linear'
            f_interp = interp1d(np.log10(yeff_arr[finite]), row[finite], kind=kind,
                               fill_value='extrapolate')
            # Find crossing by searching
            log_yeff_test = np.linspace(np.log10(yeff_arr[finite][0]),
                                       np.log10(yeff_arr[finite][-1]), 1000)
            dchi2_test = f_interp(log_yeff_test)
            idx_cross = np.where(dchi2_test > 4.61)[0]
            if len(idx_cross) > 0:
                yeff_upper_90.append((m_chi, 10**log_yeff_test[idx_cross[0]]))

    if len(yeff_upper_90) > 0:
        yeff_upper_90 = np.array(yeff_upper_90)
        
        # Plot exclusion curve
        set_plot_style(style="paper", cmap_name="plasma", base_fontsize=12, linewidth=1.8, n_colors=10)
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Exclusion region
        ax.fill_between(np.log10(yeff_upper_90[:, 0]),
                        np.log10(yeff_upper_90[:, 1]),
                        np.log10(yeff_arr[-1]),
                        alpha=0.3, color='red', label='Excluded (90% CL)')
        
        # Boundary line
        ax.plot(np.log10(yeff_upper_90[:, 0]),
                np.log10(yeff_upper_90[:, 1]),
                'r-', lw=2.5, label='90% CL upper limit')
        
        # Formatting
        ax.set_xlabel(r'$\log_{10}(m_\chi / \mathrm{GeV})$', fontsize=14)
        ax.set_ylabel(r'$\log_{10}(y_{\mathrm{eff}})$', fontsize=14)
        ax.set_title('Higgs-portal DM: Exclusion from Fermi halo excess', 
                     fontsize=13, pad=10)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        
        # Set reasonable axis limits
        ax.set_xlim(np.log10(m_chi_arr[0]), np.log10(m_chi_arr[-1]))
        ax.set_ylim(np.log10(yeff_arr[0]), np.log10(yeff_arr[-1]))
        
        plt.tight_layout()
        plt.savefig('exclusion_curve.png', dpi=300, bbox_inches='tight', **current_savefig_kwargs())
        plt.close()
    else:
        print("Warning: No 90% CL exclusion boundary found in parameter space")
