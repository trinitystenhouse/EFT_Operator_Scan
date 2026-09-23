"""
Source-agnostic spectrum abstraction for the γχ scattering constraint pipeline.

The pipeline is written against a single dataclass — :class:`SpectrumSource` —
so :mod:`make_data_driven_scattering_limits` can drive any of

  * ``halo``   — the pixel-level MCMC posterior of the Fermi-LAT galactic-centre
                 halo template (Totani-style, ROI emissivity-weighted).
  * ``dsph``   — dwarf spheroidals from the McDaniel 2024 legacy analysis
                 (per-dSph SED + J-factor from a stacked catalog, combined at
                 the likelihood level).
  * ``igrb``   — the isotropic extragalactic gamma-ray background of
                 Ackermann et al. 2015 (single spectrum + cosmological J).

by presenting a common interface to the constraint scan:

  * ``E_bins_GeV``          — energy bin centres [GeV]
  * ``phi``, ``phi_err_*``  — measured differential flux + errors,
                              units MeV / (cm² s sr) — E² dN/dE convention
  * ``tau_prefactor_K``     — scalar K [GeV/cm²] such that
                              tau(E) = K * sigma_tot(E) / m_chi
  * ``source_label``, ``source_metadata`` — bookkeeping for the boundary file.

The halo case does NOT use ``tau_prefactor_K``; it keeps the ROI-integrated
prefactor built from ``l_grid`` and ``b_grid`` for backward compatibility. The
dSph and IGRB cases DO use it (they have a single well-defined column density,
not an emissivity-weighted average over an extended ROI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# Public dataclass
# =============================================================================

@dataclass
class SpectrumSource:
    """Container for one photon-flux measurement driving the scattering scan.

    Attributes
    ----------
    E_bins_GeV : (nE,) float
        Energy bin centres [GeV].
    phi : (nE,) float
        Central flux value at each bin, E² dN/dE units [MeV cm⁻² s⁻¹ sr⁻¹]
        (the same normalisation as the halo posterior in ``totani_data_loader``).
    phi_err_sym : (nE,) float
        Symmetric 1-sigma uncertainty, 0.5 * (φ_p84 − φ_p16) where available,
        else √(σ_stat² + σ_sys²) for datasets with a Gaussian error model.
    phi_err_lo, phi_err_hi : (nE,) float
        Asymmetric (lower, upper) 1-sigma errors when the underlying dataset
        provides them; equal to ``phi_err_sym`` otherwise.
    phi_ul : (nE,) float or None
        95% CL upper limit per bin when available (dSphs) — else None.
    tau_prefactor_K : float or None
        Scalar column-density prefactor K [GeV/cm²].
          * None → use the halo ROI-integrated K from ``roi_tau_prefactor``.
          * float → bypass the ROI average and compute τ = K σ / m_χ directly.
    source_label : str
        Human-readable label, e.g. ``"dSph stack (McDaniel 2024)"``.
    source_metadata : dict
        Extra bookkeeping (dataset provenance, cuts, dSph list, IRF, ...) —
        written verbatim to the boundary npz for reproducibility.
    positive_mask : (nE,) bool
        True where φ > 0. Constraint scan may drop non-positive bins.
    finite_mask : (nE,) bool
        True where φ, φ_err are finite.
    """

    E_bins_GeV: np.ndarray
    phi: np.ndarray
    phi_err_sym: np.ndarray
    phi_err_lo: np.ndarray
    phi_err_hi: np.ndarray
    tau_prefactor_K: Optional[float]
    source_label: str
    phi_ul: Optional[np.ndarray] = None
    source_metadata: dict = field(default_factory=dict)
    dataset_kind: str = "generic"
    # Component-aware datasets (dSph stacks, extragalactic source lists ...)
    # populate this list with per-component dicts so the constraint generator
    # can loop and sum χ² with per-component J-marginalisation. Single-source
    # datasets (halo, IGRB) leave it as None.
    per_component_data: Optional[list] = None

    @property
    def positive_mask(self) -> np.ndarray:
        return self.phi > 0.0

    @property
    def finite_mask(self) -> np.ndarray:
        return (
            np.isfinite(self.phi)
            & np.isfinite(self.phi_err_sym)
            & (self.phi_err_sym > 0.0)
        )

    def summary(self) -> str:
        lines = [
            f"SpectrumSource ({self.dataset_kind}): {self.source_label}",
            f"  n_bins        : {len(self.E_bins_GeV)}",
            f"  E range       : {np.nanmin(self.E_bins_GeV):.3g} – "
            f"{np.nanmax(self.E_bins_GeV):.3g} GeV",
            f"  positive bins : {int(self.positive_mask.sum())} / "
            f"{len(self.phi)}",
            f"  tau prefactor : "
            + (
                "ROI-integrated (halo)"
                if self.tau_prefactor_K is None
                else f"{self.tau_prefactor_K:.3e} GeV/cm²"
            ),
        ]
        return "\n".join(lines)


# =============================================================================
# IGRB loader — Ackermann+ 2015 Table 3 (Foreground Model A, disc excluded)
# =============================================================================

# Cosmological baseline column density used throughout the paper's §II.A:
#   ρ_χ = 1.2e-6 GeV/cm³, L = 1.14e28 cm  →  J_cosmo = 1.37e22 GeV/cm²
IGRB_J_COSMO_GeV_cm2 = 1.37e22

# Ackermann et al. 2015, "The spectrum of isotropic diffuse gamma-ray emission
# between 100 MeV and 820 GeV", ApJ 799, 86, arXiv:1410.3696.
#
# Source: the published machine-readable Table 3 (apj504089t3_mrt.txt), kept
# verbatim at data/ackermann2015_igrb_table3_mrt.txt. Foreground model A rows.
#
# Table 3 tabulates the BAND-INTEGRATED IGRB flux f [cm^-2 s^-1 sr^-1], not a
# differential spectrum. It is converted here to E^2 dN/dE at the log-centre of
# each band, to match the HaloSpectrum.phi convention:
#
#     E2dNdE = E_c^2 * f / (E_hi - E_lo),      E_c = sqrt(E_lo * E_hi)
#
# The two uncertainty columns are kept separate because they are not the same
# kind of quantity. sig_stat is Table 3's "statistical + instrument related
# systematics" (its note 1), which is per-bin and independent. sig_fg is the
# foreground-modelling uncertainty, which is a single correlated choice of
# Galactic diffuse model across every bin, is strongly asymmetric, and is NOT
# independent per-bin noise. load_igrb_source() uses sig_stat alone; treating
# sig_fg as independent Gaussian error would double-count a correlated
# systematic that the profiled source normalisation already largely absorbs.
#
# VALIDATED: predicting each band flux from the paper's own Table 4 model-A fit
# (I_100 = 0.95e-7 MeV^-1 cm^-2 s^-1 sr^-1, gamma = 2.32, E_cut = 279 GeV)
# reproduces the tabulated f with rms pull 0.30 and a total of 7.19e-6 against
# the published (7.2 +- 0.6)e-6 cm^-2 s^-1 sr^-1 above 100 MeV.
#
# Columns:  E_min [GeV]  E_max [GeV]  E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]
#           sig_stat (stat + instrument)  sig_fg (foreground modelling)
_ACKERMANN2015_TABLE3_MODEL_A = np.array([
    [  0.1000,   0.1414, 9.4574e-04, 1.8816e-04, 1.7694e-04],
    [  0.1414,   0.2000, 8.2089e-04, 2.0829e-04, 1.6485e-04],
    [  0.2000,   0.2828, 7.2203e-04, 2.1121e-04, 1.5642e-04],
    [  0.2828,   0.4000, 6.4185e-04, 1.9207e-04, 1.5012e-04],
    [  0.4000,   0.5657, 6.1261e-04, 1.3752e-04, 1.5267e-04],
    [  0.5657,   0.8000, 6.2891e-04, 7.7841e-05, 1.6375e-04],
    [  0.8000,   1.1314, 5.2767e-04, 5.9895e-05, 1.4892e-04],
    [  1.1314,   1.6000, 4.1296e-04, 4.9660e-05, 1.3544e-04],
    [  1.6000,   2.2627, 3.2942e-04, 4.4873e-05, 1.4772e-04],
    [  2.2627,   3.2000, 3.0375e-04, 3.2043e-05, 1.4361e-04],
    [  3.2000,   4.5255, 2.5456e-04, 3.2995e-05, 1.3110e-04],
    [  4.5255,   6.4000, 2.2574e-04, 3.6843e-05, 1.1337e-04],
    [  6.4000,   9.0510, 2.0935e-04, 3.2230e-05, 1.0002e-04],
    [  9.0510,  12.8000, 2.3544e-04, 3.0542e-05, 8.1196e-05],
    [ 12.8000,  18.1019, 1.7651e-04, 2.3370e-05, 6.6536e-05],
    [ 18.1019,  25.6000, 1.6372e-04, 2.0176e-05, 5.7690e-05],
    [ 25.6000,  36.2039, 1.3556e-04, 1.6476e-05, 4.7128e-05],
    [ 36.2039,  51.2000, 1.2991e-04, 1.5420e-05, 4.0927e-05],
    [ 51.2000,  72.4077, 1.1085e-04, 1.3817e-05, 3.2925e-05],
    [ 72.4077, 102.4000, 8.9146e-05, 1.2306e-05, 2.5493e-05],
    [102.4000, 144.8155, 5.4051e-05, 9.9535e-06, 1.6913e-05],
    [144.8155, 204.8000, 4.8410e-05, 9.9578e-06, 1.2952e-05],
    [204.8000, 289.6309, 3.2976e-05, 9.3452e-06, 8.8558e-06],
    [289.6309, 409.6000, 3.1614e-05, 1.0432e-05, 8.4503e-06],
    [409.6000, 579.2619, 1.0251e-05, 7.5566e-06, 4.6744e-06],
    [579.2619, 819.2000, 8.3183e-08, 4.4182e-06, 3.6232e-06],
])


def load_igrb_source() -> SpectrumSource:
    """Return a :class:`SpectrumSource` for the Ackermann+ 2015 IGRB.

    Uses Foreground Model A (their default reconstruction). ``phi_err_sym`` is
    the statistical-plus-instrument uncertainty; see the table header for why
    the foreground-modelling uncertainty is not added to it.
    The column density is the cosmological baseline of §II.A,
    J_cosmo = 1.37 × 10²² GeV cm⁻².
    """
    tbl = _ACKERMANN2015_TABLE3_MODEL_A
    E_min, E_max = tbl[:, 0], tbl[:, 1]
    E_ref = np.sqrt(E_min * E_max)                  # log-centre of each bin
    phi = tbl[:, 2]                                 # MeV cm⁻² s⁻¹ sr⁻¹
    err = tbl[:, 3]                                 # stat + instrument only

    return SpectrumSource(
        E_bins_GeV=E_ref,
        phi=phi,
        phi_err_sym=err,
        phi_err_lo=err,
        phi_err_hi=err,
        tau_prefactor_K=IGRB_J_COSMO_GeV_cm2,
        source_label="IGRB (Ackermann+ 2015, Model A)",
        source_metadata={
            "reference": "Ackermann et al. 2015, ApJ 799, 86, arXiv:1410.3696",
            "table": "Table 3, Foreground Model A",
            "J_column_GeV_cm2": IGRB_J_COSMO_GeV_cm2,
            "J_provenance": (
                "cosmological baseline: rho_chi=1.2e-6 GeV/cm^3, L=1.14e28 cm "
                "(paper §II.A)"
            ),
            "err_convention": (
                "statistical + instrument systematics (Table 3 note 1). The "
                "foreground-modelling uncertainty is a correlated choice of "
                "Galactic diffuse model, not independent per-bin noise, and is "
                "carried in column 4 for reference rather than added here."
            ),
            "n_bins": len(tbl),
            "E_min_GeV": float(E_min.min()),
            "E_max_GeV": float(E_max.max()),
            "provenance": (
                "published machine-readable Table 3 (apj504089t3_mrt.txt), "
                "model A; verbatim copy at "
                "data/ackermann2015_igrb_table3_mrt.txt"
            ),
        },
        dataset_kind="igrb",
    )


# =============================================================================
# Halo adapter — wrap the existing HaloSpectrum in the new interface
# =============================================================================

def wrap_halo_as_source(halo, source_label: str, err_mode: str = "sym") -> SpectrumSource:
    """Wrap an existing :class:`HaloSpectrum` from ``totani_data_loader`` as a
    :class:`SpectrumSource` so the constraint scan sees a single interface.

    The halo case keeps ``tau_prefactor_K = None`` — the constraint generator
    continues to use the ROI-integrated K from ``roi_tau_prefactor``.
    """
    phi_err = getattr(halo, f"phi_err_{err_mode}")
    return SpectrumSource(
        E_bins_GeV=np.asarray(halo.E_bins_GeV, dtype=float),
        phi=np.asarray(halo.phi, dtype=float),
        phi_err_sym=np.asarray(halo.phi_err_sym, dtype=float),
        phi_err_lo=np.asarray(halo.phi_err_lo, dtype=float),
        phi_err_hi=np.asarray(halo.phi_err_hi, dtype=float),
        tau_prefactor_K=None,
        source_label=source_label,
        source_metadata={
            "nfw_label": getattr(halo, "nfw_label", ""),
            "mcmc_dir": getattr(halo, "mcmc_dir", ""),
            "n_loaded": int(getattr(halo, "n_loaded", 0)),
            "err_mode": err_mode,
        },
        dataset_kind="halo",
    )


# =============================================================================
# Dispatcher — one entry point for the constraint generator
# =============================================================================

def load_spectrum_source(dataset: str, **kwargs) -> SpectrumSource:
    """Return a :class:`SpectrumSource` for the requested dataset.

    Parameters
    ----------
    dataset : {'halo', 'dsph', 'igrb'}
    **kwargs
        Passed through to the specific loader. For ``halo`` requires
        ``halo`` (the HaloSpectrum object already loaded) and optionally
        ``err_mode``. For ``dsph`` requires ``dsph_root`` (path to the top
        of the McDaniel 2024 data release) and optionally ``dsph_selection``.

    Returns
    -------
    SpectrumSource
    """
    dataset = str(dataset).lower()
    if dataset == "halo":
        halo = kwargs.pop("halo", None)
        if halo is None:
            raise ValueError("dataset='halo' requires halo=<HaloSpectrum>")
        label = kwargs.pop("source_label", f"halo posterior ({halo.nfw_label})")
        err_mode = kwargs.pop("err_mode", "sym")
        return wrap_halo_as_source(halo, source_label=label, err_mode=err_mode)
    elif dataset == "igrb":
        return load_igrb_source()
    elif dataset == "dsph":
        # Deferred to core.dsph_sed_loader (task #17). Import lazily so the
        # halo/igrb code paths do not need the dSph FITS machinery installed.
        from core.dsph_sed_loader import load_dsph_source  # type: ignore
        return load_dsph_source(**kwargs)
    else:
        raise ValueError(
            f"Unknown dataset={dataset!r}. Choose 'halo', 'dsph', or 'igrb'."
        )
