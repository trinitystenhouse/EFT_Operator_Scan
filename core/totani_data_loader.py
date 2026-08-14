"""
totani_data_loader.py
=====================
Load the Galactic-centre halo spectrum from the pixel-level MCMC posterior of
the companion analysis (arXiv:2607.08552). Each npz stores the full posterior
over template coefficients f for one energy bin, fitted to the Fermi-LAT
Pass 8 counts through a binned Poisson likelihood.

Physical convention
-------------------
The NFW halo template is pole-normalised:

    mu_nfw[k, pixel] = (iso_target_e2 / E_ctr[k]^2)
                       * (J(l,b) / J_pole)
                       * expo[k, pixel] * omega[pixel] * dE[k]

so at the galactic pole J/J_pole = 1, and at template coefficient f:

    E^2 dN/dE |_pole = f * iso_target_e2          [MeV cm^-2 s^-1 sr^-1]

This is an exact algebraic consequence of the normalisation choice and
requires no pixel data outside the ROI.

Central value: posterior median f_p50, which guarantees p16 <= p50 <= p84 and so
avoids negative error bars near the positivity boundary.

Public API
----------
load_halo_spectrum(mcmc_dir, nfw_label=None)
    -> HaloSpectrum dataclass with E_bins_MeV, E_bins_GeV, phi, phi_p16,
       phi_p84, phi_err_sym, iso_target_e2 per bin.

available_mcmc_dirs()
    -> dict mapping profile name -> absolute path, for whichever dirs exist.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent

# Root of the companion release (arXiv:2607.08552, doi:10.5281/zenodo.21280725)
# that ships the pixel-level halo posteriors. Point HALO_POSTERIOR_ROOT at the
# directory holding the pixelwise_mcmc_results_* subdirectories, or REPO_PATH at
# a tree laid out as <REPO_PATH>/halo_posterior/mcmc/fit_results/pixelwise_mcmc/.
_ENV_ROOT = os.environ.get("HALO_POSTERIOR_ROOT")
if _ENV_ROOT:
    _PIXEL_ROOT = Path(_ENV_ROOT)
else:
    _PIXEL_ROOT = (
        Path(os.environ.get("REPO_PATH", str(_HERE.parent.parent)))
        / "halo_posterior" / "mcmc" / "fit_results" / "pixelwise_mcmc"
    )

# The three halo posteriors the paper uses: the rho^2 fit that carries the
# limits (Sec. V A), the rho^2.5 stress test drawn as the second contour of
# Fig. 3, and the disk-included refit quoted as the systematic envelope
# (Sec. V B, "weakens the exclusion by at most 0.040 dex").
_MCMC_DIRS = {
    "pixelwise_global_rho2":        _PIXEL_ROOT / "pixelwise_mcmc_results_fig6",
    "pixelwise_global_rho2.5":      _PIXEL_ROOT / "pixelwise_mcmc_results_fig5",
    "pixelwise_global_rho2_w_disk": _PIXEL_ROOT / "pixelwise_mcmc_results_fig6_w_disk",
}

_ISO_TARGET_E2_DEFAULT = 1e-4  # MeV cm^-2 s^-1 sr^-1 — fallback if not in npz


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class HaloSpectrum:
    """
    Totani halo-component spectrum, loaded from MCMC posterior files.

    All flux quantities are E^2 dN/dE at the galactic pole
    [MeV cm^-2 s^-1 sr^-1].

    Attributes
    ----------
    E_bins_MeV : (nE,)   energy bin centres [MeV]
    E_bins_GeV : (nE,)   energy bin centres [GeV]
    phi        : (nE,)   central value (posterior median f_p50 * iso_target_e2)
    phi_p16    : (nE,)   16th percentile (lower 1-sigma)
    phi_p84    : (nE,)   84th percentile (upper 1-sigma)
    phi_err_lo : (nE,)   phi - phi_p16  (always >= 0)
    phi_err_hi : (nE,)   phi_p84 - phi  (always >= 0)
    phi_err_sym: (nE,)   symmetric 1-sigma = 0.5*(phi_p84 - phi_p16)
    iso_target_e2 : (nE,) iso_target_e2 value used in each energy bin [MeV cm^-2 s^-1 sr^-1]
    f_nfw_p50  : (nE,)   raw posterior median NFW coefficient (dimensionless)
    f_nfw_p16  : (nE,)   raw 16th percentile
    f_nfw_p84  : (nE,)   raw 84th percentile
    nfw_label  : str     NFW template label matched in the npz files
    mcmc_dir   : str     directory the data was loaded from
    n_loaded   : int     number of energy bins successfully loaded
    """
    E_bins_MeV:    np.ndarray
    E_bins_GeV:    np.ndarray
    phi:           np.ndarray
    phi_p16:       np.ndarray
    phi_p84:       np.ndarray
    phi_err_lo:    np.ndarray
    phi_err_hi:    np.ndarray
    phi_err_sym:   np.ndarray
    iso_target_e2: np.ndarray
    f_nfw_p50:     np.ndarray
    f_nfw_p16:     np.ndarray
    f_nfw_p84:     np.ndarray
    nfw_label:     str
    mcmc_dir:      str
    n_loaded:      int

    @property
    def positive_mask(self) -> np.ndarray:
        """True where the central value is positive (use for chi2 fits)."""
        return self.phi > 0.0

    @property
    def finite_mask(self) -> np.ndarray:
        """True where all of phi, phi_p16, phi_p84 are finite."""
        return (
            np.isfinite(self.phi)
            & np.isfinite(self.phi_p16)
            & np.isfinite(self.phi_p84)
        )

    def to_fit_arrays(
        self,
        *,
        positive_only: bool = True,
        err_mode: str = "sym",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (E_GeV, phi, sigma) arrays ready for use in chi2 fitting,
        replacing the old PHI_TOTANI / SIGMA_TOTANI / E_BINS_GEV.

        Parameters
        ----------
        positive_only : bool
            If True (default), return only bins where phi > 0. This matches
            the convention in attenuation_eft.py and fit_totani_dm_scattering.py.
        err_mode : str
            'sym'  — symmetric 1-sigma = 0.5*(p84-p16)
            'lo'   — lower asymmetric errorbar (phi - p16)
            'hi'   — upper asymmetric errorbar (p84 - phi)
            'max'  — conservative: max(lo, hi)

        Returns
        -------
        E_GeV, phi, sigma : (n,) arrays
        """
        mask = self.finite_mask
        if positive_only:
            mask = mask & self.positive_mask

        if err_mode == "sym":
            sigma = self.phi_err_sym
        elif err_mode == "lo":
            sigma = self.phi_err_lo
        elif err_mode == "hi":
            sigma = self.phi_err_hi
        elif err_mode == "max":
            sigma = np.maximum(self.phi_err_lo, self.phi_err_hi)
        else:
            raise ValueError(f"err_mode must be 'sym', 'lo', 'hi', or 'max'; got {err_mode!r}")

        # Guard: never return zero or negative sigma
        sigma = np.where(
            (sigma > 0.0) & np.isfinite(sigma),
            sigma,
            np.nan,
        )
        mask = mask & np.isfinite(sigma)

        return (
            self.E_bins_GeV[mask],
            self.phi[mask],
            sigma[mask],
        )

    def summary(self) -> str:
        lines = [
            f"HaloSpectrum from: {self.mcmc_dir}",
            f"  NFW label : {self.nfw_label}",
            f"  nE loaded : {self.n_loaded} / {len(self.E_bins_MeV)}",
            f"  E range   : {self.E_bins_GeV[0]:.2f} – {self.E_bins_GeV[-1]:.2f} GeV",
            f"  phi range : {np.nanmin(self.phi):.3e} – {np.nanmax(self.phi):.3e} MeV cm^-2 s^-1 sr^-1",
            f"  positive bins: {int(self.positive_mask.sum())} / {len(self.phi)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_halo_spectrum(
    mcmc_dir: str | Path,
    *,
    nfw_label: Optional[str] = None,
    n_energy_bins: Optional[int] = None,
    central_stat: str = "f_p50",
) -> HaloSpectrum:
    """
    Load the NFW halo-component spectrum from the MCMC npz files.

    Parameters
    ----------
    mcmc_dir : str or Path
        Directory containing mcmc_results_k00.npz ... mcmc_results_kNN.npz.
    nfw_label : str, optional
        Exact NFW template label to look for in each npz. If None (default),
        the loader uses the first template whose name contains 'nfw'
        (case-insensitive). Raises if zero or more than one are found.
    n_energy_bins : int, optional
        Expected number of energy bins. If provided and the loader finds fewer
        or more, it warns but continues. Inferred automatically if None.
    central_stat : str
        Which posterior statistic to use as the central value.
        'f_p50' (default, recommended) — posterior median.
        'f_ml' — maximum-likelihood point (can produce negative errorbars
                 near zero; only use for diagnostics).

    Returns
    -------
    HaloSpectrum

    Raises
    ------
    FileNotFoundError
        If mcmc_dir does not exist or contains no mcmc_results_k*.npz files.
    KeyError
        If the NFW label cannot be matched in any energy bin.
    """
    mcmc_dir = Path(mcmc_dir)
    if not mcmc_dir.exists():
        raise FileNotFoundError(
            f"MCMC results directory not found: {mcmc_dir}\n"
            "Point HALO_POSTERIOR_ROOT at the companion release "
            "(doi:10.5281/zenodo.21280725); see README.md."
        )

    # Discover all result files
    npz_files = sorted(mcmc_dir.glob("mcmc_results_k*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No mcmc_results_k*.npz files found in {mcmc_dir}"
        )

    nE = len(npz_files)
    if n_energy_bins is not None and nE != n_energy_bins:
        import warnings
        warnings.warn(
            f"Expected {n_energy_bins} energy bins but found {nE} npz files in {mcmc_dir}.",
            stacklevel=2,
        )

    # Energy axis comes from the Ectr_mev field stored in each npz.
    E_bins_MeV = np.full(nE, np.nan)

    # Allocate output arrays
    f_p50   = np.full(nE, np.nan)
    f_p16   = np.full(nE, np.nan)
    f_p84   = np.full(nE, np.nan)
    iso_e2  = np.full(nE, _ISO_TARGET_E2_DEFAULT)
    matched_label: Optional[str] = None
    n_loaded = 0

    for path in npz_files:
        # Extract bin index from filename
        stem = path.stem  # e.g. 'mcmc_results_k06'
        try:
            k = int(stem.split("_k")[-1])
        except ValueError:
            continue

        if k >= nE:
            continue

        npz = np.load(path, allow_pickle=True)

        # Energy axis from npz if not loaded from FITS
        if not np.isfinite(E_bins_MeV[k]) and "Ectr_mev" in npz:
            E_bins_MeV[k] = float(npz["Ectr_mev"])

        # iso_target_e2 for this bin
        if "iso_target_e2" in npz:
            val = npz["iso_target_e2"]
            v = float(val) if val.ndim == 0 else float(val.flat[0])
            if np.isfinite(v) and v > 0.0:
                iso_e2[k] = v

        # Match NFW label
        labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]
        idx = _find_nfw_index(labels, nfw_label, k)
        if idx is None:
            continue

        # Track which label was matched (for reporting)
        if matched_label is None:
            matched_label = labels[idx]

        # Read posterior statistics
        stat_map = {
            "f_p50": "f_p50",
            "f_ml":  "f_ml",
        }
        central_key = stat_map.get(central_stat, "f_p50")
        if central_key not in npz or "f_p16" not in npz or "f_p84" not in npz:
            continue

        f_p50[k] = float(np.atleast_1d(npz[central_key])[idx])
        f_p16[k] = float(np.atleast_1d(npz["f_p16"])[idx])
        f_p84[k] = float(np.atleast_1d(npz["f_p84"])[idx])
        n_loaded += 1

    if n_loaded == 0:
        raise KeyError(
            f"No NFW halo component could be matched in any energy bin in {mcmc_dir}. "
            f"Tried label: {nfw_label!r}. "
            "Check that the MCMC runs completed and that the NFW template label is correct."
        )

    if matched_label is None:
        matched_label = nfw_label or "nfw (inferred)"

    # Convert f -> E^2 dN/dE at the galactic pole
    phi     = f_p50 * iso_e2
    phi_p16 = f_p16 * iso_e2
    phi_p84 = f_p84 * iso_e2

    phi_err_lo  = np.maximum(phi - phi_p16, 0.0)
    phi_err_hi  = np.maximum(phi_p84 - phi, 0.0)
    phi_err_sym = 0.5 * (phi_err_lo + phi_err_hi)

    return HaloSpectrum(
        E_bins_MeV    = E_bins_MeV,
        E_bins_GeV    = E_bins_MeV / 1000.0,
        phi           = phi,
        phi_p16       = phi_p16,
        phi_p84       = phi_p84,
        phi_err_lo    = phi_err_lo,
        phi_err_hi    = phi_err_hi,
        phi_err_sym   = phi_err_sym,
        iso_target_e2 = iso_e2,
        f_nfw_p50     = f_p50,
        f_nfw_p16     = f_p16,
        f_nfw_p84     = f_p84,
        nfw_label     = matched_label,
        mcmc_dir      = str(mcmc_dir),
        n_loaded      = n_loaded,
    )


def _find_nfw_index(
    labels: list[str],
    requested: Optional[str],
    k: int,
) -> Optional[int]:
    """
    Find the index of the NFW component in a list of labels.

    If `requested` is given, match it exactly first; fall back to
    case-insensitive substring match on 'nfw'. If exactly one candidate
    is found, return its index. If zero or multiple, return None.
    """
    if requested is not None and requested in labels:
        return labels.index(requested)

    # Substring search
    candidates = [i for i, lab in enumerate(labels) if "nfw" in lab.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        import warnings
        warnings.warn(
            f"Multiple NFW labels in bin k={k}: "
            f"{[labels[i] for i in candidates]}. "
            "Pass nfw_label= explicitly to resolve ambiguity. Skipping bin.",
            stacklevel=3,
        )
    return None


# ---------------------------------------------------------------------------
# Utility: list available MCMC directories
# ---------------------------------------------------------------------------

def available_mcmc_dirs() -> dict[str, str]:
    """Return a dict of profile_name -> absolute_path for existing MCMC dirs."""
    return {
        name: str(path)
        for name, path in _MCMC_DIRS.items()
        if path.exists()
    }


# ---------------------------------------------------------------------------
# Quick sanity check (run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Available MCMC directories:")
    for name, path in available_mcmc_dirs().items():
        print(f"  {name}: {path}")

    print("\nLoading the rho^2 disk-excluded posterior...")
    try:
        hs = load_halo_spectrum(_MCMC_DIRS["pixelwise_global_rho2"])
        print(hs.summary())
        print("\nFirst 5 bins:")
        for k in range(min(5, len(hs.E_bins_GeV))):
            print(
                f"  k={k:02d}  E={hs.E_bins_GeV[k]:.2f} GeV  "
                f"phi={hs.phi[k]:.3e}  +{hs.phi_err_hi[k]:.2e}/-{hs.phi_err_lo[k]:.2e}"
                f"  [MeV cm^-2 s^-1 sr^-1]"
            )
        E, phi, sigma = hs.to_fit_arrays(positive_only=True)
        print(f"\nPositive-bin fit arrays: {len(E)} bins, "
              f"E in [{E[0]:.2f}, {E[-1]:.2f}] GeV")
    except FileNotFoundError as e:
        print(f"  Could not load: {e}")
        print("  Set HALO_POSTERIOR_ROOT; see README.md.")
