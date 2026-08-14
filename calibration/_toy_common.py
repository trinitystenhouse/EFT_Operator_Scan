"""Shared toy-generation machinery for the threshold calibration.

Burn-in: the stored chains are the FULL emcee output, pre-convergence steps
included. Over the first ~30 steps the walker-mean logprob sits 5-7 orders of
magnitude below the converged plateau, and the resulting outliers inflate the
chain standard deviation to 2.8-4.8x the 68% half-width even though the stored
percentiles (which are outlier-robust) are unaffected. Untrimmed, the null
chi^2 has mean 59.9 against an expected 7. After trimming, std/half-width is
1.004-1.014 and the mean is recovered -- see the report.
"""
import numpy as np
from pathlib import Path
import sys
_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent))


def burn_in_index(logprob):
    """First step from which the walker-mean logprob sits on the converged plateau."""
    m = np.asarray(logprob, float).mean(1)
    plateau = np.median(m[len(m) // 2:])
    scale = max(abs(plateau) * 1e-6, 1.0)
    ok = m > plateau - 10 * scale
    return 0 if ok.all() else int(len(m) - np.argmin(ok[::-1]))


def load_toy_inputs(profile="pixelwise_global_rho2"):
    """Return (E, phi_true, sigma, phi_chain) on the production 8-bin selection.

    phi_chain is (nbin, nsamp): burn-in-trimmed halo-coefficient chain draws
    converted to flux through the loader's own relation, phi = f_nfw * iso_e2.
    """
    from core.totani_data_loader import _MCMC_DIRS, load_halo_spectrum
    from core.spectrum_source import wrap_halo_as_source
    D = Path(_MCMC_DIRS[profile])
    s = wrap_halo_as_source(load_halo_spectrum(D), source_label="toys")
    E = np.asarray(s.E_bins_GeV, float)
    fit = s.finite_mask & s.positive_mask
    fit[np.argsort(E)[:2]] = False          # --drop-lowest-bins 2
    fit &= E <= 200.0                       # --e-max-fit 200
    idx = np.where(fit)[0]

    flats = []
    for k in idx:
        d = np.load(D / f"mcmc_results_k{k:02d}.npz", allow_pickle=True)
        lab = [str(x) for x in np.asarray(d["labels"]).ravel()]
        j = [i for i, l in enumerate(lab) if l.lower().startswith("nfw")]
        if len(j) != 1:
            raise RuntimeError(f"bin {k}: found {len(j)} 'nfw' labels, expected 1")
        b = burn_in_index(np.asarray(d["logprob"], float))
        ch = np.asarray(d["chain"], float)[b:, :, j[0]].ravel()
        flats.append(ch * float(np.atleast_1d(d["iso_target_e2"])[0]))
    n = min(f.size for f in flats)          # ragged after per-bin trimming
    phi_chain = np.stack([f[:n] for f in flats])
    return (E[idx], np.asarray(s.phi, float)[idx],
            np.asarray(s.phi_err_sym, float)[idx], phi_chain, idx)


def profiled_chi2(toy, model, sigma):
    """chi^2 with the source normalisation A profiled analytically.

    toy   : (..., nbin) pseudo-data
    model : (..., nbin) mu_i / A, i.e. phi_true * exp(-tau)
    """
    w = 1.0 / sigma**2
    num = (toy * model * w).sum(-1)
    den = (model * model * w).sum(-1)
    A = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    r = (toy - A[..., None] * model) / sigma
    return (r * r).sum(-1)
