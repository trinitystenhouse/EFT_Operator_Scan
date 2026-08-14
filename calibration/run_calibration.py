"""Toy Monte Carlo calibration of the Delta chi^2 exclusion threshold.

Replaces the assumed asymptotic 4.61 (90% CL, 2 dof) with a measured one, and
produces median-expected contours with bands.

The critical construction: the source template Phi_src stays FIXED at the
observed p50 spectrum across all toys; only the pseudo-data fluctuates. Setting
Phi_src = Phi_toy would reproduce chi^2(tau=0) = 0 for every toy and learn
nothing -- that degeneracy is exactly what makes the published contours Asimov
expected sensitivity rather than an observed exclusion.

    Phi_true = observed halo p50
    Phi_toy  = burn-in-trimmed chain draw, independent per bin
    mu_i     = A * Phi_true,i * exp(-tau_i(m, Lambda)),  A profiled analytically
    chi2_toy = sum_i (Phi_toy,i - A_hat mu_i)^2 / sigma_i^2

Two statistics are compared:
    (a) q = chi2(m,L) - chi2_min      global grid minimum; carries scan dependence
    (b) q = chi2(m,L) - chi2(tau=0)   referenced to the null; no scan dependence

tau is cached at Lambda = 1 and rescaled, since tau ~ Lambda^-p holds to machine
precision for the closed forms (verified: max |pred/actual - 1| < 9e-16).
"""
import sys, json
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO.parent))

from _toy_common import load_toy_inputs, profiled_chi2          # noqa: E402
from core.spectral_reshaping import ReshapingConfig, compute_tau_spectrum   # noqa: E402
from core.spectrum_source import wrap_halo_as_source            # noqa: E402
from core.totani_data_loader import _MCMC_DIRS, load_halo_spectrum          # noqa: E402
from constraint_generation.make_data_driven_scattering_limits import operator_couplings  # noqa: E402

NTOY   = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED   = 20260807
M_GRID = np.logspace(-6, 8, 100)      # production scan grid
L_GRID = np.logspace(-3, 7, 100)

OPS = [   # (key, dm_type, majorana, Lambda power p in tau ~ Lambda^-p)
    ("dipole_magnetic",  "fermionic", False, 4),
    ("dipole_electric",  "fermionic", False, 4),
    ("scalar_rayleigh",  "scalar",    False, 4),
    ("rayleigh_odd",     "fermionic", True,  6),
    ("rayleigh_even",    "fermionic", True,  6),
    ("rayleigh_full",    "fermionic", True,  6),
]


def tau_reference(op, dm, maj, E, phi_true, sig, K):
    """tau_i(m_chi) at Lambda = 1, shape (nM, nbin). Corrected numerics."""
    cs, cp, cphi = operator_couplings(op, dm)
    out = np.empty((len(M_GRID), len(E)))
    for i, m in enumerate(M_GRID):
        cfg = ReshapingConfig(
            m_chi=float(m), Lambda=1.0, operator=op, dm_type=dm,
            c_s=cs, c_p=cp, c_phi=cphi, majorana=maj,
            E_bins=E, phi_0=phi_true, phi_data=phi_true, phi_err=sig,
            tau_prefactor_override=K,
            sigma_model="closed_form", roi_recovery_model="template",
            kernel_nodes="exact_u")
        out[i] = np.atleast_1d(compute_tau_spectrum(cfg, arm="attenuation"))
    return out


def main():
    E, phi_true, sig, phi_chain, idx = load_toy_inputs()
    s = wrap_halo_as_source(load_halo_spectrum(_MCMC_DIRS["pixelwise_global_rho2"]),
                            source_label="toys")
    K = s.tau_prefactor_K
    nb, ns = phi_chain.shape
    rng = np.random.default_rng(SEED)
    toys = np.stack([phi_chain[i][rng.integers(0, ns, NTOY)] for i in range(nb)], 1)

    chi2_null = profiled_chi2(toys, np.broadcast_to(phi_true, toys.shape), sig)
    print(f"[toys] N={NTOY}  bins={nb}  chi2_null mean={chi2_null.mean():.3f} "
          f"median={np.median(chi2_null):.3f}")

    results = {}
    for op, dm, maj, p in OPS:
        tau_ref = tau_reference(op, dm, maj, E, phi_true, sig, K)      # (nM, nb)
        Lpow = L_GRID[:, None] ** (-p)                                  # (nL, 1)

        # ---- pass 1: chi2_min per toy (global grid minimum) -----------------
        chi2_min = np.full(NTOY, np.inf)
        for i in range(len(M_GRID)):
            tau_i = tau_ref[i][None, :] * Lpow                           # (nL, nb)
            model = phi_true[None, :] * np.exp(-tau_i)                   # (nL, nb)
            c = profiled_chi2(toys[:, None, :], model[None, :, :], sig)  # (NTOY, nL)
            chi2_min = np.minimum(chi2_min, np.nanmin(c, axis=1))

        # ---- pass 2: per-cell 90th percentile of q, both constructions ------
        q90_a = np.empty((len(M_GRID), len(L_GRID)))
        q90_b = np.empty_like(q90_a)
        for i in range(len(M_GRID)):
            tau_i = tau_ref[i][None, :] * Lpow
            model = phi_true[None, :] * np.exp(-tau_i)
            c = profiled_chi2(toys[:, None, :], model[None, :, :], sig)  # (NTOY, nL)
            q90_a[i] = np.percentile(c - chi2_min[:, None], 90, axis=0)
            q90_b[i] = np.percentile(c - chi2_null[:, None], 90, axis=0)

        thr_calib = float(np.median(q90_b))

        # ---- pass 3: per-toy exclusion contour at each threshold ------------
        # Lambda_90(m) = largest Lambda whose q exceeds the threshold.
        contours = {}
        for tag, thr in (("nominal", 4.61), ("calib", thr_calib)):
            lam = np.full((NTOY, len(M_GRID)), np.nan)
            for i in range(len(M_GRID)):
                tau_i = tau_ref[i][None, :] * Lpow
                model = phi_true[None, :] * np.exp(-tau_i)
                c = profiled_chi2(toys[:, None, :], model[None, :, :], sig)
                sel = (c - chi2_null[:, None]) >= thr                    # (NTOY, nL)
                any_ = sel.any(1)
                last = len(L_GRID) - 1 - np.argmax(sel[:, ::-1], axis=1)
                lam[any_, i] = L_GRID[last[any_]]
            contours[tag] = lam
            np.savez_compressed(_HERE / f"toy_contours_{op}_{tag}.npz",
                                M=M_GRID, lam90=lam.astype(np.float32), threshold=thr)

        results[op] = dict(tau_ref=tau_ref, q90_a=q90_a, q90_b=q90_b, p=p,
                           thr_calib=thr_calib)
        np.savez_compressed(_HERE / f"toy_grid_{op}.npz",
                            M=M_GRID, L=L_GRID, tau_ref=tau_ref,
                            q90_a=q90_a, q90_b=q90_b, chi2_min=chi2_min,
                            chi2_null=chi2_null, p=p, thr_calib=thr_calib)
        print(f"  [{op}] q90_b median={np.median(q90_b):.3f}  "
              f"q90_a median={np.median(q90_a):.3f}  "
              f"IQR(q90_b)=[{np.percentile(q90_b,25):.2f},{np.percentile(q90_b,75):.2f}]")

    np.savez_compressed(_HERE / "calibration_summary.npz",
                        M=M_GRID, L=L_GRID, chi2_null=chi2_null,
                        **{f"{k}_q90_b": v["q90_b"] for k, v in results.items()},
                        **{f"{k}_q90_a": v["q90_a"] for k, v in results.items()})
    print("done")


if __name__ == "__main__":
    main()
