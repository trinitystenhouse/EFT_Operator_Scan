"""Null-hypothesis toys: tau cache, per-cell quantiles and the scan penalty.

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

The production contours use (a): extract_90cl_boundary() in core/attenuation_eft.py
subtracts np.nanmin(chi2_grid) before contouring, and main.tex defines
Delta chi^2 = chi^2 - chi^2_min. On the Asimov grids the two coincide
(chi2_min = chi2(tau=0) = 0), so the choice only shows up in toys, through the
scan penalty D = chi2(tau=0) - chi2_min = q_a - q_b >= 0, reported here.

REVISED 2026-09-14. This script used to (i) set thr_calib to the median of the
null-toy q90_b over the WHOLE (m, Lambda) grid -- dominated by cells with
tau ~ 0, where q_b vanishes identically, so it came out 0 to 4e-11 -- and
(ii) build per-toy contours by selecting (c - chi2_null) >= thr, statistic (b),
not the statistic the production contours use. Both are removed: the threshold
is calibrated at the signal hypothesis for both statistics in run_coverage.py,
and per-toy contours are built in run_expected.py with the production
extractor. Pipeline: run_calibration.py -> run_coverage.py -> run_expected.py.

tau is cached at Lambda = 1 and rescaled, since tau ~ Lambda^-p holds to machine
precision for the closed forms (verified: max |pred/actual - 1| < 9e-16).
"""
import sys
import argparse
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO.parent))

from _toy_common import load_toy_inputs, profiled_chi2, profile_tag   # noqa: E402

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


def tau_reference(op, dm, maj, E, phi_true, sig, K, power=2.0):
    """tau_i(m_chi) at Lambda = 1, shape (nM, nbin). Corrected numerics.

    power is the emissivity power of the template w(theta), which must match the
    halo profile the inputs come from."""
    from core.spectral_reshaping import ReshapingConfig, compute_tau_spectrum
    from constraint_generation.make_data_driven_scattering_limits import operator_couplings
    cs, cp, cphi = operator_couplings(op, dm)
    out = np.empty((len(M_GRID), len(E)))
    for i, m in enumerate(M_GRID):
        cfg = ReshapingConfig(
            m_chi=float(m), Lambda=1.0, operator=op, dm_type=dm,
            c_s=cs, c_p=cp, c_phi=cphi, majorana=maj,
            E_bins=E, phi_0=phi_true, phi_data=phi_true, phi_err=sig,
            tau_prefactor_override=K,
            sigma_model="closed_form", roi_recovery_model="template",
            roi_emissivity_power=power, kernel_nodes="exact_u")
        out[i] = np.atleast_1d(compute_tau_spectrum(cfg, arm="attenuation"))
    return out


def main():
    ap = argparse.ArgumentParser(description="Null toys: tau cache and scan penalty.")
    ap.add_argument("ntoy", nargs="?", type=int, default=1000)
    ap.add_argument("--reuse-tau", action="store_true",
                    help="load tau_ref from the existing toy_grid_<op>.npz instead of recomputing")
    ap.add_argument("--profile", default="pixelwise_global_rho2",
                    help="halo posterior; files for any other profile carry a tag, e.g. toy_grid_<op>_rho2.5.npz")
    ap.add_argument("--ops", nargs="+", default=[o[0] for o in OPS], choices=[o[0] for o in OPS])
    ap.add_argument("--nl", type=int, default=100,
                    help="Lambda points over 1e-3 - 1e7 GeV. Must match the production grids "
                         "(1200 since the 2026-09-15 refinement).")
    ap.add_argument("--chunk", type=int, default=2000, help="toys per chi^2 block, bounds memory")
    a = ap.parse_args()
    L_grid = np.logspace(-3, 7, a.nl)
    tag = profile_tag(a.profile)

    E, phi_true, sig, phi_chain, idx = load_toy_inputs(a.profile)
    nb, ns = phi_chain.shape
    rng = np.random.default_rng(SEED)
    toys = np.stack([phi_chain[i][rng.integers(0, ns, a.ntoy)] for i in range(nb)], 1)

    chi2_null = profiled_chi2(toys, np.broadcast_to(phi_true, toys.shape), sig)
    print(f"[toys] N={a.ntoy}  bins={nb}  chi2_null mean={chi2_null.mean():.3f} "
          f"median={np.median(chi2_null):.3f}")

    if not a.reuse_tau:
        from core.spectrum_source import wrap_halo_as_source
        from core.totani_data_loader import _MCMC_DIRS, load_halo_spectrum
        K = wrap_halo_as_source(load_halo_spectrum(_MCMC_DIRS[a.profile]),
                                source_label="toys").tau_prefactor_K

    results = {}
    for op, dm, maj, p in [o for o in OPS if o[0] in a.ops]:
        if a.reuse_tau:
            tau_ref = np.load(_HERE / f"toy_grid_{op}{tag}.npz")["tau_ref"]
        else:
            from core.roi_recovery import emissivity_power_for_profile
            tau_ref = tau_reference(op, dm, maj, E, phi_true, sig, K,
                                    power=emissivity_power_for_profile(a.profile))   # (nM, nb)
        Lpow = L_grid[:, None] ** (-p)                                  # (nL, 1)

        def chi2_rows(model):
            """(NTOY, nL) profiled chi^2 against one mass row, in blocks of a.chunk toys."""
            out = np.empty((a.ntoy, model.shape[0]))
            for s0 in range(0, a.ntoy, a.chunk):
                out[s0:s0 + a.chunk] = profiled_chi2(toys[s0:s0 + a.chunk, None, :], model[None, :, :], sig)
            return out

        # ---- pass 1: chi2_min per toy (global grid minimum) -----------------
        chi2_min = np.full(a.ntoy, np.inf)
        for i in range(len(M_GRID)):
            model = phi_true[None, :] * np.exp(-tau_ref[i][None, :] * Lpow)
            chi2_min = np.minimum(chi2_min, np.nanmin(chi2_rows(model), axis=1))

        # ---- pass 2: per-cell null-toy 90th percentile of q (diagnostic) ----
        q90_a = np.empty((len(M_GRID), len(L_grid)))
        q90_b = np.empty_like(q90_a)
        for i in range(len(M_GRID)):
            model = phi_true[None, :] * np.exp(-tau_ref[i][None, :] * Lpow)
            c = chi2_rows(model)
            q90_a[i] = np.percentile(c - chi2_min[:, None], 90, axis=0)
            q90_b[i] = np.percentile(c - chi2_null[:, None], 90, axis=0)

        # ---- scan penalty: how far the grid minimum undercuts tau = 0 -------
        D = chi2_null - chi2_min
        results[op] = dict(q90_a=q90_a, q90_b=q90_b, D=D)
        np.savez_compressed(_HERE / f"toy_grid_{op}{tag}.npz",
                            M=M_GRID, L=L_grid, tau_ref=tau_ref,
                            q90_a=q90_a, q90_b=q90_b, chi2_min=chi2_min,
                            chi2_null=chi2_null, p=p, scan_penalty=D, profile=a.profile)
        print(f"  [{op}] scan penalty D = chi2(tau=0) - chi2_min: p90={np.percentile(D, 90):.3f}  "
              f"P(D>=3.84)={100*np.mean(D >= 3.84):.2f}%  P(D>=4.61)={100*np.mean(D >= 4.61):.2f}%")

    np.savez_compressed(_HERE / f"calibration_summary{tag}.npz",
                        M=M_GRID, L=L_grid, chi2_null=chi2_null,
                        **{f"{k}_q90_b": v["q90_b"] for k, v in results.items()},
                        **{f"{k}_q90_a": v["q90_a"] for k, v in results.items()},
                        **{f"{k}_scan_penalty": v["D"] for k, v in results.items()})
    print("done")


if __name__ == "__main__":
    main()
