"""Expected 90% CL contours from null toys: median and 68%/95% bands.

Background-only pseudo-data (Phi_toy = burn-in-trimmed chain draw, independent
per bin; template fixed at the observed p50), fitted exactly as the production
grids are: A profiled analytically, chi2 on the production (m, Lambda) grid
(100 x 1200 since the 2026-09-15 refinement, taken from the tau cache axes), and Lambda_90(m) taken by the production upper-envelope
construction on the production 800-point mass axis (_toy_common.envelope_contour,
checked against core.attenuation_eft.extract_90cl_boundary at start-up). A band
edge therefore carries the same grid-resolution bias as the plotted contour.

Per mass, the band is a percentile of Lambda_90 over toys. A toy with no
contour at some mass excludes nothing there and enters as Lambda_90 = 0, not as
a dropped NaN, which would push the band upward exactly where sensitivity runs
out. A percentile that lands on such toys is stored as NaN.

Configurations (statistic, threshold):
    a_nominal     chi2 - chi2_min  at 4.61            production; drawn in Figs. 3-5
    a_calibrated  chi2 - chi2_min  at median q90(a)   from coverage.npz
    b_nominal     chi2 - chi2(0)   at 4.61
    b_calibrated  chi2 - chi2(0)   at median q90(b)   from coverage.npz

REVISED 2026-09-14. Previously the per-toy contour was the last excluded GRID
cell (quantised in 0.1-dex steps, coarser than the band itself), percentiles
dropped no-contour toys (nanpercentile), and the calibrated threshold was a
hard-coded 3.84. expected_contour_<op>.npz is superseded by expected_band_<op>.npz.
"""
import sys
import argparse
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO.parent))
from _toy_common import load_toy_inputs, profiled_chi2, envelope_contour, profile_tag   # noqa: E402

SEED = 20260809
BOUNDARY_SUFFIX = "_profnorm_removalfix"
PCTS = (2.5, 16.0, 50.0, 84.0, 97.5)
NPZ = {  # production Asimov contour, pixelwise rho^2 halo
    "dipole_magnetic": "fermionic_dipole_magnetic",
    "dipole_electric": "fermionic_dipole_electric",
    "scalar_rayleigh": "scalar_scalar_rayleigh",
    "rayleigh_odd":    "fermionic_rayleigh_odd_majorana",
    "rayleigh_even":   "fermionic_rayleigh_even_majorana",
    "rayleigh_full":   "fermionic_rayleigh_full_majorana",
}
FIG3_OPS = ["dipole_magnetic", "rayleigh_full", "scalar_rayleigh"]


def percentiles_with_error(lam):
    """Per-mass percentiles of (NTOY, nM) Lambda_90 (0 = no contour) and their
    order-statistic 1-sigma MC errors in dex. Zero percentiles become NaN."""
    n = lam.shape[0]
    s = np.sort(lam, axis=0)
    pct, err = [], []
    for p in PCTS:
        q = p / 100.0
        h = np.sqrt(n * q * (1 - q))
        lo = s[max(int(np.floor(n * q - h)) - 1, 0)]
        hi = s[min(int(np.ceil(n * q + h)) - 1, n - 1)]
        v = np.percentile(lam, p, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.where((lo > 0) & (hi > 0), 0.5 * np.log10(hi / lo), np.nan)
        pct.append(np.where(v > 0, v, np.nan))
        err.append(e)
    return np.array(pct), np.array(err)


def main():
    ap = argparse.ArgumentParser(description="Expected contours and bands from null toys.")
    ap.add_argument("ntoy", nargs="?", type=int, default=10000)
    ap.add_argument("--ops", nargs="+", default=FIG3_OPS, choices=list(NPZ))
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--profile", default="pixelwise_global_rho2",
                    help="halo posterior; needs toy_grid_<op><tag>.npz from run_calibration.py --profile")
    a = ap.parse_args()
    tag = profile_tag(a.profile)

    E, phi_true, sig, phi_chain, _ = load_toy_inputs(a.profile)
    nb, ns = phi_chain.shape
    # coverage.npz is the signal-hypothesis calibration on the rho^2 posterior,
    # so other profiles get the nominal-threshold configurations only.
    cov_path = _HERE / "coverage.npz"
    thr = {"a_nominal": ("a", 4.61), "b_nominal": ("b", 4.61)}
    if cov_path.exists() and not tag:
        cv = np.load(cov_path)
        if "q90_b" in cv.files:
            thr["a_calibrated"] = ("a", float(np.median(cv["q90"])))
            thr["b_calibrated"] = ("b", float(np.median(cv["q90_b"])))
    print(f"[expected] N={a.ntoy} null toys, {nb} bins; thresholds "
          + ", ".join(f"{k}={v[1]:.3f}" for k, v in thr.items()))

    streams = dict(zip(a.ops, np.random.SeedSequence(SEED).spawn(len(a.ops))))
    for op in a.ops:
        g = np.load(_HERE / f"toy_grid_{op}{tag}.npz")
        M, L, tau_ref, p = g["M"], g["L"], g["tau_ref"], float(g["p"])
        prod_path = (_REPO / "constraint_boundaries" /
                     f"mcmc_{a.profile}_halo_raw_attenuation_{NPZ[op]}{BOUNDARY_SUFFIX}_90cl.npz")
        d = np.load(prod_path, allow_pickle=True)
        T = tau_ref[:, None, :] * L[None, :, None] ** (-p)

        # ---- start-up checks: same chi2 grid, same contour as production -----
        c_as = profiled_chi2(phi_true, phi_true * np.exp(-T), sig)
        pc = d["chi2_grid"].astype(float)
        ok = np.isfinite(pc) & (pc > 1e-3) & (pc < 1e3)
        chi2_dev = float(np.max(np.abs(c_as[ok] / pc[ok] - 1)))
        x_eval, y_as = envelope_contour(M, L, c_as - c_as.min(), 4.61)
        lam_asimov = 10 ** y_as
        from core.attenuation_eft import extract_90cl_boundary
        b = extract_90cl_boundary(M, L, c_as)
        contour_dev = float(np.nanmax(np.abs(np.interp(np.log10(b[:, 0]), x_eval, y_as) - np.log10(b[:, 1]))))
        prod_dev = float(np.nanmax(np.abs(np.interp(np.log10(d["mchi_GeV"]), x_eval, y_as)
                                          - np.log10(d["lambda_GeV"]))))
        print(f"\n  [{op}] checks: chi2 grid vs production max|ratio-1|={chi2_dev:.1e}; "
              f"envelope vs extractor {contour_dev:.1e} dex; vs production contour {prod_dev:.1e} dex")
        if chi2_dev > 1e-4 or contour_dev > 1e-6 or prod_dev > 1e-4:
            raise RuntimeError(f"{op}: toy grid does not reproduce the {BOUNDARY_SUFFIX} production grid")

        # ---- toys -------------------------------------------------------------
        rng = np.random.default_rng(streams[op])
        toys = np.stack([phi_chain[k][rng.integers(0, ns, a.ntoy)] for k in range(nb)], 1)
        lam = {k: np.zeros((a.ntoy, len(x_eval))) for k in thr}
        D = np.empty(a.ntoy)
        models = phi_true * np.exp(-T)                                     # (nM, nL, nb)
        for s0 in range(0, a.ntoy, a.chunk):
            t = toys[s0:s0 + a.chunk]
            c = np.stack([profiled_chi2(t[:, None, :], models[i][None], sig) for i in range(len(M))], 1)
            cmin = c.reshape(len(t), -1).min(1)
            cnull = profiled_chi2(t, np.broadcast_to(phi_true, t.shape), sig)
            D[s0:s0 + len(t)] = cnull - cmin
            for j in range(len(t)):
                ref = {"a": cmin[j], "b": cnull[j]}
                for k, (stat, th) in thr.items():
                    _, y = envelope_contour(M, L, c[j] - ref[stat], th)
                    lam[k][s0 + j] = np.where(np.isfinite(y), 10 ** y, 0.0)

        rec = dict(m_GeV=10 ** x_eval, lam_asimov=lam_asimov, percentiles=np.array(PCTS),
                   ntoy=a.ntoy, seed=SEED, boundary_suffix=BOUNDARY_SUFFIX,
                   halo_profile=a.profile, production_file=prod_path.name,
                   scan_penalty_p90=np.percentile(D, 90))
        for k, (stat, th) in thr.items():
            pct, err = percentiles_with_error(lam[k])
            rec[f"{k}_pct"], rec[f"{k}_pct_err_dex"], rec[f"{k}_threshold"] = pct, err, th
            rec[f"{k}_null_excluded_frac"] = float(np.mean(D >= th)) if stat == "a" else 0.0
            rec[f"{k}_peak_pct"] = np.percentile(lam[k].max(1), PCTS)
        keep = D < 4.61                                  # toys whose scan does not exclude tau = 0
        rec["a_nominal_no_null_excl_pct"], _ = percentiles_with_error(lam["a_nominal"][keep])
        # plotting keys: production statistic and threshold
        P = rec["a_nominal_pct"]
        rec.update(lam_lo95=P[0], lam_lo68=P[1], lam_med=P[2], lam_hi68=P[3], lam_hi95=P[4])
        out = _HERE / f"expected_band_{op}{tag}.npz"
        np.savez_compressed(out, **rec)

        # ---- summary ------------------------------------------------------------
        inside = np.isfinite(lam_asimov)
        for k in thr:
            P, Er = rec[f"{k}_pct"], rec[f"{k}_pct_err_dex"]
            both = inside & np.isfinite(P[1]) & np.isfinite(P[3])
            w = np.log10(P[3][both] / P[1][both])
            off = np.log10(P[2][both] / lam_asimov[both])
            ipk = int(np.nanargmax(np.where(inside, lam_asimov, np.nan)))
            print(f"    {k:13s} thr={thr[k][1]:.3f}  68% width: median {np.median(w):.4f} dex "
                  f"[{w.min():.4f}, {w.max():.4f}], at Asimov peak {np.log10(P[3][ipk]/P[1][ipk]):.4f} dex; "
                  f"median-Asimov offset median {np.median(off):+.4f} [{off.min():+.4f}, {off.max():+.4f}] dex; "
                  f"max MC err on edges {np.nanmax(Er[[1, 3]][:, both]):.4f} dex; "
                  f"null excluded {100*rec[f'{k}_null_excluded_frac']:.2f}%")
        Pn, Pa = rec["a_nominal_no_null_excl_pct"], rec["a_nominal_pct"]
        both = inside & np.isfinite(Pn[3]) & np.isfinite(Pa[3])
        print(f"    band edges without null-excluding toys shift by at most "
              f"{np.nanmax(np.abs(np.log10(Pn[[1, 3]][:, both] / Pa[[1, 3]][:, both]))):.4f} dex; "
              f"scan penalty p90 = {rec['scan_penalty_p90']:.3f}")
        print(f"    wrote {out.name}")


if __name__ == "__main__":
    main()
