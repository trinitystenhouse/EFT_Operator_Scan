"""Signal-hypothesis calibration and coverage check.

A 90% CL exclusion threshold must be referenced to the hypothesis being
excluded, not to the null: we exclude (m*, L*) if q exceeds the value that
q attains only 10% of the time WHEN (m*, L*) IS TRUE. Null-generated toys
cannot supply that -- under the null, q at a high-tau cell is dominated by the
signal/null mismatch and its spread says nothing about coverage.

So toys here are generated AT the hypothesis:

    mu*      = Phi_true * exp(-tau(m*, L*))          (A = 1 at truth)
    Phi_toy  = mu* + (chain draw - Phi_true)         (observed noise, shifted)
    (a) q    = chi2(m*, L*) - chi2_min               scan minimum -- production
    (b) q    = chi2(m*, L*) - chi2(tau = 0)          null reference

Residuals are added rather than multiplied so the per-bin sigma scale is
preserved; the profiled A absorbs any residual common normalisation.

(a) is what the production contours apply: extract_90cl_boundary() subtracts
np.nanmin(chi2_grid) before contouring. chi2_min is taken over the production
grid together with the test point itself, so q >= 0.

REVISED 2026-09-14:
  * Residuals are drawn per test point from independent SeedSequence children.
    They were drawn once, before the operator loop, and reused at all twelve
    points, so the quoted quantiles shared one set of fluctuations.
  * Test points sit above the mass-flat region (--mass-set varying, default).
    The previous points, grid indices 0/14/28/42 = 1e-6 ... 0.87 GeV, all lay
    on the low-mass plateau of the dipole and scalar contours, where tau(E) at
    the contour is identical; with shared residuals the scalar Rayleigh gave
    q90 = 3.9943 four times. --mass-set legacy reruns those masses with
    independent draws.
  * L* is solved onto the Asimov contour (q = 4.61) by bisection, not taken at
    the last excluded grid cell, which sat up to one 0.1-dex step inside it.
  * Every 90% quantile carries its Monte Carlo error, and a constant-in-mass
    hypothesis is tested per operator.
"""
import sys
import argparse
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO.parent))
from _toy_common import load_toy_inputs, profiled_chi2, quantile_with_error, profile_tag   # noqa: E402

SEED = 20260808
THR_NOMINAL = 4.61
OPS = ("dipole_magnetic", "scalar_rayleigh", "rayleigh_full")   # the Fig. 3 operators
NPZ = {"dipole_magnetic": "fermionic_dipole_magnetic",
       "scalar_rayleigh": "scalar_scalar_rayleigh",
       "rayleigh_full":   "fermionic_rayleigh_full_majorana"}
VARYING_MASSES_GEV = (1e1, 1e2, 1e3, 1e4)   # four decades, above every plateau
LEGACY_INDICES = (0, 14, 28, 42)


def asimov_lambda(phi, sig, tau_row, p, L, thr=THR_NOMINAL):
    """Largest Lambda with Asimov q >= thr at one mass (Asimov chi2_min = 0)."""
    q = profiled_chi2(phi, phi * np.exp(-tau_row[None, :] * L[:, None] ** (-p)), sig)
    sel = np.where(q >= thr)[0]
    if sel.size == 0 or sel.max() == len(L) - 1:
        return np.nan
    lo, hi = np.log10(L[sel.max()]), np.log10(L[sel.max() + 1])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if profiled_chi2(phi, phi * np.exp(-tau_row * 10.0 ** (-p * mid)), sig) >= thr:
            lo = mid
        else:
            hi = mid
    return 10.0 ** lo


def constant_fit(x, y, s):
    """chi2/dof and p-value for y = const, plus the weighted slope dy/dx +- err."""
    from scipy import stats
    w = 1.0 / s**2
    ybar = np.sum(w * y) / np.sum(w)
    chi2 = float(np.sum(w * (y - ybar) ** 2))
    coef, cov = np.polyfit(x, y, 1, w=1.0 / s, cov="unscaled")
    return chi2, len(y) - 1, float(stats.chi2.sf(chi2, len(y) - 1)), coef[0], np.sqrt(cov[0, 0])


def main():
    ap = argparse.ArgumentParser(description="Signal-hypothesis threshold calibration.")
    ap.add_argument("ntoy", nargs="?", type=int, default=2000)
    ap.add_argument("--mass-set", choices=("varying", "legacy"), default="varying")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None, help="default coverage<tag>.npz (varying) / coverage_legacy<tag>.npz")
    ap.add_argument("--profile", default="pixelwise_global_rho2",
                    help="halo posterior (emissivity weighting). Non-default profiles need "
                         "toy_grid_<op><tag>.npz from run_calibration.py --profile, and write tagged outputs")
    ap.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS),
                    help="subset of operators, e.g. --ops dipole_magnetic for a spot check")
    a = ap.parse_args()
    tag = profile_tag(a.profile)

    E, phi_true, sig, phi_chain, _ = load_toy_inputs(a.profile)
    nb, ns = phi_chain.shape

    specs = []
    for op in a.ops:
        g = np.load(_HERE / f"toy_grid_{op}{tag}.npz")
        M = g["M"]
        idx = (LEGACY_INDICES if a.mass_set == "legacy" else
               [int(np.argmin(np.abs(np.log10(M) - np.log10(m)))) for m in VARYING_MASSES_GEV])
        specs += [(op, g, i) for i in idx]
    streams = np.random.SeedSequence(a.seed).spawn(len(specs))
    print(f"[coverage] N={a.ntoy} per point, {len(specs)} points, mass set '{a.mass_set}', "
          f"independent residuals per point (SeedSequence {a.seed})")

    rows = []
    for (op, g, i), ss in zip(specs, streams):
        M, L, tau_ref, p = g["M"], g["L"], g["tau_ref"], float(g["p"])
        rng = np.random.default_rng(ss)
        resid = np.stack([phi_chain[k][rng.integers(0, ns, a.ntoy)] for k in range(nb)], 1) - phi_true

        L_star = asimov_lambda(phi_true, sig, tau_ref[i], p, L)
        d = np.load(_REPO / "constraint_boundaries" /
                    f"mcmc_{a.profile}_halo_raw_attenuation_{NPZ[op]}_profnorm_removalfix_90cl.npz")
        L_prod = float(10 ** np.interp(np.log10(M[i]), np.log10(d["mchi_GeV"]), np.log10(d["lambda_GeV"])))
        tau_star = tau_ref[i] * L_star ** (-p)
        mu_star = phi_true * np.exp(-tau_star)
        toys = mu_star + resid

        chi2_star = profiled_chi2(toys, np.broadcast_to(mu_star, toys.shape), sig)
        chi2_null = profiled_chi2(toys, np.broadcast_to(phi_true, toys.shape), sig)
        chi2_min = chi2_star.copy()
        Lpow = L[:, None] ** (-p)
        for m_i in range(len(M)):
            mdl = phi_true * np.exp(-tau_ref[m_i][None, :] * Lpow)
            chi2_min = np.minimum(chi2_min, np.nanmin(profiled_chi2(toys[:, None, :], mdl[None], sig), axis=1))
        qa, qb = chi2_star - chi2_min, chi2_star - chi2_null

        q90a, ea, ba = quantile_with_error(qa, rng=rng)
        q90b, eb, bb = quantile_with_error(qb, rng=rng)
        cov = float(np.mean(qa <= THR_NOMINAL))
        cov_err = np.sqrt(cov * (1 - cov) / a.ntoy)
        rows.append(dict(op=op, m=M[i], L=L_star, L_prod=L_prod, tau_max=float(tau_star.max()),
                         q90a=q90a, q90a_err=ea, q90a_boot=ba, q90b=q90b, q90b_err=eb, q90b_boot=bb,
                         cov=cov, cov_err=cov_err))
        print(f"  {op:16s} m*={M[i]:9.3e}  L*={L_star:.4f} (prod {L_prod:.4f})  tau_max={tau_star.max():.4f}"
              f"  q90(a)={q90a:.3f}+-{ea:.3f} [boot {ba:.3f}]  q90(b)={q90b:.3f}+-{eb:.3f}"
              f"  cov@4.61={100*cov:.1f}+-{100*cov_err:.1f}%")

    col = lambda k: np.array([r[k] for r in rows])
    out = Path(a.out) if a.out else _HERE / (f"coverage{tag}.npz" if a.mass_set == "varying"
                                             else f"coverage_legacy{tag}.npz")
    np.savez_compressed(out, op=col("op"), m_star=col("m"), L_star=col("L"), L_prod=col("L_prod"),
                        tau_max=col("tau_max"), q90=col("q90a"), q90_err=col("q90a_err"),
                        q90_boot=col("q90a_boot"), q90_b=col("q90b"), q90_b_err=col("q90b_err"),
                        coverage=col("cov"), coverage_err=col("cov_err"), ntoy=a.ntoy, seed=a.seed,
                        mass_set=a.mass_set, profile=a.profile, statistic="a: chi2 - chi2_min (production); b: chi2 - chi2(tau=0)")

    q90a, ea, q90b = col("q90a"), col("q90a_err"), col("q90b")
    print(f"\n  q90(a) across {len(rows)} points: min={q90a.min():.3f} median={np.median(q90a):.3f} "
          f"max={q90a.max():.3f}; typical MC error {np.median(ea):.3f}")
    print(f"  q90(b) across {len(rows)} points: min={q90b.min():.3f} median={np.median(q90b):.3f} "
          f"max={q90b.max():.3f}")
    print(f"  (a) - (b) per point: {np.round(q90a - q90b, 3)}")
    cv = col("cov")
    print(f"  coverage at 4.61: min={100*cv.min():.1f}% median={100*np.median(cv):.1f}% max={100*cv.max():.1f}%")
    print("\n  trend with mass, q90(a) = const vs log10 m:")
    for op in tuple(a.ops) + ("ALL",):
        sel = np.ones(len(rows), bool) if op == "ALL" else (col("op") == op)
        chi2, dof, pval, slope, serr = constant_fit(np.log10(col("m")[sel]), q90a[sel], ea[sel])
        print(f"    {op:16s} chi2/dof={chi2:.2f}/{dof}  p={pval:.3f}  slope={slope:+.3f}+-{serr:.3f} per decade")
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
