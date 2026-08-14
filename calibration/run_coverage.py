"""Signal-hypothesis calibration and coverage check.

A 90% CL exclusion threshold must be referenced to the hypothesis being
excluded, not to the null: we exclude (m*, L*) if q exceeds the value that
q attains only 10% of the time WHEN (m*, L*) IS TRUE. Null-generated toys
cannot supply that -- under the null, q at a high-tau cell is dominated by the
signal/null mismatch and its spread says nothing about coverage.

So toys here are generated AT the hypothesis:

    mu*      = Phi_true * exp(-tau(m*, L*))          (A = 1 at truth)
    Phi_toy  = mu* + (chain draw - Phi_true)         (observed noise, shifted)
    q        = chi2(m*, L*) - chi2_min               construction (a)

Residuals are added rather than multiplied so the per-bin sigma scale is
preserved; the profiled A absorbs any residual common normalisation.
"""
import sys
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO.parent))
from _toy_common import load_toy_inputs, profiled_chi2      # noqa: E402

NTOY = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = 20260808


def main():
    E, phi_true, sig, phi_chain, _ = load_toy_inputs()
    nb, ns = phi_chain.shape
    rng = np.random.default_rng(SEED)
    resid = np.stack([phi_chain[i][rng.integers(0, ns, NTOY)] for i in range(nb)], 1) - phi_true

    rows = []
    for op in ("dipole_magnetic", "scalar_rayleigh", "rayleigh_full"):
        g = np.load(_HERE / f"toy_grid_{op}.npz")
        M, L, tau_ref, p = g["M"], g["L"], g["tau_ref"], int(g["p"])
        Lpow = L[:, None] ** (-float(p))

        # Test points: the cells that actually set the limit, i.e. on the Asimov
        # contour. Take the largest Lambda whose Asimov q exceeds 4.61.
        tau_grid = tau_ref[:, None, :] * Lpow[None, :, :]           # (nM, nL, nb)
        model_g = phi_true * np.exp(-tau_grid)
        chi2_asimov = profiled_chi2(phi_true, model_g, sig)          # (nM, nL)
        q_asimov = chi2_asimov - np.nanmin(chi2_asimov)
        pts = []
        for i in range(0, len(M), 14):
            sel = np.where(q_asimov[i] >= 4.61)[0]
            if sel.size:
                pts.append((i, int(sel.max())))
        pts = pts[:4]

        for (i, j) in pts:
            mu_star = phi_true * np.exp(-tau_grid[i, j])
            toys = mu_star + resid                                    # (NTOY, nb)
            # chi2 over the whole grid for each toy -> chi2_min
            chi2_min = np.full(NTOY, np.inf)
            chi2_star = profiled_chi2(toys, np.broadcast_to(mu_star, toys.shape), sig)
            for a in range(len(M)):
                mdl = phi_true * np.exp(-tau_ref[a][None, :] * Lpow)   # (nL, nb)
                c = profiled_chi2(toys[:, None, :], mdl[None, :, :], sig)
                chi2_min = np.minimum(chi2_min, np.nanmin(c, axis=1))
            q = chi2_star - chi2_min
            thr90 = float(np.percentile(q, 90))
            cover = float(np.mean(q <= 4.61))
            rows.append((op, M[i], L[j], float(tau_grid[i, j].max()), thr90, cover))
            print(f"  {op:17s} m*={M[i]:9.2e}  L*={L[j]:8.4f}  tau_max={tau_grid[i,j].max():7.4f}"
                  f"   q90={thr90:6.3f}   coverage@4.61={100*cover:5.1f}%")

    np.savez_compressed(_HERE / "coverage.npz",
                        op=np.array([r[0] for r in rows]),
                        m_star=np.array([r[1] for r in rows]),
                        L_star=np.array([r[2] for r in rows]),
                        tau_max=np.array([r[3] for r in rows]),
                        q90=np.array([r[4] for r in rows]),
                        coverage=np.array([r[5] for r in rows]))
    q90 = np.array([r[4] for r in rows]); cov = np.array([r[5] for r in rows])
    print(f"\n  calibrated q90 across {len(rows)} contour points: "
          f"min={q90.min():.3f} median={np.median(q90):.3f} max={q90.max():.3f}")
    print(f"  coverage at the assumed 4.61: min={100*cov.min():.1f}% "
          f"median={100*np.median(cov):.1f}% max={100*cov.max():.1f}%  (nominal 90%)")


if __name__ == "__main__":
    main()
