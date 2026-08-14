"""Median expected exclusion contours with 68%/95% bands, and the peak shift.

Null-generated toys (background-only), statistic (a) q = chi2 - chi2_min, which
is the construction the paper uses. For each toy the contour is the largest
Lambda whose q exceeds the threshold; median and bands are percentiles of that
contour across toys, per mass.
"""
import sys
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO.parent))
from _toy_common import load_toy_inputs, profiled_chi2      # noqa: E402

NTOY = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SEED = 20260809
THRESHOLDS = {"nominal": 4.61, "calibrated": 3.84}

OPS = ["dipole_magnetic", "dipole_electric", "scalar_rayleigh",
       "rayleigh_odd", "rayleigh_even", "rayleigh_full"]
NPZ = {  # Asimov contour written by the production pipeline
    "dipole_magnetic": "fermionic_dipole_magnetic",
    "dipole_electric": "fermionic_dipole_electric",
    "scalar_rayleigh": "scalar_scalar_rayleigh",
    "rayleigh_odd":    "fermionic_rayleigh_odd_majorana",
    "rayleigh_even":   "fermionic_rayleigh_even_majorana",
    "rayleigh_full":   "fermionic_rayleigh_full_majorana",
}


def main():
    E, phi_true, sig, phi_chain, _ = load_toy_inputs()
    nb, ns = phi_chain.shape
    rng = np.random.default_rng(SEED)
    toys = np.stack([phi_chain[i][rng.integers(0, ns, NTOY)] for i in range(nb)], 1)
    print(f"[expected] N={NTOY} toys, {nb} bins")

    out = {}
    for op in OPS:
        g = np.load(_HERE / f"toy_grid_{op}.npz")
        M, L, tau_ref, p = g["M"], g["L"], g["tau_ref"], float(g["p"])
        Lpow = L[:, None] ** (-p)

        chi2 = np.empty((NTOY, len(M), len(L)), dtype=np.float32)
        for i in range(len(M)):
            mdl = phi_true * np.exp(-tau_ref[i][None, :] * Lpow)
            chi2[:, i, :] = profiled_chi2(toys[:, None, :], mdl[None, :, :], sig).astype(np.float32)
        chi2_min = chi2.reshape(NTOY, -1).min(1)
        q = chi2 - chi2_min[:, None, None]

        rec = {"M": M}
        for tag, thr in THRESHOLDS.items():
            sel = q >= thr
            any_ = sel.any(2)
            last = len(L) - 1 - np.argmax(sel[:, :, ::-1], axis=2)
            lam = np.where(any_, L[last], np.nan)                       # (NTOY, nM)
            with np.errstate(invalid="ignore"):
                pc = np.nanpercentile(lam, [2.5, 16, 50, 84, 97.5], axis=0)
            rec[f"{tag}_pct"] = pc
            rec[f"{tag}_thr"] = thr
            rec[f"{tag}_peak"] = np.nanpercentile(np.nanmax(lam, axis=1),
                                                  [2.5, 16, 50, 84, 97.5])
        out[op] = rec
        np.savez_compressed(_HERE / f"expected_contour_{op}.npz", **rec)
        del chi2, q

        # Asimov peak from the production grid
        pth = (_REPO / "constraint_boundaries" /
               f"mcmc_pixelwise_global_rho2_halo_raw_attenuation_{NPZ[op]}"
               f"_90cl.npz")
        if pth.exists():
            d = np.load(pth, allow_pickle=True)
            asim = float(np.asarray(d["lambda_GeV"]).max())
        else:
            asim = np.nan
        med = rec["nominal_peak"][2]
        cal = rec["calibrated_peak"][2]
        print(f"  {op:17s} Asimov peak={asim:7.4f}   median exp (4.61)={med:7.4f} "
              f"({100*(med/asim-1):+6.1f}%)   median exp (3.84)={cal:7.4f} "
              f"({100*(cal/asim-1):+6.1f}%)")
    print("done")


if __name__ == "__main__":
    main()
