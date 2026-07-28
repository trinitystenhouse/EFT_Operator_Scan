#!/usr/bin/env python3
"""
Quick overlay of the m_ann benchmark family on the (m_chi, Lambda) plane for
the Blois Fig. 2 paper-summary operators (dipole_magnetic Dirac,
anapole Majorana, scalar Rayleigh).

Reads the NPZs produced by `constraint_generation/run_mann_benchmark_family.py`
and lays each operator on its own subplot, coloured by m_ann, dashed for bb /
solid for WW. Missing contours (the empty 'no 90% CL found' cases) are listed
in the per-panel legend so the absence is explicit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
BOUNDARY_DIR = _HERE / "constraint_boundaries"
OUT_DIR = _HERE / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


OPERATORS = [
    ("Dirac Magnetic Dipole (dim-5)", "fermionic_dipole_magnetic"),
    ("Majorana Anapole / Axial Charge Radius (dim-6)", "fermionic_anapole_majorana"),
    ("Scalar Rayleigh (dim-7)", "scalar_scalar_rayleigh"),
]

CHANNELS = [
    ("WW", "-"),
    ("bb", "--"),
]

M_ANN = [100, 500, 700, 1000]

# A perceptually-ordered colour per m_ann.
MANN_COLORS = {
    100: "#0571b0",
    500: "#92c5de",
    700: "#f4a582",
    1000: "#ca0020",
}


def find_pppc_npz(operator_tag: str, channel: str, m_ann: int) -> Path | None:
    path = (
        BOUNDARY_DIR
        / f"mcmc_rho2_pppc_{channel}_mann{m_ann}_raw_attenuation_{operator_tag}_90cl.npz"
    )
    return path if path.exists() else None


def find_halo_npz(operator_tag: str) -> Path | None:
    """The Blois Fig. 2 black dashed curve uses source=halo, not PPPC."""
    path = BOUNDARY_DIR / f"mcmc_rho2_halo_raw_attenuation_{operator_tag}_90cl.npz"
    return path if path.exists() else None


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), sharey=True)
    fig.suptitle(
        "Source=halo reference (Blois Fig. 2) vs PPPC m_ann family "
        "(rho^2 halo, 90% CL attenuation)",
        fontsize=11,
    )

    for ax, (title, operator_tag) in zip(axes, OPERATORS):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$m_\chi$ [GeV]")
        ax.grid(True, which="both", alpha=0.2)

        # ---- Reference: source=halo (what Blois Fig. 2 shows) ----
        halo_path = find_halo_npz(operator_tag)
        if halo_path is not None:
            z = np.load(halo_path)
            mchi_h = z["mchi_GeV"]
            lam_h = z["lambda_plot_GeV"]
            if len(mchi_h) >= 5:
                order = np.argsort(mchi_h)
                ax.plot(
                    mchi_h[order],
                    lam_h[order],
                    color="black",
                    ls="--",
                    lw=2.4,
                    label="source=halo (Blois Fig. 2)",
                    zorder=10,
                )

        # ---- PPPC m_ann family overlay ----
        missing = []
        for channel, ls in CHANNELS:
            for m_ann in M_ANN:
                path = find_pppc_npz(operator_tag, channel, m_ann)
                if path is None:
                    missing.append(f"{channel} {m_ann} GeV")
                    continue
                z = np.load(path)
                mchi = z["mchi_GeV"]
                lam = z["lambda_plot_GeV"]
                # Drop the trivial single-point "contours"; they aren't really curves.
                if len(mchi) < 5:
                    missing.append(f"{channel} {m_ann} GeV (n_pts={len(mchi)})")
                    continue
                # Sort by mchi so the curve doesn't self-intersect from the contour ordering.
                order = np.argsort(mchi)
                ax.plot(
                    mchi[order],
                    lam[order],
                    color=MANN_COLORS[m_ann],
                    ls=ls,
                    lw=1.5,
                    label=f"PPPC {channel}, $m_{{ann}}$={m_ann} GeV",
                    alpha=0.85,
                )

        if missing:
            ax.text(
                0.02, 0.02,
                "PPPC: no usable contour:\n" + "\n".join(f"  {m}" for m in missing),
                transform=ax.transAxes,
                fontsize=7.0,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff5e6", ec="#cc7700", lw=0.6),
            )

        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7.0, framealpha=0.9)

    axes[0].set_ylabel(r"$\Lambda / C^{1/n}$ [GeV]")

    out_pdf = OUT_DIR / "mann_family_overlay.pdf"
    out_png = OUT_DIR / "mann_family_overlay.png"
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
