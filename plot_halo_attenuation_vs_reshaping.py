#!/usr/bin/env python3
"""
Compare 90% CL exclusion in (m_chi, Lambda/C^(1/n)) for the source=halo Totani
spectrum from:

    (a) spectral attenuation only (the Blois Fig. 2 black dashed curve), and
    (b) spectral attenuation + single-scatter reshaping.

For each of the three paper-summary operators, we scan the same (m_chi, Lambda)
grid and overlay:

    * the attenuation 90% CL contour (solid black curve),
    * the reshaping 90% CL excluded cells (red squares), and
    * the multi-scatter regime (light grey shading) where tau > 0.3 and the
      single-scatter reshaping model is not a valid description.

The honest finding (see Blois proceedings): within the single-scatter-valid
regime, the reshaping channel does not extend the morphological exclusion
beyond what attenuation already excludes -- the excluded reshaping cells sit on
or below the multi-scatter boundary, where the model is on the edge of its
validity domain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from core.attenuation_eft import (  # noqa: E402
    _paper_y_axis_values,
    extract_90cl_boundary,
)
from core.spectral_reshaping import scan_reshaping_chi2  # noqa: E402
from core.totani_data_loader import _MCMC_DIRS, load_halo_spectrum  # noqa: E402


OPERATORS = [
    {
        "title": "Dirac Magnetic Dipole (dim-5)",
        "dm_type": "fermionic",
        "operator": "dipole_magnetic",
        "majorana": False,
        "c_s": 1.0, "c_p": 0.0, "c_phi": 1.0,
    },
    {
        "title": "Majorana Anapole / Axial Charge Radius (dim-6)",
        "dm_type": "fermionic",
        "operator": "anapole",
        "majorana": True,
        "c_s": 0.0, "c_p": 1.0, "c_phi": 1.0,
    },
    {
        "title": "Scalar Rayleigh (dim-7)",
        "dm_type": "scalar",
        "operator": "scalar_rayleigh",
        "majorana": False,
        "c_s": 0.0, "c_p": 0.0, "c_phi": 1.0,
    },
]


# Wide grid: low enough Lambda to catch the reshaping-excluded sliver, high
# enough m_chi to span the literature comparator range used in Fig. 2.
MCHI_GRID = np.logspace(-6, 8, 36)
LAMBDA_GRID = np.logspace(-5, 4, 36)


def load_halo_inputs():
    halo = load_halo_spectrum(_MCMC_DIRS["rho2"])
    phi_err = halo.phi_err_sym
    fit_mask = (
        halo.finite_mask
        & np.isfinite(phi_err)
        & (phi_err > 0)
        & halo.positive_mask
    )
    return {
        "E_bins": halo.E_bins_GeV[fit_mask],
        "phi_src": halo.phi.copy()[fit_mask],
        "phi_data": halo.phi[fit_mask],
        "phi_err": phi_err[fit_mask],
    }


def scan_one(operator_cfg, inputs):
    """Run the joint attenuation + reshaping scan for one operator."""
    result = scan_reshaping_chi2(
        MCHI_GRID,
        LAMBDA_GRID,
        dm_type=operator_cfg["dm_type"],
        operator=operator_cfg["operator"],
        c_s=operator_cfg["c_s"],
        c_p=operator_cfg["c_p"],
        c_phi=operator_cfg["c_phi"],
        majorana=operator_cfg["majorana"],
        E_bins=inputs["E_bins"],
        phi_0=inputs["phi_src"],
        phi_data=inputs["phi_data"],
        phi_err=inputs["phi_err"],
        n_theta=64,
        apply_roi_weight=True,
        roi_half_angle_deg=60.0,
        also_compute_attenuation=True,
        fit_normalization=False,
        max_tau_single_scatter=0.3,
        require_lambda_gt_mdm=False,
        verbose=False,
    )
    return result


def to_plot_y(lambda_arr, operator_cfg):
    return _paper_y_axis_values(
        np.asarray(lambda_arr, dtype=float),
        operator_cfg["dm_type"],
        operator_cfg["operator"],
        c_s=operator_cfg["c_s"],
        c_p=operator_cfg["c_p"] if operator_cfg["c_p"] > 0 else 1.0,
        c_phi=operator_cfg["c_phi"],
    )


def cell_corners(mchi, lam_plot, i, j):
    """Return (mchi_lo, mchi_hi, lam_lo, lam_hi) for cell (i, j) using log midpoints."""
    def mid(arr, k):
        if k == 0:
            return arr[k] * np.sqrt(arr[k] / arr[k + 1])
        if k == len(arr) - 1:
            return arr[k] * np.sqrt(arr[k] / arr[k - 1])
        return np.sqrt(arr[k - 1] * arr[k])

    def hi(arr, k):
        if k == len(arr) - 1:
            return arr[k] * np.sqrt(arr[k] / arr[k - 1])
        return np.sqrt(arr[k] * arr[k + 1])

    return mid(mchi, i), hi(mchi, i), mid(lam_plot, j), hi(lam_plot, j)


def main() -> int:
    inputs = load_halo_inputs()
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), sharey=False)
    fig.suptitle(
        "Source=halo Totani spectrum: 90% CL exclusion from attenuation vs "
        "attenuation+reshaping",
        fontsize=11,
    )

    for ax, op in zip(axes, OPERATORS):
        print(f"scanning {op['operator']}{' (majorana)' if op['majorana'] else ''} ...")
        result = scan_one(op, inputs)
        ca = result["chi2_attenuation"]
        cr = result["chi2_reshaping"]
        lam_plot_grid = to_plot_y(LAMBDA_GRID, op)

        # --- Attenuation 90% CL boundary (the Blois Fig. 2 black curve) ---
        boundary_att = extract_90cl_boundary(MCHI_GRID, LAMBDA_GRID, ca)
        if boundary_att.shape[0] > 0:
            mchi_b = boundary_att[:, 0]
            lam_b_plot = to_plot_y(boundary_att[:, 1], op)
            order = np.argsort(mchi_b)
            ax.plot(
                mchi_b[order], lam_b_plot[order],
                color="black", lw=2.6, ls="-",
                label="Attenuation only (90% CL)",
                zorder=10,
            )

        # --- Multi-scatter regime (single-scatter cap trips: chi2_resh = NaN) ---
        # Plot one shaded patch per NaN cell; use very low alpha so the
        # accumulated colour reads as 'forbidden region'.
        nan_mask = ~np.isfinite(cr)
        for i in range(len(MCHI_GRID)):
            for j in range(len(LAMBDA_GRID)):
                if nan_mask[i, j]:
                    x0, x1, y0, y1 = cell_corners(MCHI_GRID, lam_plot_grid, i, j)
                    ax.fill_between(
                        [x0, x1], [y0, y0], [y1, y1],
                        color="0.55", alpha=0.10, lw=0, zorder=1,
                    )

        # --- Reshaping 90% CL excluded cells ---
        cr_min = np.nanmin(cr) if np.any(np.isfinite(cr)) else np.nan
        resh_excl = np.isfinite(cr) & (cr - cr_min > 4.61) if np.isfinite(cr_min) else np.zeros_like(cr, dtype=bool)
        n_excl = int(resh_excl.sum())
        for i in range(len(MCHI_GRID)):
            for j in range(len(LAMBDA_GRID)):
                if resh_excl[i, j]:
                    x0, x1, y0, y1 = cell_corners(MCHI_GRID, lam_plot_grid, i, j)
                    ax.fill_between(
                        [x0, x1], [y0, y0], [y1, y1],
                        color="#d7191c", alpha=0.55, lw=0, zorder=4,
                    )

        # Legend proxies (so the categorical fills appear in the legend).
        ax.fill_between(
            [], [], [], color="0.55", alpha=0.10,
            label="Multi-scatter regime ($\\tau > 0.3$,\nsingle-scatter reshaping invalid)",
        )
        ax.fill_between(
            [], [], [], color="#d7191c", alpha=0.55,
            label=f"Reshaping 90% CL excluded ({n_excl} cells)" if n_excl else "Reshaping 90% CL: no cells excluded",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(op["title"], fontsize=10)
        ax.set_xlabel(r"$m_\chi$ [GeV]")
        ax.set_ylabel(r"$\Lambda / C^{1/n}$ [GeV]")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_xlim(MCHI_GRID.min(), MCHI_GRID.max())
        ax.set_ylim(lam_plot_grid.min(), lam_plot_grid.max())
        ax.legend(loc="upper right", fontsize=7.0, framealpha=0.9)

    out_dir = _HERE / "plots"
    out_dir.mkdir(exist_ok=True)
    out_pdf = out_dir / "halo_attenuation_vs_reshaping.pdf"
    out_png = out_dir / "halo_attenuation_vs_reshaping.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
