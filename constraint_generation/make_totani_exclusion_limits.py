#!/usr/bin/env python3
"""
make_totani_exclusion_limits.py
================================
Extract 95% CL exclusion boundaries from tension_scan_29gev_v2 scan grids
and produce conference-style overlay plots matching totani_operator_overlays.pdf.

Logic
-----
The scan stores chi2(m_ann, Lambda) for each operator. A point is EXCLUDED
at 95% CL if chi2 > chi2_min + delta_chi2. Small Lambda (strong scattering)
over-attenuates the halo flux, worsening chi2 — so the boundary traces the
maximum Lambda that the data rules out at each m_ann.

Output
------
  constraint_boundaries/totani_halo_exclusion_<op>_95cl.npz  (one per operator)
  plots/totani_exclusion_overlays_conference.pdf / .png

Usage
-----
  cd Totani_Scattering/
  python constraint_generation/make_totani_exclusion_limits.py
  python constraint_generation/make_totani_exclusion_limits.py --delta-chi2 3.84   # 1-dof
  python constraint_generation/make_totani_exclusion_limits.py --scan-dir results/tension_scan_29gev_v2
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_WORKSPACE = _ROOT.parent
BOUNDARY_DIR     = _ROOT / "constraint_boundaries"
SCAN_DIR_DEFAULT = _ROOT / "results" / "tension_scan"
OUTPUT_DIR       = _ROOT / "plots"

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_WORKSPACE))
from constraints_data.limits import LIMITS_BY_OPERATOR

# ---------------------------------------------------------------------------
# Operator metadata: scan-dir-key → limits-registry operator key.
# ---------------------------------------------------------------------------
SCAN_TO_OPERATOR_KEY = {
    "dipole_magnetic_fermionic":        "dipole_magnetic",
    "dipole_electric_fermionic":        "dipole_electric",
    "charge_radius_fermionic":          "charge_radius",
    "anapole_fermionic":                "anapole",
    "anapole_fermionic_majorana":       "anapole_majorana",
    "rayleigh_even_fermionic":          "rayleigh_even",
    "rayleigh_odd_fermionic":           "rayleigh_odd",
    "rayleigh_even_fermionic_majorana": "rayleigh_even_majorana",
    "rayleigh_odd_fermionic_majorana":  "rayleigh_odd_majorana",
    "rayleigh_full_scalar":             "scalar_rayleigh",
}


def panel_metadata_from_registry(operator_key: str) -> tuple[str, str, str, bool]:
    spec = LIMITS_BY_OPERATOR[operator_key]
    return (
        operator_key,
        str(spec["operator"]),
        str(spec["dm_type"]),
        bool(spec["majorana"]),
    )

# ---------------------------------------------------------------------------
# Load scan npz
# ---------------------------------------------------------------------------
def load_scan_grid(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    f = data.files

    def get(key, default=None):
        return np.asarray(data[key], dtype=float) if key in f else default

    # chi2 grid is saved as 3D (n_mass, n_scat, n_coupling); squeeze scat dim if fixed
    chi2 = get("chi2")
    if chi2 is not None and chi2.ndim == 3:
        chi2 = np.nanmin(chi2, axis=1)   # (n_mass, n_coupling)

    chi2_dof = get("chi2_per_dof")
    if chi2_dof is not None and chi2_dof.ndim == 3:
        chi2_dof = np.nanmin(chi2_dof, axis=1)

    return {
        "ann_masses": get("ann_masses_GeV", np.array([])),
        "couplings":  get("couplings",      np.array([])),
        "chi2":       chi2,
        "chi2_dof":   chi2_dof,
    }


# ---------------------------------------------------------------------------
# Extract boundary
# ---------------------------------------------------------------------------
def extract_exclusion_boundary(
    ann_masses: np.ndarray,
    couplings:  np.ndarray,
    chi2_grid:  np.ndarray,
    delta_chi2: float = 5.99,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mchi, lambda_boundary) arrays tracing the 95% CL exclusion contour.

    At each m_ann we find the largest Lambda where chi2 > chi2_min + delta_chi2.
    Points with small Lambda (strong scattering) are excluded because they
    suppress the halo flux below what Totani measures.
    """
    chi2_min  = np.nanmin(chi2_grid)
    threshold = chi2_min + delta_chi2

    mchi_out   = []
    lambda_out = []

    for i, m in enumerate(ann_masses):
        row = chi2_grid[i, :]
        excluded = np.isfinite(row) & (row > threshold)
        if not np.any(excluded):
            continue

        # Boundary = rightmost excluded index (largest Lambda that is excluded)
        idx = np.where(excluded)[0][-1]

        # Sub-grid interpolation
        if idx < len(couplings) - 1 and np.isfinite(row[idx]) and np.isfinite(row[idx + 1]):
            dchi = row[idx + 1] - row[idx]
            if abs(dchi) > 1e-30:
                frac = (threshold - row[idx]) / dchi
                log_lam = np.log10(couplings[idx]) + frac * (
                    np.log10(couplings[idx + 1]) - np.log10(couplings[idx])
                )
                lam = 10.0 ** log_lam
            else:
                lam = couplings[idx]
        else:
            lam = couplings[idx]

        mchi_out.append(m)
        lambda_out.append(lam)

    if len(mchi_out) < 2:
        return np.array([]), np.array([])

    return np.array(mchi_out), np.array(lambda_out)


# ---------------------------------------------------------------------------
# Save boundary npz (format expected by make_paper_style_operator_overlays.py)
# ---------------------------------------------------------------------------
def save_boundary(
    out_path:   Path,
    mchi:       np.ndarray,
    lam:        np.ndarray,
    operator:   str,
    dm_type:    str,
    delta_chi2: float,
    scan_floor_limited: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label = rf"Totani halo excl. (this work, $\Delta\chi^2={delta_chi2:.2f}$)"
    np.savez(
        str(out_path),
        mchi_GeV            = mchi,
        lambda_GeV          = lam,
        lambda_plot_GeV     = lam,         # C=1 throughout, so lambda_plot = lambda_raw
        paper_label         = label,
        operator            = operator,
        dm_type             = dm_type,
        validity_guides     = "yes",
        boundary_extraction = "chi2_threshold_95cl",
        scan_floor_limited  = scan_floor_limited,
        omega_max_for_validity = np.float64(494.28),  # max LAT energy bin in scan [GeV]
        eft_kinematic_factor   = np.float64(1.0),
    )
    print(f"  Saved: {out_path.name}  ({len(mchi)} pts)")


# ---------------------------------------------------------------------------
# Find scan directories
# ---------------------------------------------------------------------------
def find_scan_dirs(scan_root: Path) -> dict[str, Path]:
    found = {}
    for d in sorted(scan_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        for suffix in ("_WW_rho2", "_bb_rho2", "_tautau_rho2"):
            if name.endswith(suffix):
                found[name[: -len(suffix)]] = d
                break
    return found


# ---------------------------------------------------------------------------
# Process all operators
# ---------------------------------------------------------------------------
def process_all(scan_root: Path, delta_chi2: float) -> list[str]:
    scan_dirs   = find_scan_dirs(scan_root)
    saved_keys  = []

    for scan_key, scan_dir in scan_dirs.items():
        if "higgs_portal" in scan_key:
            print(f"  Skip {scan_key}  (y_eff axis, not Lambda)")
            continue

        npz_path = scan_dir / "scan_grid.npz"
        if not npz_path.exists():
            print(f"  Skip {scan_key}  (no scan_grid.npz)")
            continue

        meta = None
        for k, operator_key in SCAN_TO_OPERATOR_KEY.items():
            if scan_key.startswith(k):
                meta = panel_metadata_from_registry(operator_key)
                break
        if meta is None:
            print(f"  Skip {scan_key}  (no panel config)")
            continue

        panel_key, operator, dm_type, majorana = meta
        majorana_suffix = "_majorana" if majorana else ""
        print(f"\n[{scan_key}]")

        grid = load_scan_grid(npz_path)

        # Prefer raw chi2 over chi2/dof so threshold is in chi2 units
        chi2_grid = grid["chi2"] if (grid["chi2"] is not None and grid["chi2"].size > 0) \
                    else grid["chi2_dof"]

        if chi2_grid is None or chi2_grid.size == 0:
            print("  No chi2 data")
            continue

        ann_masses = grid["ann_masses"]
        couplings  = grid["couplings"]

        if ann_masses.size == 0 or couplings.size == 0:
            print("  Empty mass/coupling axes")
            continue

        if chi2_grid.shape != (len(ann_masses), len(couplings)):
            print(f"  Shape mismatch: chi2={chi2_grid.shape}, "
                  f"masses={len(ann_masses)}, couplings={len(couplings)}")
            continue

        mchi, lam = extract_exclusion_boundary(ann_masses, couplings, chi2_grid, delta_chi2)

        if len(mchi) == 0:
            print("  No exclusion boundary (tau << 1 everywhere in EFT-valid region)")
            mchi = np.array([np.nan])
            lam  = np.array([np.nan])
            floor = True
        else:
            print(f"  Boundary: m=[{mchi.min():.2e},{mchi.max():.2e}] GeV, "
                  f"Lambda=[{lam.min():.2e},{lam.max():.2e}] GeV")
            floor = False

        out_name = f"totani_halo_exclusion_{dm_type}_{operator}{majorana_suffix}_95cl.npz"
        save_boundary(
            BOUNDARY_DIR / out_name,
            mchi, lam, operator, dm_type, delta_chi2,
            scan_floor_limited=floor,
        )
        saved_keys.append(panel_key)

    return saved_keys


# ---------------------------------------------------------------------------
# Conference overlay plot
# ---------------------------------------------------------------------------
def make_conference_overlay(
    saved_panel_keys: list[str],
    out_name:    str   = "totani_exclusion_overlays_conference",
    delta_chi2:  float = 5.99,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available"); return

    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_WORKSPACE))
    try:
        from make_paper_style_operator_overlays import (
            PANEL_CONFIGS,
            THIS_WORK_COLOR, GUIDE_COLOR,
            canonical_legend_label, legend_sort_key,
            plot_panel, finite_positive_curve,
            OUTPUT_DIR as OVERLAY_DIR,
        )
        from helpers.trinity_plotting import save_figure, set_plot_style
    except ImportError as e:
        print(f"Cannot import overlay machinery: {e}"); return

    EXCL_COLOR = "#FF6B35"    # warm orange — distinct from cyan/teal THIS_WORK palette
    EXCL_LABEL = f"Totani halo exclusion (this work, 95% CL, Δχ²={delta_chi2:.2f})"
    DECONV_TAU_COLOR = "#8A2BE2"
    DECONV_TAU_LABEL = "Deconv spectral-modification limit"
    DECONV_TENSION_COLOR = "#2E8B57"
    DECONV_TENSION_LABEL = "Deconv Δtension=0 boundary"

    def _deconv_operator_tokens(cfg: dict) -> list[str]:
        op = str(cfg["operator"])
        if op == "scalar_rayleigh":
            return ["rayleigh_full", "scalar_rayleigh"]
        return [op]

    def _find_deconv_boundary(cfg: dict, limit_kind: str) -> Path | None:
        """Find deconv boundary files with robust fallback over naming variants."""
        dm = str(cfg["dm_type"])
        op_tokens = _deconv_operator_tokens(cfg)
        candidates: list[Path] = []

        # Preferred naming from make_deconv_exclusion_limits.py
        for op_token in op_tokens:
            candidates.append(
                BOUNDARY_DIR / f"deconv_{limit_kind}_limit_{op_token}_{dm}_WW_rho2.npz"
            )

        # Fallback: any matching channel/profile for this operator+dm type.
        if not any(p.exists() for p in candidates):
            for op_token in op_tokens:
                candidates.extend(
                    sorted(BOUNDARY_DIR.glob(f"deconv_{limit_kind}_limit_{op_token}_{dm}_*.npz"))
                )

        for path in candidates:
            if path.exists():
                return path
        return None

    # Plot all registered operators that have panel support; highlight scan results.
    operators = [op for op in LIMITS_BY_OPERATOR if op in PANEL_CONFIGS]
    n = len(operators)
    ncols = 3
    nrows = -(-n // ncols)   # ceiling division

    set_plot_style(
        style="light", cmap_name="plasma",
        base_fontsize=17, linewidth=2.4, n_colors=14,
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 5.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.patch.set_facecolor("white")

    all_handles, all_labels, seen = [], [], set()

    for i, op in enumerate(operators):
        ax  = axes[i]
        cfg = PANEL_CONFIGS[op]

        # Draw the standard panel (EFT validity, unitarity, literature, legacy Totani)
        handles = plot_panel(ax, op, show_legend=False)

        # Overlay our new exclusion curve
        majorana_suffix = "_majorana" if cfg["majorana"] else ""
        excl_path = BOUNDARY_DIR / (
            f"totani_halo_exclusion_{cfg['dm_type']}_{cfg['operator']}"
            f"{majorana_suffix}_95cl.npz"
        )

        has_curve = False
        if excl_path.exists():
            d     = np.load(excl_path, allow_pickle=True)
            mchi  = np.asarray(d["mchi_GeV"],        dtype=float)
            lam   = np.asarray(d["lambda_plot_GeV"], dtype=float)
            mchi, lam, _ = finite_positive_curve(mchi, lam)

            if len(mchi) > 0:
                h_excl, = ax.loglog(
                    mchi, lam,
                    color=EXCL_COLOR, lw=2.8, ls="-", zorder=5,
                    label=EXCL_LABEL,
                )
                # Shade excluded region (below curve = stronger scattering = excluded)
                ylim_lo = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-8
                ax.fill_between(
                    mchi, ylim_lo, lam,
                    color=EXCL_COLOR, alpha=0.13, zorder=4,
                )
                handles.append(h_excl)
                has_curve = True

        if not has_curve:
            ax.text(
                0.5, 0.44,
                "No EFT-valid exclusion\n(tau << 1 everywhere)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color=EXCL_COLOR,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    fc=(1.0, 1.0, 1.0, 0.88),
                    ec=EXCL_COLOR,
                    alpha=0.95,
                ),
            )
            flag = "no EFT-valid boundary"
        else:
            flag = "orange: excluded region"
        ax.text(0.03, 0.05, flag, transform=ax.transAxes,
                fontsize=9, color=EXCL_COLOR, va="bottom")

        # Overlay deconvolution-derived boundaries, if present.
        tau_path = _find_deconv_boundary(cfg, "tau")
        if tau_path is not None:
            d_tau = np.load(tau_path, allow_pickle=True)
            m_tau = np.asarray(d_tau["mchi_GeV"], dtype=float)
            l_tau = np.asarray(d_tau["lambda_lim_GeV"], dtype=float)
            m_tau, l_tau, _ = finite_positive_curve(m_tau, l_tau)
            print(f"  [{op}] deconv tau: {len(m_tau)} points from {tau_path.name}")
            if len(m_tau) > 0:
                tau_kwargs = {}
                if len(m_tau) <= 2:
                    tau_kwargs = {"marker": "o", "ms": 6.0, "mec": "none"}
                h_tau, = ax.loglog(
                    m_tau,
                    l_tau,
                    color=DECONV_TAU_COLOR,
                    lw=2.4,
                    ls="--",
                    zorder=5.5,
                    label=DECONV_TAU_LABEL,
                    **tau_kwargs,
                )
                handles.append(h_tau)

        tens_path = _find_deconv_boundary(cfg, "tension")
        if tens_path is not None:
            d_tens = np.load(tens_path, allow_pickle=True)
            m_tens = np.asarray(d_tens["mchi_GeV"], dtype=float)
            l_tens = np.asarray(d_tens["lambda_lim_GeV"], dtype=float)
            m_tens, l_tens, _ = finite_positive_curve(m_tens, l_tens)
            print(f"  [{op}] deconv tension: {len(m_tens)} points from {tens_path.name}")
            if len(m_tens) > 0:
                tens_kwargs = {}
                if len(m_tens) <= 2:
                    tens_kwargs = {"marker": "s", "ms": 6.0, "mec": "none"}
                h_tens, = ax.loglog(
                    m_tens,
                    l_tens,
                    color=DECONV_TENSION_COLOR,
                    lw=2.4,
                    ls=":",
                    zorder=5.6,
                    label=DECONV_TENSION_LABEL,
                    **tens_kwargs,
                )
                handles.append(h_tens)

        for h in handles:
            lbl   = h.get_label()
            canon = canonical_legend_label(lbl)
            if lbl and canon not in seen:
                seen.add(canon)
                all_labels.append(canon)
                all_handles.append(h)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    # Add exclusion patch to legend if not already present
    if EXCL_LABEL not in seen:
        patch = mpatches.Patch(color=EXCL_COLOR, label=EXCL_LABEL)
        all_handles.append(patch)
        all_labels.append(EXCL_LABEL)

    # Sort legend
    pairs = sorted(zip(all_labels, all_handles), key=lambda p: legend_sort_key(p[0]))
    all_labels  = [l for l, _ in pairs]
    all_handles = [h for _, h in pairs]

    ncols_leg = 4 if len(all_handles) <= 12 else 5
    fig.legend(
        all_handles, all_labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.018),
        ncol=ncols_leg, fontsize=10, frameon=False,
        columnspacing=1.1, handlelength=2.0, handletextpad=0.5,
    )

    fig.suptitle(
        "EFT photon–DM operator constraints from Totani 20 GeV halo excess",
        fontsize=22, y=0.988,
    )
    fig.text(
        0.5, 0.118,
        rf"Orange: 95% CL exclusion from Totani halo attenuation "
        rf"($\Delta\chi^2={delta_chi2:.2f}$, this work).",
        ha="center", va="center", fontsize=11, color=EXCL_COLOR,
    )
    fig.text(
        0.5, 0.096,
        "Where no boundary appears, tau << 1 for all EFT-valid parameters - "
        "a theoretical no-go for this class of models.",
        ha="center", va="center", fontsize=11, color=GUIDE_COLOR,
    )
    fig.text(
        0.5, 0.074,
        "Purple dashed: deconv spectral-modification limit; green dotted: deconv Δtension=0 boundary.",
        ha="center", va="center", fontsize=10.5, color="#444444",
    )

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.22, 1, 0.965])
    out_path = OVERLAY_DIR / out_name
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"\nSaved: {out_path}.pdf / .png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract Totani halo exclusion limits and make conference overlay plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scan-dir",   type=Path,  default=SCAN_DIR_DEFAULT,
                   help="Root dir containing per-operator scan_grid.npz files.")
    p.add_argument("--delta-chi2", type=float, default=5.99,
                   help="Δχ² threshold: 5.99 = 2-dof 95%% CL; 3.84 = 1-dof 95%% CL.")
    p.add_argument("--out-name",   default="totani_exclusion_overlays_conference",
                   help="Output plot filename (no extension) inside plots/.")
    p.add_argument("--no-plot",    action="store_true", default=False,
                   help="Extract boundaries only, skip the overlay plot.")
    p.add_argument(
        "--style",
        default=None,
        help="Plot style: paper, conference/dark_transparent, or conference_light/light_transparent.",
    )
    args = p.parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style

    if not args.scan_dir.exists():
        print(f"ERROR: scan dir not found: {args.scan_dir}"); return 1

    print(f"Scan root : {args.scan_dir}")
    print(f"Δchi²     : {args.delta_chi2}  (95% CL)")
    print(f"Boundaries→ {BOUNDARY_DIR}\n")

    saved = process_all(args.scan_dir, args.delta_chi2)
    print(f"\nExtracted boundaries for {len(saved)} operators.")

    if not args.no_plot:
        print("\nGenerating conference overlay…")
        make_conference_overlay(saved, out_name=args.out_name, delta_chi2=args.delta_chi2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
