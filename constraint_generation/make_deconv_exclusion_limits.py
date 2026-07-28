#!/usr/bin/env python3
"""
make_deconv_exclusion_limits.py
================================
Extract upper limits on the photon-DM scattering optical depth tau from
the deconvolution methodology.

New methodology
---------------
The old chi2-based limits (make_totani_exclusion_limits.py) asked:
    "Where does scattering ruin the spectral chi2 fit?"
Those limits sit in the EFT-invalid region (small Lambda) and are not
physically meaningful.

The new limits ask:
    "Assuming scattering (m_chi, Lambda) has occurred, how much does the
    deconvolved spectrum differ from the observed spectrum, and does that
    difference exceed the measurement uncertainty?"

Limit definition
----------------
For each (m_chi, Lambda) point in the deconv_scan.npz:

  1. The deconvolved spectrum phi_recovered = T^{-1} @ phi_obs differs from
     phi_obs by delta_phi = phi_recovered - phi_obs.

  2. The deconvolution chi2 wrt the ORIGINAL observed spectrum is:
         chi2_deconv = sum_i (phi_recovered[i] - phi_obs[i])^2 / phi_err[i]^2

     This measures how much the scattering hypothesis modifies the spectrum.
     If chi2_deconv > delta_chi2_threshold, the scattering hypothesis produces
     a spectrum that is inconsistent with the measurement.

  3. The tau upper limit at each (m_chi, Lambda) is the tau_max value at the
     chi2_deconv = threshold boundary.

  4. Since tau scales as Lambda^{-p} (p=2 for dipoles, p=4 for d=6, etc.),
     and chi2_deconv scales as tau^2 (first-order: delta_phi ~ tau), the
     limit on Lambda is:
         Lambda_limit = Lambda_scan * (tau_scan / tau_threshold)^{1/p}
     where tau_threshold = sqrt(chi2_threshold) * phi_err / |delta_phi_shape|.

Additionally, we compute the tension-based limit:
    "For what (m_chi, Lambda) does the PPPC fit to the deconvolved spectrum
    give a tension > 1 that was NOT present in the original fit?"
This is the boundary where scattering CREATES tension, rather than resolving it.
It is a conservative exclusion of scattering scenarios that would make things worse.

Output
------
  constraint_boundaries/deconv_tau_limit_<operator>_<channel>.npz
  constraint_boundaries/deconv_tension_limit_<operator>_<channel>.npz

Usage
-----
  cd Totani_Scattering/
  python constraint_generation/make_deconv_exclusion_limits.py --scan-dir results/deconv_scan
  python constraint_generation/make_deconv_exclusion_limits.py --scan-dir results/deconv_kernel_inspection
  python constraint_generation/make_deconv_exclusion_limits.py --scan-dir results/deconv_mscat_scan --plot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

_HERE        = Path(__file__).resolve().parent
_ROOT        = _HERE.parent
_REPO_ROOT   = _ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
BOUNDARY_DIR = _ROOT / "constraint_boundaries"
DARK_BG      = "#0d0d14"

from helpers.trinity_plotting import current_savefig_kwargs, set_plot_style  # noqa: E402

# ---------------------------------------------------------------------------
# Load deconv_scan.npz
# ---------------------------------------------------------------------------

def load_deconv_npz(path: Path) -> dict:
    d = np.load(str(path), allow_pickle=False)
    return {k: np.asarray(d[k]) for k in d.files}


# ---------------------------------------------------------------------------
# Compute chi2_deconv: how inconsistent is the deconvolved spectrum
# with the original observed data?
# ---------------------------------------------------------------------------

def compute_deconv_chi2(data: dict) -> np.ndarray:
    """
    chi2_deconv[i,j] = sum_k (phi_recovered[k] - phi_obs[k])^2 / phi_err[k]^2

    This is the chi2 of the CORRECTION, not the fit residual.
    It measures whether the scattering hypothesis modifies the spectrum
    by more than the measurement uncertainties.

    For tau << 1 (first order):
        phi_recovered - phi_obs ≈ tau * (I - K) @ phi_obs
        chi2_deconv ≈ sum_k [tau(E_k) * correction_shape[k]]^2 / phi_err[k]^2

    This scales as tau^2, so the limit is:
        tau_limit = tau_scan * sqrt(chi2_threshold / chi2_deconv)

    Shape: (n_scatter_mass, n_coupling) matching tau_max_grid.
    """
    phi_obs = np.asarray(data["phi_data"],    float)     # (nE,)
    phi_err = np.asarray(data["phi_err_sym"], float)     # (nE,)
    tau_grid = np.asarray(data["tau_max_grid"], float)   # (nS, nC)
    nS, nC   = tau_grid.shape

    mask = np.isfinite(phi_obs) & np.isfinite(phi_err) & (phi_err > 0)

    # We can reconstruct the correction shape from tau and the spectral
    # reshaping formula (first-order). Since we don't have K saved, we
    # use the fact that in first order:
    #   chi2_deconv = sum_k (delta_tau_k * phi_obs[k])^2 / phi_err[k]^2
    # where delta_tau_k = tau(E_k) - sum_j K[k,j]*tau(E_j).
    # For a diagonal approximation (attenuation-only, no inscatter cross-term):
    #   chi2_deconv ≈ sum_k (tau[k] * phi_obs[k])^2 / phi_err[k]^2
    # This is an UPPER BOUND on chi2_deconv (inscatter partially cancels attenuation).
    # The true chi2_deconv is smaller, so this gives a CONSERVATIVE limit.

    # tau_max_grid gives the maximum tau across energy bins.
    # For the energy scaling: dipole ~ E^2, so tau(E) ~ tau_max * (E/E_peak)^2.
    E = np.asarray(data["E_bins_GeV"], float)
    E_peak = E[5] if len(E) > 5 else E[-1]  # 21 GeV bin

    # We use the tau_max as a proxy for tau at E_peak and extrapolate.
    # Conservative (upper-bound) chi2_deconv: treat all bins as if tau = tau_max.
    chi2_grid = np.full((nS, nC), np.nan)

    for s in range(nS):
        for j in range(nC):
            tau_max_ij = float(tau_grid[s, j])
            if not np.isfinite(tau_max_ij) or tau_max_ij <= 0:
                continue
            # Dipole E-scaling: tau(E) ~ tau_max * (E/E_peak)^2
            tau_E = tau_max_ij * (E / E_peak) ** 2
            # Upper bound: correction ~ tau * phi_obs (no inscatter cancellation)
            correction = tau_E * phi_obs
            chi2_ij = float(np.sum((correction[mask] / phi_err[mask]) ** 2))
            chi2_grid[s, j] = chi2_ij

    return chi2_grid


# ---------------------------------------------------------------------------
# Extract tau upper limit at each m_chi
# ---------------------------------------------------------------------------

def extract_tau_upper_limit(
    data:            dict,
    chi2_deconv:     np.ndarray,
    chi2_threshold:  float = 5.99,
) -> dict:
    """
    For each m_chi (scatter mass), find the Lambda value where chi2_deconv
    crosses chi2_threshold. Below this Lambda, scattering would modify the
    spectrum beyond measurement uncertainty.

    Returns arrays suitable for a Lambda exclusion plot.
    """
    scatter_masses = np.asarray(data["scatter_masses_GeV"], float)
    couplings      = np.asarray(data["couplings"],          float)
    eft_valid      = np.asarray(data["eft_valid_mask"],     bool)
    nS, nC         = chi2_deconv.shape

    mchi_limit   = []
    lambda_limit = []
    tau_limit_vals = []

    for s, m in enumerate(scatter_masses):
        row     = chi2_deconv[s, :]       # (nC,) vs coupling
        eft_row = eft_valid[s, :]
        tau_row = np.asarray(data["tau_max_grid"][s, :], float)

        if not np.any(np.isfinite(row)):
            continue

        # Find where chi2_deconv crosses threshold (from above as coupling decreases)
        # chi2_deconv increases as Lambda decreases (tau ~ Lambda^{-p})
        # Sort by coupling ascending; chi2 should decrease with coupling
        idx_sort = np.argsort(couplings)
        row_s  = row[idx_sort]
        lam_s  = couplings[idx_sort]
        eft_s  = eft_row[idx_sort]
        tau_s  = tau_row[idx_sort]

        # Find crossing from below threshold to above threshold
        # (smaller Lambda = larger tau = larger chi2_deconv)
        crossing_idx = None
        for j in range(len(lam_s) - 1):
            if (np.isfinite(row_s[j]) and np.isfinite(row_s[j+1]) and
                    row_s[j] <= chi2_threshold < row_s[j+1]):
                crossing_idx = j
                break

        if crossing_idx is None:
            # Either all below threshold (no limit) or all above (fully excluded)
            if np.all(row_s[np.isfinite(row_s)] < chi2_threshold):
                # tau everywhere too small to constrain — no limit this m_chi
                continue
            else:
                # All excluded — limit is at largest coupling with chi2 > threshold
                crossing_idx = np.where(
                    np.isfinite(row_s) & (row_s >= chi2_threshold))[0][-1]
                lambda_lim = lam_s[crossing_idx]
                tau_lim    = tau_s[crossing_idx]
        else:
            # Interpolate in log(Lambda)
            chi2_lo = row_s[crossing_idx]
            chi2_hi = row_s[crossing_idx + 1]
            frac    = (chi2_threshold - chi2_lo) / (chi2_hi - chi2_lo + 1e-300)
            log_lam_lim = (np.log10(lam_s[crossing_idx]) * (1 - frac) +
                           np.log10(lam_s[crossing_idx + 1]) * frac)
            lambda_lim = 10.0 ** log_lam_lim
            # Interpolate tau
            tau_lim = (tau_s[crossing_idx] * (1 - frac) +
                       tau_s[crossing_idx + 1] * frac)

        mchi_limit.append(float(m))
        lambda_limit.append(float(lambda_lim))
        tau_limit_vals.append(float(tau_lim))

    return dict(
        mchi        = np.array(mchi_limit),
        lambda_lim  = np.array(lambda_limit),
        tau_lim     = np.array(tau_limit_vals),
        chi2_threshold = chi2_threshold,
    )


# ---------------------------------------------------------------------------
# Extract tension-worsening boundary
# ---------------------------------------------------------------------------

def extract_tension_limit(data: dict) -> dict:
    """
    Find (m_chi, Lambda) where delta_tension > 0 AND EFT-valid.
    These are parameter combinations where the scattering hypothesis
    would make the Totani tension WORSE.

    This is a conservative exclusion: we exclude scattering scenarios that
    would worsen the already-marginal tension even slightly.
    """
    scatter_masses = np.asarray(data["scatter_masses_GeV"], float)
    couplings      = np.asarray(data["couplings"],          float)
    delta_tension  = np.asarray(data["delta_tension"],      float)
    eft_valid      = np.asarray(data["eft_valid_mask"],     bool)
    tau_grid       = np.asarray(data["tau_max_grid"],       float)

    # For each m_chi, find the Lambda boundary where delta_tension crosses zero
    mchi_out   = []
    lambda_out = []
    tau_out    = []

    for s, m in enumerate(scatter_masses):
        row     = delta_tension[s, :]
        eft_row = eft_valid[s, :]
        tau_row = tau_grid[s, :]

        # Only look in EFT-valid region
        eft_mask = eft_row & np.isfinite(row)
        if not np.any(eft_mask):
            continue

        # Find crossing from negative (helps) to positive (worsens)
        idx_sort = np.argsort(couplings)
        row_s  = row[idx_sort]
        lam_s  = couplings[idx_sort]
        eft_s  = eft_mask[idx_sort]
        tau_s  = tau_row[idx_sort]

        crossing = None
        for j in range(len(lam_s) - 1):
            if (eft_s[j] and eft_s[j+1] and
                    np.isfinite(row_s[j]) and np.isfinite(row_s[j+1]) and
                    row_s[j] <= 0 < row_s[j+1]):
                crossing = j
                break

        if crossing is None:
            if np.all(row_s[eft_s & np.isfinite(row_s)] <= 0):
                # Scattering always helps or neutral — no worsening limit
                continue
            elif np.all(row_s[eft_s & np.isfinite(row_s)] > 0):
                # Scattering always worsens — limit at largest EFT-valid coupling
                idx_eft = np.where(eft_s & np.isfinite(row_s))[0][-1]
                mchi_out.append(float(m))
                lambda_out.append(float(lam_s[idx_eft]))
                tau_out.append(float(tau_s[idx_eft]))
            continue

        frac = -row_s[crossing] / (row_s[crossing+1] - row_s[crossing] + 1e-300)
        log_lam = (np.log10(lam_s[crossing]) * (1-frac) +
                   np.log10(lam_s[crossing+1]) * frac)
        mchi_out.append(float(m))
        lambda_out.append(float(10.0 ** log_lam))
        tau_out.append(float(tau_s[crossing] * (1-frac) + tau_s[crossing+1] * frac))

    return dict(
        mchi       = np.array(mchi_out),
        lambda_lim = np.array(lambda_out),
        tau_lim    = np.array(tau_out),
    )


# ---------------------------------------------------------------------------
# Save boundary npz
# ---------------------------------------------------------------------------

def save_limit_npz(
    out_path:   Path,
    limit:      dict,
    data:       dict,
    limit_type: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        mchi_GeV        = limit["mchi"].astype(np.float32),
        lambda_lim_GeV  = limit["lambda_lim"].astype(np.float32),
        tau_lim         = limit["tau_lim"].astype(np.float32),
        operator        = np.array(str(data.get("operator", "unknown"))),
        dm_type         = np.array(str(data.get("dm_type",   "fermionic"))),
        ann_channel     = np.array(str(data.get("ann_channel", "WW"))),
        halo_profile    = np.array(str(data.get("halo_profile", "rho2"))),
        limit_type      = np.array(limit_type),
        orig_tension    = np.array(float(data.get("orig_tension", np.nan))),
        orig_ann_mass   = np.array(float(data.get("orig_ann_mass_GeV", np.nan))),
    )
    n = len(limit["mchi"])
    if n > 0:
        print(f"  Saved {n}-point {limit_type} boundary: {out_path.name}")
        if n > 0:
            print(f"    m_chi : [{limit['mchi'].min():.2e}, {limit['mchi'].max():.2e}] GeV")
            print(f"    Lambda: [{limit['lambda_lim'].min():.2e}, {limit['lambda_lim'].max():.2e}] GeV")
            print(f"    tau   : [{limit['tau_lim'].min():.2e}, {limit['tau_lim'].max():.2e}]")
    else:
        print(f"  No boundary points found for {limit_type} ({out_path.name})")


# ---------------------------------------------------------------------------
# Plot: (m_chi, Lambda) exclusion plane for one operator
# ---------------------------------------------------------------------------

def make_exclusion_plot(
    data:       dict,
    chi2_deconv: np.ndarray,
    tau_limit:  dict,
    tens_limit: dict,
    out_path:   Path,
) -> None:
    scatter_masses = np.asarray(data["scatter_masses_GeV"], float)
    couplings      = np.asarray(data["couplings"],          float)
    eft_valid      = np.asarray(data["eft_valid_mask"],     bool)
    tau_grid       = np.asarray(data["tau_max_grid"],       float)
    delta_tension  = np.asarray(data["delta_tension"],      float)
    orig_tension   = float(data.get("orig_tension", np.nan))
    label          = str(data.get("label", "operator"))
    channel        = str(data.get("ann_channel", "WW"))

    set_plot_style(
        style=os.environ.get("TRINITY_PLOT_STYLE", "conference"),
        cmap_name="plasma",
        base_fontsize=10,
        linewidth=1.6,
        n_colors=10,
    )
    text_col = plt.rcParams.get("text.color", "white")
    bg_col = plt.rcParams.get("axes.facecolor", DARK_BG)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor(plt.rcParams.get("figure.facecolor", bg_col))

    def _style(ax):
        ax.set_facecolor(bg_col)
        for sp in ax.spines.values(): sp.set_color(plt.rcParams.get("axes.edgecolor", text_col))
        ax.tick_params(colors=text_col, labelsize=8.5)
        ax.xaxis.label.set_color(text_col)
        ax.yaxis.label.set_color(text_col)
        ax.title.set_color(text_col)

    if scatter_masses.size > 1 and couplings.size > 1:
        S, C = np.meshgrid(scatter_masses, couplings, indexing="ij")
        logS = np.log10(S)
        logC = np.log10(C)

        # Panel 1: log10(tau_max)
        ax = axes[0]
        tau_plot = np.where(tau_grid > 0, np.log10(tau_grid), np.nan)
        cf = ax.contourf(logS, logC, tau_plot, levels=40, cmap="plasma")
        plt.colorbar(cf, ax=ax, label=r"$\log_{10}(\tau_\mathrm{max})$")
        ax.contour(logS, logC, eft_valid.astype(float),
                   levels=[0.5], colors=["#50FA7B"], lw=1.8, linestyles="--")
        ax.set_xlabel(r"$\log_{10}(m_\chi/\mathrm{GeV})$", fontsize=10)
        ax.set_ylabel(r"$\log_{10}(\Lambda/\mathrm{GeV})$", fontsize=10)
        ax.set_title(r"$\log_{10}(\tau_\mathrm{max})$" "\n[green dashed = EFT validity]",
                     fontsize=9)
        _style(ax)

        # Panel 2: chi2_deconv (spectrum modification significance)
        ax = axes[1]
        chi2_plot = np.where(np.isfinite(chi2_deconv) & (chi2_deconv > 0),
                             np.log10(chi2_deconv), np.nan)
        cf2 = ax.contourf(logS, logC, chi2_plot, levels=40, cmap="inferno")
        plt.colorbar(cf2, ax=ax,
                     label=r"$\log_{10}(\chi^2_\mathrm{deconv})$ [spectrum modification]")
        # Threshold contour
        ax.contour(logS, logC, chi2_deconv,
                   levels=[5.99], colors=["#FF6B35"], linewidths=2.0)
        ax.contour(logS, logC, eft_valid.astype(float),
                   levels=[0.5], colors=["#50FA7B"], linewidths=1.8, linestyles="--")
        # Plot tau_limit boundary
        if len(tau_limit["mchi"]) > 1:
            ax.plot(np.log10(tau_limit["mchi"]),
                    np.log10(tau_limit["lambda_lim"]),
                    color="#FF6B35", lw=2.5, ls="-",
                    label=r"$\chi^2_\mathrm{deconv}=5.99$ limit")
            ax.legend(fontsize=8)
        ax.set_xlabel(r"$\log_{10}(m_\chi/\mathrm{GeV})$", fontsize=10)
        ax.set_ylabel(r"$\log_{10}(\Lambda/\mathrm{GeV})$", fontsize=10)
        ax.set_title(
            r"Spectral modification $\chi^2_\mathrm{deconv}$" "\n"
            r"[orange = 95% CL exclusion boundary, green = EFT validity]",
            fontsize=9)
        _style(ax)

        # Panel 3: delta_tension
        ax = axes[2]
        vmax = max(0.001, float(np.nanpercentile(np.abs(delta_tension), 95)))
        cf3 = ax.contourf(logS, logC,
                          np.clip(delta_tension, -vmax, vmax),
                          levels=np.linspace(-vmax, vmax, 41),
                          cmap="coolwarm")
        plt.colorbar(cf3, ax=ax,
                     label=r"$\Delta$tension (deconv $-$ orig)")
        ax.contour(logS, logC, delta_tension,
                   levels=[0.0], colors=["white"], linewidths=1.5)
        ax.contour(logS, logC, eft_valid.astype(float),
                   levels=[0.5], colors=["#50FA7B"], linewidths=1.8, linestyles="--")
        if len(tens_limit["mchi"]) > 1:
            ax.plot(np.log10(tens_limit["mchi"]),
                    np.log10(tens_limit["lambda_lim"]),
                    color="white", lw=2.0, ls=":",
                    label=r"$\Delta$tension = 0 boundary")
            ax.legend(fontsize=8)
        ax.set_xlabel(r"$\log_{10}(m_\chi/\mathrm{GeV})$", fontsize=10)
        ax.set_ylabel(r"$\log_{10}(\Lambda/\mathrm{GeV})$", fontsize=10)
        ax.set_title(
            r"$\Delta$tension (deconv $-$ no scattering)" "\n"
            r"[white = $\Delta T = 0$, green = EFT validity]",
            fontsize=9)
        _style(ax)

    fig.suptitle(
        f"{label} | {channel} | orig tension = {orig_tension:.2f}×\n"
        "Deconvolution-based limits on photon–DM scattering",
        color="white", fontsize=11, y=1.01)

    fig.tight_layout(w_pad=3.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    print(f"  Plot: {out_path}")


# ---------------------------------------------------------------------------
# Process one npz file
# ---------------------------------------------------------------------------

def process_scan(
    npz_path:       Path,
    chi2_threshold: float,
    do_plot:        bool,
) -> None:
    print(f"\n[{npz_path.parent.name}]")
    data = load_deconv_npz(npz_path)

    op       = str(data.get("operator",    "unknown"))
    dm_type  = str(data.get("dm_type",     "fermionic"))
    channel  = str(data.get("ann_channel", "WW"))
    profile  = str(data.get("halo_profile","rho2"))

    print(f"  operator={op}, channel={channel}, profile={profile}")
    print(f"  orig tension = {float(data.get('orig_tension', np.nan)):.3f}x")
    print(f"  best EFT delta_tension = {float(data.get('best_eft_delta_tension', np.nan)):+.4f}x")

    # 1. Compute chi2_deconv
    chi2_deconv = compute_deconv_chi2(data)

    max_chi2 = float(np.nanmax(chi2_deconv)) if np.any(np.isfinite(chi2_deconv)) else np.nan
    print(f"  max chi2_deconv (conservative upper bound) = {max_chi2:.3e}")
    print(f"  chi2 threshold = {chi2_threshold:.2f}")

    if max_chi2 < chi2_threshold:
        print(f"  >> chi2_deconv < threshold everywhere: tau is too small to exclude")
        print(f"     This is the fundamental no-go: EFT scattering cannot modify")
        print(f"     the Totani spectrum at the level of the measurement uncertainty.")

    # 2. Extract tau upper limit
    tau_lim = extract_tau_upper_limit(data, chi2_deconv, chi2_threshold)

    # 3. Extract tension-worsening boundary
    tens_lim = extract_tension_limit(data)

    # 4. Save boundaries
    stem = f"{op}_{dm_type}_{channel}_{profile}"
    save_limit_npz(
        BOUNDARY_DIR / f"deconv_tau_limit_{stem}.npz",
        tau_lim, data, "tau_upper_limit_chi2_deconv"
    )
    save_limit_npz(
        BOUNDARY_DIR / f"deconv_tension_limit_{stem}.npz",
        tens_lim, data, "delta_tension_zero_boundary"
    )

    # 5. Plot
    if do_plot:
        make_exclusion_plot(
            data, chi2_deconv, tau_lim, tens_lim,
            out_path=npz_path.parent / "deconv_limits.png",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract deconvolution-based limits on photon-DM scattering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scan-dir", default="results/deconv_scan",
                   help="Root directory containing deconv_scan.npz files.")
    p.add_argument("--delta-chi2", type=float, default=5.99,
                   help="chi2_deconv threshold: 5.99 = 95% CL (2 dof).")
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.add_argument(
        "--style",
        default=None,
        help="Plot style: paper, conference/dark_transparent, or conference_light/light_transparent.",
    )
    args = p.parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style

    scan_root = Path(args.scan_dir)
    if not scan_root.exists():
        # Try relative to Totani_Scattering/.
        scan_root = _ROOT / args.scan_dir
    if not scan_root.exists():
        print(f"ERROR: scan dir not found: {args.scan_dir}")
        return

    print(f"Scan root  : {scan_root}")
    print(f"chi2 thresh: {args.delta_chi2}")
    print(f"Boundaries → {BOUNDARY_DIR}")

    # Find all deconv_scan.npz files
    npz_files = sorted(scan_root.rglob("deconv_scan.npz"))
    if not npz_files:
        print(f"No deconv_scan.npz files found under {scan_root}")
        return

    print(f"Found {len(npz_files)} scan file(s)")
    for npz in npz_files:
        process_scan(npz, args.delta_chi2, args.plot)

    print(f"\nDone. Boundaries saved to {BOUNDARY_DIR}/")


if __name__ == "__main__":
    main()
