#!/usr/bin/env python3
r"""
make_uv_baseline_tau_limits.py
==============================

Set photon-DM scattering constraints on UV completions WITHOUT using the Totani
halo morphology fit. Instead the constraint is the model-independent statement

    tau_max(E_gamma) >= tau_needed   for some E_gamma in the LAT band,

evaluated along a fixed, clean astrophysical baseline:

  * "gc"     : Galactic-centre NFW column  (J_gc ~ 1.05e22 GeV/cm^2)
  * "cosmo"  : cosmological line of sight   (J_cosmo ~ 1.37e22 GeV/cm^2)

i.e. exactly the logic of make_combined_fermion_scalar_tau_grid.py, but reduced
to a single exclusion *boundary* per operator (largest Lambda that still reaches
the threshold at each m_chi), then translated into the PHYSICAL parameters of
the UV completion via the matching maps derived in make_uv_completion_limits.py.

This is the "reach on a clean baseline" constraint: no MCMC, no halo posterior,
no spectral fit. The threshold tau ~ 1e-2 is the generic floor at which a ~1%
energy-dependent flux deficit becomes distinguishable from a smooth power law in
the LAT band (tau ~ 1e-3 for the projected CTA reach).

Two completions (Higgs portal handled separately):
  1. Charged-messenger loop   -> Dirac magnetic / electric dipole (dim-5)
  2. Kinetic-mixing dark U(1)' -> charge radius / anapole (dim-6)

For each, plots are produced in UV-physical axes:
  * M_mediator vs m_chi   (coupling fixed)
  * coupling  vs M_mediator  (m_chi fixed)

PLACEMENT
---------
    Totani_Scattering/constraint_generation/make_uv_baseline_tau_limits.py

Requires make_uv_completion_limits.py to be in the same folder (it imports the
matching maps from it). Run from Totani_Scattering/.

USAGE
-----
    python constraint_generation/make_uv_baseline_tau_limits.py \
        --theory all --baseline gc --tau-needed 1e-2 --include-electric --include-charge-radius

    python constraint_generation/make_uv_baseline_tau_limits.py \
        --theory dipole --baseline cosmo --tau-needed 1e-3
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_WORKSPACE = _ROOT.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_WORKSPACE))

from helpers.trinity_plotting import save_figure, set_paper_style, set_plot_style  # noqa: E402

# Reuse the cross-section / column machinery already validated in the repo.
from core.attenuation_eft import (  # noqa: E402
    sigma_tot_fermionic_array,
    compute_J_los,
    KPC_TO_CM,
)

# Reuse the UV matching maps and constraint-boundary rescaling helpers from the
# sibling script so the physics lives in exactly one place.
_UV_PATH = _HERE / "make_uv_completion_limits.py"
if not _UV_PATH.exists():
    raise FileNotFoundError(
        f"Expected make_uv_completion_limits.py next to this script at {_UV_PATH}. "
        "Place both files in constraint_generation/."
    )
_spec = importlib.util.spec_from_file_location("uv_completion_maps", _UV_PATH)
uv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uv)

PLOTDIR = _ROOT / "plots"

COL_FERMI_REACH = "#9B6DFF"
COL_CTA_REACH = "#7DDC84"
COL_CUSTOM_REACH = "black"
COL_VISIBLE_FILL = "#B18CFF"
COL_EXCLUDED_FILL = "#8B93A4"
COL_PANEL_BG = "#FFFFFF"

# ---------------------------------------------------------------------------
# Baseline column densities J = int rho dl  [GeV/cm^2]
# ---------------------------------------------------------------------------
# Cosmological uniform column from the paper (rho_chi * L):
#   rho_chi = 1.2e-6 GeV/cm^3, L = 1.14e28 cm  ->  J_cosmo = 1.37e22 GeV/cm^2
J_COSMO_GEV_CM2 = 1.37e22

# Galactic-centre column: integrate the Totani NFW profile along the b=0,l=0
# sightline. compute_J_los(power=1) returns int rho dl in GeV/cm^2 already.
def gc_column_density() -> float:
    """J_gc = int rho dl toward the Galactic centre [GeV/cm^2]."""
    # A small offset off the exact GC avoids the r->0 cusp; 0.5 deg is well
    # inside the LAT ROI and representative of the bright inner column.
    return float(compute_J_los(l_deg=0.5, b_deg=0.5, power=1, n_points=2000))


def baseline_column(name: str) -> tuple[float, str]:
    if name == "cosmo":
        return J_COSMO_GEV_CM2, r"cosmological column $J_{\rm cosmo}=1.37\times10^{22}\,{\rm GeV\,cm^{-2}}$"
    if name == "gc":
        J = gc_column_density()
        return J, rf"GC NFW column $J_{{\rm gc}}={J:.2e}\,{{\rm GeV\,cm^{{-2}}}}$"
    raise ValueError("baseline must be 'gc' or 'cosmo'")


# ---------------------------------------------------------------------------
# tau and the EFT-plane reach boundary
# ---------------------------------------------------------------------------

def tau_max_over_band(m_chi, Lambda, J, *, operator, c_s, c_p, E_band, n_theta=160):
    r"""tau_max = J/m_chi * max_E sigma_tot(E).

    sigma_tot_fermionic_array already returns cm^2; J is GeV/cm^2; m_chi GeV.
    """
    sig = sigma_tot_fermionic_array(
        E_band, float(m_chi), float(c_s), float(c_p), float(Lambda),
        operator=operator, majorana=False, n_theta=n_theta,
    )
    sig = np.nan_to_num(np.asarray(sig, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    sig_max = float(np.max(sig)) if sig.size else 0.0
    return (J / float(m_chi)) * sig_max


def reach_boundary_eft(m_chi_arr, *, operator, c_s, c_p, J, tau_needed,
                       E_band, log10_lambda_min=-3.0, log10_lambda_max=8.0,
                       n_lambda=220, n_theta=160):
    r"""For each m_chi, return the largest Lambda with tau_max >= tau_needed.

    Since sigma ~ Lambda^{-2n} (n = lambda_power/2), tau decreases monotonically
    with Lambda at fixed m_chi, so the crossing tau_max(Lambda) = tau_needed is a
    single upper boundary. We bracket it on a log grid and refine by log-linear
    interpolation in tau.
    """
    m_chi_arr = np.asarray(m_chi_arr, dtype=float)
    lam_grid = np.logspace(log10_lambda_min, log10_lambda_max, n_lambda)

    mchi_out, lam_out = [], []
    for m_chi in m_chi_arr:
        taus = np.array([
            tau_max_over_band(m_chi, lam, J, operator=operator,
                              c_s=c_s, c_p=c_p, E_band=E_band, n_theta=n_theta)
            for lam in lam_grid
        ])
        good = np.isfinite(taus) & (taus > 0)
        if not np.any(good):
            continue
        meets = good & (taus >= tau_needed)
        if not np.any(meets):
            continue
        idx_last = int(np.max(np.where(meets)[0]))
        if idx_last >= len(lam_grid) - 1:
            mchi_out.append(m_chi); lam_out.append(lam_grid[idx_last]); continue
        # log-linear interpolation between idx_last and idx_last+1
        x0, x1 = np.log10(lam_grid[idx_last]), np.log10(lam_grid[idx_last + 1])
        t0, t1 = taus[idx_last], taus[idx_last + 1]
        if t0 <= 0 or t1 <= 0 or t0 == t1:
            mchi_out.append(m_chi); lam_out.append(lam_grid[idx_last]); continue
        yT = np.log10(tau_needed)
        xT = x0 + (x1 - x0) * (yT - np.log10(t0)) / (np.log10(t1) - np.log10(t0))
        mchi_out.append(m_chi); lam_out.append(10.0 ** xT)

    return np.asarray(mchi_out, dtype=float), np.asarray(lam_out, dtype=float)


# ---------------------------------------------------------------------------
# Operator -> (c_s, c_p) benchmark, matching the repo convention
# ---------------------------------------------------------------------------
def operator_couplings(operator: str) -> tuple[float, float]:
    if operator in ("dipole_magnetic", "charge_radius"):
        return 1.0, 0.0
    if operator in ("dipole_electric", "anapole"):
        return 0.0, 1.0
    raise ValueError(f"Unsupported operator: {operator}")


# ===========================================================================
# Plotting (UV-physical axes), reusing uv.* maps and saved constraints
# ===========================================================================

def _reach_style(tau_needed: float) -> tuple[str, str]:
    tau_needed = float(tau_needed)
    if np.isclose(tau_needed, 1e-2, rtol=0.05):
        return COL_FERMI_REACH, r"Fermi-LAT visibility reach ($\tau_{\max}=10^{-2}$)"
    if np.isclose(tau_needed, 1e-3, rtol=0.05):
        return COL_CTA_REACH, r"CTA visibility reach ($\tau_{\max}=10^{-3}$)"
    return COL_CUSTOM_REACH, rf"Visibility reach ($\tau_{{\max}}={tau_needed:g}$)"


def _format_panel(ax):
    ax.set_facecolor(COL_PANEL_BG)
    ax.grid(True, alpha=0.25, ls=":", color="#AAAAAA", which="both")
    ax.tick_params(labelsize=10)


def _add_caption(ax, text):
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        color="black",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#AAAAAA", "pad": 3.0},
        zorder=8,
    )


def _dedupe_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    out_handles, out_labels, seen = [], [], set()
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in seen:
            continue
        seen.add(label)
        out_handles.append(handle)
        out_labels.append(label)
    ax.legend(out_handles, out_labels, fontsize=8.2, loc="best", framealpha=0.86)


def _positive_limits(*arrays, pad=0.25):
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        good = np.isfinite(arr) & (arr > 0.0)
        if np.any(good):
            vals.append(arr[good])
    if not vals:
        return 1e-3, 1e3
    vals = np.concatenate(vals)
    lo = 10.0 ** (np.floor(np.log10(np.nanmin(vals))) - pad)
    hi = 10.0 ** (np.ceil(np.log10(np.nanmax(vals))) + pad)
    return lo, hi


def _positive_x_limits(*arrays, pad=0.10):
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        good = np.isfinite(arr) & (arr > 0.0)
        if np.any(good):
            vals.append(arr[good])
    if not vals:
        return None
    vals = np.concatenate(vals)
    lo = 10.0 ** (np.log10(np.nanmin(vals)) - pad)
    hi = 10.0 ** (np.log10(np.nanmax(vals)) + pad)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= lo:
        return None
    return lo, hi


def _apply_curve_limits(ax, *, x_arrays=(), y_arrays=(), include_y=None):
    xlim = _positive_x_limits(*x_arrays)
    if xlim is not None:
        ax.set_xlim(*xlim)
    y_inputs = list(y_arrays)
    if include_y is not None:
        y_inputs.append(np.asarray(include_y, dtype=float))
    if y_inputs:
        ylo, yhi = _positive_limits(*y_inputs, pad=0.20)
        ax.set_ylim(ylo, yhi)


def _constraint_mmed_curves(curves, transform):
    out = []
    for curve in curves:
        x = np.asarray(curve["mchi_GeV"], dtype=float)
        y = transform(curve["lambda_GeV"], x)
        good = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        if np.count_nonzero(good) < 2:
            continue
        out.append((x[good], y[good], curve))
    return out


def _constraint_coupling_curves(curves, transform, x_grid, *, mchi_benchmark):
    out = []
    for curve in curves:
        lam_at_mb = uv._interp_log_boundary(curve, mchi_benchmark)
        if lam_at_mb is None:
            continue
        y = transform(lam_at_mb, x_grid)
        good = np.isfinite(x_grid) & np.isfinite(y) & (x_grid > 0.0) & (y > 0.0)
        if np.count_nonzero(good) < 2:
            continue
        out.append((x_grid[good], y[good], curve))
    return out


def _draw_existing_constraint_regions(ax, plotted_curves, *, direction):
    if not plotted_curves:
        return
    ymin, ymax = ax.get_ylim()
    first_fill = True
    for x, y, _ in plotted_curves:
        if direction == "below":
            ax.fill_between(
                x, ymin, y,
                facecolor=COL_EXCLUDED_FILL,
                alpha=0.14,
                hatch="////",
                edgecolor=COL_EXCLUDED_FILL,
                linewidth=0.0,
                label="Already excluded by saved constraints" if first_fill else None,
                zorder=3,
            )
        else:
            ax.fill_between(
                x, y, ymax,
                facecolor=COL_EXCLUDED_FILL,
                alpha=0.14,
                hatch="////",
                edgecolor=COL_EXCLUDED_FILL,
                linewidth=0.0,
                label="Already excluded by saved constraints" if first_fill else None,
                zorder=3,
            )
        first_fill = False


def _draw_constraint_lines(ax, plotted_curves, *, max_labels=10):
    used = set()
    for idx, (x, y, curve) in enumerate(plotted_curves):
        label = str(curve["label"])
        if label in used:
            label = curve["name"].replace("_", " ")
        used.add(label)
        ax.plot(
            x,
            y,
            color=curve["color"],
            ls=curve["linestyle"],
            lw=1.7,
            alpha=0.96,
            label=label if idx < max_labels else None,
            zorder=4,
        )


def _draw_visibility_region(ax, x, y, *, tau_needed, direction):
    color, label = _reach_style(tau_needed)
    ymin, ymax = ax.get_ylim()
    if direction == "below":
        ax.fill_between(
            x, ymin, y,
            color=COL_VISIBLE_FILL,
            alpha=0.16,
            label=r"Optical-depth visible region ($\tau_{\max}\geq\tau_{\rm req}$)",
            zorder=2,
        )
    else:
        ax.fill_between(
            x, y, ymax,
            color=COL_VISIBLE_FILL,
            alpha=0.16,
            label=r"Optical-depth visible region ($\tau_{\max}\geq\tau_{\rm req}$)",
            zorder=2,
        )
    ax.plot(x, y, color=color, lw=3.0, label=label, zorder=6)


def plot_dipole_uv(operator, *, mchi_eft, lam_eft, lam_benchmark, mchi_benchmark,
                   x_ratio, Q, baseline_label, tau_needed, outtag,
                   constraint_halo_profile, constraint_model_kind):
    is_electric = operator == "dipole_electric"
    op_title = "Electric dipole" if is_electric else "Magnetic dipole"
    cap = (
        rf"{baseline_label}"
        "\n"
        rf"criterion: $\max_E\tau(E)\geq {tau_needed:g}$"
    )

    order = np.argsort(mchi_eft)
    mchi_eft, lam_eft = mchi_eft[order], lam_eft[order]
    constraint_boundaries = uv.load_operator_boundaries(
        operator,
        halo_profile=constraint_halo_profile,
        model_kind=constraint_model_kind,
    )

    # ---- Panel A: M_F vs m_chi at fixed lambda ----
    M_F = uv.uv_dipole_from_lambda(lam_eft, mchi_eft, x_ratio=x_ratio, Q=Q,
                                   solve_for="M_F", lam=lam_benchmark)
    good = np.isfinite(M_F) & (M_F > 0)
    constraint_curves = _constraint_mmed_curves(
        constraint_boundaries,
        lambda Lambda, mchi: uv.uv_dipole_from_lambda(
            Lambda, mchi, x_ratio=x_ratio, Q=Q, solve_for="M_F", lam=lam_benchmark
        ),
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    uv._setup_axes(ax, r"$m_\chi$ [GeV]", r"$M_F$ [GeV]",
                   f"{op_title} UV completion: charged messenger\n"
                   rf"$\lambda={lam_benchmark:g}$, $x={x_ratio:g}$, $Q={Q:g}$")
    _format_panel(ax)
    _apply_curve_limits(
        ax,
        x_arrays=[mchi_eft[good], *[x for x, _, _ in constraint_curves]],
        y_arrays=[M_F[good], *[y for _, y, _ in constraint_curves]],
    )
    if np.any(good):
        _draw_visibility_region(ax, mchi_eft[good], M_F[good], tau_needed=tau_needed, direction="below")
    _draw_existing_constraint_regions(ax, constraint_curves, direction="below")
    _draw_constraint_lines(ax, constraint_curves)
    _add_caption(ax, cap)
    _dedupe_legend(ax)
    out_a = PLOTDIR / f"uv_baseline_{outtag}_Mmed_vs_mchi"
    save_figure(fig, str(out_a)); plt.close(fig)
    print(f"  saved {out_a}.png/.pdf")

    # ---- Panel B: lambda vs M_F at fixed m_chi ----
    if len(mchi_eft) >= 2:
        lam_at_mb = 10.0 ** float(np.interp(np.log10(mchi_benchmark),
                                            np.log10(mchi_eft), np.log10(lam_eft)))
    else:
        lam_at_mb = float(lam_eft[0]) if len(lam_eft) else np.nan
    M_F_grid = np.logspace(1, 5, 400)
    lam_coupling = uv.uv_dipole_from_lambda(lam_at_mb, mchi_benchmark, x_ratio=x_ratio,
                                            Q=Q, solve_for="lam", M_F=M_F_grid)
    good = np.isfinite(lam_coupling) & (lam_coupling > 0)
    constraint_curves = _constraint_coupling_curves(
        constraint_boundaries,
        lambda Lambda, M_F: uv.uv_dipole_from_lambda(
            Lambda, mchi_benchmark, x_ratio=x_ratio, Q=Q, solve_for="lam", M_F=M_F
        ),
        M_F_grid,
        mchi_benchmark=mchi_benchmark,
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    uv._setup_axes(ax, r"$M_F$ [GeV]", r"Yukawa coupling $\lambda$",
                   f"{op_title} UV completion: charged messenger\n"
                   rf"$m_\chi={mchi_benchmark:g}$ GeV, $x={x_ratio:g}$, $Q={Q:g}$")
    _format_panel(ax)
    _apply_curve_limits(
        ax,
        x_arrays=[M_F_grid[good], *[x for x, _, _ in constraint_curves]],
        y_arrays=[lam_coupling[good], *[y for _, y, _ in constraint_curves]],
        include_y=[np.sqrt(4 * np.pi)],
    )
    if np.any(good):
        _draw_visibility_region(
            ax, M_F_grid[good], lam_coupling[good], tau_needed=tau_needed, direction="above"
        )
    _draw_existing_constraint_regions(ax, constraint_curves, direction="above")
    _draw_constraint_lines(ax, constraint_curves)
    ax.axhline(np.sqrt(4 * np.pi), color=uv.COL_PERTURB, lw=1.8, ls=":",
               label=r"Perturbativity $\lambda=\sqrt{4\pi}$")
    _add_caption(ax, cap)
    _dedupe_legend(ax)
    out_b = PLOTDIR / f"uv_baseline_{outtag}_g_vs_Mmed"
    save_figure(fig, str(out_b)); plt.close(fig)
    print(f"  saved {out_b}.png/.pdf")


def plot_darkphoton_uv(operator, *, mchi_eft, lam_eft, gD_benchmark, eps_benchmark,
                       mchi_benchmark, baseline_label, tau_needed, outtag,
                       constraint_halo_profile, constraint_model_kind):
    op_title = "Anapole" if operator == "anapole" else "Charge radius"
    cap = (
        rf"{baseline_label}"
        "\n"
        rf"criterion: $\max_E\tau(E)\geq {tau_needed:g}$"
    )

    order = np.argsort(mchi_eft)
    mchi_eft, lam_eft = mchi_eft[order], lam_eft[order]
    constraint_boundaries = uv.load_operator_boundaries(
        operator,
        halo_profile=constraint_halo_profile,
        model_kind=constraint_model_kind,
    )

    # ---- Panel A: m_A' vs m_chi at fixed (eps, g_D) ----
    m_Ap = uv.uv_darkphoton_from_lambda(lam_eft, solve_for="m_Ap",
                                        eps=eps_benchmark, g_D=gD_benchmark)
    good = np.isfinite(m_Ap) & (m_Ap > 0)
    constraint_curves = _constraint_mmed_curves(
        constraint_boundaries,
        lambda Lambda, _mchi: uv.uv_darkphoton_from_lambda(
            Lambda, solve_for="m_Ap", eps=eps_benchmark, g_D=gD_benchmark
        ),
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    uv._setup_axes(ax, r"$m_\chi$ [GeV]", r"$m_{A'}$ [GeV]",
                   f"{op_title} UV completion: kinetic-mixing dark $U(1)'$\n"
                   rf"$g_D={gD_benchmark:g}$, $\epsilon={eps_benchmark:g}$")
    _format_panel(ax)
    _apply_curve_limits(
        ax,
        x_arrays=[mchi_eft[good], *[x for x, _, _ in constraint_curves]],
        y_arrays=[m_Ap[good], *[y for _, y, _ in constraint_curves]],
    )
    if np.any(good):
        _draw_visibility_region(ax, mchi_eft[good], m_Ap[good], tau_needed=tau_needed, direction="below")
    _draw_existing_constraint_regions(ax, constraint_curves, direction="below")
    _draw_constraint_lines(ax, constraint_curves)
    _add_caption(ax, cap)
    _dedupe_legend(ax)
    out_a = PLOTDIR / f"uv_baseline_{outtag}_Mmed_vs_mchi"
    save_figure(fig, str(out_a)); plt.close(fig)
    print(f"  saved {out_a}.png/.pdf")

    # ---- Panel B: g_D vs m_A' at fixed eps and m_chi ----
    if len(mchi_eft) >= 2:
        lam_at_mb = 10.0 ** float(np.interp(np.log10(mchi_benchmark),
                                            np.log10(mchi_eft), np.log10(lam_eft)))
    else:
        lam_at_mb = float(lam_eft[0]) if len(lam_eft) else np.nan
    m_Ap_grid = np.logspace(-2, 4, 400)
    gD = uv.uv_darkphoton_from_lambda(lam_at_mb, solve_for="g_D",
                                      eps=eps_benchmark, m_Ap=m_Ap_grid)
    good = np.isfinite(gD) & (gD > 0)
    constraint_curves = _constraint_coupling_curves(
        constraint_boundaries,
        lambda Lambda, m_Ap: uv.uv_darkphoton_from_lambda(
            Lambda, solve_for="g_D", eps=eps_benchmark, m_Ap=m_Ap
        ),
        m_Ap_grid,
        mchi_benchmark=mchi_benchmark,
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    uv._setup_axes(ax, r"$m_{A'}$ [GeV]", r"dark gauge coupling $g_D$",
                   f"{op_title} UV completion: kinetic-mixing dark $U(1)'$\n"
                   rf"$\epsilon={eps_benchmark:g}$, $m_\chi={mchi_benchmark:g}$ GeV")
    _format_panel(ax)
    _apply_curve_limits(
        ax,
        x_arrays=[m_Ap_grid[good], *[x for x, _, _ in constraint_curves]],
        y_arrays=[gD[good], *[y for _, y, _ in constraint_curves]],
        include_y=[np.sqrt(4 * np.pi)],
    )
    if np.any(good):
        _draw_visibility_region(ax, m_Ap_grid[good], gD[good], tau_needed=tau_needed, direction="above")
    _draw_existing_constraint_regions(ax, constraint_curves, direction="above")
    _draw_constraint_lines(ax, constraint_curves)
    ax.axhline(np.sqrt(4 * np.pi), color=uv.COL_PERTURB, lw=1.8, ls=":",
               label=r"Perturbativity $g_D=\sqrt{4\pi}$")
    _add_caption(ax, cap)
    _dedupe_legend(ax)
    out_b = PLOTDIR / f"uv_baseline_{outtag}_g_vs_Mmed"
    save_figure(fig, str(out_b)); plt.close(fig)
    print(f"  saved {out_b}.png/.pdf")


# ===========================================================================
# Driver
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--theory", default="all", choices=["dipole", "darkphoton", "all"])
    p.add_argument("--baseline", default="gc", choices=["gc", "cosmo"],
                   help="Clean astrophysical column for the optical depth.")
    p.add_argument("--constraint-halo-profile", default="rho2",
                   help="Halo-profile tag used when selecting saved mcmc_* constraints to overlay.")
    p.add_argument("--constraint-model-kind", default="raw_attenuation",
                   choices=["raw_attenuation", "spectral_reshaping"],
                   help="Model-kind tag used when selecting saved mcmc_* constraints to overlay.")
    p.add_argument("--tau-needed", type=float, default=1e-2,
                   help="Observability threshold (1e-2 ~ Fermi-LAT, 1e-3 ~ CTA).")
    p.add_argument("--e-min", type=float, default=50.6, help="LAT band lower edge [GeV].")
    p.add_argument("--e-max", type=float, default=494.0, help="LAT band upper edge [GeV].")
    p.add_argument("--n-e", type=int, default=24, help="Energy points across the band.")
    p.add_argument("--n-theta", type=int, default=160)

    p.add_argument("--mchi-min", type=float, default=1e-6)
    p.add_argument("--mchi-max", type=float, default=1e8)
    p.add_argument("--n-mchi", type=int, default=40)
    p.add_argument("--log10-lambda-min", type=float, default=-3.0)
    p.add_argument("--log10-lambda-max", type=float, default=8.0)
    p.add_argument("--n-lambda", type=int, default=220)

    # dipole benchmarks
    p.add_argument("--dipole-lambda", type=float, default=1.0)
    p.add_argument("--dipole-mchi", type=float, default=100.0)
    p.add_argument("--dipole-x", type=float, default=1.0)
    p.add_argument("--dipole-Q", type=float, default=1.0)
    p.add_argument("--include-electric", action="store_true")

    # dark-photon benchmarks
    p.add_argument("--dp-gD", type=float, default=1.0)
    p.add_argument("--dp-eps", type=float, default=1e-3)
    p.add_argument("--dp-mchi", type=float, default=100.0)
    p.add_argument("--include-charge-radius", action="store_true")

    p.add_argument("--style", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style
    style = args.style or "paper"
    if str(style).lower() == "paper":
        set_paper_style(base_fontsize=10, linewidth=1.6, n_colors=14, cmap_name="plasma")
    else:
        set_plot_style(style=style, cmap_name="plasma", base_fontsize=12, linewidth=1.8)
    PLOTDIR.mkdir(parents=True, exist_ok=True)

    J, baseline_label = baseline_column(args.baseline)
    E_band = np.logspace(np.log10(args.e_min), np.log10(args.e_max), int(args.n_e))
    m_chi_arr = np.logspace(np.log10(args.mchi_min), np.log10(args.mchi_max), int(args.n_mchi))

    print(f"Baseline: {baseline_label}")
    print(f"  J = {J:.3e} GeV/cm^2 ;  tau_needed = {args.tau_needed:g} ;  "
          f"E in [{args.e_min:g},{args.e_max:g}] GeV")

    do_dipole = args.theory in ("dipole", "all")
    do_dp = args.theory in ("darkphoton", "all")

    def boundary(op):
        cs, cp = operator_couplings(op)
        return reach_boundary_eft(
            m_chi_arr, operator=op, c_s=cs, c_p=cp, J=J, tau_needed=args.tau_needed,
            E_band=E_band, log10_lambda_min=args.log10_lambda_min,
            log10_lambda_max=args.log10_lambda_max, n_lambda=args.n_lambda,
            n_theta=args.n_theta,
        )

    if do_dipole:
        print("Charged-messenger -> magnetic dipole")
        mchi_b, lam_b = boundary("dipole_magnetic")
        if len(mchi_b):
            plot_dipole_uv("dipole_magnetic", mchi_eft=mchi_b, lam_eft=lam_b,
                           lam_benchmark=args.dipole_lambda, mchi_benchmark=args.dipole_mchi,
                           x_ratio=args.dipole_x, Q=args.dipole_Q,
                           baseline_label=baseline_label, tau_needed=args.tau_needed,
                           outtag="dipole_magnetic",
                           constraint_halo_profile=args.constraint_halo_profile,
                           constraint_model_kind=args.constraint_model_kind)
        else:
            print("  [WARN] no points reach threshold for magnetic dipole")
        if args.include_electric:
            print("Charged-messenger -> electric dipole")
            mchi_b, lam_b = boundary("dipole_electric")
            if len(mchi_b):
                plot_dipole_uv("dipole_electric", mchi_eft=mchi_b, lam_eft=lam_b,
                               lam_benchmark=args.dipole_lambda, mchi_benchmark=args.dipole_mchi,
                               x_ratio=args.dipole_x, Q=args.dipole_Q,
                               baseline_label=baseline_label, tau_needed=args.tau_needed,
                               outtag="dipole_electric",
                               constraint_halo_profile=args.constraint_halo_profile,
                               constraint_model_kind=args.constraint_model_kind)
            else:
                print("  [WARN] no points reach threshold for electric dipole")

    if do_dp:
        print("Kinetic-mixing dark photon -> anapole")
        mchi_b, lam_b = boundary("anapole")
        if len(mchi_b):
            plot_darkphoton_uv("anapole", mchi_eft=mchi_b, lam_eft=lam_b,
                               gD_benchmark=args.dp_gD, eps_benchmark=args.dp_eps,
                               mchi_benchmark=args.dp_mchi, baseline_label=baseline_label,
                               tau_needed=args.tau_needed, outtag="darkphoton_anapole",
                               constraint_halo_profile=args.constraint_halo_profile,
                               constraint_model_kind=args.constraint_model_kind)
        else:
            print("  [WARN] no points reach threshold for anapole")
        if args.include_charge_radius:
            print("Kinetic-mixing dark photon -> charge radius")
            mchi_b, lam_b = boundary("charge_radius")
            if len(mchi_b):
                plot_darkphoton_uv("charge_radius", mchi_eft=mchi_b, lam_eft=lam_b,
                                   gD_benchmark=args.dp_gD, eps_benchmark=args.dp_eps,
                                   mchi_benchmark=args.dp_mchi, baseline_label=baseline_label,
                                   tau_needed=args.tau_needed, outtag="darkphoton_charge_radius",
                                   constraint_halo_profile=args.constraint_halo_profile,
                                   constraint_model_kind=args.constraint_model_kind)
            else:
                print("  [WARN] no points reach threshold for charge radius")

    print("\nDone. Plots in", PLOTDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
