import argparse
import importlib.util
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from helpers.fermi_plotting import (
    CONF_ANNOT_FS,
    CONF_HEADER_FS,
    CONF_LABEL_FS,
    CONF_LEGEND_FS,
    CONF_TICK_FS,
    CONF_TITLE_FS,
    add_hatched_region_from_contour,
    latex_sci,
    operator_title,
)
from helpers.trinity_plotting import save_figure, set_plot_style, set_paper_style


REPO_DIR = Path(__file__).resolve().parent.parent
_SPECTRUM_BUNDLED = Path(__file__).resolve().parent / "data" / "fermi_halo_spectrum.txt"
DEFAULT_SPECTRUM = (
    _SPECTRUM_BUNDLED if _SPECTRUM_BUNDLED.exists()
    else REPO_DIR / "fermi_data" / "york" / "processed" / "spectrum_data.txt"
)

FERMIONIC_SCRIPT = REPO_DIR / "Fermionic_DM_Eff_Operator" / "Fermi-LAT_analysis_eff_coupling_fermionic.py"
SCALAR_SCRIPT = REPO_DIR / "Scalar_DM_Eff_Operator" / "Fermi-LAT_analysis_eff_coupling_scalar.py"

OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

TOP_ROW_PANELS = [
    ("rayleigh_full", "dirac", "Rayleigh Full"),
    ("charge_radius", "dirac", "Charge Radius"),
    ("anapole", "dirac", "Anapole"),
    ("dipole_electric", "dirac", "EL Dipole"),
    ("dipole_magnetic", "dirac", "MA Dipole"),
]

BOTTOM_ROW_PANELS = [
    ("rayleigh_full", "majorana", "Rayleigh Full"),
    ("charge_radius", "majorana", "Charge Radius"),
    ("anapole", "majorana", "Anapole"),
    ("rayleigh_scalar", "scalar", "Scalar Rayleigh"),
    (None, None, None),
]


def compute_panelc_j_gal():
    kpc_to_cm = 3.0857e21
    rho_s = 0.184
    r_s_nfw = 24.42
    r_sun = 8.5
    l_los = np.concatenate(
        [
            np.linspace(0.001, 0.5, 20_000),
            np.linspace(0.5, r_sun, 20_000),
        ]
    )
    r_los = np.sqrt(l_los**2 + r_sun**2 - 2.0 * l_los * r_sun)
    r_los = np.maximum(r_los, 0.001)
    rho_los = rho_s / ((r_los / r_s_nfw) * (1.0 + r_los / r_s_nfw) ** 2)
    return float(np.trapezoid(rho_los, l_los * kpc_to_cm))


def unitarity_lambda_curve(operator, mchi_grid):
    mchi_grid = np.asarray(mchi_grid, dtype=float)
    xgrid = np.where(mchi_grid > 0.0, mchi_grid, np.nan)
    if operator in ("dipole_magnetic", "dipole_electric"):
        return np.sqrt(16.0 * np.pi * xgrid), "Unitarity (dipole)"
    if operator in ("charge_radius", "anapole"):
        return (16.0 * np.pi * xgrid**2) ** 0.25, "Unitarity (dim-6)"
    if "rayleigh" in str(operator):
        return (128.0 * np.pi**2 * xgrid**2) ** (1.0 / 6.0), "Unitarity (Rayleigh)"
    return None, None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_baseline(args):
    baseline = args.baseline
    if baseline == "gc":
        return {
            "label": baseline,
            "components": [(baseline, float(args.fermionic.rho_chi_gc), float(args.fermionic.L_gc))],
        }
    if baseline == "cosmic":
        return {
            "label": baseline,
            "components": [(baseline, float(args.fermionic.rho_chi_cosmic), float(args.fermionic.L_cosmic))],
        }
    if baseline == "gc_nfw":
        return {
            "label": baseline,
            "components": [("gc_nfw", 1.0, compute_panelc_j_gal())],
        }
    if baseline == "panelc":
        return {
            "label": baseline,
            "components": [
                ("gc_nfw", 1.0, compute_panelc_j_gal()),
                ("cosmic", float(args.fermionic.rho_chi_cosmic), float(args.fermionic.L_cosmic)),
            ],
        }
    if args.rho_chi is None or args.L_cm is None:
        raise ValueError("For --baseline custom you must provide both --rho-chi and --L-cm.")
    return {
        "label": baseline,
        "components": [(baseline, float(args.rho_chi), float(args.L_cm))],
    }


def build_energy_grid(args, energies):
    tau_mode = str(args.tau_energy_mode)
    if tau_mode == "dip":
        e_eval = np.asarray([float(args.dip_energy)], dtype=float)
        label = rf"$E={float(args.dip_energy):g}\,\mathrm{{GeV}}$"
        omega_max = max(float(np.max(energies)), float(args.dip_energy))
        return e_eval, label, omega_max

    if tau_mode not in {"band", "anywhere"}:
        raise ValueError(f"Unknown tau_energy_mode={tau_mode!r}")

    emin = float(args.tau_energy_min) if args.tau_energy_min is not None else float(np.min(energies))
    emax = float(args.tau_energy_max) if args.tau_energy_max is not None else float(np.max(energies))
    if emin <= 0.0 or emax <= 0.0 or emin >= emax:
        raise ValueError(f"Invalid tau energy band: Emin={emin}, Emax={emax} (need 0 < Emin < Emax).")

    n_energy = int(max(2, args.tau_energy_n))
    e_eval = np.logspace(np.log10(emin), np.log10(emax), n_energy)
    dip_pct = 100.0 * float(args.dip_depth)
    label = (
        f"{dip_pct:g}% dip anywhere"
        "\n"
        rf"in $E\in[{emin:g},{emax:g}]\,\mathrm{{GeV}}$"
    )
    return e_eval, label, float(emax)


def _compute_single_grid(panel, args, e_eval, omega_max_for_validity, rho_chi, l_cm):
    operator, dm_kind, _ = panel
    if operator is None:
        return None

    common = dict(
        E_eval=np.asarray(e_eval, dtype=float),
        rho_chi=float(rho_chi),
        L_cm=float(l_cm),
        omega_max_for_validity=float(omega_max_for_validity),
        eft_kinematic_factor=float(args.eft_kinematic_factor),
        log10_Lambda_min=float(args.log10_Lambda_min),
        log10_Lambda_max=float(args.log10_Lambda_max),
        log10_mchi_min=float(args.log10_mchi_min),
        log10_mchi_max=float(args.log10_mchi_max),
        n_Lambda=int(args.tau_grid_n_lambda),
        n_mchi=int(args.tau_grid_n_mchi),
    )

    if dm_kind == "scalar":
        return args.scalar.compute_max_tau_grid(**common)

    return args.fermionic.compute_max_tau_grid(
        **common,
        operator=str(operator),
        fermion_type=str(dm_kind),
    )


def compute_panel_grid(panel, args, e_eval, omega_max_for_validity, baseline_spec):
    operator, _, _ = panel
    if operator is None:
        return None

    component_grids = []
    for _, rho_chi, l_cm in baseline_spec["components"]:
        component_grids.append(
            _compute_single_grid(panel, args, e_eval, omega_max_for_validity, rho_chi, l_cm)
        )

    if len(component_grids) == 1:
        return component_grids[0]

    merged = dict(component_grids[0])
    tau_stack = np.stack([np.asarray(grid["tau_grid"], dtype=float) for grid in component_grids], axis=0)
    merged["tau_grid"] = np.max(tau_stack, axis=0)
    merged["eft_valid_grid"] = np.asarray(component_grids[0]["eft_valid_grid"], dtype=bool)
    return merged


def build_meta_text(args, baseline_spec, tau_needed, tau_energy_label):
    lines = []
    for label, rho_chi, l_cm in baseline_spec["components"]:
        if label == "gc_nfw":
            lines.append(rf"$J_{{\rm gal,NFW}}={latex_sci(l_cm)}\ \mathrm{{GeV/cm^2}}$")
        else:
            lines.append(rf"{label}: $\rho_\chi={latex_sci(rho_chi)}\ \mathrm{{GeV/cm^3}}$")
            lines.append(rf"{label}: $L={latex_sci(l_cm)}\ \mathrm{{cm}}$")
    if len(baseline_spec["components"]) > 1:
        lines.append(r"$\tau=\max(\tau_i)$ over listed baselines")
    lines.extend(
        [
            rf"{tau_energy_label}",
            rf"validity: $\Lambda^2 \geq {float(args.eft_kinematic_factor):g}\,\max(s_\mathrm{{max}}, |t|_\mathrm{{max}})$,",
            "unitarity",
            r"benchmark couplings: $c_s=c_p=1$",
            r"or $c_\phi=1$"
        ]
    )
    return "\n".join(lines)


def draw_panel(
    ax,
    grid,
    panel,
    row_idx,
    col_idx,
    show_ylabel,
    show_xlabel,
    vmin,
    vmax,
    tau_needed,
    tau_energy_label,
    plot_log10_lambda_min,
    plot_log10_lambda_max,
):
    operator, dm_kind, _ = panel
    lambda_grid = np.asarray(grid["Lambda_grid"], dtype=float)
    mchi_grid = np.asarray(grid["mchi_grid"], dtype=float)
    tau_grid = np.asarray(grid["tau_grid"], dtype=float)
    eft_valid_grid = np.asarray(grid["eft_valid_grid"], dtype=bool)
    x = np.log10(lambda_grid)
    y = np.log10(mchi_grid)
    X, Y = np.meshgrid(x, y, indexing="xy")

    log10_tau = np.log10(np.asarray(tau_grid, dtype=float) + 1e-30)

    im = ax.pcolormesh(
        x,
        y,
        log10_tau.T,
        cmap="plasma",
        vmin=float(vmin),
        vmax=float(vmax),
        shading="auto",
    )

    eft_invalid = (~eft_valid_grid.T).astype(float)
    if np.any(eft_invalid > 0.0):
        ax.contourf(
            x,
            y,
            eft_invalid,
            levels=[0.5, 1.5],
            colors=["gray"],
            alpha=0.30,
            zorder=2,
        )

    ax.contour(
        x,
        y,
        eft_valid_grid.T.astype(float),
        levels=[0.5],
        colors="r",
        linewidths=2.5,
        linestyles="--",
        zorder=4,
    )

    tau_field_all = np.asarray(tau_grid, dtype=float).T
    tau_finite = np.isfinite(tau_field_all)

    tau_fermi = 1e-2
    tau_cta = 1e-3
    tau_ge_fermi = tau_finite & (tau_field_all >= float(tau_fermi))
    if np.any(tau_ge_fermi):
        ax.contour(X, Y, tau_ge_fermi.astype(float), levels=[0.5], colors=["cyan"], linewidths=3.0, zorder=6)

    tau_ge_cta = tau_finite & (tau_field_all >= float(tau_cta))
    if np.any(tau_ge_cta):
        ax.contour(X, Y, tau_ge_cta.astype(float), levels=[0.5], colors=["lime"], linewidths=3.0, zorder=5)

    if float(tau_needed) > 0.0:
        tau_field = np.asarray(tau_grid, dtype=float).T
        eft_mask = np.asarray(eft_valid_grid, dtype=bool).T
        tau_ge_needed = np.isfinite(tau_field) & (tau_field >= float(tau_needed))
        if np.any(tau_ge_needed):
            ax.contour(X, Y, tau_ge_needed.astype(float), levels=[0.5], colors=["white"], linewidths=2.5, linestyles="--", zorder=5)

        lam_unit, _ = unitarity_lambda_curve(operator, mchi_grid)
        if lam_unit is not None:
            unitary_valid_mask = (lambda_grid[None, :] >= lam_unit[:, None])
        else:
            unitary_valid_mask = np.ones_like(tau_field, dtype=bool)

        overlap = eft_mask & unitary_valid_mask & np.isfinite(tau_field) & (tau_field >= float(tau_needed))
        has_overlap = bool(np.any(overlap))
        if has_overlap:
            add_hatched_region_from_contour(
                ax=ax,
                X=X,
                Y=Y,
                Z=overlap.astype(float),
                level=0.5,
                upper_level=1.5,
                hatch="////",
                edgecolor="cyan",
                zorder=3,
                outline_lw=2.0,
            )
        else:
            ax.text(
                0.02,
                0.98,
                "No EFT-valid & testable region",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="w",
                fontsize=CONF_ANNOT_FS - 4,
                bbox={"facecolor": "k", "alpha": 0.35, "edgecolor": "none"},
                zorder=6,
            )

    lam_unit, _ = unitarity_lambda_curve(operator, mchi_grid)
    if lam_unit is not None:
        valid_unit = np.isfinite(lam_unit) & (lam_unit > 0.0)
        if np.any(valid_unit):
            ax.plot(
                np.log10(lam_unit[valid_unit]),
                np.log10(mchi_grid[valid_unit]),
                color="#B8B8B8",
                lw=2.0,
                ls=":",
                zorder=4,
            )

    cdm_bound = 1e-3
    if float(np.min(mchi_grid)) <= float(cdm_bound) <= float(np.max(mchi_grid)):
        ax.axhline(float(np.log10(cdm_bound)), color="yellow", lw=2.0, ls=":", alpha=0.8, zorder=5)

    title = "Scalar Rayleigh" if dm_kind == "scalar" else operator_title(str(operator))
    ax.set_title(title, fontsize=CONF_TITLE_FS - 6)
    ax.grid(True, alpha=0.20, ls=":", color="gray", which="both")

    if show_ylabel:
        dm_label = "Dirac" if row_idx == 0 else "Majorana / Scalar"
        ax.set_ylabel(rf"{dm_label}" "\n" r"$\log_{10}(m_\chi/\mathrm{GeV})$", fontsize=CONF_LABEL_FS)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    if show_xlabel:
        ax.set_xlabel(r"$\log_{10}(\Lambda/\mathrm{GeV})$", fontsize=CONF_LABEL_FS)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    ax.set_xlim(float(plot_log10_lambda_min), float(plot_log10_lambda_max))
    ax.set_ylim(-10.0, 6.0)

    coeff_text = r"$c_\phi=1$" if dm_kind == "scalar" else r"$c_s=c_p=1$"
    ax.text(
        0.98,
        0.02,
        rf"{coeff_text}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=CONF_ANNOT_FS - 4,
        bbox={"facecolor": "none", "edgecolor": "white", "alpha": 0.5, "boxstyle": "round,pad=0.3"},
        zorder=7,
    )

    ax.tick_params(labelsize=CONF_TICK_FS)
    return im


def main():
    parser = argparse.ArgumentParser(
        description="Make a combined 2x5 tau-grid figure for fermionic and scalar EFT operators."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=str(DEFAULT_SPECTRUM),
        help="Input spectrum file [E, F, Ferr columns].",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="gc",
        choices=["gc", "cosmic", "gc_nfw", "panelc", "custom"],
        help="Baseline choice for rho_chi and path length.",
    )
    parser.add_argument("--rho-chi", type=float, default=None, help="Custom rho_chi in GeV/cm^3.")
    parser.add_argument("--L-cm", type=float, default=None, help="Custom path length in cm.")
    parser.add_argument("--dip-energy", type=float, default=500.0, help="Dip energy in GeV for tau-energy-mode=dip.")
    parser.add_argument(
        "--tau-energy-mode",
        type=str,
        default="anywhere",
        choices=["anywhere", "band", "dip"],
        help="Use 'anywhere'/'band' to search for a dip anywhere in an energy range, or 'dip' for one fixed energy.",
    )
    parser.add_argument("--tau-energy-min", type=float, default=None, help="Minimum energy in GeV for anywhere/band mode.")
    parser.add_argument("--tau-energy-max", type=float, default=None, help="Maximum energy in GeV for anywhere/band mode.")
    parser.add_argument("--tau-energy-n", type=int, default=60, help="Number of energy points for anywhere/band mode.")
    parser.add_argument(
        "--eft-kinematic-factor",
        type=float,
        default=1.0,
        help="Require Lambda^2 >= factor*max(s_max,|t|_max).",
    )
    parser.add_argument("--log10-Lambda-min", type=float, default=-3.0, help="Minimum log10(Lambda/GeV).")
    parser.add_argument("--log10-Lambda-max", type=float, default=7.0, help="Maximum log10(Lambda/GeV).")
    parser.add_argument("--log10-mchi-min", type=float, default=-10.0, help="Minimum log10(mchi/GeV).")
    parser.add_argument("--log10-mchi-max", type=float, default=6.0, help="Maximum log10(mchi/GeV).")
    parser.add_argument("--tau-grid-n-lambda", type=int, default=40, help="Number of Lambda grid points.")
    parser.add_argument("--tau-grid-n-mchi", type=int, default=40, help="Number of mchi grid points.")
    parser.add_argument(
        "--dip-depth",
        type=float,
        default=0.01,
        help="Target fractional dip depth used for the tau_needed overlay.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="combined_fermion_scalar_tau_grid",
        help="Output basename inside Totani_Scattering/plots.",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="Plot style: paper, conference/dark_transparent, or conference_light/light_transparent.",
    )
    args = parser.parse_args()
    if args.style:
        os.environ["TRINITY_PLOT_STYLE"] = args.style

    args.fermionic = load_module("fermionic_tau_module", FERMIONIC_SCRIPT)
    args.scalar = load_module("scalar_tau_module", SCALAR_SCRIPT)

    set_paper_style(base_fontsize=9, linewidth=1.5, n_colors=12, cmap_name="plasma")

    data = np.loadtxt(args.filename)
    energies = np.asarray(data[:, 0], dtype=float)
    e_eval, tau_energy_label, omega_max_for_validity = build_energy_grid(args, energies)
    baseline_spec = resolve_baseline(args)
    dip_depth = min(max(float(args.dip_depth), 0.0), 0.999999)
    tau_needed = -np.log(1.0 - dip_depth) if dip_depth > 0.0 else 0.0
    header_text = build_meta_text(args, baseline_spec, tau_needed, tau_energy_label)

    panels = [TOP_ROW_PANELS, BOTTOM_ROW_PANELS]
    panel_grids = []
    finite_logs = []

    for row in panels:
        row_grids = []
        for panel in row:
            grid = compute_panel_grid(panel, args, e_eval, omega_max_for_validity, baseline_spec)
            row_grids.append(grid)
            if grid is None:
                continue
            finite = np.log10(np.asarray(grid["tau_grid"], dtype=float) + 1e-30)
            finite = finite[np.isfinite(finite)]
            if finite.size > 0:
                finite_logs.append(finite)
        panel_grids.append(row_grids)

    if not finite_logs:
        raise RuntimeError("No finite tau-grid values were produced.")

    vmin = min(-20.0, float(min(np.min(arr) for arr in finite_logs)))
    vmax = max(0.0, float(max(np.max(arr) for arr in finite_logs)))
    plot_log10_lambda_min = max(float(args.log10_Lambda_min), -3.0)
    plot_log10_lambda_max = float(args.log10_Lambda_max)

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(30.0, 14.0),
        gridspec_kw={"wspace": 0.10, "hspace": 0.14},
    )

    last_im = None
    for row_idx, row in enumerate(panels):
        for col_idx, panel in enumerate(row):
            ax = axes[row_idx, col_idx]
            grid = panel_grids[row_idx][col_idx]
            operator, dm_kind, title = panel

            if grid is None:
                ax.set_axis_off()
                continue

            last_im = draw_panel(
                ax=ax,
                grid=grid,
                panel=panel,
                row_idx=row_idx,
                col_idx=col_idx,
                show_ylabel=(col_idx == 0),
                show_xlabel=(row_idx == 1),
                vmin=vmin,
                vmax=vmax,
                tau_needed=tau_needed,
                tau_energy_label=tau_energy_label,
                plot_log10_lambda_min=plot_log10_lambda_min,
                plot_log10_lambda_max=plot_log10_lambda_max,
            )

    if last_im is None:
        raise RuntimeError("Figure was created without any plotted panels.")

    legend_handles = [
        Patch(facecolor="gray", edgecolor="gray", alpha=0.30, label="EFT invalid"),
        Line2D([0], [0], color="r", lw=2.5, ls="--", label="EFT validity limit"),
        Line2D([0], [0], color="#B8B8B8", lw=2.0, ls=":", label="Unitarity guide"),
        Patch(facecolor="none", edgecolor="cyan", hatch="////", label=r"EFT-valid + testable ($\tau\geq\tau_{\rm needed}$)"),
        Line2D([0], [0], color="cyan", lw=3.0, label=r"Fermi-LAT reach ($\tau=10^{-2}$)"),
        Line2D([0], [0], color="lime", lw=3.0, label=r"CTA reach ($\tau=10^{-3}$)"),
        Line2D([0], [0], color="yellow", lw=2.0, ls=":", label=r"CDM bound ($m_\chi\geq 1\,\mathrm{MeV}$)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.46, 0.992),
        ncol=4,
        fontsize=CONF_LEGEND_FS - 2,
        frameon=True,
        framealpha=0.25,
    )

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.022, pad=0.04, aspect=35)
    cbar.set_label(r"$\log_{10}(\tau_{\max})$", fontsize=CONF_LABEL_FS)
    cbar.ax.tick_params(labelsize=CONF_TICK_FS)

    info_ax = axes[1, 4]
    info_ax.set_axis_off()
    info_ax.text(
        0.04,
        0.96,
        header_text,
        transform=info_ax.transAxes,
        ha="left",
        va="top",
        fontsize=CONF_ANNOT_FS,
        linespacing=1.35,
    )

    fig.subplots_adjust(top=0.88, left=0.07, right=0.86, bottom=0.10)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(OUTPUT_DIR / args.outfile))
    plt.close(fig)
    print(f"Saved combined tau-grid figure: {OUTPUT_DIR / args.outfile}")


if __name__ == "__main__":
    raise SystemExit(main())
