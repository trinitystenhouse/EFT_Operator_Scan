# Reproducing the results and figures

Exact commands behind every figure and quoted number in "Photon–Dark Matter
Elastic Scattering: An Effective-Operator Scan and First Operator-Resolved
Limits from the Galactic Halo".

There are three stages, in increasing order of what they require:

| Stage | Produces | External data needed |
|-------|----------|----------------------|
| 1 | The five figures | none |
| 2 | The 90% CL exclusion grids the figures read | companion halo posterior |
| 3 | The Sec. II D threshold calibration | companion halo posterior |

Each stage's outputs are committed, so you can start at any stage. Run
everything from the repository root.

Every script runs the published configuration with no flags. Where a command
below passes one, it is selecting a *different* product (another profile, another
dataset), not correcting a default.

---

## Stage 0 — Install

Python 3.9 or later:

```bash
pip install -r requirements.txt
python -m pytest tests/
```

Expected: 85 passing. These pin the real-photon amplitudes (including the
identically-vanishing anapole and charge-radius cases), the closed-form cross
sections against resolved quadrature, kernel normalisation and closure, the
static-target null limits of the transport, and the EFT-validity wedge.

---

## Stage 1 — The figures

The exclusion grids in `constraint_boundaries/` are committed, so this works from
a fresh clone with no downloads:

```bash
python make_paper_results_figures.py
```

Writes `.pdf` and `.png` for all five figures into `paper_plots/`. Options:

```bash
python make_paper_results_figures.py --only 2 5        # a subset
python make_paper_results_figures.py --out-dir /tmp/x  # elsewhere
python make_paper_results_figures.py --copy-to ../paper/figures
```

Which figure is which is tabulated in [README.md](README.md).

---

## Stage 2 — Regenerating the exclusion grids

### Get the halo posterior

Dataset (a) of Sec. II E, released with the companion paper:

> T. R. Stenhouse, C. Ghag, F. F. Deppisch, "The 20 GeV Galactic Halo Excess:
> Pixel-Level Confirmation and Consistency with Sub-TeV WIMP Annihilation,"
> arXiv:2607.08552. Release:
> [doi:10.5281/zenodo.21280725](https://doi.org/10.5281/zenodo.21280725).

```bash
export HALO_POSTERIOR_ROOT=/path/to/archive/pixelwise_mcmc
```

That directory must hold `pixelwise_mcmc_results_fig6/` (ρ², disk excluded — the
published profile), and for the variants below `pixelwise_mcmc_results_fig5/`
(ρ<sup>2.5</sup>) and `pixelwise_mcmc_results_fig6_w_disk/` (disk included). Each
contains `mcmc_results_k00.npz` … `mcmc_results_k12.npz`, one per energy bin.

Only Stage 2 and Stage 3 need this. Stage 1 and the tests do not.

### The published scan

```bash
python constraint_generation/make_data_driven_scattering_limits.py --run-all
```

This is the command behind every halo grid in `constraint_boundaries/`. Run bare
it uses the published configuration:

| setting | value | where it comes from |
|---|---|---|
| `--halo-profile` | `pixelwise_global_rho2` | Sec. V A |
| `--source` | `measured` (Φ<sub>src</sub> = Φ<sub>data</sub>) | Sec. II D |
| `--fit-normalization` | on (A profiled analytically) | Eq. (II.14) |
| `--drop-lowest-bins` | 2 (the 1.51 and 2.55 GeV bins) | Sec. IV B |
| `--e-max-fit` | 200 GeV → ω<sub>max</sub> = 168.9 GeV | Sec. IV B |
| grid | 100 × 100 over m<sub>χ</sub> ∈ [10⁻⁶, 10⁸], Λ ∈ [10⁻³, 10⁷] GeV | Sec. IV C |
| `--delta-chi2` | 4.61 (90% CL, two parameters) | Sec. II D |
| ROI recovery | `template` for the halo, `unity` for the IGRB | Sec. V D |

Eight bins survive the cuts, with centres 4.31, 7.28, 12.3, 20.8, 35.1, 59.2, 100
and 169 GeV, and a combined statistical uncertainty of 2.576%.

`--run-all` covers the operator/DM-type combinations, including both parity
pieces of the Rayleigh family and their combined `rayleigh_full` form for Dirac
and Majorana. Each writes one `.npz` per operator for both the `raw_attenuation`
and `spectral_reshaping` observables.

### The other products the figures use

```bash
# rho^2.5 profile — the second (magenta) contour in Fig. 3
python constraint_generation/make_data_driven_scattering_limits.py \
    --halo-profile pixelwise_global_rho2.5 --run-all

# IGRB cross-check — the magenta contours in Fig. 4
python constraint_generation/make_data_driven_scattering_limits.py \
    --dataset igrb --run-all

# Planck 2018 CMB elastic-scattering bounds — Fig. 3
python constraint_generation/cmb_constraints.py
```

Two-component annihilation-template variants (Sec. V D), at the companion
paper's best-fit annihilator masses:

```bash
python constraint_generation/make_data_driven_scattering_limits.py \
    --source pppc --channel WW --ann-mass 550 --run-all
python constraint_generation/make_data_driven_scattering_limits.py \
    --source pppc --channel bb --ann-mass 720 --run-all
```

These are reported in the text but not drawn: the fit is poor (χ²/dof ≈ 9) and
its minimum sits at an interior point with τ > 0, so its 90% CL region is a
confidence interval around a preferred non-zero optical depth rather than a
bound.

The systematic envelope of Sec. V B (at most 0.040 dex, 8.8% in Λ) is the
difference against the disk-included refit:

```bash
python constraint_generation/make_data_driven_scattering_limits.py \
    --halo-profile pixelwise_global_rho2_w_disk --out-suffix _wdisk --run-all
```

### Reading the output

Each `.npz` records the grid it was built on in `grid_mchi_GeV` and
`grid_lambda_GeV`, and its configuration in `halo_profile`, `roi_recovery_model`,
`drop_lowest_bins`, `e_max_fit`, `fit_normalization`, `delta_chi2_threshold` and
`omega_max_for_validity`. Check those before comparing a new scan against a
committed one.

**Sub-step quantities need a finer grid.** The production Λ axis is 100 points
over ten decades, 26% per step. Any quoted change in Λ smaller than that is not
resolved at production resolution: pass `--nl 400` or finer before quoting one.
This is why the band-truncation costs of Sec. IV B were re-measured at
successively finer resolution until they stabilised.

Useful variants: drop `--run-all` and pass `--operator` / `--dm-type` /
`--majorana` to scan a single operator; `--err-mode` to switch the posterior
error convention; `--out-suffix` to write alongside the published grids rather
than over them; `--no-plot` to skip the diagnostic figure.

After regenerating, rerun Stage 1 to redraw the figures.

---

## Stage 3 — The exclusion-threshold calibration

Sec. II D calibrates Δχ² against pseudo-experiments drawn from the per-bin
posterior chains rather than assuming the asymptotic value. Run in order:

```bash
python calibration/run_calibration.py   # toy grids + the calibrated q90
python calibration/run_coverage.py      # coverage of the adopted threshold
python calibration/run_expected.py      # median-expected contours
```

`run_calibration.py` writes `toy_grid_<operator>.npz` and
`calibration_summary.npz` into `calibration/`, which the other two read.

The chains are the full emcee output including pre-convergence steps, so the
toys trim on log-probability first (`_toy_common.burn_in_index`); drawing from
them untrimmed inflates the variance by about an order of magnitude while
leaving the stored percentiles untouched, which is why it does not show up in
the central values.

Expected: a calibrated 90% quantile between 3.51 and 4.02 with median 3.84
across twelve points on the exclusion contour, and 92–95% coverage at the
adopted 4.61.

---

## Directory reference

| Path | Role |
|------|------|
| `core/` | Cross sections, optical depths, transport, validity wedge, loaders |
| `constraint_generation/` | Stage 2 scripts that turn spectra into exclusion grids |
| `calibration/` | Stage 3 pseudo-experiment calibration |
| `constraint_boundaries/` | Committed 90% CL grids (`.npz`) |
| `constraints_data/` | Digitised external limits plus the `limits.py` registry |
| `data/` | PPPC4DMID gamma-ray yield table and the full-band LAT energy grid |
| `helpers/` | Plot styling |
| `tests/` | Regression tests |

Adding an operator means touching four registries that all key on operator name:
`PANEL_CONFIGS` in `make_sensitivity_map.py` (Fig. 2), `PANEL_CONFIGS` in
`make_paper_style_operator_overlays.py` (Fig. 3), `PANEL_CONFIGS` in
`make_multi_dataset_overlay.py` (Fig. 4), and `OPERATOR_SPECS` plus the relevant
`GENERATED_LIMIT_SPECS` groups in `constraints_data/limits.py`. Missing the last
of these silently drops the external-limit overlays from the panel rather than
raising.
