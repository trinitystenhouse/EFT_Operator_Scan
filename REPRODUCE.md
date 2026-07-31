# Reproducing the results and figures

This file gives the exact commands behind every number and figure in
"Photon-Dark Matter Elastic Scattering: An Effective-Operator Scan and First
Limits from the Galactic Halo".

There are three stages, in increasing order of what they require:

| Stage | Produces | External data needed |
|-------|----------|----------------------|
| 1 | The five paper figures | none |
| 2 | The 90% CL exclusion grids the figures read | companion MCMC halo posterior |
| 3 | The observed halo spectrum | raw *Fermi*-LAT Pass 8 event files |

Stage 1 is enough to reproduce the paper as published. Stages 2 and 3
rebuild its inputs from scratch. Each stage's outputs are committed, so you can
start at any stage and work downwards.

Run everything from the repository root, so that `core/`, `helpers/` and
`totani_helpers/` resolve as packages.

---

## Stage 0 — Install

Python 3.9 or later:

```bash
pip install -r requirements.txt
```

Then confirm the installation with the regression tests, which need no
external data and take a couple of seconds:

```bash
python -m pytest tests/
```

Expected: 23 passing. These pin the real-photon amplitudes (including the
identically-vanishing anapole and charge-radius cases) and the multi-dSph
limit machinery.

---

## Stage 1 — The paper figures

The digitised exclusion grids in `constraint_boundaries/` and the observed
halo spectrum in `data/fermi_halo_spectrum.txt` are both committed, so this
works from a fresh clone with no downloads:

```bash
python make_paper_results_figures.py
```

This writes both `.pdf` and `.png` for all five figures into `paper_plots/`,
overwriting the committed copies with byte-comparable output. Options:

```bash
python make_paper_results_figures.py --only 2 5        # a subset
python make_paper_results_figures.py --out-dir /tmp/x  # elsewhere
python make_paper_results_figures.py --copy-to ../paper/paper_plots
```

`bash run_paper_plots.sh` is a thin wrapper over the same entry point.

`make_paper_results_figures.py` is the single source of truth for figure
styling: every font size, colour, panel geometry, contour level and legend
placement lives in the constants at the top of that file, not in the
sub-modules it calls.

### Which figure is which

| Paper | File | Shows | Drawn by |
|-------|------|-------|----------|
| Fig. 1 | `f1_uv_complete_tau_vs_mchi` | Maximum perturbative τ vs m<sub>χ</sub> for the dark-Higgs and gravitational channels | `make_uv_complete_tau_vs_mchi.py` |
| Fig. 2 | `f2_sensitivity_map` | 6-panel f<sub>required</sub> heatmap over the operator basis | `make_sensitivity_map.py` |
| Fig. 3 | `f3_halo_constraints` | 3-panel halo exclusion vs collider / direct / indirect / cosmology bounds | `make_paper_style_operator_overlays.py` |
| Fig. 4 | `f4_multi_dataset_overlay` | Halo vs IGRB cross-dataset overlay, plus PPPC source variants | `make_multi_dataset_overlay.py` |
| Fig. 5 | `f5_uv_translation_bounds` | Halo bound translated onto the dark-Higgs and EW-doublet completions | `make_uv_translation_bounds.py` |

Two notes on what you will see in the console. Fig. 3 prints
`[note] legacy boundary absent, skipping: totani_fermionic_*_90cl.npz` for
some operators — expected, and not an error: those legacy tension-scan files
were withdrawn and the panels draw the current data-driven boundaries
instead. The Planck CMB curve is absent from the Rayleigh panels because the
Rayleigh CMB bound sits near Λ ~ 10<sup>-14</sup> GeV, roughly ten decades
below the plotted window.

---

## Stage 2 — Regenerating the exclusion grids

This is the stage that needs the external dataset.

### Get the halo posterior

The pixel-level MCMC Galactic-centre halo posterior is dataset (D1) of the
paper and is released with the **companion** paper, not this repository:

> T. R. Stenhouse, C. Ghag, F. F. Deppisch, "The 20 GeV Galactic Halo Excess:
> Pixel-Level Confirmation and Consistency with Sub-TeV WIMP Annihilation,"
> arXiv:2607.08552.
>
> Release: `trinitystenhouse/Totani-Reanalysis`,
> [doi:10.5281/zenodo.21280725](https://doi.org/10.5281/zenodo.21280725)
> (CC-BY-4.0). That archive holds the companion analysis scripts and its
> data-acquisition instructions; run its MCMC pipeline, or take the posterior
> files it ships, to get the `mcmc_results_k*.npz` needed below.

Once you have the posterior, point this code at it. The simplest route is the
environment variable, which overrides all path guessing:

```bash
export TOTANI_MCMC_DIR=/path/to/archive/pixelwise_mcmc_results_fig6
```

That directory must contain `mcmc_results_k00.npz` … `mcmc_results_k12.npz`
(13 energy bins). `pixelwise_mcmc_results_fig6` is the ρ² disk-excluded fit —
the profile used for the published limits.

Alternatively set `REPO_PATH` to the archive root and let the loader find the
profiles itself; `core/totani_data_loader.py::_MCMC_DIRS` maps every profile
name (`rho2`, `rho2.5`, `rho1`, and their `global_`/`pixelwise_global_`
variants) to its expected subdirectory.

`core/attenuation_eft.py` loads the posterior **at import time** and raises
`FileNotFoundError` if it is missing. That is deliberate: a misconfigured scan
must fail loudly rather than quietly fall back to superseded hand-digitised
bin centres. So this error means the path above is wrong, not that the code is
broken. Only Stage 2 imports that module — Stage 1 and the tests do not.

### Run the scan

The whole operator set against the published halo profile:

```bash
python constraint_generation/make_data_driven_scattering_limits.py \
    --dataset halo --halo-profile pixelwise_global_rho2 \
    --source measured --run-all
```

`--run-all` covers all twelve operator/DM-type combinations, including both
the CP-even/odd Rayleigh pieces and their combined (`rayleigh_full`) forms for
Dirac and Majorana. Each run writes one `.npz` per operator into
`constraint_boundaries/`, for both the `raw_attenuation` and
`spectral_reshaping` observables.

The other products the figures use:

```bash
# rho^2.5 profile — the second contour in Fig. 3
python constraint_generation/make_data_driven_scattering_limits.py \
    --dataset halo --halo-profile pixelwise_global_rho2.5 --source measured --run-all

# IGRB cross-check — the magenta contours in Fig. 4
python constraint_generation/make_data_driven_scattering_limits.py \
    --dataset igrb --source measured --run-all

# Two-component PPPC source variants — the dash-dot contours in Fig. 4
python constraint_generation/make_data_driven_scattering_limits.py \
    --dataset halo --halo-profile pixelwise_global_rho2 \
    --source pppc --channel WW --ann-mass 550 --run-all
python constraint_generation/make_data_driven_scattering_limits.py \
    --dataset halo --halo-profile pixelwise_global_rho2 \
    --source pppc --channel bb --ann-mass 720 --run-all

# Planck CMB bounds
python constraint_generation/cmb_constraints.py
```

**Pass the grid size explicitly.** The script defaults to `--nm 32 --nl 32`,
but the published halo grids are **100 × 100**. On a 32-point Λ axis spanning
ten decades each cell is a factor of 2.05, so a scan left on the defaults
quantises the contour far more coarsely than the paper's. To reproduce the
published ρ² grids, add:

```
--mchi-min 1e-6 --mchi-max 1e8 --nm 100 --lambda-min 1e-3 --lambda-max 1e7 --nl 100
```

Each stored `.npz` records the grid it was built on in `grid_mchi_GeV` and
`grid_lambda_GeV`; check those before comparing a new scan against a committed
one. Other defaults match the paper: `--delta-chi2 4.61` (90% CL, two
parameters of interest), 160 scattering-angle points, and emissivity-weighted
ROI averaging over a 60° half-angle. The scan uses the ROI-averaged linear
column J<sub>ROI</sub>, not the order-of-magnitude baselines of the paper's
Sec. II — see `core/attenuation_eft.py::roi_tau_prefactor`.

Useful variants: drop `--run-all` and pass `--operator`/`--dm-type`/
`--majorana` to scan a single operator; `--err-mode` to switch the posterior
error convention; `--no-plot` to skip the diagnostic figure. The PPPC channel
and annihilator mass appear in the output filename as
`pppc_<channel>_mann<mass>`, so you can check which variants you already have
by listing `constraint_boundaries/`.

After regenerating, rerun Stage 1 to redraw the figures from the new grids.

---

## Stage 3 — Rebuilding the observed spectrum

Only needed to regenerate `data/fermi_halo_spectrum.txt` from raw photons.
The committed file is the version used in the paper.

Raw *Fermi*-LAT Pass 8 photon and spacecraft files are public from the Fermi
Science Support Center, <https://fermi.gsfc.nasa.gov/ssc/>. The loaders in
`core/attenuation_eft.py::load_fermi_spectrum_energies` and the two
figure-side defaults prefer the committed copy and fall back to
`../fermi_data/york/processed/spectrum_data.txt` alongside the repository, so
place a regenerated file at either location, or pass an explicit path.

Format: whitespace-separated columns of energy [GeV], E²dN/dE
[GeV cm⁻² s⁻¹], error, and counts, with a leading `#` comment line.

---

## Directory reference

| Path | Role |
|------|------|
| `core/` | Cross sections, optical depths, validity masks, dataset loaders |
| `constraint_generation/` | Stage 2 scripts that turn spectra into exclusion grids |
| `constraint_boundaries/` | Committed 90% CL grids (`.npz`), one per operator/dataset/profile |
| `constraints_data/` | Digitised external limits plus the `limits.py` registry |
| `data/` | PPPC4DMID yield tables, benchmark curves, observed halo spectrum |
| `helpers/`, `totani_helpers/` | Plot styling and posterior/FITS I/O |
| `paper_plots/` | Committed copies of the five figures |
| `tests/` | Regression tests |

Adding an operator means touching four registries that all key on operator
name: `PANEL_CONFIGS` in `make_sensitivity_map.py` (Fig. 2), `PANEL_CONFIGS`
in `make_paper_style_operator_overlays.py` (Fig. 3), `PANEL_CONFIGS` in
`make_multi_dataset_overlay.py` (Fig. 4), and `OPERATOR_SPECS` plus the
relevant `GENERATED_LIMIT_SPECS` groups in `constraints_data/limits.py`.
Missing the last of these silently drops the external-limit overlays from the
panel rather than raising.
