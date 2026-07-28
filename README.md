# EFT_Operator_Scan

Analysis code accompanying **"Photon-Dark Matter Elastic Scattering: An
Effective-Operator Scan and First Limits from the Galactic Halo"**
(Stenhouse, Acar, Bashkanov, Deppisch, Ghag, Isaacson, Watts; submitted to
Phys. Rev. D; [arXiv:XXXX.XXXXX] once posted). The code computes
gamma-chi elastic-scattering optical depths for an effective-operator basis
spanning dimension 5-7 dipole, anapole, charge-radius and Rayleigh operators
(and a set of illustrative UV completions), turns those optical depths into
chi-squared / f_required sensitivity maps against Fermi-LAT gamma-ray
spectra, and produces the figures used in the paper. The primary
astrophysical input — a pixel-level MCMC posterior for the 20 GeV Galactic
Centre halo excess — comes from a companion analysis and is not part of this
repository; see "External data" below.

## Requirements and installation

Python 3.9 or later. Install the dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the third-party packages actually imported by the
code: `numpy`, `scipy`, `matplotlib`, `pandas`, `astropy`, `pytest`, plus
`Pillow`, `tqdm` and `cycler` (used by the bundled plotting-style helpers).
Nothing is version-pinned; no script relies on a specific release.

The two small in-repository support packages `helpers/` (the paper's
font/colour scheme) and `totani_helpers/` (posterior I/O) are included, so
running the figure scripts requires nothing beyond `requirements.txt`. Run
the scripts from the repository root, or add it to `PYTHONPATH`, so these
packages resolve.

## Repository layout

```
EFT_Operator_Scan/
├── core/                     physics library
│   ├── attenuation_eft.py       EFT operator cross sections and optical-depth (tau) machinery
│   ├── cross_sections.py        legacy gravitational / Higgs-portal cross sections
│   ├── kinematics.py             Compton kinematics
│   ├── spectral_reshaping.py     Compton transfer kernel and reshaping scan
│   ├── spectrum_source.py        common interface over halo / IGRB / dSph spectra
│   ├── totani_data_loader.py     loads the halo MCMC posterior (external data, see below)
│   ├── dsph_sed_loader.py        dSph SED loader (see README_dsph_sources.md)
│   ├── dsph_limits.py / dsph_limits_multi.py   dSph exclusion limits
│   ├── eft_validity.py           EFT/unitarity validity masks
│   └── plot_reshaping.py         reshaping diagnostic plots
│
├── constraint_generation/    scripts that turn spectra into constraint-boundary products
│   ├── make_data_driven_scattering_limits.py   master halo/IGRB scan driver
│   ├── make_uv_completion_limits.py            UV-completion translations
│   ├── make_uv_baseline_tau_limits.py          perturbative baseline tau limits
│   ├── make_deconv_exclusion_limits.py         deconvolution-derived tau/tension boundaries
│   ├── make_totani_exclusion_limits.py         chi-squared exclusion boundaries (legacy scan grids)
│   ├── cmb_constraints.py                      Planck 2018 CMB bounds (Boddy & Gluscevic 2018)
│   ├── run_mann_benchmark_family.py            PPPC annihilator benchmark family
│   └── dummy_constraints.py, *.csv             placeholder/collider limit inputs
│
├── constraint_boundaries/    generated 90% CL exclusion grids, one .npz per operator/dataset/profile —
│                             INCLUDED so the figures below reproduce without rerunning the MCMC scan
├── constraints_data/         digitised external limits (LZ, XENON1T, LEP, Fermi dSph, HESS, ...) plus
│                             the limits.py / limits2.py registry (see constraints_data/README.md)
├── data/                     PPPC4DMID gamma-ray yield tables, Boddy & Gluscevic (2019) benchmark curves,
│                             and fermi_halo_spectrum.txt (the observed 20 GeV halo SED used by Figs 3-4)
├── helpers/                  plotting-style helpers (font/colour scheme, save utilities)
├── totani_helpers/           posterior / FITS I/O helpers used by the halo-scan loaders
├── tests/                    pytest regression tests (EFT real-photon amplitudes, multi-dSph limits)
├── paper_plots/              pre-built copies of the five paper figures (pdf/png)
│
├── make_uv_complete_tau_vs_mchi.py         Fig. 1 (Sec. II)   perturbative UV-complete tau vs m_chi
├── make_sensitivity_map.py                 Fig. 2 (Sec. III)  f_required sensitivity heatmap, full operator basis
├── make_paper_style_operator_overlays.py   Fig. 3 (Sec. IV.C) halo exclusion overlay with external bounds
├── make_multi_dataset_overlay.py           Fig. 4 (Sec. IV.D) halo + IGRB cross-dataset overlay
├── make_uv_translation_bounds.py           Fig. 5 (Sec. V)    halo bound translated onto two UV completions
├── make_paper_results_figures.py           single entry point that calls the five scripts above with the
│                                           exact styling/geometry used in the paper (current source of truth)
├── run_paper_plots.sh                      earlier per-figure shell wrapper (see note below)
│
├── make_combined_fermion_scalar_tau_grid.py, make_uv_complete_theory_limits.py, ...
│                             supplementary/exploratory plots not in the main figure set
├── attenuation.py, deconvolve_totani_spectrum.py, fit_totani_dm_scattering.py,
│   scan_deconvolved_scattering_fit.py, run_reshaping.py, run_global_scattering_grid.py, ...
│                             analysis scripts used to build the constraint_boundaries/ products
│                             and to cross-check the deconvolution/reshaping pipeline
└── README_dsph_sources.md    provenance of the dSph SED inputs (see that file directly)
```

## Reproducing the paper figures

The digitised 90% CL exclusion grids in `constraint_boundaries/` and the
observed halo spectrum in `data/fermi_halo_spectrum.txt` ship with this
repository, so all five figures rebuild from a fresh clone with no external
download:

```bash
python make_paper_results_figures.py
```

This writes `f1_uv_complete_tau_vs_mchi` through `f5_uv_translation_bounds`
(`.pdf` and `.png`) to `paper_plots/`, matching the copies already committed
there. Use `--only 2 4` to build a subset, or `--copy-to <dir>` to also copy
the output elsewhere. `run_paper_plots.sh` is a thin wrapper around the same
entry point.

The regression tests likewise need no external data:

```bash
python -m pytest tests/
```

Individual constraint-boundary products (the `.npz` files already present in
`constraint_boundaries/`) can be regenerated with, e.g.:

```bash
python constraint_generation/make_data_driven_scattering_limits.py \
    --dataset halo --halo-profile pixelwise_global_rho2 --source measured --run-all
```

This step does require the MCMC halo posterior described next.

## External data (not included in this repository)

Neither of the items below is needed to reproduce the paper figures or run
the tests — both of those work from a fresh clone. They are needed only to
**regenerate the exclusion grids from scratch** (the halo scan).

### (a) Pixel-level MCMC Galactic-centre halo posterior — needed only to rerun the scan

The halo optical-depth scan (`core/attenuation_eft.py`, driven by
`constraint_generation/make_data_driven_scattering_limits.py`) reads the
20 GeV halo excess directly from the pixel-level MCMC posterior of the
companion paper, T. R. Stenhouse, C. Ghag, F. F. Deppisch, *"The 20 GeV
Galactic Halo Excess: Pixel-Level Confirmation and Consistency with Sub-TeV
WIMP Annihilation,"* arXiv:2607.08552 (submitted to Phys. Rev. D). Obtain it
from that paper's own Zenodo archive: **[DOI: 10.5281/zenodo.XXXXXXX —
placeholder, see companion paper for the definitive record]**.

`core/attenuation_eft.py` loads these posteriors at import time and raises
`FileNotFoundError` if they cannot be found — deliberately, so that a scan
run fails loudly rather than silently falling back to superseded
hand-digitised values. This only affects code paths that recompute optical
depths; the figure scripts and tests do not import this module and run
without the posterior.

Place the archive so the posterior files resolve to one of:

```
<REPO_PATH>/Totani_reanalysis/mcmc/fit_results/pixelwise_mcmc/pixelwise_mcmc_results_fig6/
    mcmc_results_k00.npz ... mcmc_results_k12.npz      (preferred: pixel-level fit, NFW rho^2, disk excluded)

<REPO_PATH>/Totani_paper_check/mcmc/mcmc_results_fig6/
    mcmc_results_k00.npz ... mcmc_results_k12.npz      (legacy fallback: single-likelihood fit)
```

where `REPO_PATH` is, by default, the parent directory of wherever you
cloned this repository (i.e. these two directories are expected as
*siblings* of the repository root, not inside it) — or set the environment
variable `REPO_PATH` to point at wherever you unpacked the archive. For the
specific module-level load in `core/attenuation_eft.py`, you can instead set
`TOTANI_MCMC_DIR` to the exact directory containing the `mcmc_results_k*.npz`
files:

```bash
export TOTANI_MCMC_DIR=/path/to/zenodo_archive/pixelwise_mcmc_results_fig6
python make_paper_results_figures.py
```

Other halo profiles (rho^2.5, rho^1, disk-included variants) and datasets
(fig2_3, fig4) follow the same `mcmc_results_k*.npz` naming under the
corresponding subdirectory — see `core/totani_data_loader._MCMC_DIRS` for the
full mapping.

### (b) Raw Fermi-LAT Pass 8 photon data — optional

The **processed** halo spectrum (`data/fermi_halo_spectrum.txt`, energy in
the first column) ships with this repository and is what the figures use, so
no download is required to reproduce them. The underlying raw Fermi-LAT
Pass 8 photon and spacecraft files are public from the Fermi Science Support
Center (<https://fermi.gsfc.nasa.gov/ssc/>) and are needed only if you want
to rebuild that processed spectrum from the event data. The loaders
(`core.attenuation_eft.load_fermi_spectrum_energies` and the two figure-side
defaults) prefer the bundled copy and fall back to a
`fermi_data/york/processed/spectrum_data.txt` alongside the repository if it
is absent; you can also pass an explicit path.

## Citation

See `CITATION.cff`. In brief: cite the paper above once the arXiv number is
assigned, and this software release via its Zenodo DOI (`.zenodo.json`); if
your work also depends on the halo posterior itself, cite the companion
paper (arXiv:2607.08552) and its own data release.

## License

MIT — see `LICENSE`.
