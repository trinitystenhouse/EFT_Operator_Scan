# EFT_Operator_Scan

Analysis code for **"Photon–Dark Matter Elastic Scattering: An Effective-Operator
Scan and First Operator-Resolved Limits from the Galactic Halo"** (Stenhouse,
Acar, Bashkanov, Deppisch, Ghag and Watts).

The code computes elastic γχ → γχ optical depths for the dimension-5 to
dimension-7 operator basis coupling scalar, Majorana and Dirac dark matter to
the photon, transports gamma-ray spectra through the resulting attenuation and
redistribution kernel, turns that into 90% CL exclusion contours and
exposure-multiplier maps against *Fermi*-LAT data, and draws the paper's five
figures.

## Install

Python 3.9 or later:

```bash
pip install -r requirements.txt
```

Run everything from the repository root so that `core/`, `helpers/` and
`constraints_data/` resolve as packages.

The regression tests need no external data:

```bash
python -m pytest tests/
```

## The five figures

All five rebuild from a fresh clone with no downloads — the 90% CL exclusion
grids they read are committed in `constraint_boundaries/`:

```bash
python make_paper_results_figures.py
```

This writes `.pdf` and `.png` for each figure to `paper_plots/`. Use
`--only 2 4` for a subset, `--out-dir` to write elsewhere.

| Paper | File | Shows | Drawn by |
|-------|------|-------|----------|
| Fig. 1 | `f1_uv_complete_tau_vs_mchi` | Maximum perturbative τ vs m<sub>χ</sub>, dark-Higgs and gravitational channels (Sec. III) | `make_uv_complete_tau_vs_mchi.py` |
| Fig. 2 | `f2_sensitivity_map` | 4-panel f<sub>required</sub> map over the surviving operators (Sec. IV C) | `make_sensitivity_map.py` |
| Fig. 3 | `f3_halo_constraints` | Halo exclusion vs collider / direct / indirect / cosmology bounds (Sec. V C) | `make_paper_style_operator_overlays.py` |
| Fig. 4 | `f4_multi_dataset_overlay` | Halo vs IGRB cross-dataset consistency (Sec. V D) | `make_multi_dataset_overlay.py` |
| Fig. 5 | `f5_uv_translation_bounds` | Halo bound translated onto the dark-Higgs and EW-doublet completions (Sec. VI D) | `make_uv_translation_bounds.py` |

`make_paper_results_figures.py` is the single source of truth for figure
styling: every font size, colour, panel geometry, contour level and legend
placement lives in the constants at the top of that file, not in the modules it
calls.

## Repository layout

```
core/                       physics library
  attenuation_eft.py          operator amplitudes, cross sections, ROI optical depth
  cross_sections.py           Higgs-portal and gravitational cross sections (Sec. III)
  sigma_closed_form.py        closed-form sigma_tot, Eqs. (IV.12)–(IV.17)
  kinematics.py               Compton kinematics and the redistribution kernel, Eq. (II.13)
  roi_recovery.py             w(theta) from the rho^2 template over the real ROI
  spectral_reshaping.py       transport, Eq. (II.10), and the (m_chi, Lambda) scan
  spectrum_source.py          common interface over the halo and IGRB spectra
  eft_validity.py             EFT-validity wedge, Eq. (IV.18)
  totani_data_loader.py       reads the halo MCMC posterior (external, see below)

constraint_generation/
  make_data_driven_scattering_limits.py   the exclusion-grid scan
  cmb_constraints.py                      Planck 2018 CMB elastic-scattering bounds

calibration/                threshold calibration against pseudo-experiments (Sec. II D)
constraint_boundaries/      committed 90% CL grids, one .npz per operator/dataset/profile
constraints_data/           digitised external limits plus the limits.py registry
data/                       PPPC4DMID gamma-ray yield table and the full-band LAT energy grid
helpers/                    the paper's plotting style
tests/                      regression tests

make_paper_results_figures.py   entry point that draws all five figures
make_uv_complete_tau_vs_mchi.py, make_sensitivity_map.py,
make_paper_style_operator_overlays.py, make_multi_dataset_overlay.py,
make_uv_translation_bounds.py   the per-figure modules it calls
```

## Regenerating the exclusion grids

This is the only step that needs data from outside the repository. Run bare, the
scan reproduces the published halo grid: the pixel-level Galactic-centre
posterior under NFW ρ², source normalisation profiled, the two lowest bins
dropped, a 200 GeV ceiling, and a 100 × 100 grid.

```bash
python constraint_generation/make_data_driven_scattering_limits.py --run-all
```

`--run-all` covers the operator set; each run writes one `.npz` per operator into
`constraint_boundaries/` for both the `raw_attenuation` and `spectral_reshaping`
observables. See [REPRODUCE.md](REPRODUCE.md) for the other products the figures
use (the ρ<sup>2.5</sup> profile, the IGRB cross-check, the CMB bounds) and for
the flags behind every quoted number.

### External data: the halo posterior

The pixel-level MCMC Galactic-centre halo posterior is released with the
companion paper, not with this repository:

> T. R. Stenhouse, C. Ghag and F. F. Deppisch, "The 20 GeV Galactic Halo Excess:
> Pixel-Level Confirmation and Consistency with Sub-TeV WIMP Annihilation,"
> arXiv:2607.08552; release
> [doi:10.5281/zenodo.21280725](https://doi.org/10.5281/zenodo.21280725).

Point this code at it:

```bash
export HALO_POSTERIOR_ROOT=/path/to/archive/pixelwise_mcmc
```

That directory must contain `pixelwise_mcmc_results_fig6/` (NFW ρ², disk
excluded — the profile behind the published limits), and, for the two variants
the paper also quotes, `pixelwise_mcmc_results_fig5/` (ρ<sup>2.5</sup>) and
`pixelwise_mcmc_results_fig6_w_disk/` (the disk-included systematic of Sec. V B).
Each holds `mcmc_results_k00.npz` … `mcmc_results_k12.npz`.

The underlying *Fermi*-LAT Pass 8 photon and spacecraft files are public from the
Fermi Science Support Center (<https://fermi.gsfc.nasa.gov/ssc/>) and are needed
only to rebuild the posterior itself, which is the companion analysis rather than
this one.

## Citation

> T. R. Stenhouse, A. Acar, M. Bashkanov, F. F. Deppisch, C. Ghag and
> D. P. Watts, "Photon–Dark Matter Elastic Scattering: An Effective-Operator Scan
> and First Operator-Resolved Limits from the Galactic Halo" (2026).

If your work depends on the Galactic-centre halo posterior itself, please also
cite the companion analysis and its release, above.

## License

MIT — see `LICENSE`.
