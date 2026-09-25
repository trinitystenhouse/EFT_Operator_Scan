# Data and code deposit: photon-dark matter elastic scattering, effective-operator scan

Deposit for **"Photon-Dark Matter Elastic Scattering: An Effective-Operator Scan
and First Operator-Resolved Sensitivity Estimates from the Galactic Halo"**
(arXiv:2608.29546). Concept DOI 10.5281/zenodo.21703809.

This deposit exists to satisfy one sentence of the paper's Data Availability
statement, and its contents are scoped to exactly that sentence:

> The analysis code and figure-generation scripts that reproduce every figure in
> this paper are archived on Zenodo, together with the production sensitivity
> grids, the calibration and expected-band outputs, and the fixed random seeds
> with which the pseudo-experiments were drawn. Every figure in this paper can be
> redrawn from that deposit alone. The per-bin posterior chains that drive the
> halo bounds, and from which the pseudo-experiments are drawn, are those of the
> companion paper and are available with that work.

**Every figure in the paper can be redrawn from this deposit alone**, with no
network access and no other repository. That claim is tested: see "Clean-room
check" below.

All paths are relative to this directory.

## Redrawing the figures

    bash scripts/reproduce_production.sh          # or: python3 make_paper_results_figures.py

Requires Python 3 with numpy, scipy, matplotlib and contourpy. Nothing else, and
no environment variables. The five figures are written to `paper_plots/`.

`--only N` builds a subset of the figures, for any N and in any order, for
example `python3 make_paper_results_figures.py --only 3`. The output is identical
to that of the full build.

`EFT_IGRB_BOUNDARY_SUFFIX` should be left unset: the IGRB grid tag defaults to
`_profnorm_removalfix_igrbfix2`. Every curve is checked against the `f_bin`
recorded in its own grid, and a grid with inconsistent binning aborts the build.

## Contents

| directory | files | what it is |
|---|---|---|
| (root) | 6 + 3 | the figure and overlay scripts, plus `README.md`, `LICENSE` and `MANIFEST.tsv` |
| `core/` | 11 | the analysis library: cross sections, kinematics, removal, bin geometry, ROI recovery, spectra |
| `constraint_generation/` | 1 | `make_data_driven_scattering_limits.py`, the constraint-scan driver |
| `helpers/` | 2 | the plotting style module imported by every figure script |
| `constraints_data/` | 57 | the loader, the digitised external limit curves, and `PROVENANCE.tsv` |
| `data/` | 2 | Slatyer 2016 energy-deposition efficiency tables, used by the CMB comparison |
| `constraint_boundaries/` | 15 | the thirteen production 90% CL grids, plus two auxiliary curves Fig. 3 draws |
| `constraint_boundaries/toys/` | 26 | the four toy scripts and the calibration and expected-band outputs |
| `scripts/` | 1 | `reproduce_production.sh` |

Every code file is the production version used for the paper; checksums are
listed in `MANIFEST.tsv`. The deposit contains the code and data required for the
five figures, the thirteen grids and the toy outputs. The dwarf-spheroidal data
path is not included, so `--dataset dsph` will not run; the paper derives no dSph
bound (Sec. II E, Sec. V C).

### The production grids

Ten halo grids, `mcmc_<profile>_halo_raw_attenuation_<operator>_profnorm_removalfix_90cl.npz`
for profiles `pixelwise_global_rho2` and `pixelwise_global_rho2.5` and the Dirac
magnetic dipole, scalar Rayleigh, and Majorana Rayleigh even, odd and full.

Three IGRB grids, `igrb_ackermann2015a_measured_raw_attenuation_<operator>_profnorm_removalfix_igrbfix2_90cl.npz`
for the Dirac magnetic dipole, Majorana Rayleigh full and scalar Rayleigh.

Every grid is 100 masses by 1200 Lambda values and records the settings it was
built with, including `roi_recovery_model`, `sigma_model`, `kernel_nodes` and
the bin geometry (`f_bin` per bin, `f_bin_mode`, `f_bin_median` and the bin
ratios). The energy-migration fraction is derived from each dataset's own binning:
0.2305 on the halo grid (bin ratio 1.6890) and 0.1591 on the IGRB grid (bin ratio
1.4142). The IGRB grids also carry the closure clip described in the paper's
Sec. V C.

Two further files sit alongside them because Fig. 3 draws them:
`cmb_fermionic_dipole_magnetic_planck2018.npz` (our CMB energy-injection
comparison curve) and `totani_scalar_rayleigh_90cl.npz` (a legacy scalar
Rayleigh boundary used for one overlay).

### The toy outputs and the seeds

| file | content |
|---|---|
| `toy_grid_<op>[_rho2.5].npz` | tau cache at Lambda = 1 on the 100-point mass axis, the 1200-point Lambda axis, and null-toy summary statistics |
| `calibration_summary[_rho2.5].npz` | null-toy summary across operators |
| `coverage[_rho2.5].npz` | signal-hypothesis calibration and coverage at twelve test points, 2x10^4 toys per point |
| `expected_band_<op>[_rho2.5].npz` | median, 68% and 95% expected contours from 10^4 null toys: for the three Fig. 3 operators on both profiles, and `expected_band_rayleigh_even.npz` (rho^2) for Fig. 5 (left) |

**The pseudo-experiments themselves are not stored, and do not need to be.** A
toy is a vector of eight per-bin fluxes drawn from the posterior chains; the
scripts draw them in memory and keep only derived statistics. They are
reproducible from three ingredients: the companion paper's chains, the scripts
here, and the seeds fixed at the top of those scripts:

| script | seed | what it draws |
|---|---|---|
| `run_calibration.py` | **20260807** | null toys for the threshold calibration and the tau caches |
| `run_coverage.py` | **20260808** | signal-hypothesis toys for coverage |
| `run_expected.py` | **20260809** | null toys for the expected bands |

`run_coverage.py` spawns one independent stream per test point from a
`numpy.random.SeedSequence` on that seed, so the twelve points are reproducible
individually. `run_expected.py` likewise spawns one stream per operator, in the
order the operators are given to `--ops`.

The commands that produced the expected-band files, from the root of this
deposit (each needs the companion paper's chains, see below):

| file | command |
|---|---|
| `expected_band_{dipole_magnetic,rayleigh_full,scalar_rayleigh}.npz` | `python3 constraint_boundaries/toys/run_expected.py 10000` |
| `expected_band_{dipole_magnetic,rayleigh_full,scalar_rayleigh}_rho2.5.npz` | `python3 constraint_boundaries/toys/run_expected.py 10000 --profile pixelwise_global_rho2.5` |
| `expected_band_rayleigh_even.npz` | `python3 constraint_boundaries/toys/run_expected.py 10000 --ops rayleigh_even --chi2-check-tol 1e-3` |

Before drawing toys, `run_expected.py` checks that the chi2 grid rebuilt from
`toy_grid_<op>.npz` matches the production grid's `chi2_grid`, to a relative
tolerance set by `--chi2-check-tol` (default 1e-4). For `rayleigh_even` the two
agree to 4.8e-4, and the rebuilt 90% CL contour matches the production contour
to 1.7e-5 dex, so that file is generated with `--chi2-check-tol 1e-3`.
`--out-dir` writes the output to another directory.

### Figure 5 (left)

The dark-Higgs portal panel maps the rho^2 parity-even fermionic Rayleigh
contour (`mcmc_pixelwise_global_rho2_halo_raw_attenuation_fermionic_rayleigh_even_majorana_profnorm_removalfix_90cl.npz`)
through the matching of the paper's Sec. VI A, one curve per benchmark mass
(`FIG5_MCHI_LIST` in `make_paper_results_figures.py`), each with its 68%
expected band from `expected_band_rayleigh_even.npz`. The matching is in
`make_uv_translation_bounds.py` (`dark_higgs_lambda_R`, `dark_higgs_bound`).

## Related material

The per-bin posterior chains (48 MB) that drive the halo bounds, and from which
the pseudo-experiments are drawn, are the product of the companion paper
(arXiv:2607.08552) and are available with that work. They are not needed to
redraw the figures. Regenerating the grids or the toys does require them, and
`reproduce_production.sh all` reads their location from `CHAINS_ROOT`.

## The digitised external limit curves

`constraints_data/` holds 54 text files of **other groups' published limits,
digitised by us** from their figures, together with the loader that reads them.
They are what Figs. 3 and 5 draw as comparison curves, and the build reads 53 of
the 54 (the remaining file is a directory README).

`constraints_data/PROVENANCE.tsv` gives, for every curve: the source as recorded
in that file's own header, its arXiv identifier, the figure it was digitised
from, and its status in the pipeline. All 53 curves carry a source header; 50 of
them carry an arXiv identifier. The statuses are:

| status | files | meaning |
|---|---|---|
| loaded | 36 | read into the registry and available to the figures |
| refused by the loader guard | 15 | read, then REFUSED and never plotted, because the header does not establish a verifiable digitisation (for example `anapole/hambye_2021_anapole.txt`, whose header records that the arXiv number it once carried was a placeholder). They are shipped because the loader reads them and prints its refusal, so the deposit reproduces the published console output exactly, and because the refusal is itself part of the provenance record |
| registered, not loaded | 2 | listed in the registry but not loaded into any panel |

**These are other people's numbers.** Our digitisation carries our reading error,
not theirs. Anyone using a curve from this deposit should cite the source paper
given in `PROVENANCE.tsv`, not this deposit.

## Clean-room check

This deposit was copied to an empty directory with nothing else on the path and
all five figures were rebuilt there, with no environment variables set and no
access to the analysis repository. Each is **byte-identical** (PNG) and
identical in PDF content stream to the figure in the manuscript; the PDFs differ
only in their embedded creation timestamps.

## Licence

`LICENSE` sets three terms, because this deposit holds three kinds of material:

| material | terms |
|---|---|
| the code (every `.py` and `.sh`) | MIT |
| the data this work produced -- the thirteen grids, the toy calibration and expected-band outputs, `MANIFEST.tsv`, this README | CC-BY-4.0 |
| third-party material -- the 53 digitised curves in `constraints_data/` and the two Slatyer 2016 tables in `data/` | not ours to license; cite the source paper given in `constraints_data/PROVENANCE.tsv` |

The per-bin posterior chains are not in this deposit and are covered by the
companion paper's terms.

For the Zenodo record itself, CC-BY-4.0 is the licence that matches the bulk of
the deposited data; the MIT grant on the code and the third-party exclusion are
stated in `LICENSE`.

## Provenance and integrity

`MANIFEST.tsv` lists the relative path, size in bytes and SHA-256 of every file
in this deposit.
