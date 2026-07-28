# Constraint Generation

This folder contains scripts that compute and save constraint boundary products.
The saved `.npz` files still live in the project-level directory:

```text
Totani_Scattering/constraint_boundaries/
```

That path is intentional because `make_paper_style_operator_overlays.py` reads
all generated boundaries from there.

Run these commands from `Totani_Scattering/`.

## Files

| File | Computed constraint | Where it comes from | Saved products |
| --- | --- | --- | --- |
| `make_data_driven_scattering_limits.py` | Halo-component photon-DM scattering limits. It can compute `raw_attenuation` limits and `spectral_reshaping` limits. | The default `--source halo` uses the MCMC-derived Totani halo posterior loaded by `core/totani_data_loader.py`. With `--source pppc`, it uses a PPPC annihilation spectrum from `data/AtProduction_gammas.dat` and fits the normalization at each scattering point. The transfer/reshaping machinery comes from `core/spectral_reshaping.py`; EFT and unitarity masks come from `core/eft_validity.py`. | `constraint_boundaries/mcmc_<profile>_<source>_<model_kind>_<dm_type>_<operator>_90cl.npz`; plot overlays in `plots/mcmc_<profile>_<source>_limits_<dm_type>_<operator>.png/.pdf`. |
| `cmb_constraints.py` | CMB power-spectrum bounds on photon-DM EFT operators. | Planck 2018 bounds as parameterised by Boddy & Gluscevic 2018, arXiv:1801.08609. The script evaluates the EFT cross section at the present CMB temperature and compares `sigma_0 / m_chi` to the published velocity/temperature-scaling limits. | `constraint_boundaries/cmb_<dm_type>_<operator>_planck2018.npz`. Majorana variants add `_majorana`. |
| `make_deconv_exclusion_limits.py` | Deconvolution-derived tau and tension boundaries. | Existing `deconv_scan.npz` products from `deconvolve_totani_spectrum.py`. The tau limit asks when the deconvolved spectrum differs from the observed spectrum by more than the chosen chi2 threshold. The tension limit asks when deconvolution makes the PPPC fit more dSph-tensioned. | `constraint_boundaries/deconv_tau_limit_<operator>_<dm_type>_<channel>_<profile>.npz` and `constraint_boundaries/deconv_tension_limit_<operator>_<dm_type>_<channel>_<profile>.npz`. |
| `make_totani_exclusion_limits.py` | Older chi2 exclusion boundaries from scattering scan grids. | Existing `scan_grid.npz` files under `results/tension_scan/` or another `--scan-dir`. A point is excluded when scattering worsens the fit beyond `chi2_min + delta_chi2`. This is retained for comparison with older plots. | `constraint_boundaries/totani_halo_exclusion_<dm_type>_<operator>_95cl.npz`; optional conference overlay plots in `plots/`. |

## Example Commands

```bash
python constraint_generation/make_data_driven_scattering_limits.py --halo-profile rho2 --source halo --dm-type fermionic --operator dipole_magnetic
python constraint_generation/cmb_constraints.py
python constraint_generation/make_deconv_exclusion_limits.py --scan-dir results/deconv_scan --plot
python constraint_generation/make_totani_exclusion_limits.py --scan-dir results/tension_scan
```

## Path Conventions

- Script location: `constraint_generation/`
- Generated constraints: `constraint_boundaries/`
- Generated comparison plots: `plots/`
- Literature/digitised external curves: `constraints_data/`

The path handling in these scripts is rooted at `Totani_Scattering/`, so moving
the scripts into this folder does not change where the generated constraints are
saved.
