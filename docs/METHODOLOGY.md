# Methodology and provenance

Where every number in the manuscript comes from: which script, which inputs,
which flags. Written against the pruned release layout, not the working tree.

---

## 1. Layout

```
core/                     physics
  kinematics.py           Compton relation, Mandelstam invariants, redistribution matrix
  sigma_closed_form.py    closed-form removal cross sections F_i, two-channel sigma_rem
  roi_recovery.py         template-computed angular recovery weight w(theta)
  spectral_reshaping.py   transfer matrix, chi-squared, profiled normalisation
  attenuation_eft.py      optical depth, 90% CL boundary extraction
  eft_validity.py         validity wedge
  spectrum_source.py      reference spectra (halo posterior, IGRB)
  totani_data_loader.py   halo posterior loader
  cross_sections.py       amplitudes
constraint_generation/
  make_data_driven_scattering_limits.py   produces every *_90cl.npz
  cmb_constraints.py                      CMB elastic curves in Fig. 3
constraints_data/         digitised external limits + the provenance guard
calibration/              toy calibration of the exclusion threshold
constraint_boundaries/    the 14 grids the figures are built from
data/                     external inputs
```

Figure drivers are the five `make_*.py` at the root.

---

## 2. The headline limits

| quantity | value | grid |
|---|---|---|
| Dirac dipoles, peak | Λ ≃ 0.3215 GeV at m_χ ≃ 60.2 GeV | `mcmc_pixelwise_global_rho2_halo_raw_attenuation_fermionic_dipole_magnetic_90cl.npz` |
| dimension-6 scalar Rayleigh | Λ ≃ 0.2074 GeV at m_χ ≃ 0.221 GeV | `..._scalar_scalar_rayleigh_90cl.npz` |
| dimension-7 Rayleigh | Λ ≃ 0.79–1.0607 GeV at m_χ ≃ 0.12–0.58 TeV | `..._fermionic_rayleigh_full_majorana_90cl.npz` |

Read as `max(lambda_plot_GeV)` and the `mchi_GeV` at that index. These are
production-grid values; a twelvefold refinement moves them down 1–5%, which
§IV C states.

**The halo grids carry `fit_normalization: True`.** The source normalisation is
profiled at every scan point. Confirm it in the `.npz` metadata rather than
assuming, because the generator's own default for `--dataset halo` is *off*.

---

## 3. Removal, and why the text and the code must be read together

The attenuation observable needs a definition of removal. Two channels act, and
they are combined rather than chosen between:

```
sigma_rem = [F_i(u*) - F_i(u_max)]                      energy migration, hard
          - int_0^{u*} (dF_i/du) [1 - w(theta(u))] du   angular escape, smooth
```

with `u* = min(u_E, u_max)`, `u_E = f/(1-f)`, `f = 0.231`, and
`cos theta(u) = 1 - (m_chi/omega) u`. Implemented in
`core/sigma_closed_form.py::sigma_removal_smooth_w_cm2`.

Two traps.

**The ROI recovery model is per-dataset physics, not a numerical knob.** The
halo uses `--roi-recovery-model template`: a photon deflected out of the region
of interest is genuinely lost. The **IGRB must use `unity`**: for an isotropic
source against a uniform cosmological scatterer, out-scattering along any line
of sight is compensated by in-scattering from the surrounding sky, so angular
escape removes nothing and energy migration is the only channel. Running the
IGRB with `template` leaves the smooth term alive after the energy channel has
closed; it then grows without bound with mass, and the contour runs to the edge
of the grid instead of terminating. **The choice is not recorded in the `.npz`
metadata.** It cannot be recovered from a saved grid.

**The removal channel closes at high mass.** Above `m_chi = 2 omega (1-f)/f`,
about 1.1 TeV at `omega_max = 172.2 GeV`, even a full backscatter shifts a
photon by less than its bin width, so `sigma_rem` is exactly zero. This is a
physical result, not a failure, and it is what terminates the IGRB contour.

---

## 4. Reference spectra

**Halo.** Pixel-level MCMC posterior, `pixelwise_global_rho2`, disk excluded.
The loader offers about twenty variants and the name is not recorded in the
manuscript; this is the one.

**IGRB.** Ackermann et al. 2015, ApJ 799, 86 (arXiv:1410.3696), **published
machine-readable Table 3**, foreground model A, kept verbatim at
`data/ackermann2015_igrb_table3_mrt.txt`.

Table 3 tabulates band-integrated flux in cm⁻² s⁻¹ sr⁻¹. It is converted to
`E^2 dN/dE` at each band's log-centre to match the halo convention:

```
E2dNdE = E_c^2 * f_band / (E_hi - E_lo),   E_c = sqrt(E_lo * E_hi)
```

Per-bin errors are Table 3's **statistical + instrument** column (its note 1).
The foreground-modelling column is carried alongside but *not* added: it is a
single correlated choice of Galactic diffuse model across every bin, strongly
asymmetric, and not independent per-bin noise. Adding it in quadrature would
double-count a systematic the profiled normalisation already largely absorbs.

Validated by predicting each band flux from the paper's own Table 4 model-A fit
(I₁₀₀ = 0.95×10⁻⁷ MeV⁻¹cm⁻²s⁻¹sr⁻¹, γ = 2.32, E_cut = 279 GeV): rms pull 0.30,
total 7.19×10⁻⁶ against the published (7.2 ± 0.6)×10⁻⁶ above 100 MeV.

> Earlier releases carried a reconstructed spectrum here that was not Table 3
> and whose shape was wrong — fitted index 2.88 against the published 2.32.
> See `docs/FINDING_2026-08-13_igrb_spectrum_not_table3.md` in the analysis
> repository. Any result quoted against that spectrum is superseded.

`omega_max = 172.2 GeV` is the log-centre of the highest band surviving the
200 GeV truncation. It is not the truncation energy.

---

## 5. Cross-dataset agreement

Halo and IGRB agree within a factor of **1.4** across every operator below
m_χ ~ 100 GeV, flat in mass. The ratio climbs above ~300 GeV as the IGRB
approaches its closure at 1.1 TeV, because the IGRB bound weakens there while
the halo bound, which retains angular deflection, does not.

Reproduce with `compare_igrb_to_halo.py` in the analysis repository.

---

## 6. The exclusion threshold

Δχ² = 4.61 is adopted, not assumed correct. Wilks' conditions fail here: τ ≥ 0
puts the null on a boundary, and m_χ is unidentified under the null. Toys drawn
from the per-bin posterior chains (`calibration/`) give a calibrated 90%
quantile of 3.51–4.02 across twelve contour points, so 4.61 over-covers at
92–95% against a nominal 90%. It is retained over the calibrated 3.84 because
the scan-referenced statistic would otherwise exclude the no-scattering
hypothesis itself in 2.7% of experiments.

**Trim the chains on log-probability before drawing.** The stored `emcee`
output includes pre-convergence steps; untrimmed, the variance inflates about
ninefold. Stored percentiles are unaffected, which is why it is easy to miss.

---

## 7. Other quoted numbers

| number | source |
|---|---|
| truncation cost 16% (dim-7), 5.2% (dipole), 5.7% (scalar) | refined-grid rerun; the production Λ grid samples at 26%/step and cannot measure these |
| disk systematic 0.040 dex, 8.8% in Λ | full rescan on the disk-included posterior |
| J_ROI ≈ 4.8×10²² GeV cm⁻² (ρ²), 5.4×10²² (ρ^2.5) | ROI integration, `core/attenuation_eft.py` |
| J_cosmo = 1.37×10²² GeV cm⁻² | ρ_χ = 1.2×10⁻⁶ GeV cm⁻³, L = 1.14×10²⁸ cm |
| f = 0.231 | fractional loss carrying a photon from a bin centre to its lower edge on the native grid, ratio 1.68903 |
| w(θ) half-recovery 48° | `core/roi_recovery.py`, ρ² template over the ROI |
| Λ_therm = 600 / 320 / 5 GeV | **hard-coded constants** in the figure driver, not computed by any script here |

---

## 8. Data provenance guard

`constraints_data/limits.py` refuses any constraint file whose own header admits
it is not a digitisation. Fifteen files are currently refused and their curves
are absent from the figures by design. There is deliberately **no override
flag**: if a curve is wanted back, digitise it properly.

The guard covers `constraints_data/` only. It did **not** cover
`core/spectrum_source.py`, which is how a reference spectrum carrying a
"verify before final submission" note in its own metadata loaded silently for
several releases. Extending it to reference spectra would close that class.

The scalar-Rayleigh curves are correctly attributed to **Figure 1** of Barducci
et al. (not Figure 9) and carry only 4–7 points each; they are flagged as below
publication grade in that directory's README.
