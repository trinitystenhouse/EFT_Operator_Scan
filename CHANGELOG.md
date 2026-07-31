# Changelog

## v2 — Scalar Rayleigh normalisation correction

**Anyone using the v1 archive (doi:10.5281/zenodo.21703810) should update.**
The scalar Rayleigh limits in v1 are too weak by a factor of 2.83 in Lambda.
All other operators are unaffected.

### Corrected

`core/attenuation_eft.py::dsigma_dOmega_scalar` computed the spin-averaged
amplitude as

    |M|^2 = c_phi^2 t^2 / (4 Lambda^4)

which is 64x below the value implied by the operator it is meant to represent,

    O = (c_phi / Lambda^2) phi^2 F_mu_nu F^mu_nu.

The correct initial-photon-averaged result is `16 c_phi^2 t^2 / Lambda^4`. The
combinatorial factors are 2 from the `phi^2` contraction, 2 from assigning the
two photons to the two field-strength tensors, and 2 from the
`F^(1).F^(2)` contraction, giving `M = (8 c_phi / Lambda^2) [(k1.k2)(e1.e2) -
(k1.e2)(k2.e1)]`; summing over final and averaging over the two initial photon
polarisations with `sum_pol |...|^2 = 2 (k1.k2)^2` and `k1.k2 = -t/2` yields the
factor of 16.

The real-scalar Rayleigh operator carries no factor of 1/4, following Barducci
et al., arXiv:2501.09073 (Eqs. 2.1/2.3/2.5). This differs deliberately from the
fermionic Rayleigh operators in the same code, which retain the 1/4 of Weiner &
Yavin, arXiv:1206.2910, so that Lambda coincides with their Lambda_R.

### Regenerated

The three scalar boundary grids the paper figures use were rescanned, each on
its original grid so that the normalisation is the only change:

- `mcmc_pixelwise_global_rho2_halo_...` (100 x 100)
- `mcmc_pixelwise_global_rho2.5_halo_...` (32 x 32)
- `igrb_ackermann2015a_measured_...` (32 x 32)

Figures 2, 3 and 4 were rebuilt from them. Fermionic grids are byte-identical
to v1; Figures 1 and 5 are unchanged.

### Verification

On the identical 100 x 100 halo grid, the optical depth scaled by exactly 64
and the 90% CL boundary by `64^(1/4) = 2.83`, to within 4.5% — sub-cell on a
0.101 dex Lambda axis. The scalar reach moves from Lambda ~ 0.16 to ~0.44 GeV.
It remains below the EFT-validity wedge for `m_chi >= 1 MeV` by 0.38 dex, and
enters the valid region only over 1-95 keV, far below the cold-dark-matter
floor. No conclusion of the accompanying paper changes.

`tests/test_eft_realphoton_amplitudes.py::test_scalar_rayleigh_barducci_normalisation`
pins the corrected value; the scalar frozen reference was updated and the two
fermionic frozen references were left untouched.

### Note on regenerating

`constraint_generation/make_data_driven_scattering_limits.py` defaults to
`--nm 32 --nl 32`. The published halo grids are 100 x 100. Pass the grid
arguments explicitly when reproducing, or the contour will be quantised at a
coarser resolution than the paper's; see REPRODUCE.md.

## v1 — Initial public release

doi:10.5281/zenodo.21703810. Superseded by v2 for scalar Rayleigh results.
