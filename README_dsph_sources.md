# Velocity-dependent dSph limits — usage

Adds p-wave / d-wave / Sommerfeld dwarf-spheroidal limits to the tension scan,
plus a source switch and the `np.inf -> np.nan` out-of-range fix.

New files:
- `core/dsph_limits_multi.py` — source-aware loader (drop-in superset of `core/dsph_limits.py`).
- `extract_boddy_limits.py` — builds the velocity-dependent limit tables.
- `data/boddy2019_<wave>_<channel>.txt` — generated tables (swave/pwave/dwave/sommerfeld × bb/tautau/WW).
- `tests/test_dsph_limits_multi.py` — unit tests.

## 1. Sources

| source | meaning |
|---|---|
| `hoof` | Hoof et al. 2020 s-wave (legacy default) |
| `mcdaniel` | McDaniel et al. 2024 s-wave (bb/tautau; Hoof fallback) |
| `boddy_swave` | Boddy 2019 s-wave (passthrough rescale, factor 1) |
| `boddy_pwave` | p-wave, σv ∝ v² |
| `boddy_dwave` | d-wave, σv ∝ v⁴ |
| `boddy_somm` | Sommerfeld (Coulomb), σv ∝ 1/v |

Out-of-range masses now return **`np.nan`** (the old `np.inf` gave `tension = sv/inf = 0`,
i.e. a spurious "resolved" — the suspected origin of the old "resolved above ~650 GeV" result,
especially with McDaniel's 1 TeV ceiling vs a 2 TeV scan).

## 2. What the tables store

Each velocity-dependent table is the **halo-comparable** 95% UL on the *halo-frame* ⟨σv⟩
implied by the dwarf non-detection under that velocity law, so the existing
`tension = ⟨σv⟩_halo / ⟨σv⟩_dSph_UL` stays a like-for-like ratio. Derivation in the
`extract_boddy_limits.py` docstring. Closed forms (u ≡ ⟨(v/c)²⟩):

```
s-wave : factor = 1
p-wave : factor = u_halo / u_dwarf              (EXACT in J-weighted moments)
d-wave : factor = (u_halo / u_dwarf)² · (K_h/K_d)
Somm   : factor = sqrt(u_dwarf / u_halo)        (strengthens the limit)
```

## 3. Real-data workflow (publication-grade)

Both velocity moments come from real data: dwarf side from Boddy 2019 posteriors, halo side
from an isotropic Jeans solve on the NFW you fit. Three steps:

```
# (0) one-time: get the Boddy J-factor posteriors (git-lfs)
#     git clone https://github.com/apace7/J-Factor-Scaling.git   (placed at ../J-Factor-Scaling)
#     cd J-Factor-Scaling && git lfs install && git lfs pull

# (1) dwarf moments: stacked J-weighted <v^2>,<v^4>,<1/v> over 25 dwarfs at 0.5 deg
python extract_boddy_moments.py --aperture 0d5            # -> boddy_moments_0d5.csv

# (2) halo moments: rho^2-weighted relative-velocity moments over the ROI, SAME NFW
python compute_u_halo.py --gamma 1 --rs 21 --rvir 402 --r-sun 8 --rho-sun 0.38 \
       --baryons hernquist --out halo_moments_bary.csv     # -> u_halo_*

# (3) build the halo-comparable limit tables: factor_n = <m_n>_halo / <m_n>_dSph
python extract_boddy_limits.py --mode boddy \
       --boddy-csv boddy_moments_0d5.csv --halo-csv halo_moments_bary.csv
```

Real moments (0.5 deg dwarfs; NFW+baryons halo): u_dwarf_pwave = 9.71e-9 (v_rms 29.5 km/s);
u_halo_pwave = 1.92e-6 (v_rel,rms 415 km/s); => p-wave factor ≈ 197.

Full-scan tension with the real tables (rho2, spectral best-fit mass), in
`results/tension_velocity/<source>/`:

| channel | s-wave (hoof) | p-wave | d-wave | Sommerfeld |
|---|---|---|---|---|
| bb     | 4.33× | 0.022× | ~0 | 85× |
| WW     | 3.36× | 0.017× | ~0 | 66× |
| tautau | 2.72× | 0.014× | ~0 | 53× |

p-wave **over-resolves** the dwarf tension by ~20–50× (robust to the halo potential, see below);
Sommerfeld-alone makes it far worse. So the binding constraint on p-wave is relic abundance, not
dwarfs (see `velocity_dependent_dsph_and_dm_models.md` §2–3).

### Model choice (answer: don't change the density, fix the potential)
- **Reuse the same (g)NFW** you fit — consistency between the rho^2 weight and the velocity field
  requires the same rho(r). Use the same gamma as the fit you quote (gamma=1, fig6).
- **Include baryons in the Jeans potential.** NFW self-gravity alone gives sigma_r(R_sun)=124 km/s
  (below the observed local ~150–160); +baryons gives 161, matching. The rho^2-weighted signal comes
  from r_eff ≈ 5.6 kpc where disk+bulge dominate, so baryons raise u_halo ~2×. The p-wave conclusion
  (<1) is unchanged either way; for the paper, replace the Hernquist proxy with a full MW mass model
  (McMillan 2017 / galpy).
- Systematics that set the error band (none flip p-wave above 1): isotropy (beta=0), rho_sun
  normalisation, the baryon model, and Boddy's J-factor posterior spread. p-wave's u_halo is rigorous
  (<v_rel^2>=6 sigma^2 needs no shape assumption); d-wave/Sommerfeld use a Maxwellian closure — swap to
  Eddington for shape-exact d/Sommerfeld numbers.

### Consistency: s-wave limits on s-wave data, p-wave on p-wave
`python verify_dsph_consistency.py` asserts the pairing so a mismatch can't slip in. The decomposition:
- the **absolute Fermi sensitivity** is a *velocity-independent photon-flux* limit, taken from the
  s-wave dwarf analysis (Hoof, the `--s-source` anchor) — correct for every wave because dN/dE is the
  same;
- the **velocity weighting** is that wave's own data: dwarf `u_dwarf_n = j_n/j_s` (Boddy p/d/Sommerfeld
  effective J) and halo `u_halo_n` (Jeans).

So `L_n = sigma_v0_UL · u_halo_n/u_dwarf_n`, and the scan tension reduces to
`t_n = a_n j_n / (sigma_v0_UL j_s)` = predicted *n-wave* dwarf signal over the *n-wave* dwarf limit.
The audit confirms `boddy_swave == hoof` and `boddy_pwave` is a distinct table (×197), per wave/channel/mass.

Anchor justification (checked against the paper, arXiv:1909.13197):
- Boddy state their **s-wave limits are "consistent with those found previously in the literature"**
  (their Sec. IV). Hoof is a member of that same literature, so the Hoof s-wave anchor is consistent
  with Boddy's own s-wave analysis; the residual is within J-factor scatter, not a systematic offset.
- **Velocity convention confirmed:** Boddy write sigma_A v = (sigma_A v)_0 S(v/c) with v the *relative*
  velocity of the pair (their Sec. II). This matches the halo treatment <v_rel^2> = 6 sigma^2, so the
  dwarf and halo moments are on the same footing (would be 2x off if one used single-particle v).
- The repo ships **J-factors only** (Table I + posteriors), not the limit curves; Boddy made the limits
  with the public **MADHAT** code (ref [12]). The *fully* self-consistent anchor is therefore to
  regenerate Boddy's s-wave limit by running MADHAT on the released J-factors (exact j_s cancellation),
  rather than digitising a figure. Short of that, the Hoof anchor is validated above. If you do produce
  a Boddy s-wave curve, drop it as `data/boddy2019_swave_<ch>.txt` and rebuild the velocity tables with
  `--s-source boddy_swave --waves pwave dwave sommerfeld` (won't overwrite your swave file).

## 4. Analytic quick mode (fallback, no Boddy data needed)

```
python extract_boddy_limits.py --v-halo-kms 175 --v-dwarf-kms 7     # provisional moments
```
Right physics, provisional numbers; table headers carry a `provisional analytic moments` note. Use
only for exploration — quote the §3 real-data tables in the paper.

## 5. `--dsph-source` is already wired (no edits needed)

The switch is live in every script that computes tension; `core/dsph_limits.py` is now a thin
re-export of `dsph_limits_multi` so all importers get the Boddy sources + the nan fix at once:

| script | how to use |
|---|---|
| `scan_tension_resolution.py` | `--dsph-source boddy_pwave` |
| `scan_deconvolved_scattering_fit.py` | `--dsph-source boddy_pwave` |
| `Totani_paper_check/run_pppc_mass_scan.py` | `--dsph-source boddy_pwave` |
| `Totani_paper_check/figures/plot_pppc_mass_scan.py` | `--dsph-source boddy_pwave` |
| `Totani_paper_check/figures/plot_scattering_summary.py` | `--dsph-source boddy_pwave` |

`plot_scan_overview.py` / `compile_best_fits.py` need nothing — they read tension from the NPZ
the scans write, so they inherit whatever source produced them.

Example (reproduces the §3 table):
```
python scan_tension_resolution.py --ann-channel bb,WW,tautau --halo-profile rho2 \
  --ann-mass-min 100 --ann-mass-max 5000 --n-ann-mass 80 --cl-threshold 0.95 \
  --dsph-source boddy_pwave --output-dir results/tension_velocity/boddy_pwave
```

Note: under a p-wave hypothesis the *halo* ⟨σv⟩ you feed in is itself a·⟨v²⟩_halo, and the
relic-consistency check (the note's §2: thermal p-wave undershoots the halo by ~1e6) is a separate
calculation — the dwarf tension and the relic budget are independent constraints.

## 5b. New scripts (this work)
- `extract_boddy_moments.py` — Boddy `.npy` posteriors -> stacked dwarf moments CSV.
- `compute_u_halo.py` — isotropic Jeans on the NFW -> ROI rho^2-weighted halo moments CSV.
- `extract_boddy_limits.py` — `--mode boddy --boddy-csv … --halo-csv …` -> halo-comparable tables
  (analytic `--v-halo-kms` fallback retained).

## 6. References
- Boddy, Pace, Runburg, Strigari 2019, PRD 102, 023029 — arXiv:1909.13197 (effective J-factors).
- Boddy et al. 2017, PRD 95, 123008 — arXiv:1702.00408 (Sommerfeld J-factors).
- Hoof, Geringer-Sameth, Trotta 2020 — arXiv:1812.06986. McDaniel et al. 2024 — arXiv:2311.04982.
