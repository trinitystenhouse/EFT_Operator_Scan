#!/usr/bin/env bash
# Reproduce the figures, and optionally the production grids and toy outputs, of
#   "Photon-Dark Matter Elastic Scattering: An Effective-Operator Scan and First
#    Operator-Resolved Sensitivity Estimates from the Galactic Halo"
#
# Usage, from the root of this deposit:
#   bash scripts/reproduce_production.sh figures     # default: redraw all five figures
#   bash scripts/reproduce_production.sh all         # regenerate everything first
#
# Variables (all optional):
#   PYTHON       interpreter. Defaults to python3. Needs numpy, scipy,
#                matplotlib and contourpy.
#   CHAINS_ROOT  ONLY for "all". Directory containing
#                Totani_reanalysis/mcmc/fit_results/pixelwise_mcmc/. The per-bin
#                posterior chains are NOT part of this deposit: they belong to
#                the companion paper (arXiv:2607.08552) and are available with
#                that work. Exported as REPO_PATH, which
#                core/totani_data_loader.py reads.
#
# "figures" needs nothing but this deposit. "all" needs the companion's chains,
# because both the grids and the toys are built from the per-bin posteriors.
#
# Approximate wall times on one Apple silicon laptop core per job:
#   halo grid at nl = 1200            15 to 25 min each, ten grids
#   IGRB grid at nl = 1200            about 50 min each, three grids
#   toy calibration (tau caches)      70 to 80 min per profile
#   signal-hypothesis coverage        about 2 h per profile
#   expected bands                    about 10 min per profile
#   figures                           a few minutes
# Serially, "all" takes roughly 12 to 14 hours.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODE="${1:-figures}"
cd "${ROOT}"

GEN=constraint_generation/make_data_driven_scattering_limits.py
TOYS=constraint_boundaries/toys
PROFILES=(pixelwise_global_rho2 pixelwise_global_rho2.5)

HALO_OPS=(
    "fermionic dipole_magnetic"
    "scalar scalar_rayleigh"
    "fermionic rayleigh_even --majorana"
    "fermionic rayleigh_odd --majorana"
    "fermionic rayleigh_full --majorana"
)
IGRB_OPS=(
    "fermionic dipole_magnetic"
    "fermionic rayleigh_full --majorana"
    "scalar scalar_rayleigh"
)
HALO_COMMON=(
    --dataset halo --source measured
    --sigma-model closed_form --kernel-nodes exact_u --roi-recovery-model template
    --fit-normalization --drop-lowest-bins 2 --e-max-fit 200 --delta-chi2 4.61
    --nm 100 --nl 1200 --out-suffix _profnorm_removalfix --no-plot
)
# The IGRB uses unity ROI recovery, not template: for an isotropic source against
# a uniform cosmological scatterer, out-scattering is compensated by in-scattering,
# so angular escape removes nothing and energy migration is the only channel.
# f_bin is NOT passed: it is derived from each dataset's own bin edges
# (core/bin_geometry.py), which is 0.2305 for the halo grid and 0.1591 for the
# IGRB's.
IGRB_COMMON=(
    --dataset igrb --source measured
    --sigma-model closed_form --kernel-nodes exact_u --roi-recovery-model unity
    --fit-normalization --e-max-fit 200 --delta-chi2 4.61
    --nm 100 --nl 1200 --out-suffix _profnorm_removalfix_igrbfix2 --no-plot
)

if [ "${MODE}" = "all" ]
then
    if [ -n "${CHAINS_ROOT:-}" ]; then export REPO_PATH="${CHAINS_ROOT}"; fi
    if [ -z "${REPO_PATH:-}" ]
    then
        echo "ERROR: regenerating grids and toys needs the companion paper's per-bin" >&2
        echo "posterior chains, which are not in this deposit. Set CHAINS_ROOT." >&2
        exit 2
    fi

    echo "[1/4] halo grids, ten runs"
    for profile in "${PROFILES[@]}"; do
        for spec in "${HALO_OPS[@]}"; do
            read -r -a parts <<< "${spec}"; extra=("${parts[@]:2}")
            echo "  halo ${profile} ${parts[1]} ${extra[*]:-}"
            "${PYTHON}" "${GEN}" "${HALO_COMMON[@]}" --halo-profile "${profile}" \
                --dm-type "${parts[0]}" --operator "${parts[1]}" ${extra[@]+"${extra[@]}"}
        done
    done

    echo "[2/4] IGRB grids, three runs"
    for spec in "${IGRB_OPS[@]}"; do
        read -r -a parts <<< "${spec}"; extra=("${parts[@]:2}")
        echo "  igrb ${parts[1]} ${extra[*]:-}"
        "${PYTHON}" "${GEN}" "${IGRB_COMMON[@]}" \
            --dm-type "${parts[0]}" --operator "${parts[1]}" ${extra[@]+"${extra[@]}"}
    done

    echo "[3/4] toy calibration and tau caches (seed 20260807), then coverage"
    "${PYTHON}" "${TOYS}/run_calibration.py" 20000 --nl 1200
    "${PYTHON}" "${TOYS}/run_calibration.py" 20000 --nl 1200 --profile pixelwise_global_rho2.5
    # run_expected.py for rho^2 reads coverage.npz, so coverage runs first.
    "${PYTHON}" "${TOYS}/run_coverage.py" 20000                                    # seed 20260808
    "${PYTHON}" "${TOYS}/run_coverage.py" 20000 --profile pixelwise_global_rho2.5
    "${PYTHON}" "${TOYS}/run_expected.py" 10000                                    # seed 20260809
    "${PYTHON}" "${TOYS}/run_expected.py" 10000 --profile pixelwise_global_rho2.5
fi

echo "[figures] all five"
# --only N builds a subset and reproduces the published file exactly, for any N.
# The figure numbers follow the manuscript.
#
# No environment variable is needed. The IGRB grid tag defaults to
# _profnorm_removalfix_igrbfix2 inside make_paper_results_figures.py, and each
# curve is checked against the f_bin recorded in its grid, so a grid with
# inconsistent binning aborts the build instead of being plotted.
"${PYTHON}" make_paper_results_figures.py --out-dir paper_plots

echo "done. Figures are in ${ROOT}/paper_plots"
