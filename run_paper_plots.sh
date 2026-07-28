#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_paper_plots.sh — regenerate every figure in the paper.
#
# Thin wrapper around make_paper_results_figures.py, which is the single
# source of truth for all five figures. The digitised exclusion grids in
# constraint_boundaries/ and the observed halo spectrum in data/ ship with
# the repository, so this reproduces Figs 1-5 with no external download.
#
# Usage:
#   bash run_paper_plots.sh                 # all figures -> paper_plots/
#   bash run_paper_plots.sh --only 3 4      # a subset
#   PAPER_PLOTS_DIR=/path/out bash run_paper_plots.sh
# ---------------------------------------------------------------------------
set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${PAPER_PLOTS_DIR:-$REPO_ROOT/paper_plots}"

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT" python3 make_paper_results_figures.py --out-dir "$OUT_DIR" "$@"
