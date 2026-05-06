#!/usr/bin/env bash
#
# Run all Tier 1 and Tier 2 methods end-to-end.
#
#   Tier 1 (mean, max, rank_avg)           — no-training combiners
#   Tier 2 (weighted_mean, logreg, xgb, mlp) — learned stackers, with --tune
#
# Usage:
#   ./run_all.sh                              # default run name (timestamp)
#   ./run_all.sh --run-name my_experiment     # custom run name
#   ./run_all.sh --no-tune                    # skip Tier 2 hyperparam search
#   ./run_all.sh --data-dir data/final_data_v2 --run-name v2_baseline
#
# Submissions + logs land in:
#   results/<run-name>/{method}_test_submission.tsv
#   results/<run-name>/logs.txt

set -euo pipefail

# ---- defaults ---------------------------------------------------------------

RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="data/final_data"
TUNE="--tune"

TIER1_METHODS=(mean max rank_avg)
TIER1_METHODS=(rank_avg)
# TIER2_METHODS=(weighted_mean logreg xgb mlp)
TIER2_METHODS=()

# ---- parse args -------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)   RUN_NAME="$2";  shift 2 ;;
    --data-dir)   DATA_DIR="$2";  shift 2 ;;
    --no-tune)    TUNE="";        shift   ;;
    -h|--help)
      grep '^#' "$0" | head -n 20 | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      echo "run './run_all.sh --help' for usage" >&2
      exit 1 ;;
  esac
done

# ---- sanity ----------------------------------------------------------------

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data dir '$DATA_DIR' not found." >&2
  echo "Did you run 'python -m src.create_ensemble_datasets ...' first?" >&2
  exit 1
fi

RESULTS_DIR="results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

echo "===================================================================="
echo "Run name:   $RUN_NAME"
echo "Data dir:   $DATA_DIR"
echo "Tuning:     ${TUNE:-(disabled)}"
echo "Results:    $RESULTS_DIR/"
echo "===================================================================="

# ---- Tier 1 ----------------------------------------------------------------

for method in "${TIER1_METHODS[@]}"; do
  echo ""
  echo ">>> TIER 1 :: $method"
  python -u -m main \
    --tier 1 --method "$method" \
    --data-dir "$DATA_DIR" \
    --run-name "$RUN_NAME"
done

# ---- Tier 2 ----------------------------------------------------------------

for method in "${TIER2_METHODS[@]}"; do
  echo ""
  echo ">>> TIER 2 :: $method ${TUNE:+(tuned)}"
  python -u -m main \
    --tier 2 --method "$method" \
    --data-dir "$DATA_DIR" \
    --run-name "$RUN_NAME" \
    $TUNE
done

echo ""
echo "===================================================================="
echo "Done. All results under $RESULTS_DIR/"
echo "  $(ls "$RESULTS_DIR"/*.tsv 2>/dev/null | wc -l | tr -d ' ') submission files"
echo "  log: $RESULTS_DIR/logs.txt"
echo "===================================================================="