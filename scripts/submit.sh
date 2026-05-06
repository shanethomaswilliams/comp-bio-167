#!/bin/bash
# ╔═══════════════════════════════════════════════════════════════╗
# ║  CAFA-5 Submission — Edit these variables, then run:          ║
# ║  ./submit.sh                                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

# ── EDIT THESE ──────────────────────────────────────────────────
TSV_PATH="/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/go_dag_output/logistic_regression_propagated.tsv"
NAME="logreg_godag_v2"
DESCRIPTION=""
# ────────────────────────────────────────────────────────────────

# Optional overrides
USERNAME="shanetwilliams"
RETRIES=20   # max push attempts while waiting for dataset
INTERVAL=30  # seconds between retries (use 10 if versioning an existing dataset)

# ── Don't edit below ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CMD="python3 \"${SCRIPT_DIR}/submit_cafa5.py\" \
  \"${TSV_PATH}\" \
  --name \"${NAME}\" \
  --description \"${DESCRIPTION}\" \
  --username \"${USERNAME}\" \
  --retries ${RETRIES} \
  --interval ${INTERVAL}"

echo "Running: ${CMD}"
echo ""
eval ${CMD}