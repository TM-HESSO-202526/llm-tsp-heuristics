#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

SIGNAL_MODE="${SIGNAL_MODE:-all}"
INSTANCE_ROOT="${INSTANCE_ROOT:?INSTANCE_ROOT is required}"
CANDIDATE_CACHE_DIR="${CANDIDATE_CACHE_DIR:-}"
EDGE_PRIOR_CACHE_DIR="${EDGE_PRIOR_CACHE_DIR:-${EDGE_PRIOR_DIR:-}}"
OPTIMA_CSV="${OPTIMA_CSV:-data/tsp_instances_opt.csv}"
OUT_ROOT="${OUT_ROOT:-$HOME/workspace/TM/final-results/tsp_eval}"
OUT_DIR="${OUT_DIR:-}"
RESUME="${RESUME:-0}"
REPS="${REPS:-2}"
MAX_HEURISTICS="${MAX_HEURISTICS:-1000}"
MAX_INSTANCES="${MAX_INSTANCES:-1000}"
TIMEOUT_S="${TIMEOUT_S:-300}"
INSTANCES="${INSTANCES:-ALL}"
SPLITS="${SPLITS:-all}"
GLOBAL_SEED="${GLOBAL_SEED:-12345}"
MAX_CANDIDATES="${MAX_CANDIDATES:-20}"
PRIOR_MODE="${PRIOR_MODE:-frequency}"
ALLOW_INTERFACE_MISMATCH="${ALLOW_INTERFACE_MISMATCH:-0}"

mkdir -p "$OUT_ROOT"

echo "Signal mode:     $SIGNAL_MODE"
echo "Repetitions:     $REPS"
echo "Max heuristics:  $MAX_HEURISTICS"
echo "Max instances:   $MAX_INSTANCES"
echo "Instances:       $INSTANCES"
echo "Splits:          $SPLITS"
echo "Timeout seconds: $TIMEOUT_S"
echo "Instance root:   $INSTANCE_ROOT"
echo "Candidate cache: ${CANDIDATE_CACHE_DIR:-none}"
echo "Edge-prior dir:  ${EDGE_PRIOR_CACHE_DIR:-none}"
if [ -n "$OUT_DIR" ]; then
  echo "Output folder:   $OUT_DIR"
fi
if [ "$RESUME" = "1" ]; then
  RESUME_FLAG="--resume"
else
  RESUME_FLAG=""
fi

if [ "$ALLOW_INTERFACE_MISMATCH" = "1" ]; then
  INTERFACE_FLAG="--allow-interface-mismatch"
else
  INTERFACE_FLAG=""
fi

python server_eval/run_selected_tsp_eval.py \
  --signal-mode "$SIGNAL_MODE" \
  --selected-root "experiments/selected_tsp_heuristics_final_by_signal" \
  --instance-root "$INSTANCE_ROOT" \
  --candidate-cache-dir "$CANDIDATE_CACHE_DIR" \
  --edge-prior-cache-dir "$EDGE_PRIOR_CACHE_DIR" \
  --optima-csv "$OPTIMA_CSV" \
  --instances "$INSTANCES" \
  --splits "$SPLITS" \
  --repetitions "$REPS" \
  --max-heuristics "$MAX_HEURISTICS" \
  --max-instances "$MAX_INSTANCES" \
  --timeout-s "$TIMEOUT_S" \
  --global-seed "$GLOBAL_SEED" \
  --max-candidates "$MAX_CANDIDATES" \
  --prior-mode "$PRIOR_MODE" \
  --output-root "$OUT_ROOT" \
  --output-dir "$OUT_DIR" \
  $RESUME_FLAG \
  $INTERFACE_FLAG
