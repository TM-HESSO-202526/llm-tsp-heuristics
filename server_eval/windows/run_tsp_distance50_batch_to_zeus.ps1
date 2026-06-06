# ============================================================
# Final TSP distance-only 50-repetition batch launcher for IICT Zeus
#
# One command controls the whole final distance-only TSP protocol:
#   launch   -> starts/resumes one master scheduler on zeus
#   status   -> reports all selected LLM heuristics and external baselines
#   download -> downloads the whole run folder, including partial artifacts
#
# Usage from Windows PowerShell:
#   cd D:\Users\antho\TM\llm-tsp-heuristics\server_eval\windows
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#
#   .\run_tsp_distance50_batch_to_zeus.ps1 -Action launch -StartNewRun
#   .\run_tsp_distance50_batch_to_zeus.ps1 -Action status
#   .\run_tsp_distance50_batch_to_zeus.ps1 -Action download
#
# Protocol encoded here:
# - distance-only signal regime
# - 50 repetitions
# - 14 TSP instances up to pla85900
# - 10 cores concurrently
# - one core = one job
# - one job = one selected LLM heuristic or one external baseline over all instances/repetitions
# - automatic queue filling: when one job finishes, the scheduler starts the next pending job
# ============================================================

param(
    [ValidateSet("launch", "status", "download")]
    [string]$Action = "status",

    [switch]$StartNewRun,

    [switch]$NoUploadInputs,

    [switch]$NoGitPull,

    [switch]$NoSetupEnv,

    [switch]$DryRun
)

# ------------------------------
# User / server settings
# ------------------------------
$AAI_USERNAME = "anthony.atallah"
$SERVER_NAME = "zeus"
$REPO_URL = "https://github.com/TM-HESSO-202526/llm-tsp-heuristics.git"

# ------------------------------
# Private local inputs on your PC
# ------------------------------
$LOCAL_INPUT_DIR = "D:\Users\antho\TM\server_eval_tsp_inputs"
$LOCAL_TSP_INSTANCE_DIR = "$LOCAL_INPUT_DIR\TSP_instances"
$LOCAL_RESULTS_DIR = "D:\Users\antho\TM\server_eval_results"

# ------------------------------
# Final protocol settings
# ------------------------------
$RUN_LABEL = "distance50_selected_and_baselines"
$SIGNAL_MODE = "distance_only"
$REPS = 50
$INSTANCES = "dsj1000,pr1002,d1291,fl1400,pcb1173,rl1304,u1817,rl1889,pr2392,pcb3038,pla7397,usa13509,pla33810,pla85900"
$SPLITS = "all"
$INSTANCE_COUNT = 14
$EXPECTED_TASKS = $INSTANCE_COUNT * $REPS
$MAX_INSTANCES = 1000
$TIMEOUT_S = 7200
$GLOBAL_SEED = 12345

# 10 cores out of 40. Change only if needed.
$CORES_TO_USE = @(0,1,2,3,4,5,6,7,8,9)
$SCHEDULER_SLEEP_S = 60

# ------------------------------
# Remote paths
# ------------------------------
$REMOTE = "$AAI_USERNAME@$SERVER_NAME.iict-heig-vd.in"
$REMOTE_INPUT_DIR = "/home/$AAI_USERNAME/data-local/TM/tsp_input"
$REMOTE_INSTANCE_DIR = "$REMOTE_INPUT_DIR/TSP_instances"
$REMOTE_RESULTS_ROOT = "/home/$AAI_USERNAME/workspace/TM/final-results/tsp_distance50_batch"

function BoolToBash($b) { if ($b) { return "1" } else { return "0" } }

$START_NEW_RUN_BASH = BoolToBash $StartNewRun
$GIT_PULL_BASH = BoolToBash (-not $NoGitPull)
$SETUP_ENV_BASH = BoolToBash (-not $NoSetupEnv)
$DRY_RUN_BASH = BoolToBash $DryRun
$CORES_CSV = ($CORES_TO_USE -join ",")

Write-Host "=== TSP distance-only 50-repetition batch ==="
Write-Host "Remote:        $REMOTE"
Write-Host "Action:        $Action"
Write-Host "Run label:     $RUN_LABEL"
Write-Host "Signal mode:   $SIGNAL_MODE"
Write-Host "Reps:          $REPS"
Write-Host "Expected rows: $EXPECTED_TASKS per job"
Write-Host "Instances:     $INSTANCES"
Write-Host "Cores:         $CORES_CSV"
Write-Host "Start new run: $StartNewRun"
Write-Host "Dry run:       $DryRun"
Write-Host ""

if ($Action -eq "launch" -and (-not $NoUploadInputs)) {
    Write-Host "=== Local input checks ==="
    if (!(Test-Path $LOCAL_TSP_INSTANCE_DIR)) {
        Write-Host "ERROR: Missing local input folder: $LOCAL_TSP_INSTANCE_DIR"
        exit 1
    }
    foreach ($name in $INSTANCES.Split(",")) {
        $p = Join-Path $LOCAL_TSP_INSTANCE_DIR ("$name.tsp")
        if (!(Test-Path $p)) {
            Write-Host "ERROR: Missing TSP instance: $p"
            exit 1
        }
        Write-Host "OK: $p"
    }
}

Write-Host "=== Creating remote folders on $REMOTE ==="
ssh $REMOTE "mkdir -p /home/$AAI_USERNAME/workspace/TM $REMOTE_INSTANCE_DIR $REMOTE_RESULTS_ROOT"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Action -eq "launch" -and (-not $NoUploadInputs)) {
    Write-Host "=== Uploading TSP instances ==="
    foreach ($name in $INSTANCES.Split(",")) {
        $p = Join-Path $LOCAL_TSP_INSTANCE_DIR ("$name.tsp")
        scp "$p" "${REMOTE}:${REMOTE_INSTANCE_DIR}/"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

$remoteScriptTemplate = @'
#!/usr/bin/env bash
set -euo pipefail

AAI_USERNAME="__AAI_USERNAME__"
REPO_URL="__REPO_URL__"
ACTION="__ACTION__"
RUN_LABEL="__RUN_LABEL__"
SIGNAL_MODE="__SIGNAL_MODE__"
REPS="__REPS__"
EXPECTED_TASKS="__EXPECTED_TASKS__"
MAX_INSTANCES="__MAX_INSTANCES__"
TIMEOUT_S="__TIMEOUT_S__"
INSTANCES="__INSTANCES__"
SPLITS="__SPLITS__"
GLOBAL_SEED="__GLOBAL_SEED__"
CORES_CSV="__CORES_CSV__"
SCHEDULER_SLEEP_S="__SCHEDULER_SLEEP_S__"
START_NEW_RUN="__START_NEW_RUN__"
GIT_PULL="__GIT_PULL__"
SETUP_ENV="__SETUP_ENV__"
DRY_RUN="__DRY_RUN__"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

WORK_ROOT="/home/${AAI_USERNAME}/workspace/TM"
REPO_DIR="${WORK_ROOT}/llm-tsp-heuristics"
INPUT_DIR="/home/${AAI_USERNAME}/data-local/TM/tsp_input"
INSTANCE_ROOT="${INPUT_DIR}/TSP_instances"
OUT_ROOT="${WORK_ROOT}/final-results/tsp_distance50_batch"
LATEST_FILE="${OUT_ROOT}/LATEST_${RUN_LABEL}.txt"
SCHEDULER_SESSION="tsp50_scheduler"

mkdir -p "$WORK_ROOT" "$INSTANCE_ROOT" "$OUT_ROOT"

if [ "$ACTION" = "launch" ]; then
  cd "$WORK_ROOT"
  if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" llm-tsp-heuristics
  fi
  cd "$REPO_DIR"
  if [ "$GIT_PULL" = "1" ]; then
    git pull || true
  fi
  find server_eval -type f -name '*.sh' -exec sed -i 's/\r$//' {} \; 2>/dev/null || true
  if [ "$SETUP_ENV" = "1" ]; then
    bash server_eval/setup_server_env.sh
    source "/home/${AAI_USERNAME}/data-local/TM/venvs/tsp-final-eval/bin/activate" 2>/dev/null || true
    python -m pip install -q -r requirements.txt || true
    python -m pip install -q -e . || true
  fi
else
  cd "$REPO_DIR"
fi

source "/home/${AAI_USERNAME}/data-local/TM/venvs/tsp-final-eval/bin/activate" 2>/dev/null || true

if [ ! -d "$INSTANCE_ROOT" ]; then echo "ERROR: missing $INSTANCE_ROOT"; exit 2; fi

if [ "$ACTION" = "launch" ] && { [ "$START_NEW_RUN" = "1" ] || [ ! -f "$LATEST_FILE" ]; }; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_ROOT="${OUT_ROOT}/${RUN_LABEL}_${STAMP}"
  mkdir -p "$RUN_ROOT"
  echo "$RUN_ROOT" > "$LATEST_FILE"
elif [ -f "$LATEST_FILE" ]; then
  RUN_ROOT="$(cat "$LATEST_FILE")"
  mkdir -p "$RUN_ROOT"
else
  echo "ERROR: no LATEST run exists. Use -Action launch first."
  exit 2
fi

JOB_LIST="$RUN_ROOT/job_list.tsv"
JOB_STATE="$RUN_ROOT/job_state"
mkdir -p "$JOB_STATE"

write_job_list() {
  cat > "$JOB_LIST" <<'JOBS'
H_TSPD_02_normal_raw_nn2opt_best_101102_iter003	heuristic	distance_only	02_normal_raw_nn2opt_best_101102_iter003
H_TSPD_03_family_focus_grid_best_100159_iter072	heuristic	distance_only	03_family_focus_grid_best_100159_iter072
H_TSPD_04_family_focus_convex_faithful_095803_iter031	heuristic	distance_only	04_family_focus_convex_faithful_095803_iter031
H_TSPD_05_family_focus_voronoi_best_100159_iter037	heuristic	distance_only	05_family_focus_voronoi_best_100159_iter037
H_TSPD_07_family_focus_region_endpoint_fast_100159_iter177	heuristic	distance_only	07_family_focus_region_endpoint_fast_100159_iter177
H_TSPD_08_expo_distance_only_geostabilizer_399e	heuristic	distance_only	08_expo_distance_only_geostabilizer_399e
H_TSPD_09_family_focus_mst_diagnostic_100159_iter007	heuristic	distance_only	09_family_focus_mst_diagnostic_100159_iter007
H_TSPD_10_family_focus_fast_convex_095803_iter026	heuristic	distance_only	10_family_focus_fast_convex_095803_iter026
H_TSPD_11_family_focus_convex_constructive_095803_iter021	heuristic	distance_only	11_family_focus_convex_constructive_095803_iter021
B_TSPD_01_kdtree_nearest_neighbor_fixed_start	baseline	distance_only	01_kdtree_nearest_neighbor_fixed_start
B_TSPD_02_kdtree_nearest_neighbor_multistart	baseline	distance_only	02_kdtree_nearest_neighbor_multistart
B_TSPD_03_x_axis_sweep	baseline	distance_only	03_x_axis_sweep
B_TSPD_04_pca_sweep	baseline	distance_only	04_pca_sweep
B_TSPD_05_angular_sweep	baseline	distance_only	05_angular_sweep
B_TSPD_06_morton_z_order	baseline	distance_only	06_morton_z_order
B_TSPD_07_grid_serpentine	baseline	distance_only	07_grid_serpentine
B_TSPD_08_morton_bounded_local_2opt	baseline	distance_only	08_morton_bounded_local_2opt
JOBS
}

if [ "$ACTION" = "launch" ] && { [ "$START_NEW_RUN" = "1" ] || [ ! -f "$JOB_LIST" ]; }; then
  write_job_list
fi

if [ ! -f "$JOB_LIST" ]; then
  echo "ERROR: missing $JOB_LIST"
  exit 2
fi

cat > "$RUN_ROOT/run_config.json" <<EOF
{
  "run_label": "$RUN_LABEL",
  "signal_mode": "$SIGNAL_MODE",
  "repetitions": $REPS,
  "instances": "$INSTANCES",
  "splits": "$SPLITS",
  "expected_tasks_per_job": $EXPECTED_TASKS,
  "cores_csv": "$CORES_CSV",
  "timeout_s": $TIMEOUT_S,
  "global_seed": $GLOBAL_SEED,
  "run_root": "$RUN_ROOT",
  "hostname": "$(hostname)",
  "updated_at": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

sanitize_session() {
  local s
  s="tsp50_$1"
  echo "$s" | tr -c 'A-Za-z0-9_' '_' | cut -c1-80
}

row_count() {
  local job="$1"
  local raw="$RUN_ROOT/$job/raw_results.csv"
  if [ ! -f "$raw" ]; then echo 0; return; fi
  local lines
  lines=$(wc -l < "$raw" 2>/dev/null || echo 0)
  if [ "$lines" -le 0 ]; then echo 0; else echo $((lines - 1)); fi
}

is_session_alive() {
  local sess="$1"
  tmux has-session -t "$sess" 2>/dev/null
}

job_status() {
  local job="$1"
  local sess
  sess=$(sanitize_session "$job")
  local rows
  rows=$(row_count "$job")
  if [ "$rows" -ge "$EXPECTED_TASKS" ]; then
    echo "COMPLETE"
  elif is_session_alive "$sess"; then
    echo "RUNNING"
  elif [ "$rows" -gt 0 ]; then
    echo "MISSING/INCOMPLETE"
  else
    echo "PENDING"
  fi
}

active_job_sessions() {
  tmux ls 2>/dev/null | cut -d: -f1 | grep '^tsp50_' | grep -v '^tsp50_scheduler$' || true
}

active_count() {
  active_job_sessions | wc -l
}

free_core() {
  IFS=',' read -ra CORES <<< "$CORES_CSV"
  local used=""
  local sess
  while read -r sess; do
    [ -z "$sess" ] && continue
    if [ -f "$JOB_STATE/${sess}.core" ]; then
      used="$used $(cat "$JOB_STATE/${sess}.core")"
    fi
  done < <(active_job_sessions)
  local c
  for c in "${CORES[@]}"; do
    if ! echo " $used " | grep -q " $c "; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

cat > "$RUN_ROOT/run_one_job.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

JOB="$1"
KIND="$2"
OBJECTIVE="$3"
METHOD="$4"
CORE="$5"

AAI_USERNAME="__AAI_USERNAME__"
REPO_DIR="/home/${AAI_USERNAME}/workspace/TM/llm-tsp-heuristics"
INSTANCE_ROOT="/home/${AAI_USERNAME}/data-local/TM/tsp_input/TSP_instances"
RUN_ROOT="__RUN_ROOT__"
REPS="__REPS__"
EXPECTED_TASKS="__EXPECTED_TASKS__"
TIMEOUT_S="__TIMEOUT_S__"
INSTANCES="__INSTANCES__"
SPLITS="__SPLITS__"
GLOBAL_SEED="__GLOBAL_SEED__"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

source "/home/${AAI_USERNAME}/data-local/TM/venvs/tsp-final-eval/bin/activate" 2>/dev/null || true
cd "$REPO_DIR"

OUT_DIR="$RUN_ROOT/$JOB"
LOG="$RUN_ROOT/${JOB}.log"
mkdir -p "$OUT_DIR"

if [ "$KIND" = "heuristic" ]; then
  SELECTED_ROOT="experiments/selected_tsp_heuristics_final_by_signal"
elif [ "$KIND" = "baseline" ]; then
  SELECTED_ROOT="server_eval/tsp_external_baselines"
else
  echo "ERROR: unknown KIND=$KIND" | tee -a "$LOG"
  exit 2
fi

{
  echo "================================================================================================"
  echo "START $JOB kind=$KIND objective=$OBJECTIVE method=$METHOD core=$CORE"
  date
  hostname
  echo "OUT_DIR=$OUT_DIR"
  echo "EXPECTED_TASKS=$EXPECTED_TASKS"
  echo "INSTANCES=$INSTANCES"
  echo "REPS=$REPS"
  echo "TIMEOUT_S=$TIMEOUT_S"
  echo "SELECTED_ROOT=$SELECTED_ROOT"
  echo "================================================================================================"
} | tee -a "$LOG"

python server_eval/run_selected_tsp_eval.py \
  --signal-mode "$OBJECTIVE" \
  --selected-root "$SELECTED_ROOT" \
  --heuristic-ids "$METHOD" \
  --instance-root "$INSTANCE_ROOT" \
  --optima-csv "data/tsp_instances_opt.csv" \
  --instances "$INSTANCES" \
  --splits "$SPLITS" \
  --repetitions "$REPS" \
  --max-heuristics 1000 \
  --max-instances 1000 \
  --timeout-s "$TIMEOUT_S" \
  --dense-distance-threshold 20000 \
  --global-seed "$GLOBAL_SEED" \
  --output-dir "$OUT_DIR" \
  --resume \
  2>&1 | tee -a "$LOG"

{
  echo
  echo "================================================================================================"
  echo "DONE $JOB"
  date
  wc -l "$OUT_DIR/raw_results.csv" 2>/dev/null || true
  echo "================================================================================================"
} | tee -a "$LOG"
EOS

sed -i \
  -e "s#__AAI_USERNAME__#${AAI_USERNAME}#g" \
  -e "s#__RUN_ROOT__#${RUN_ROOT}#g" \
  -e "s#__REPS__#${REPS}#g" \
  -e "s#__EXPECTED_TASKS__#${EXPECTED_TASKS}#g" \
  -e "s#__TIMEOUT_S__#${TIMEOUT_S}#g" \
  -e "s#__INSTANCES__#${INSTANCES}#g" \
  -e "s#__SPLITS__#${SPLITS}#g" \
  -e "s#__GLOBAL_SEED__#${GLOBAL_SEED}#g" \
  "$RUN_ROOT/run_one_job.sh"
chmod +x "$RUN_ROOT/run_one_job.sh"

print_status() {
  echo "=== TSP DISTANCE-ONLY 50REPS STATUS ==="
  date
  hostname
  echo "RUN_ROOT=$RUN_ROOT"
  echo "JOB_LIST=$JOB_LIST"
  echo "EXPECTED_TASKS=$EXPECTED_TASKS"
  echo "CORES_CSV=$CORES_CSV"
  echo
  printf "%-72s %-10s %-15s %-54s %12s %s\n" "JOB" "KIND" "SIGNAL" "METHOD" "ROWS" "STATUS"
  printf '%0.s-' {1..180}; echo
  local complete=0 running=0 pending=0 incomplete=0 total=0
  while IFS=$'\t' read -r job kind objective method; do
    [ -z "$job" ] && continue
    total=$((total + 1))
    local rows st
    rows=$(row_count "$job")
    st=$(job_status "$job")
    case "$st" in
      COMPLETE) complete=$((complete + 1));;
      RUNNING) running=$((running + 1));;
      PENDING) pending=$((pending + 1));;
      *) incomplete=$((incomplete + 1));;
    esac
    printf "%-72s %-10s %-15s %-54s %5s/%-6s %s\n" "$job" "$kind" "$objective" "$method" "$rows" "$EXPECTED_TASKS" "$st"
  done < "$JOB_LIST"
  echo
  echo "TOTAL_JOBS=$total COMPLETE=$complete RUNNING=$running PENDING=$pending INCOMPLETE=$incomplete"
  echo
  echo "=== tmux sessions ==="
  tmux ls 2>/dev/null | grep '^tsp50_' || true
  echo
  echo "RUN_ROOT=$RUN_ROOT"
}

start_one_job() {
  local job="$1" kind="$2" objective="$3" method="$4" core="$5"
  local sess
  sess=$(sanitize_session "$job")
  echo "$core" > "$JOB_STATE/${sess}.core"
  echo "$(date '+%Y-%m-%d %H:%M:%S') START $job core=$core session=$sess" | tee -a "$RUN_ROOT/scheduler.log"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: would launch $job on core $core"
    return 0
  fi
  tmux new -d -s "$sess" "taskset -c '$core' bash '$RUN_ROOT/run_one_job.sh' '$job' '$kind' '$objective' '$method' '$core'"
}

write_scheduler() {
cat > "$RUN_ROOT/scheduler.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="__RUN_ROOT__"
JOB_LIST="$RUN_ROOT/job_list.tsv"
EXPECTED_TASKS="__EXPECTED_TASKS__"
CORES_CSV="__CORES_CSV__"
SLEEP_S="__SCHEDULER_SLEEP_S__"
JOB_STATE="$RUN_ROOT/job_state"
mkdir -p "$JOB_STATE"
cd "__REPO_DIR__"
source "/home/__AAI_USERNAME__/data-local/TM/venvs/tsp-final-eval/bin/activate" 2>/dev/null || true

sanitize_session() { local s="tsp50_$1"; echo "$s" | tr -c 'A-Za-z0-9_' '_' | cut -c1-80; }
row_count() { local raw="$RUN_ROOT/$1/raw_results.csv"; [ -f "$raw" ] || { echo 0; return; }; local lines; lines=$(wc -l < "$raw" 2>/dev/null || echo 0); [ "$lines" -le 0 ] && echo 0 || echo $((lines - 1)); }
is_session_alive() { tmux has-session -t "$1" 2>/dev/null; }
active_job_sessions() { tmux ls 2>/dev/null | cut -d: -f1 | grep '^tsp50_' | grep -v '^tsp50_scheduler$' || true; }
active_count() { active_job_sessions | wc -l; }
free_core() {
  IFS=',' read -ra CORES <<< "$CORES_CSV"
  local used="" sess c
  while read -r sess; do [ -z "$sess" ] && continue; [ -f "$JOB_STATE/${sess}.core" ] && used="$used $(cat "$JOB_STATE/${sess}.core")"; done < <(active_job_sessions)
  for c in "${CORES[@]}"; do if ! echo " $used " | grep -q " $c "; then echo "$c"; return 0; fi; done
  return 1
}
start_one_job() {
  local job="$1" kind="$2" objective="$3" method="$4" core="$5" sess
  sess=$(sanitize_session "$job")
  echo "$core" > "$JOB_STATE/${sess}.core"
  echo "$(date '+%Y-%m-%d %H:%M:%S') START $job core=$core session=$sess" | tee -a "$RUN_ROOT/scheduler.log"
  tmux new -d -s "$sess" "taskset -c '$core' bash '$RUN_ROOT/run_one_job.sh' '$job' '$kind' '$objective' '$method' '$core'"
}

while true; do
  total=0; done_jobs=0
  while IFS=$'\t' read -r job kind objective method; do
    [ -z "$job" ] && continue
    total=$((total + 1))
    rows=$(row_count "$job")
    sess=$(sanitize_session "$job")
    if [ "$rows" -ge "$EXPECTED_TASKS" ]; then
      done_jobs=$((done_jobs + 1))
      continue
    fi
    if is_session_alive "$sess"; then
      continue
    fi
    while [ "$(active_count)" -ge "$(echo "$CORES_CSV" | awk -F, '{print NF}')" ]; do sleep "$SLEEP_S"; done
    core=$(free_core || true)
    [ -z "${core:-}" ] && { sleep "$SLEEP_S"; continue; }
    start_one_job "$job" "$kind" "$objective" "$method" "$core"
  done < "$JOB_LIST"
  if [ "$done_jobs" -ge "$total" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') ALL JOBS COMPLETE" | tee -a "$RUN_ROOT/scheduler.log"
    break
  fi
  sleep "$SLEEP_S"
done
EOS
sed -i \
  -e "s#__RUN_ROOT__#${RUN_ROOT}#g" \
  -e "s#__EXPECTED_TASKS__#${EXPECTED_TASKS}#g" \
  -e "s#__CORES_CSV__#${CORES_CSV}#g" \
  -e "s#__SCHEDULER_SLEEP_S__#${SCHEDULER_SLEEP_S}#g" \
  -e "s#__REPO_DIR__#${REPO_DIR}#g" \
  -e "s#__AAI_USERNAME__#${AAI_USERNAME}#g" \
  "$RUN_ROOT/scheduler.sh"
chmod +x "$RUN_ROOT/scheduler.sh"
}

case "$ACTION" in
  launch)
    echo "=== TSP DISTANCE-ONLY BATCH LAUNCH ==="
    date
    hostname
    echo "RUN_ROOT=$RUN_ROOT"
    echo "EXPECTED_TASKS=$EXPECTED_TASKS"
    echo "CORES_CSV=$CORES_CSV"
    echo "INSTANCES=$INSTANCES"
    echo "REPS=$REPS"
    echo
    write_scheduler
    if tmux has-session -t "$SCHEDULER_SESSION" 2>/dev/null; then
      echo "Scheduler already running: $SCHEDULER_SESSION"
    else
      tmux new -d -s "$SCHEDULER_SESSION" "bash '$RUN_ROOT/scheduler.sh'"
      echo "Started scheduler: $SCHEDULER_SESSION"
    fi
    print_status
    ;;
  status)
    print_status
    ;;
  download)
    print_status
    ;;
  *)
    echo "ERROR: unknown ACTION=$ACTION"
    exit 2
    ;;
esac

echo "=== Remote run root ==="
echo "$RUN_ROOT"
echo "=== DONE ==="
'@

$remoteScript = $remoteScriptTemplate
$remoteScript = $remoteScript.Replace("__AAI_USERNAME__", $AAI_USERNAME)
$remoteScript = $remoteScript.Replace("__REPO_URL__", $REPO_URL)
$remoteScript = $remoteScript.Replace("__ACTION__", $Action)
$remoteScript = $remoteScript.Replace("__RUN_LABEL__", $RUN_LABEL)
$remoteScript = $remoteScript.Replace("__SIGNAL_MODE__", $SIGNAL_MODE)
$remoteScript = $remoteScript.Replace("__REPS__", [string]$REPS)
$remoteScript = $remoteScript.Replace("__EXPECTED_TASKS__", [string]$EXPECTED_TASKS)
$remoteScript = $remoteScript.Replace("__MAX_INSTANCES__", [string]$MAX_INSTANCES)
$remoteScript = $remoteScript.Replace("__TIMEOUT_S__", [string]$TIMEOUT_S)
$remoteScript = $remoteScript.Replace("__INSTANCES__", $INSTANCES)
$remoteScript = $remoteScript.Replace("__SPLITS__", $SPLITS)
$remoteScript = $remoteScript.Replace("__GLOBAL_SEED__", [string]$GLOBAL_SEED)
$remoteScript = $remoteScript.Replace("__CORES_CSV__", $CORES_CSV)
$remoteScript = $remoteScript.Replace("__SCHEDULER_SLEEP_S__", [string]$SCHEDULER_SLEEP_S)
$remoteScript = $remoteScript.Replace("__START_NEW_RUN__", $START_NEW_RUN_BASH)
$remoteScript = $remoteScript.Replace("__GIT_PULL__", $GIT_PULL_BASH)
$remoteScript = $remoteScript.Replace("__SETUP_ENV__", $SETUP_ENV_BASH)
$remoteScript = $remoteScript.Replace("__DRY_RUN__", $DRY_RUN_BASH)
$remoteScript = $remoteScript -replace "`r`n", "`n"

$localScript = "$env:TEMP\launch_tsp_distance50_batch.sh"
[System.IO.File]::WriteAllText($localScript, $remoteScript, (New-Object System.Text.UTF8Encoding($false)))

Write-Host "=== Uploading remote launcher ==="
scp $localScript "${REMOTE}:/home/$AAI_USERNAME/launch_tsp_distance50_batch.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Running remote launcher ==="
$remoteOutput = ssh $REMOTE "sed -i 's/
`$//' /home/$AAI_USERNAME/launch_tsp_distance50_batch.sh; chmod +x /home/$AAI_USERNAME/launch_tsp_distance50_batch.sh; bash /home/$AAI_USERNAME/launch_tsp_distance50_batch.sh"
$remoteExit = $LASTEXITCODE
$remoteOutput | ForEach-Object { Write-Host $_ }
if ($remoteExit -ne 0) { exit $remoteExit }

if ($Action -eq "download") {
    $runRootLine = ($remoteOutput | Where-Object { $_ -like "RUN_ROOT=*" } | Select-Object -Last 1)
    if ($null -eq $runRootLine) {
        Write-Host "ERROR: Could not parse RUN_ROOT from remote output."
        exit 3
    }
    $RUN_ROOT_REMOTE = $runRootLine.Replace("RUN_ROOT=", "").Trim()
    $folderName = Split-Path $RUN_ROOT_REMOTE -Leaf
    $localOut = Join-Path $LOCAL_RESULTS_DIR "tsp_distance50_batch"
    New-Item -ItemType Directory -Force -Path $localOut | Out-Null
    Write-Host "=== Downloading result folder back to local PC ==="
    scp -r "${REMOTE}:${RUN_ROOT_REMOTE}" "$localOut\"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "=== Local copy ==="
    Write-Host "$localOut\$folderName"
}
