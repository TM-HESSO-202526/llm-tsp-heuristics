# ============================================================
# Final TSP C++ distance-only 50-repetition batch launcher for Zeus
#
# Commands:
#   .\run_tsp_cpp_distance50_batch_to_zeus.ps1 -Action launch -StartNewRun
#   .\run_tsp_cpp_distance50_batch_to_zeus.ps1 -Action status
#   .\run_tsp_cpp_distance50_batch_to_zeus.ps1 -Action download
#
# Protocol:
# - selected distance-only LLM heuristics translated to C++
# - distance-only external baselines implemented in the same C++ evaluator
# - 14 instances including pla33810 and pla85900
# - 50 repetitions
# - 10 concurrent cores
# - one core = one job; one job = one method over all instances/repetitions
# ============================================================

param(
    [ValidateSet("launch", "status", "download")]
    [string]$Action = "status",
    [switch]$StartNewRun,
    [switch]$NoUploadInputs,
    [switch]$NoGitPull,
    [switch]$DryRun
)

$AAI_USERNAME = "anthony.atallah"
$SERVER_NAME = "zeus"
$REMOTE = "$AAI_USERNAME@$SERVER_NAME.iict-heig-vd.in"
$REPO_URL = "https://github.com/TM-HESSO-202526/llm-tsp-heuristics.git"

$LOCAL_INPUT_DIR = "D:\Users\antho\TM\server_eval_tsp_inputs"
$LOCAL_TSP_INSTANCE_DIR = "$LOCAL_INPUT_DIR\TSP_instances"
$LOCAL_RESULTS_DIR = "D:\Users\antho\TM\server_eval_results"
$LOCAL_REPO_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LOCAL_CPP_EVAL = Join-Path $LOCAL_REPO_ROOT "server_eval\tsp_cpp_distance_eval.cpp"

$RUN_LABEL = "cpp_distance50_direct_translation"
$REPS = 50
$INSTANCES = "dsj1000,pr1002,d1291,fl1400,pcb1173,rl1304,u1817,rl1889,pr2392,pcb3038,pla7397,usa13509,pla33810,pla85900"
$INSTANCE_COUNT = 14
$EXPECTED_TASKS = $INSTANCE_COUNT * $REPS
$TIMEOUT_S = 900
$GLOBAL_SEED = 12345
$CORES_TO_USE = @(0,1,2,3,4,5,6,7,8,9)
$SCHEDULER_SLEEP_S = 30

$REMOTE_INPUT_DIR = "/home/$AAI_USERNAME/data-local/TM/tsp_input"
$REMOTE_INSTANCE_DIR = "$REMOTE_INPUT_DIR/TSP_instances"
$REMOTE_RESULTS_ROOT = "/home/$AAI_USERNAME/workspace/TM/final-results/tsp_cpp_distance50_batch"
$CORES_CSV = ($CORES_TO_USE -join ",")

function B($x) { if ($x) { "1" } else { "0" } }
$START_NEW_RUN_BASH = B $StartNewRun
$GIT_PULL_BASH = B (-not $NoGitPull)
$DRY_RUN_BASH = B $DryRun

Write-Host "=== TSP C++ distance-only 50-repetition batch ==="
Write-Host "Remote:        $REMOTE"
Write-Host "Action:        $Action"
Write-Host "Run label:     $RUN_LABEL"
Write-Host "Reps:          $REPS"
Write-Host "Expected rows: $EXPECTED_TASKS per job"
Write-Host "Instances:     $INSTANCES"
Write-Host "Cores:         $CORES_CSV"
Write-Host "Timeout/row:   $TIMEOUT_S s"
Write-Host "Start new run: $StartNewRun"
Write-Host "Dry run:       $DryRun"
Write-Host ""

if ($Action -eq "launch" -and (-not $NoUploadInputs)) {
    Write-Host "=== Local input checks ==="
    if (!(Test-Path $LOCAL_TSP_INSTANCE_DIR)) { Write-Host "ERROR missing $LOCAL_TSP_INSTANCE_DIR"; exit 1 }
    foreach ($name in $INSTANCES.Split(",")) {
        $p = Join-Path $LOCAL_TSP_INSTANCE_DIR ("$name.tsp")
        if (!(Test-Path $p)) { Write-Host "ERROR missing $p"; exit 1 }
        Write-Host "OK: $p"
    }
}

Write-Host "=== Creating remote folders on $REMOTE ==="
ssh $REMOTE "mkdir -p /home/$AAI_USERNAME/workspace/TM $REMOTE_INSTANCE_DIR $REMOTE_RESULTS_ROOT /tmp/tsp_cpp_launcher"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Action -eq "launch" -and (-not $NoUploadInputs)) {
    Write-Host "=== Uploading TSP instances ==="
    foreach ($name in $INSTANCES.Split(",")) {
        $p = Join-Path $LOCAL_TSP_INSTANCE_DIR ("$name.tsp")
        scp "$p" "${REMOTE}:${REMOTE_INSTANCE_DIR}/"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

$remoteScript = @'
#!/usr/bin/env bash
set -euo pipefail

AAI_USERNAME="__AAI_USERNAME__"
REPO_URL="__REPO_URL__"
ACTION="__ACTION__"
RUN_LABEL="__RUN_LABEL__"
REPS="__REPS__"
EXPECTED_TASKS="__EXPECTED_TASKS__"
TIMEOUT_S="__TIMEOUT_S__"
INSTANCES="__INSTANCES__"
GLOBAL_SEED="__GLOBAL_SEED__"
CORES_CSV="__CORES_CSV__"
SCHEDULER_SLEEP_S="__SCHEDULER_SLEEP_S__"
START_NEW_RUN="__START_NEW_RUN__"
GIT_PULL="__GIT_PULL__"
DRY_RUN="__DRY_RUN__"
SELF_SCRIPT="$0"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

WORK_ROOT="/home/${AAI_USERNAME}/workspace/TM"
REPO_DIR="${WORK_ROOT}/llm-tsp-heuristics"
INSTANCE_ROOT="/home/${AAI_USERNAME}/data-local/TM/tsp_input/TSP_instances"
OUT_ROOT="${WORK_ROOT}/final-results/tsp_cpp_distance50_batch"
LATEST_FILE="${OUT_ROOT}/LATEST_${RUN_LABEL}.txt"
SCHEDULER_SESSION="tspcpp50_scheduler"

mkdir -p "$WORK_ROOT" "$INSTANCE_ROOT" "$OUT_ROOT"

if [ "$ACTION" = "launch" ]; then
  cd "$WORK_ROOT"
  if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" llm-tsp-heuristics
  fi
  cd "$REPO_DIR"
  if [ "$GIT_PULL" = "1" ]; then git pull || true; fi
  if [ -f /tmp/tsp_cpp_launcher/tsp_cpp_distance_eval.cpp ]; then
    echo "=== installing uploaded strict C++ evaluator ==="
    cp /tmp/tsp_cpp_launcher/tsp_cpp_distance_eval.cpp server_eval/tsp_cpp_distance_eval.cpp
  fi
  echo "=== compiling C++ distance evaluator ==="
  test -f server_eval/tsp_cpp_distance_eval.cpp || { echo "ERROR missing server_eval/tsp_cpp_distance_eval.cpp"; exit 2; }
  g++ -std=c++17 -O3 -march=native -DNDEBUG -o server_eval/tsp_cpp_distance_eval server_eval/tsp_cpp_distance_eval.cpp
else
  cd "$REPO_DIR"
fi

if [ "$ACTION" = "launch" ] && { [ "$START_NEW_RUN" = "1" ] || [ ! -f "$LATEST_FILE" ]; }; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_ROOT="${OUT_ROOT}/${RUN_LABEL}_${STAMP}"
  mkdir -p "$RUN_ROOT"
  echo "$RUN_ROOT" > "$LATEST_FILE"
elif [ -f "$LATEST_FILE" ]; then
  RUN_ROOT="$(cat "$LATEST_FILE")"
  mkdir -p "$RUN_ROOT"
else
  echo "ERROR: no LATEST run exists. Use -Action launch -StartNewRun first."
  exit 2
fi

JOB_LIST="$RUN_ROOT/job_list.tsv"
JOB_STATE="$RUN_ROOT/job_state"
mkdir -p "$JOB_STATE"

write_job_list() {
cat > "$JOB_LIST" <<'JOBS'
H_TSPD_CPP_02_normal_raw_nn2opt_best_101102_iter003	heuristic	distance_only	02_normal_raw_nn2opt_best_101102_iter003
H_TSPD_CPP_03_family_focus_grid_best_100159_iter072	heuristic	distance_only	03_family_focus_grid_best_100159_iter072
H_TSPD_CPP_04_family_focus_convex_faithful_095803_iter031	heuristic	distance_only	04_family_focus_convex_faithful_095803_iter031
H_TSPD_CPP_05_family_focus_voronoi_best_100159_iter037	heuristic	distance_only	05_family_focus_voronoi_best_100159_iter037
H_TSPD_CPP_07_family_focus_region_endpoint_fast_100159_iter177	heuristic	distance_only	07_family_focus_region_endpoint_fast_100159_iter177
H_TSPD_CPP_08_expo_distance_only_geostabilizer_399e	heuristic	distance_only	08_expo_distance_only_geostabilizer_399e
H_TSPD_CPP_09_family_focus_mst_diagnostic_100159_iter007	heuristic	distance_only	09_family_focus_mst_diagnostic_100159_iter007
H_TSPD_CPP_10_family_focus_fast_convex_095803_iter026	heuristic	distance_only	10_family_focus_fast_convex_095803_iter026
H_TSPD_CPP_11_family_focus_convex_constructive_095803_iter021	heuristic	distance_only	11_family_focus_convex_constructive_095803_iter021
B_TSPD_CPP_01_kdtree_nearest_neighbor_fixed_start	baseline	distance_only	01_kdtree_nearest_neighbor_fixed_start
B_TSPD_CPP_02_kdtree_nearest_neighbor_multistart	baseline	distance_only	02_kdtree_nearest_neighbor_multistart
B_TSPD_CPP_03_x_axis_sweep	baseline	distance_only	03_x_axis_sweep
B_TSPD_CPP_04_pca_sweep	baseline	distance_only	04_pca_sweep
B_TSPD_CPP_05_angular_sweep	baseline	distance_only	05_angular_sweep
B_TSPD_CPP_06_morton_z_order	baseline	distance_only	06_morton_z_order
B_TSPD_CPP_07_grid_serpentine	baseline	distance_only	07_grid_serpentine
B_TSPD_CPP_08_morton_bounded_local_2opt	baseline	distance_only	08_morton_bounded_local_2opt
JOBS
}

if [ "$ACTION" = "launch" ] && { [ "$START_NEW_RUN" = "1" ] || [ ! -f "$JOB_LIST" ]; }; then write_job_list; fi
if [ ! -f "$JOB_LIST" ]; then echo "ERROR missing $JOB_LIST"; exit 2; fi

cat > "$RUN_ROOT/run_config.json" <<EOF
{"run_label":"$RUN_LABEL","language":"cpp","repetitions":$REPS,"instances":"$INSTANCES","expected_tasks_per_job":$EXPECTED_TASKS,"cores_csv":"$CORES_CSV","timeout_s":$TIMEOUT_S,"global_seed":$GLOBAL_SEED,"run_root":"$RUN_ROOT","updated_at":"$(date '+%Y-%m-%d %H:%M:%S')"}
EOF

sanitize_session(){ local s="tspcpp50_$1"; echo "$s" | tr -c 'A-Za-z0-9_' '_' | cut -c1-80; }
row_count(){ local raw="$RUN_ROOT/$1/raw_results.csv"; if [ ! -f "$raw" ]; then echo 0; else local l; l=$(wc -l < "$raw" 2>/dev/null || echo 0); if [ "$l" -le 0 ]; then echo 0; else echo $((l-1)); fi; fi; }
is_session_alive(){ tmux has-session -t "$1" 2>/dev/null; }

status_one(){
  local job="$1" kind="$2" signal="$3" method="$4"
  local rows; rows=$(row_count "$job")
  local sess; sess=$(sanitize_session "$job")
  local st="PENDING"
  if [ "$rows" -ge "$EXPECTED_TASKS" ]; then st="COMPLETE"; elif is_session_alive "$sess"; then st="RUNNING"; elif [ "$rows" -gt 0 ]; then st="INCOMPLETE"; fi
  printf "%-78s %-10s %-14s %-62s %5s/%-5s %s\n" "$job" "$kind" "$signal" "$method" "$rows" "$EXPECTED_TASKS" "$st"
}

print_status(){
  echo "=== TSP C++ DISTANCE-ONLY 50REPS STATUS ==="
  date; hostname
  echo "RUN_ROOT=$RUN_ROOT"
  echo "JOB_LIST=$JOB_LIST"
  echo "EXPECTED_TASKS=$EXPECTED_TASKS"
  echo "CORES_CSV=$CORES_CSV"
  echo
  printf "%-78s %-10s %-14s %-62s %s\n" "JOB" "KIND" "SIGNAL" "METHOD" "ROWS STATUS"
  printf '%0.s-' {1..180}; echo
  local total=0 complete=0 running=0 pending=0 incomplete=0
  while IFS=$'\t' read -r job kind signal method; do
    [ -z "${job:-}" ] && continue
    local rows; rows=$(row_count "$job")
    local sess; sess=$(sanitize_session "$job")
    local st="PENDING"
    if [ "$rows" -ge "$EXPECTED_TASKS" ]; then st="COMPLETE"; complete=$((complete+1));
    elif is_session_alive "$sess"; then st="RUNNING"; running=$((running+1));
    elif [ "$rows" -gt 0 ]; then st="INCOMPLETE"; incomplete=$((incomplete+1));
    else pending=$((pending+1)); fi
    total=$((total+1))
    printf "%-78s %-10s %-14s %-62s %5s/%-5s %s\n" "$job" "$kind" "$signal" "$method" "$rows" "$EXPECTED_TASKS" "$st"
  done < "$JOB_LIST"
  echo
  echo "TOTAL_JOBS=$total COMPLETE=$complete RUNNING=$running PENDING=$pending INCOMPLETE=$incomplete"
  echo
  echo "=== tmux sessions ==="; tmux ls 2>/dev/null | grep -E 'tspcpp50_|tspcpp50_scheduler' || true
}

make_run_one(){
cat > "$RUN_ROOT/run_one_job.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
JOB="$1"; KIND="$2"; SIGNAL="$3"; METHOD="$4"; CORE="$5"
REPO_DIR="__REPO_DIR__"
RUN_ROOT="__RUN_ROOT__"
INSTANCE_ROOT="__INSTANCE_ROOT__"
INSTANCES="__INSTANCES__"
REPS="__REPS__"
TIMEOUT_S="__TIMEOUT_S__"
GLOBAL_SEED="__GLOBAL_SEED__"
LOG="$RUN_ROOT/${JOB}.log"
OUT="$RUN_ROOT/$JOB"
mkdir -p "$OUT"
cd "$REPO_DIR"
echo "=== START $JOB ===" | tee -a "$LOG"
date | tee -a "$LOG"
echo "core=$CORE kind=$KIND method=$METHOD" | tee -a "$LOG"
taskset -c "$CORE" ./server_eval/tsp_cpp_distance_eval \
  --job "$JOB" --kind "$KIND" --signal "$SIGNAL" --method "$METHOD" \
  --instance-root "$INSTANCE_ROOT" --optima-csv data/tsp_instances_opt.csv \
  --instances "$INSTANCES" --reps "$REPS" --timeout-s "$TIMEOUT_S" \
  --global-seed "$GLOBAL_SEED" --out-dir "$OUT" 2>&1 | tee -a "$LOG"
echo "=== DONE $JOB ===" | tee -a "$LOG"
date | tee -a "$LOG"
SH
sed -i "s#__REPO_DIR__#$REPO_DIR#g; s#__RUN_ROOT__#$RUN_ROOT#g; s#__INSTANCE_ROOT__#$INSTANCE_ROOT#g; s#__INSTANCES__#$INSTANCES#g; s#__REPS__#$REPS#g; s#__TIMEOUT_S__#$TIMEOUT_S#g; s#__GLOBAL_SEED__#$GLOBAL_SEED#g" "$RUN_ROOT/run_one_job.sh"
chmod +x "$RUN_ROOT/run_one_job.sh"
}

scheduler_loop(){
  make_run_one
  echo "=== SCHEDULER START ===" | tee -a "$RUN_ROOT/scheduler.log"
  while true; do
    local active=0
    while IFS=$'\t' read -r job kind signal method; do
      [ -z "${job:-}" ] && continue
      local sess; sess=$(sanitize_session "$job")
      if is_session_alive "$sess"; then active=$((active+1)); fi
    done < "$JOB_LIST"
    local max_active; max_active=$(echo "$CORES_CSV" | awk -F, '{print NF}')
    if [ "$active" -lt "$max_active" ]; then
      IFS=',' read -ra cores <<< "$CORES_CSV"
      for core in "${cores[@]}"; do
        local used=0
        while IFS=$'\t' read -r job kind signal method; do
          local sess; sess=$(sanitize_session "$job")
          if is_session_alive "$sess" && tmux capture-pane -pt "$sess" 2>/dev/null | grep -q "core=$core"; then used=1; fi
        done < "$JOB_LIST"
        [ "$used" = "1" ] && continue
        while IFS=$'\t' read -r job kind signal method; do
          [ -z "${job:-}" ] && continue
          local rows; rows=$(row_count "$job")
          local sess; sess=$(sanitize_session "$job")
          if [ "$rows" -lt "$EXPECTED_TASKS" ] && ! is_session_alive "$sess"; then
            echo "$(date '+%F %T') launching $job on core $core" | tee -a "$RUN_ROOT/scheduler.log"
            tmux new -d -s "$sess" "bash '$RUN_ROOT/run_one_job.sh' '$job' '$kind' '$signal' '$method' '$core'"
            active=$((active+1)); used=1; break
          fi
        done < "$JOB_LIST"
        [ "$active" -ge "$max_active" ] && break
      done
    fi
    local remaining=0
    while IFS=$'\t' read -r job kind signal method; do
      [ -z "${job:-}" ] && continue
      local rows; rows=$(row_count "$job")
      local sess; sess=$(sanitize_session "$job")
      if [ "$rows" -lt "$EXPECTED_TASKS" ] || is_session_alive "$sess"; then remaining=$((remaining+1)); fi
    done < "$JOB_LIST"
    if [ "$remaining" -eq 0 ]; then echo "=== SCHEDULER DONE ===" | tee -a "$RUN_ROOT/scheduler.log"; break; fi
    sleep "$SCHEDULER_SLEEP_S"
  done
}

if [ "${1:-}" = "scheduler-child" ]; then
  scheduler_loop
elif [ "$ACTION" = "launch" ]; then
  if [ "$DRY_RUN" = "1" ]; then print_status; exit 0; fi
  if tmux has-session -t "$SCHEDULER_SESSION" 2>/dev/null; then echo "Scheduler already running."; else tmux new -d -s "$SCHEDULER_SESSION" "bash '$SELF_SCRIPT' scheduler-child"; fi
  print_status
elif [ "$ACTION" = "status" ]; then
  print_status
elif [ "$ACTION" = "download" ]; then
  print_status
else
  echo "unknown action $ACTION"; exit 2
fi
'@

$remoteScript = $remoteScript.Replace("__AAI_USERNAME__", $AAI_USERNAME)
$remoteScript = $remoteScript.Replace("__REPO_URL__", $REPO_URL)
$remoteScript = $remoteScript.Replace("__ACTION__", $Action)
$remoteScript = $remoteScript.Replace("__RUN_LABEL__", $RUN_LABEL)
$remoteScript = $remoteScript.Replace("__REPS__", [string]$REPS)
$remoteScript = $remoteScript.Replace("__EXPECTED_TASKS__", [string]$EXPECTED_TASKS)
$remoteScript = $remoteScript.Replace("__TIMEOUT_S__", [string]$TIMEOUT_S)
$remoteScript = $remoteScript.Replace("__INSTANCES__", $INSTANCES)
$remoteScript = $remoteScript.Replace("__GLOBAL_SEED__", [string]$GLOBAL_SEED)
$remoteScript = $remoteScript.Replace("__CORES_CSV__", $CORES_CSV)
$remoteScript = $remoteScript.Replace("__SCHEDULER_SLEEP_S__", [string]$SCHEDULER_SLEEP_S)
$remoteScript = $remoteScript.Replace("__START_NEW_RUN__", $START_NEW_RUN_BASH)
$remoteScript = $remoteScript.Replace("__GIT_PULL__", $GIT_PULL_BASH)
$remoteScript = $remoteScript.Replace("__DRY_RUN__", $DRY_RUN_BASH)

$tmp = New-TemporaryFile
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($tmp.FullName, ($remoteScript -replace "`r`n", "`n"), $utf8NoBom)

if ($Action -eq "launch") {
    Write-Host "=== Uploading strict C++ evaluator ==="
    if (!(Test-Path $LOCAL_CPP_EVAL)) { Write-Host "ERROR missing $LOCAL_CPP_EVAL"; exit 1 }
    scp "$LOCAL_CPP_EVAL" "${REMOTE}:/tmp/tsp_cpp_launcher/tsp_cpp_distance_eval.cpp"
    if ($LASTEXITCODE -ne 0) { Remove-Item $tmp.FullName -Force; exit $LASTEXITCODE }
}

Write-Host "=== Uploading remote launcher ==="
scp $tmp.FullName "${REMOTE}:/tmp/tsp_cpp_launcher/launch_tsp_cpp_distance50_batch.sh"
if ($LASTEXITCODE -ne 0) { Remove-Item $tmp.FullName -Force; exit $LASTEXITCODE }
Remove-Item $tmp.FullName -Force

Write-Host "=== Running remote launcher ==="
ssh $REMOTE "chmod +x /tmp/tsp_cpp_launcher/launch_tsp_cpp_distance50_batch.sh && bash /tmp/tsp_cpp_launcher/launch_tsp_cpp_distance50_batch.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Action -eq "download") {
    Write-Host "=== Downloading result folder back to local PC ==="
    $localOut = Join-Path $LOCAL_RESULTS_DIR "tsp_cpp_distance50_batch"
    New-Item -ItemType Directory -Force -Path $localOut | Out-Null
    $remoteRunRoot = ssh $REMOTE "cat ${REMOTE_RESULTS_ROOT}/LATEST_${RUN_LABEL}.txt"
    $remoteRunRoot = $remoteRunRoot.Trim()
    scp -r "${REMOTE}:${remoteRunRoot}" "$localOut\"
}
