# ============================================================
# TSP C++ signal-regime 50-repetition batch launcher for Zeus
#
# Commands:
#   .\run_tsp_cpp_signal50_batch_to_zeus.ps1 -Action launch -StartNewRun
#   .\run_tsp_cpp_signal50_batch_to_zeus.ps1 -Action status
#   .\run_tsp_cpp_signal50_batch_to_zeus.ps1 -Action download
#
# Protocol:
# - direct C++ translations of selected/log-recovered TSP signal heuristics
# - C1, C1a: candidate-list regime (C1 recovered from actual LLM logs, iter_001_d854c319152c7853)
# - P1, P2, P3: edge-prior regime
# - 12 instances up to usa13509 only
# - 50 repetitions
# - one core = one method/job over all instances/repetitions
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
$LOCAL_CANDIDATE_CACHE_DIR = "$LOCAL_INPUT_DIR\LKH_candidate_cache"
$LOCAL_EDGE_PRIOR_CACHE_DIR = "$LOCAL_INPUT_DIR\LKH_edge_prior_cache"
$LOCAL_RESULTS_DIR = "D:\Users\antho\TM\server_eval_results"
$LOCAL_REPO_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LOCAL_CPP_EVAL = Join-Path $LOCAL_REPO_ROOT "server_eval\tsp_cpp_signal_eval.cpp"
$LOCAL_PREP_SCRIPT = Join-Path $LOCAL_REPO_ROOT "server_eval\prepare_tsp_signal_cpp_inputs.py"

$RUN_LABEL = "cpp_signal50_candidate_prior"
$REPS = 50
$INSTANCES = "dsj1000,pr1002,d1291,fl1400,pcb1173,rl1304,u1817,rl1889,pr2392,pcb3038,pla7397,usa13509"
$INSTANCE_COUNT = 12
$EXPECTED_TASKS = $INSTANCE_COUNT * $REPS
$TIMEOUT_S = 900
$GLOBAL_SEED = 12345
$MAX_CANDIDATES = 20
$CORES_TO_USE = @(0,1,2,3,4)
$SCHEDULER_SLEEP_S = 30

$REMOTE_INPUT_DIR = "/home/$AAI_USERNAME/data-local/TM/tsp_input"
$REMOTE_INSTANCE_DIR = "$REMOTE_INPUT_DIR/TSP_instances"
$REMOTE_CANDIDATE_DIR = "$REMOTE_INPUT_DIR/LKH_candidate_cache"
$REMOTE_EDGE_PRIOR_DIR = "$REMOTE_INPUT_DIR/LKH_edge_prior_cache"
$REMOTE_PRIOR_TXT_DIR = "$REMOTE_INPUT_DIR/LKH_edge_prior_cpp_txt"
$REMOTE_RESULTS_ROOT = "/home/$AAI_USERNAME/workspace/TM/final-results/tsp_cpp_signal50_batch"
$CORES_CSV = ($CORES_TO_USE -join ",")

function B($x) { if ($x) { "1" } else { "0" } }
$START_NEW_RUN_BASH = B $StartNewRun
$GIT_PULL_BASH = B (-not $NoGitPull)
$DRY_RUN_BASH = B $DryRun

Write-Host "=== TSP C++ signal-regime 50-repetition batch ==="
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
    if (!(Test-Path $LOCAL_CANDIDATE_CACHE_DIR)) { Write-Host "ERROR missing $LOCAL_CANDIDATE_CACHE_DIR"; exit 1 }
    if (!(Test-Path $LOCAL_EDGE_PRIOR_CACHE_DIR)) { Write-Host "ERROR missing $LOCAL_EDGE_PRIOR_CACHE_DIR"; exit 1 }
    foreach ($name in $INSTANCES.Split(",")) {
        $p = Join-Path $LOCAL_TSP_INSTANCE_DIR ("$name.tsp")
        if (!(Test-Path $p)) { Write-Host "ERROR missing $p"; exit 1 }
        Write-Host "OK: $p"
    }
}

Write-Host "=== Creating remote folders on $REMOTE ==="
ssh $REMOTE "mkdir -p /home/$AAI_USERNAME/workspace/TM $REMOTE_INSTANCE_DIR $REMOTE_CANDIDATE_DIR $REMOTE_EDGE_PRIOR_DIR $REMOTE_PRIOR_TXT_DIR $REMOTE_RESULTS_ROOT /tmp/tsp_cpp_signal_launcher"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Action -eq "launch" -and (-not $NoUploadInputs)) {
    Write-Host "=== Uploading TSP instances ==="
    foreach ($name in $INSTANCES.Split(",")) {
        $p = Join-Path $LOCAL_TSP_INSTANCE_DIR ("$name.tsp")
        scp "$p" "${REMOTE}:${REMOTE_INSTANCE_DIR}/"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    Write-Host "=== Uploading candidate cache directory ==="
    scp -r "$LOCAL_CANDIDATE_CACHE_DIR" "${REMOTE}:${REMOTE_INPUT_DIR}/"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "=== Uploading edge-prior cache directory ==="
    scp -r "$LOCAL_EDGE_PRIOR_CACHE_DIR" "${REMOTE}:${REMOTE_INPUT_DIR}/"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Action -eq "launch") {
    if (!(Test-Path $LOCAL_CPP_EVAL)) { Write-Host "ERROR missing $LOCAL_CPP_EVAL"; exit 1 }
    if (!(Test-Path $LOCAL_PREP_SCRIPT)) { Write-Host "ERROR missing $LOCAL_PREP_SCRIPT"; exit 1 }
    scp "$LOCAL_CPP_EVAL" "${REMOTE}:/tmp/tsp_cpp_signal_launcher/tsp_cpp_signal_eval.cpp"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    scp "$LOCAL_PREP_SCRIPT" "${REMOTE}:/tmp/tsp_cpp_signal_launcher/prepare_tsp_signal_cpp_inputs.py"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
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
MAX_CANDIDATES="__MAX_CANDIDATES__"
CORES_CSV="__CORES_CSV__"
SCHEDULER_SLEEP_S="__SCHEDULER_SLEEP_S__"
START_NEW_RUN="__START_NEW_RUN__"
GIT_PULL="__GIT_PULL__"
DRY_RUN="__DRY_RUN__"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

WORK_ROOT="/home/${AAI_USERNAME}/workspace/TM"
REPO_DIR="${WORK_ROOT}/llm-tsp-heuristics"
INSTANCE_ROOT="/home/${AAI_USERNAME}/data-local/TM/tsp_input/TSP_instances"
CANDIDATE_ROOT="/home/${AAI_USERNAME}/data-local/TM/tsp_input/LKH_candidate_cache"
EDGE_PRIOR_ROOT="/home/${AAI_USERNAME}/data-local/TM/tsp_input/LKH_edge_prior_cache"
PRIOR_TXT_ROOT="/home/${AAI_USERNAME}/data-local/TM/tsp_input/LKH_edge_prior_cpp_txt"
OUT_ROOT="${WORK_ROOT}/final-results/tsp_cpp_signal50_batch"
LATEST_FILE="${OUT_ROOT}/LATEST_${RUN_LABEL}.txt"
SCHEDULER_SESSION="tspsig50_scheduler"

mkdir -p "$WORK_ROOT" "$INSTANCE_ROOT" "$CANDIDATE_ROOT" "$EDGE_PRIOR_ROOT" "$PRIOR_TXT_ROOT" "$OUT_ROOT"

if [ "$ACTION" = "launch" ]; then
  cd "$WORK_ROOT"
  if [ ! -d "$REPO_DIR/.git" ]; then git clone "$REPO_URL" llm-tsp-heuristics; fi
  cd "$REPO_DIR"
  if [ "$GIT_PULL" = "1" ]; then git pull || true; fi
  if [ -f /tmp/tsp_cpp_signal_launcher/tsp_cpp_signal_eval.cpp ]; then
    cp /tmp/tsp_cpp_signal_launcher/tsp_cpp_signal_eval.cpp server_eval/tsp_cpp_signal_eval.cpp
  fi
  if [ -f /tmp/tsp_cpp_signal_launcher/prepare_tsp_signal_cpp_inputs.py ]; then
    cp /tmp/tsp_cpp_signal_launcher/prepare_tsp_signal_cpp_inputs.py server_eval/prepare_tsp_signal_cpp_inputs.py
  fi
  echo "=== preparing C++ prior text files ==="
  python3 server_eval/prepare_tsp_signal_cpp_inputs.py \
    --instances "$INSTANCES" \
    --instance-root "$INSTANCE_ROOT" \
    --edge-prior-cache-dir "$EDGE_PRIOR_ROOT" \
    --out-dir "$PRIOR_TXT_ROOT" \
    --prior-mode frequency --topk 5
  echo "=== compiling C++ signal evaluator ==="
  g++ -std=c++17 -O3 -march=native -DNDEBUG -o server_eval/tsp_cpp_signal_eval server_eval/tsp_cpp_signal_eval.cpp
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
TSPC_CPP_C1_candidate_nn_constructive	heuristic	candidate_list	C1_candidate_nn_constructive
TSPC_CPP_C1a_candidate_cleanup	heuristic	candidate_list	C1a_candidate_cleanup
TSPP_CPP_P1_quality_prior	heuristic	edge_prior	P1_quality_prior
TSPP_CPP_P2_prior_dominant	heuristic	edge_prior	P2_prior_dominant
TSPP_CPP_P3_fast_prior_lookahead	heuristic	edge_prior	P3_fast_prior_lookahead
JOBS
}

if [ "$ACTION" = "launch" ] && { [ "$START_NEW_RUN" = "1" ] || [ ! -f "$JOB_LIST" ]; }; then write_job_list; fi
if [ ! -f "$JOB_LIST" ]; then echo "ERROR missing $JOB_LIST"; exit 2; fi

cat > "$RUN_ROOT/run_config.json" <<EOF
{"run_label":"$RUN_LABEL","language":"cpp","repetitions":$REPS,"instances":"$INSTANCES","expected_tasks_per_job":$EXPECTED_TASKS,"cores_csv":"$CORES_CSV","timeout_s":$TIMEOUT_S,"global_seed":$GLOBAL_SEED,"candidate_root":"$CANDIDATE_ROOT","edge_prior_root":"$EDGE_PRIOR_ROOT","prior_txt_root":"$PRIOR_TXT_ROOT","run_root":"$RUN_ROOT","updated_at":"$(date '+%Y-%m-%d %H:%M:%S')"}
EOF

sanitize_session(){ local s="tspsig50_$1"; echo "$s" | tr -c 'A-Za-z0-9_' '_' | cut -c1-80; }
row_count(){ local raw="$RUN_ROOT/$1/raw_results.csv"; if [ ! -f "$raw" ]; then echo 0; else local l; l=$(wc -l < "$raw" 2>/dev/null || echo 0); if [ "$l" -le 0 ]; then echo 0; else echo $((l-1)); fi; fi; }
is_session_alive(){ tmux has-session -t "$1" 2>/dev/null; }

print_status(){
  echo "=== TSP C++ SIGNAL 50REPS STATUS ==="; date; hostname
  echo "RUN_ROOT=$RUN_ROOT"; echo "JOB_LIST=$JOB_LIST"; echo "EXPECTED_TASKS=$EXPECTED_TASKS"; echo "CORES_CSV=$CORES_CSV"; echo
  printf "%-45s %-10s %-15s %-34s %s\n" "JOB" "KIND" "SIGNAL" "METHOD" "ROWS STATUS"
  printf '%0.s-' {1..130}; echo
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
    printf "%-45s %-10s %-15s %-34s %5s/%-5s %s\n" "$job" "$kind" "$signal" "$method" "$rows" "$EXPECTED_TASKS" "$st"
  done < "$JOB_LIST"
  echo; echo "TOTAL_JOBS=$total COMPLETE=$complete RUNNING=$running PENDING=$pending INCOMPLETE=$incomplete"; echo
  echo "=== tmux sessions ==="; tmux ls 2>/dev/null | grep -E 'tspsig50_|tspsig50_scheduler' || true
}

make_run_one(){
cat > "$RUN_ROOT/run_one_job.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
JOB="$1"; KIND="$2"; SIGNAL="$3"; METHOD="$4"; CORE="$5"
REPO_DIR="__REPO_DIR__"
RUN_ROOT="__RUN_ROOT__"
INSTANCE_ROOT="__INSTANCE_ROOT__"
CANDIDATE_ROOT="__CANDIDATE_ROOT__"
PRIOR_TXT_ROOT="__PRIOR_TXT_ROOT__"
INSTANCES="__INSTANCES__"
REPS="__REPS__"
TIMEOUT_S="__TIMEOUT_S__"
GLOBAL_SEED="__GLOBAL_SEED__"
MAX_CANDIDATES="__MAX_CANDIDATES__"
LOG="$RUN_ROOT/${JOB}.log"
OUT="$RUN_ROOT/$JOB"
mkdir -p "$OUT"
cd "$REPO_DIR"
echo "=== START $JOB ===" | tee -a "$LOG"
date | tee -a "$LOG"
echo "core=$CORE kind=$KIND signal=$SIGNAL method=$METHOD" | tee -a "$LOG"
taskset -c "$CORE" ./server_eval/tsp_cpp_signal_eval \
  --job "$JOB" --kind "$KIND" --signal "$SIGNAL" --method "$METHOD" \
  --instance-root "$INSTANCE_ROOT" --candidate-root "$CANDIDATE_ROOT" --prior-txt-root "$PRIOR_TXT_ROOT" \
  --optima-csv data/tsp_instances_opt.csv --instances "$INSTANCES" --reps "$REPS" \
  --timeout-s "$TIMEOUT_S" --global-seed "$GLOBAL_SEED" --max-candidates "$MAX_CANDIDATES" \
  --out-dir "$OUT" 2>&1 | tee -a "$LOG"
echo "=== DONE $JOB ===" | tee -a "$LOG"
SH
sed -i "s#__REPO_DIR__#$REPO_DIR#g; s#__RUN_ROOT__#$RUN_ROOT#g; s#__INSTANCE_ROOT__#$INSTANCE_ROOT#g; s#__CANDIDATE_ROOT__#$CANDIDATE_ROOT#g; s#__PRIOR_TXT_ROOT__#$PRIOR_TXT_ROOT#g; s#__INSTANCES__#$INSTANCES#g; s#__REPS__#$REPS#g; s#__TIMEOUT_S__#$TIMEOUT_S#g; s#__GLOBAL_SEED__#$GLOBAL_SEED#g; s#__MAX_CANDIDATES__#$MAX_CANDIDATES#g" "$RUN_ROOT/run_one_job.sh"
chmod +x "$RUN_ROOT/run_one_job.sh"
}

make_scheduler(){
cat > "$RUN_ROOT/scheduler.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="__RUN_ROOT__"
JOB_LIST="$RUN_ROOT/job_list.tsv"
CORES_CSV="__CORES_CSV__"
EXPECTED_TASKS="__EXPECTED_TASKS__"
SLEEP_S="__SCHEDULER_SLEEP_S__"
IFS=',' read -r -a CORES <<< "$CORES_CSV"
sanitize_session(){ local s="tspsig50_$1"; echo "$s" | tr -c 'A-Za-z0-9_' '_' | cut -c1-80; }
row_count(){ local raw="$RUN_ROOT/$1/raw_results.csv"; if [ ! -f "$raw" ]; then echo 0; else local l; l=$(wc -l < "$raw" 2>/dev/null || echo 0); if [ "$l" -le 0 ]; then echo 0; else echo $((l-1)); fi; fi; }
is_alive(){ tmux has-session -t "$1" 2>/dev/null; }
while true; do
  all_done=1
  used=()
  while IFS=$'\t' read -r job kind signal method; do
    [ -z "${job:-}" ] && continue
    sess=$(sanitize_session "$job")
    rows=$(row_count "$job")
    if [ "$rows" -ge "$EXPECTED_TASKS" ]; then continue; fi
    all_done=0
    if is_alive "$sess"; then continue; fi
    for core in "${CORES[@]}"; do
      busy=0
      for u in "${used[@]:-}"; do [ "$u" = "$core" ] && busy=1; done
      if [ "$busy" = 0 ]; then
        used+=("$core")
        echo "[$(date)] launching $job on core $core, existing rows=$rows"
        tmux new -d -s "$sess" "$RUN_ROOT/run_one_job.sh '$job' '$kind' '$signal' '$method' '$core'"
        break
      fi
    done
  done < "$JOB_LIST"
  if [ "$all_done" = 1 ]; then echo "[$(date)] all jobs complete"; break; fi
  sleep "$SLEEP_S"
done
SH
sed -i "s#__RUN_ROOT__#$RUN_ROOT#g; s#__CORES_CSV__#$CORES_CSV#g; s#__EXPECTED_TASKS__#$EXPECTED_TASKS#g; s#__SCHEDULER_SLEEP_S__#$SCHEDULER_SLEEP_S#g" "$RUN_ROOT/scheduler.sh"
chmod +x "$RUN_ROOT/scheduler.sh"
}

if [ "$ACTION" = "status" ]; then print_status; exit 0; fi
if [ "$ACTION" = "download" ]; then print_status; exit 0; fi
if [ "$ACTION" = "launch" ]; then
  print_status
  make_run_one
  make_scheduler
  if [ "$DRY_RUN" = "1" ]; then echo "DRY RUN: not launching scheduler"; exit 0; fi
  tmux kill-session -t "$SCHEDULER_SESSION" 2>/dev/null || true
  tmux new -d -s "$SCHEDULER_SESSION" "$RUN_ROOT/scheduler.sh"
  echo "Launched scheduler session: $SCHEDULER_SESSION"
  print_status
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
$remoteScript = $remoteScript.Replace("__MAX_CANDIDATES__", [string]$MAX_CANDIDATES)
$remoteScript = $remoteScript.Replace("__CORES_CSV__", $CORES_CSV)
$remoteScript = $remoteScript.Replace("__SCHEDULER_SLEEP_S__", [string]$SCHEDULER_SLEEP_S)
$remoteScript = $remoteScript.Replace("__START_NEW_RUN__", $START_NEW_RUN_BASH)
$remoteScript = $remoteScript.Replace("__GIT_PULL__", $GIT_PULL_BASH)
$remoteScript = $remoteScript.Replace("__DRY_RUN__", $DRY_RUN_BASH)

$tmp = Join-Path $env:TEMP "run_tsp_cpp_signal50_zeus.sh"
$remoteScript | Set-Content -Encoding UTF8 -Path $tmp
scp "$tmp" "${REMOTE}:/tmp/tsp_cpp_signal_launcher/run_tsp_cpp_signal50_zeus.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ssh $REMOTE "bash /tmp/tsp_cpp_signal_launcher/run_tsp_cpp_signal50_zeus.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Action -eq "download") {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $dest = Join-Path $LOCAL_RESULTS_DIR "tsp_cpp_signal50_batch_$stamp"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $runRoot = ssh $REMOTE "cat $REMOTE_RESULTS_ROOT/LATEST_${RUN_LABEL}.txt"
    scp -r "${REMOTE}:$runRoot" "$dest\"
    Write-Host "Downloaded to $dest"
}
