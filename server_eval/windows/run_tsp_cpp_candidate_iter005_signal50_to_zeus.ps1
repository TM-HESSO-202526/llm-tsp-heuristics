# ============================================================
# TSP C++ candidate-list iter005 50-repetition launcher for Zeus
#
# Commands:
#   .\run_tsp_cpp_candidate_iter005_signal50_to_zeus.ps1 -Action launch -StartNewRun -NoGitPull
#   .\run_tsp_cpp_candidate_iter005_signal50_to_zeus.ps1 -Action status
#   .\run_tsp_cpp_candidate_iter005_signal50_to_zeus.ps1 -Action download
#
# Protocol:
# - direct C++ translation of actual LLM log heuristic iter_005_d41c0705c632564f
# - candidate-list regime only, no edge prior
# - 12 instances up to usa13509 only
# - 50 repetitions
# - logs construction candidate-list moves vs full-distance fallback moves
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
$LOCAL_RESULTS_DIR = "D:\Users\antho\TM\server_eval_results"
$LOCAL_REPO_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LOCAL_CPP_EVAL = Join-Path $LOCAL_REPO_ROOT "server_eval\tsp_cpp_signal_eval.cpp"

$RUN_LABEL = "cpp_candidate_iter005_signal50"
$REPS = 50
$INSTANCES = "dsj1000,pr1002,d1291,fl1400,pcb1173,rl1304,u1817,rl1889,pr2392,pcb3038,pla7397,usa13509"
$INSTANCE_COUNT = 12
$EXPECTED_TASKS = $INSTANCE_COUNT * $REPS
$TIMEOUT_S = 900
$GLOBAL_SEED = 12345
$MAX_CANDIDATES = 20
$CORE_TO_USE = 0

$REMOTE_INPUT_DIR = "/home/$AAI_USERNAME/data-local/TM/tsp_input"
$REMOTE_INSTANCE_DIR = "$REMOTE_INPUT_DIR/TSP_instances"
$REMOTE_CANDIDATE_DIR = "$REMOTE_INPUT_DIR/LKH_candidate_cache"
$REMOTE_RESULTS_ROOT = "/home/$AAI_USERNAME/workspace/TM/final-results/tsp_cpp_candidate_iter005_signal50"

function B($x) { if ($x) { "1" } else { "0" } }
$START_NEW_RUN_BASH = B $StartNewRun
$GIT_PULL_BASH = B (-not $NoGitPull)
$DRY_RUN_BASH = B $DryRun

Write-Host "=== TSP C++ candidate-list iter005 50-repetition run ==="
Write-Host "Remote:        $REMOTE"
Write-Host "Action:        $Action"
Write-Host "Run label:     $RUN_LABEL"
Write-Host "Method:        C1b_candidate_iter005_mnnls_cr"
Write-Host "Reps:          $REPS"
Write-Host "Expected rows: $EXPECTED_TASKS"
Write-Host "Instances:     $INSTANCES"
Write-Host "Core:          $CORE_TO_USE"
Write-Host "Timeout/row:   $TIMEOUT_S s"
Write-Host "Start new run: $StartNewRun"
Write-Host "Dry run:       $DryRun"
Write-Host ""

if ($Action -eq "launch" -and (-not $NoUploadInputs)) {
    Write-Host "=== Local input checks ==="
    if (!(Test-Path $LOCAL_TSP_INSTANCE_DIR)) { Write-Host "ERROR missing $LOCAL_TSP_INSTANCE_DIR"; exit 1 }
    if (!(Test-Path $LOCAL_CANDIDATE_CACHE_DIR)) { Write-Host "ERROR missing $LOCAL_CANDIDATE_CACHE_DIR"; exit 1 }
    foreach ($name in $INSTANCES.Split(",")) {
        $p = Join-Path $LOCAL_TSP_INSTANCE_DIR ("$name.tsp")
        if (!(Test-Path $p)) { Write-Host "ERROR missing $p"; exit 1 }
        Write-Host "OK: $p"
    }
}

Write-Host "=== Creating remote folders on $REMOTE ==="
ssh $REMOTE "mkdir -p /home/$AAI_USERNAME/workspace/TM $REMOTE_INSTANCE_DIR $REMOTE_CANDIDATE_DIR $REMOTE_RESULTS_ROOT /tmp/tsp_cpp_candidate_iter005_launcher"
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
}

if ($Action -eq "launch") {
    if (!(Test-Path $LOCAL_CPP_EVAL)) { Write-Host "ERROR missing $LOCAL_CPP_EVAL"; exit 1 }
    scp "$LOCAL_CPP_EVAL" "${REMOTE}:/tmp/tsp_cpp_candidate_iter005_launcher/tsp_cpp_signal_eval.cpp"
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
CORE_TO_USE="__CORE_TO_USE__"
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
OUT_ROOT="${WORK_ROOT}/final-results/tsp_cpp_candidate_iter005_signal50"
LATEST_FILE="${OUT_ROOT}/LATEST_${RUN_LABEL}.txt"
JOB="TSPC_CPP_C1b_iter005_mnnls_cr"
KIND="heuristic"
SIGNAL="candidate_list"
METHOD="C1b_candidate_iter005_mnnls_cr"
SESSION="tspc1b_iter005"

mkdir -p "$WORK_ROOT" "$INSTANCE_ROOT" "$CANDIDATE_ROOT" "$OUT_ROOT"

if [ "$ACTION" = "launch" ]; then
  cd "$WORK_ROOT"
  if [ ! -d "$REPO_DIR/.git" ]; then git clone "$REPO_URL" llm-tsp-heuristics; fi
  cd "$REPO_DIR"
  if [ "$GIT_PULL" = "1" ]; then git pull || true; fi
  if [ -f /tmp/tsp_cpp_candidate_iter005_launcher/tsp_cpp_signal_eval.cpp ]; then
    cp /tmp/tsp_cpp_candidate_iter005_launcher/tsp_cpp_signal_eval.cpp server_eval/tsp_cpp_signal_eval.cpp
  fi
  echo "=== compiling C++ signal evaluator ==="
  g++ -std=c++17 -O3 -march=native -DNDEBUG -o server_eval/tsp_cpp_signal_eval server_eval/tsp_cpp_signal_eval.cpp
else
  cd "$REPO_DIR"
fi

if [ "$ACTION" = "launch" ] && { [ "$START_NEW_RUN" = "1" ] || [ ! -f "$LATEST_FILE" ]; }; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_ROOT="${OUT_ROOT}/${RUN_LABEL}_${STAMP}"
  mkdir -p "$RUN_ROOT/$JOB"
  echo "$RUN_ROOT" > "$LATEST_FILE"
elif [ -f "$LATEST_FILE" ]; then
  RUN_ROOT="$(cat "$LATEST_FILE")"
  mkdir -p "$RUN_ROOT/$JOB"
else
  echo "ERROR: no LATEST run exists. Use -Action launch -StartNewRun first."
  exit 2
fi

raw_rows(){ local raw="$RUN_ROOT/$JOB/raw_results.csv"; if [ ! -f "$raw" ]; then echo 0; else local l; l=$(wc -l < "$raw" 2>/dev/null || echo 0); if [ "$l" -le 0 ]; then echo 0; else echo $((l-1)); fi; fi; }
print_status(){
  echo "=== TSP C++ CANDIDATE ITER005 STATUS ==="; date; hostname
  echo "RUN_ROOT=$RUN_ROOT"; echo "JOB=$JOB"; echo "EXPECTED_TASKS=$EXPECTED_TASKS"; echo "CORE=$CORE_TO_USE"; echo
  rows=$(raw_rows)
  st="PENDING"
  if [ "$rows" -ge "$EXPECTED_TASKS" ]; then st="COMPLETE"; elif tmux has-session -t "$SESSION" 2>/dev/null; then st="RUNNING"; elif [ "$rows" -gt 0 ]; then st="INCOMPLETE"; fi
  printf "%-35s %-10s %-15s %-34s %5s/%-5s %s\n" "JOB" "KIND" "SIGNAL" "METHOD" "ROWS" "" "STATUS"
  printf '%0.s-' {1..120}; echo
  printf "%-35s %-10s %-15s %-34s %5s/%-5s %s\n" "$JOB" "$KIND" "$SIGNAL" "$METHOD" "$rows" "$EXPECTED_TASKS" "$st"
  echo; echo "=== tmux sessions ==="; tmux ls 2>/dev/null | grep -E 'tspc1b_iter005' || true
}

if [ "$ACTION" = "status" ]; then print_status; exit 0; fi
if [ "$ACTION" = "download" ]; then print_status; exit 0; fi

if [ "$ACTION" = "launch" ]; then
  cat > "$RUN_ROOT/run_one_job.sh" <<SH
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
LOG="$RUN_ROOT/${JOB}.log"
OUT="$RUN_ROOT/$JOB"
mkdir -p "\$OUT"
echo "=== START $JOB ===" | tee -a "\$LOG"
date | tee -a "\$LOG"
echo "core=$CORE_TO_USE signal=$SIGNAL method=$METHOD" | tee -a "\$LOG"
taskset -c "$CORE_TO_USE" ./server_eval/tsp_cpp_signal_eval \\
  --job "$JOB" --kind "$KIND" --signal "$SIGNAL" --method "$METHOD" \\
  --instance-root "$INSTANCE_ROOT" --candidate-root "$CANDIDATE_ROOT" --prior-txt-root "$CANDIDATE_ROOT" \\
  --optima-csv data/tsp_instances_opt.csv --instances "$INSTANCES" --reps "$REPS" \\
  --timeout-s "$TIMEOUT_S" --global-seed "$GLOBAL_SEED" --max-candidates "$MAX_CANDIDATES" \\
  --out-dir "\$OUT" 2>&1 | tee -a "\$LOG"
echo "=== DONE $JOB ===" | tee -a "\$LOG"
SH
  chmod +x "$RUN_ROOT/run_one_job.sh"
  print_status
  if [ "$DRY_RUN" = "1" ]; then echo "DRY RUN: not launching"; exit 0; fi
  rows=$(raw_rows)
  if [ "$rows" -ge "$EXPECTED_TASKS" ]; then echo "Already complete"; print_status; exit 0; fi
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new -d -s "$SESSION" "$RUN_ROOT/run_one_job.sh"
  echo "Launched tmux session: $SESSION"
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
$remoteScript = $remoteScript.Replace("__CORE_TO_USE__", [string]$CORE_TO_USE)
$remoteScript = $remoteScript.Replace("__START_NEW_RUN__", $START_NEW_RUN_BASH)
$remoteScript = $remoteScript.Replace("__GIT_PULL__", $GIT_PULL_BASH)
$remoteScript = $remoteScript.Replace("__DRY_RUN__", $DRY_RUN_BASH)

$tmp = Join-Path $env:TEMP "run_tsp_cpp_candidate_iter005_zeus.sh"
$remoteScript = $remoteScript -replace "`r`n", "`n"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($tmp, $remoteScript, $utf8NoBom)
scp "$tmp" "${REMOTE}:/tmp/tsp_cpp_candidate_iter005_launcher/run_tsp_cpp_candidate_iter005_zeus.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ssh $REMOTE "bash /tmp/tsp_cpp_candidate_iter005_launcher/run_tsp_cpp_candidate_iter005_zeus.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Action -eq "download") {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $dest = Join-Path $LOCAL_RESULTS_DIR "tsp_cpp_candidate_iter005_signal50_$stamp"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $runRoot = ssh $REMOTE "cat $REMOTE_RESULTS_ROOT/LATEST_${RUN_LABEL}.txt"
    scp -r "${REMOTE}:$runRoot" "$dest\"
    Write-Host "Downloaded to $dest"
}
