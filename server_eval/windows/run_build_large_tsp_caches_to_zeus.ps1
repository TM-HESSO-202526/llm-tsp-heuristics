# ============================================================
# Build large TSP LKH/POPMUSIC candidate and edge-prior caches on IICT server
#
# Current protocol:
#   1) finish usa13509 edge prior first, reusing existing partial tours/work dir
#   2) then build pla33810 candidate .cand, then edge-prior .npz
#   3) then build pla85900 candidate .cand, then edge-prior .npz
#
# Edge-prior tours are parallelized inside build_large_tsp_caches.py.
# Default uses 10 cores on zeus: 20..29.
# ============================================================

$AAI_USERNAME = "anthony.atallah"
$SERVER_NAME = "zeus"
$REPO_URL = "https://github.com/TM-HESSO-202526/llm-tsp-heuristics.git"

# Local input root on your PC:
# D:\Users\antho\TM\server_eval_tsp_inputs\TSP_instances
# D:\Users\antho\TM\server_eval_tsp_inputs\LKH_candidate_cache
# D:\Users\antho\TM\server_eval_tsp_inputs\LKH_edge_prior_cache
$LOCAL_INPUT_DIR = "D:\Users\antho\TM\server_eval_tsp_inputs"
$LOCAL_TSP_INSTANCE_DIR = "$LOCAL_INPUT_DIR\TSP_instances"
$LOCAL_CANDIDATE_CACHE_DIR = "$LOCAL_INPUT_DIR\LKH_candidate_cache"
$LOCAL_EDGE_PRIOR_CACHE_DIR = "$LOCAL_INPUT_DIR\LKH_edge_prior_cache"

# Sequential order. Do not change unless you really want a different chain.
$INSTANCES = @("usa13509", "pla33810", "pla85900")

# Historical cache settings used by the project.
$RUNS = 30
$TOPK = 5
$BASE_SEED = 12345
$CANDIDATE_TIMEOUT_S = 14400

# Parallelization for edge-prior tour runs.
# Use 10 cores, avoiding low core IDs often used by clustering jobs.
$PARALLEL_PRIOR_RUNS = 10
$CORE_LIST = "20,21,22,23,24,25,26,27,28,29"

# usa13509 already has partial historical POPMUSIC work/tours. Keep historical mode to reuse them.
$USA_PRIOR_METHOD = "historical_popmusic"
$USA_TIME_LIMIT_S = 86400
$USA_SUBPROCESS_TIMEOUT_S = 90000

# For pla33810 and pla85900, build the .cand first, then build tour-frequency prior using that candidate file.
# This avoids rebuilding POPMUSIC candidates inside every short prior run.
$LARGE_PRIOR_METHOD = "cached_candidate_lkh"
$LARGE_TIME_LIMIT_S = 600
$LARGE_SUBPROCESS_TIMEOUT_S = 1200

# Usually leave these at 0. The script resumes missing work and skips complete files.
$FORCE_CANDIDATE = "0"
$FORCE_PRIOR = "0"

$REMOTE = "$AAI_USERNAME@$SERVER_NAME.iict-heig-vd.in"
$REMOTE_INPUT_DIR = "/home/$AAI_USERNAME/data-local/TM/tsp_input"
$REMOTE_INSTANCE_DIR = "$REMOTE_INPUT_DIR/TSP_instances"
$REMOTE_CANDIDATE_CACHE_DIR = "$REMOTE_INPUT_DIR/LKH_candidate_cache"
$REMOTE_EDGE_PRIOR_CACHE_DIR = "$REMOTE_INPUT_DIR/LKH_edge_prior_cache"
$REMOTE_REPO_ROOT = "/home/$AAI_USERNAME/workspace/TM/llm-tsp-heuristics"
$REMOTE_RESULTS_ROOT = "/home/$AAI_USERNAME/workspace/TM/final-results/tsp_cache_build"
$TMUX_SESSION = "build_large_tsp_caches"

Write-Host "=== Local TSP checks ==="
foreach ($name in $INSTANCES) {
    $p = "$LOCAL_TSP_INSTANCE_DIR\$name.tsp"
    if (!(Test-Path $p)) {
        Write-Host "ERROR: missing $p"
        exit 1
    }
    Write-Host "OK: $p"
}

Write-Host "=== Creating remote folders on $REMOTE ==="
ssh $REMOTE "mkdir -p $REMOTE_INSTANCE_DIR $REMOTE_CANDIDATE_CACHE_DIR $REMOTE_EDGE_PRIOR_CACHE_DIR $REMOTE_RESULTS_ROOT /home/$AAI_USERNAME/workspace/TM"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Uploading selected .tsp files ==="
foreach ($name in $INSTANCES) {
    scp "$LOCAL_TSP_INSTANCE_DIR\$name.tsp" "${REMOTE}:${REMOTE_INSTANCE_DIR}/"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "=== Uploading existing local candidate/prior caches if present ==="
foreach ($name in $INSTANCES) {
    $cand = "$LOCAL_CANDIDATE_CACHE_DIR\${name}_cand-popmusic-k20-s14-sol20-nn5-tr1.cand"
    if (Test-Path $cand) {
        Write-Host "Uploading existing candidate cache: $cand"
        scp "$cand" "${REMOTE}:${REMOTE_CANDIDATE_CACHE_DIR}/"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    $npz = "$LOCAL_EDGE_PRIOR_CACHE_DIR\${name}_popmusic_edge_prior_runs30_topk5.npz"
    if (Test-Path $npz) {
        Write-Host "Uploading existing prior cache: $npz"
        scp "$npz" "${REMOTE}:${REMOTE_EDGE_PRIOR_CACHE_DIR}/"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

# Upload tethys partial usa13509 work directory if available locally.
# This contains the already generated usa13509 run_XXX.tour files.
$TETHYS_TAR = "$LOCAL_INPUT_DIR\tethys_usa13509_cache_artifacts\tethys_usa13509_cache_artifacts_20260531_183524.tar.gz"
if (Test-Path $TETHYS_TAR) {
    Write-Host "=== Uploading/extracting tethys usa13509 partial artifacts ==="
    scp "$TETHYS_TAR" "${REMOTE}:${REMOTE_INPUT_DIR}/"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $remoteExtract = @'
set -euo pipefail
REMOTE_INPUT_DIR="__REMOTE_INPUT_DIR__"
REMOTE_CANDIDATE_CACHE_DIR="__REMOTE_CANDIDATE_CACHE_DIR__"
REMOTE_EDGE_PRIOR_CACHE_DIR="__REMOTE_EDGE_PRIOR_CACHE_DIR__"
TAR="$REMOTE_INPUT_DIR/tethys_usa13509_cache_artifacts_20260531_183524.tar.gz"
TMP="$REMOTE_INPUT_DIR/tethys_usa13509_extract_tmp"
rm -rf "$TMP"
mkdir -p "$TMP"
tar -xzf "$TAR" -C "$TMP"
ROOT="$TMP/tethys_usa13509_cache_artifacts_20260531_183524"
if [ -d "$ROOT/LKH_candidate_cache" ]; then
  cp -n "$ROOT"/LKH_candidate_cache/* "$REMOTE_CANDIDATE_CACHE_DIR"/ 2>/dev/null || true
fi
if [ -d "$ROOT/LKH_edge_prior_cache" ]; then
  cp -rn "$ROOT"/LKH_edge_prior_cache/* "$REMOTE_EDGE_PRIOR_CACHE_DIR"/ 2>/dev/null || true
fi
echo "Existing usa13509 tours:"
find "$REMOTE_EDGE_PRIOR_CACHE_DIR/usa13509_popmusic_edge_prior_runs30_topk5_work" -name 'run_*.tour' 2>/dev/null | wc -l || true
rm -rf "$TMP"
'@
    $remoteExtract = $remoteExtract.Replace("__REMOTE_INPUT_DIR__", $REMOTE_INPUT_DIR)
    $remoteExtract = $remoteExtract.Replace("__REMOTE_CANDIDATE_CACHE_DIR__", $REMOTE_CANDIDATE_CACHE_DIR)
    $remoteExtract = $remoteExtract.Replace("__REMOTE_EDGE_PRIOR_CACHE_DIR__", $REMOTE_EDGE_PRIOR_CACHE_DIR)
    # Send through a real remote file and normalize line endings.
    # Piping Get-Content directly to ssh can leave Windows CRLF characters,
    # which makes bash read "pipefail\r" and fail with "invalid option name".
    $remoteExtract = $remoteExtract -replace "`r`n", "`n"
    $remoteExtract = $remoteExtract -replace "`r", ""
    $tmpExtract = New-TemporaryFile
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($tmpExtract.FullName, $remoteExtract, $utf8NoBom)
    scp $tmpExtract "${REMOTE}:/home/$AAI_USERNAME/extract_tethys_usa13509.sh"
    Remove-Item $tmpExtract
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    ssh $REMOTE "sed -i 's/\r$//' /home/$AAI_USERNAME/extract_tethys_usa13509.sh; bash /home/$AAI_USERNAME/extract_tethys_usa13509.sh"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "=== Preparing remote sequential cache-build job ==="
$fallbackFlag = ""
$forceCandFlag = ""
if ($FORCE_CANDIDATE -eq "1") { $forceCandFlag = "--force-candidate" }
$forcePriorFlag = ""
if ($FORCE_PRIOR -eq "1") { $forcePriorFlag = "--force-prior" }

$remoteCommands = @'
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

AAI_USERNAME="__AAI_USERNAME__"
REPO_URL="__REPO_URL__"
REMOTE_INPUT_DIR="__REMOTE_INPUT_DIR__"
REMOTE_INSTANCE_DIR="__REMOTE_INSTANCE_DIR__"
REMOTE_CANDIDATE_CACHE_DIR="__REMOTE_CANDIDATE_CACHE_DIR__"
REMOTE_EDGE_PRIOR_CACHE_DIR="__REMOTE_EDGE_PRIOR_CACHE_DIR__"
REMOTE_RESULTS_ROOT="__REMOTE_RESULTS_ROOT__"
RUNS="__RUNS__"
TOPK="__TOPK__"
BASE_SEED="__BASE_SEED__"
CANDIDATE_TIMEOUT_S="__CANDIDATE_TIMEOUT_S__"
PARALLEL_PRIOR_RUNS="__PARALLEL_PRIOR_RUNS__"
CORE_LIST="__CORE_LIST__"
USA_PRIOR_METHOD="__USA_PRIOR_METHOD__"
USA_TIME_LIMIT_S="__USA_TIME_LIMIT_S__"
USA_SUBPROCESS_TIMEOUT_S="__USA_SUBPROCESS_TIMEOUT_S__"
LARGE_PRIOR_METHOD="__LARGE_PRIOR_METHOD__"
LARGE_TIME_LIMIT_S="__LARGE_TIME_LIMIT_S__"
LARGE_SUBPROCESS_TIMEOUT_S="__LARGE_SUBPROCESS_TIMEOUT_S__"
FORCE_CANDIDATE_FLAG="__FORCE_CANDIDATE_FLAG__"
FORCE_PRIOR_FLAG="__FORCE_PRIOR_FLAG__"

mkdir -p /home/$AAI_USERNAME/workspace/TM "$REMOTE_RESULTS_ROOT"
cd /home/$AAI_USERNAME/workspace/TM
if [ ! -d llm-tsp-heuristics/.git ]; then
  git clone "$REPO_URL"
fi
cd llm-tsp-heuristics
git pull --quiet 2>/dev/null || true
sed -i 's/\r$//' server_eval/*.sh 2>/dev/null || true

bash server_eval/setup_server_env.sh
source /home/$AAI_USERNAME/data-local/TM/venvs/tsp-final-eval/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
LOG="$REMOTE_RESULTS_ROOT/build_large_tsp_caches_chain_${TS}.log"
echo "$LOG" > "$REMOTE_RESULTS_ROOT/LATEST_large_tsp_cache_chain_log.txt"

echo "=== LARGE TSP CACHE CHAIN ===" | tee -a "$LOG"
date | tee -a "$LOG"
hostname | tee -a "$LOG"
echo "CORE_LIST=$CORE_LIST" | tee -a "$LOG"
echo "PARALLEL_PRIOR_RUNS=$PARALLEL_PRIOR_RUNS" | tee -a "$LOG"

run_cache() {
  local instance="$1"
  local method="$2"
  local time_limit="$3"
  local timeout="$4"
  echo | tee -a "$LOG"
  echo "================================================================================================" | tee -a "$LOG"
  echo "START $instance method=$method" | tee -a "$LOG"
  echo "================================================================================================" | tee -a "$LOG"
  date | tee -a "$LOG"

  set +e
  python -u server_eval/build_large_tsp_caches.py \
    --instances "$instance" \
    --instance-root "$REMOTE_INSTANCE_DIR" \
    --candidate-cache-dir "$REMOTE_CANDIDATE_CACHE_DIR" \
    --edge-prior-dir "$REMOTE_EDGE_PRIOR_CACHE_DIR" \
    --lkh-binary "/home/$AAI_USERNAME/data-local/TM/tools/lkh/LKH" \
    --runs "$RUNS" \
    --topk "$TOPK" \
    --base-seed "$BASE_SEED" \
    --time-limit-s "$time_limit" \
    --subprocess-timeout-s "$timeout" \
    --candidate-timeout-s "$CANDIDATE_TIMEOUT_S" \
    --prior-method "$method" \
    --parallel-prior-runs "$PARALLEL_PRIOR_RUNS" \
    --core-list "$CORE_LIST" \
    $FORCE_CANDIDATE_FLAG \
    $FORCE_PRIOR_FLAG \
    2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e

  if [ "$rc" -ne 0 ]; then
    echo "FAILED $instance with exit code $rc; stopping chain." | tee -a "$LOG"
    exit "$rc"
  fi

  echo "DONE $instance" | tee -a "$LOG"
  date | tee -a "$LOG"
}

# Required order: finish usa first, then pla33810, then pla85900.
run_cache "usa13509" "$USA_PRIOR_METHOD" "$USA_TIME_LIMIT_S" "$USA_SUBPROCESS_TIMEOUT_S"
run_cache "pla33810" "$LARGE_PRIOR_METHOD" "$LARGE_TIME_LIMIT_S" "$LARGE_SUBPROCESS_TIMEOUT_S"
run_cache "pla85900" "$LARGE_PRIOR_METHOD" "$LARGE_TIME_LIMIT_S" "$LARGE_SUBPROCESS_TIMEOUT_S"

echo | tee -a "$LOG"
echo "=== FINAL OUTPUT CHECK ===" | tee -a "$LOG"
ls -lh "$REMOTE_CANDIDATE_CACHE_DIR"/usa13509_cand-popmusic-k20-s14-sol20-nn5-tr1.cand \
       "$REMOTE_CANDIDATE_CACHE_DIR"/pla33810_cand-popmusic-k20-s14-sol20-nn5-tr1.cand \
       "$REMOTE_CANDIDATE_CACHE_DIR"/pla85900_cand-popmusic-k20-s14-sol20-nn5-tr1.cand \
       "$REMOTE_EDGE_PRIOR_CACHE_DIR"/usa13509_popmusic_edge_prior_runs30_topk5.npz \
       "$REMOTE_EDGE_PRIOR_CACHE_DIR"/pla33810_popmusic_edge_prior_runs30_topk5.npz \
       "$REMOTE_EDGE_PRIOR_CACHE_DIR"/pla85900_popmusic_edge_prior_runs30_topk5.npz 2>&1 | tee -a "$LOG"

echo "DONE ALL" | tee -a "$LOG"
'@

$remoteCommands = $remoteCommands.Replace("__AAI_USERNAME__", $AAI_USERNAME)
$remoteCommands = $remoteCommands.Replace("__REPO_URL__", $REPO_URL)
$remoteCommands = $remoteCommands.Replace("__REMOTE_INPUT_DIR__", $REMOTE_INPUT_DIR)
$remoteCommands = $remoteCommands.Replace("__REMOTE_INSTANCE_DIR__", $REMOTE_INSTANCE_DIR)
$remoteCommands = $remoteCommands.Replace("__REMOTE_CANDIDATE_CACHE_DIR__", $REMOTE_CANDIDATE_CACHE_DIR)
$remoteCommands = $remoteCommands.Replace("__REMOTE_EDGE_PRIOR_CACHE_DIR__", $REMOTE_EDGE_PRIOR_CACHE_DIR)
$remoteCommands = $remoteCommands.Replace("__REMOTE_RESULTS_ROOT__", $REMOTE_RESULTS_ROOT)
$remoteCommands = $remoteCommands.Replace("__RUNS__", [string]$RUNS)
$remoteCommands = $remoteCommands.Replace("__TOPK__", [string]$TOPK)
$remoteCommands = $remoteCommands.Replace("__BASE_SEED__", [string]$BASE_SEED)
$remoteCommands = $remoteCommands.Replace("__CANDIDATE_TIMEOUT_S__", [string]$CANDIDATE_TIMEOUT_S)
$remoteCommands = $remoteCommands.Replace("__PARALLEL_PRIOR_RUNS__", [string]$PARALLEL_PRIOR_RUNS)
$remoteCommands = $remoteCommands.Replace("__CORE_LIST__", $CORE_LIST)
$remoteCommands = $remoteCommands.Replace("__USA_PRIOR_METHOD__", $USA_PRIOR_METHOD)
$remoteCommands = $remoteCommands.Replace("__USA_TIME_LIMIT_S__", [string]$USA_TIME_LIMIT_S)
$remoteCommands = $remoteCommands.Replace("__USA_SUBPROCESS_TIMEOUT_S__", [string]$USA_SUBPROCESS_TIMEOUT_S)
$remoteCommands = $remoteCommands.Replace("__LARGE_PRIOR_METHOD__", $LARGE_PRIOR_METHOD)
$remoteCommands = $remoteCommands.Replace("__LARGE_TIME_LIMIT_S__", [string]$LARGE_TIME_LIMIT_S)
$remoteCommands = $remoteCommands.Replace("__LARGE_SUBPROCESS_TIMEOUT_S__", [string]$LARGE_SUBPROCESS_TIMEOUT_S)
$remoteCommands = $remoteCommands.Replace("__FORCE_CANDIDATE_FLAG__", $forceCandFlag)
$remoteCommands = $remoteCommands.Replace("__FORCE_PRIOR_FLAG__", $forcePriorFlag)
$remoteCommands = $remoteCommands -replace "`r`n", "`n"
$remoteCommands = $remoteCommands -replace "`r", ""

$tmp = New-TemporaryFile
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($tmp.FullName, $remoteCommands, $utf8NoBom)
scp $tmp "${REMOTE}:/home/$AAI_USERNAME/build_large_tsp_caches_job.sh"
Remove-Item $tmp
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ssh $REMOTE "sed -i 's/\r$//' /home/$AAI_USERNAME/build_large_tsp_caches_job.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Starting detached tmux session: $TMUX_SESSION ==="
ssh $REMOTE "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true; tmux new -d -s $TMUX_SESSION 'bash /home/$AAI_USERNAME/build_large_tsp_caches_job.sh'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== STARTED ==="
Write-Host "Attach/live view:"
Write-Host ("  ssh {0} 'tmux attach -t {1}'" -f $REMOTE, $TMUX_SESSION)
Write-Host "Monitor logs:"
Write-Host ("  ssh {0} 'LOG=`$(cat {1}/LATEST_large_tsp_cache_chain_log.txt); tail -f `$LOG'" -f $REMOTE, $REMOTE_RESULTS_ROOT)
Write-Host "Check output files:"
Write-Host ("  ssh {0} 'ls -lh {1}/usa13509_cand-popmusic-k20-s14-sol20-nn5-tr1.cand {1}/pla33810_cand-popmusic-k20-s14-sol20-nn5-tr1.cand {1}/pla85900_cand-popmusic-k20-s14-sol20-nn5-tr1.cand {2}/usa13509_popmusic_edge_prior_runs30_topk5.npz {2}/pla33810_popmusic_edge_prior_runs30_topk5.npz {2}/pla85900_popmusic_edge_prior_runs30_topk5.npz 2>/dev/null'" -f $REMOTE, $REMOTE_CANDIDATE_CACHE_DIR, $REMOTE_EDGE_PRIOR_CACHE_DIR)
