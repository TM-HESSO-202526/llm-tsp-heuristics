# ============================================================
# Build large TSP LKH/POPMUSIC candidate and edge-prior caches on IICT server
#
# Default target: Zeus. Change $SERVER_NAME if you use another server.
# Uploads .tsp files from local PC, then starts a detached tmux job on server.
# ============================================================

$AAI_USERNAME = "anthony.atallah"
$SERVER_NAME = "tethys"  # change manually if you use another server
$REPO_URL = "https://github.com/TM-HESSO-202526/llm-tsp-heuristics.git"

# Local input root on your PC. Put pla7397.tsp and usa13509.tsp here:
# D:\Users\antho\TM\server_eval_tsp_inputs\TSP_instances
$LOCAL_INPUT_DIR = "D:\Users\antho\TM\server_eval_tsp_inputs"
$LOCAL_TSP_INSTANCE_DIR = "$LOCAL_INPUT_DIR\TSP_instances"

# Instances to build on the server.
$INSTANCES = "pla7397,usa13509"

# Historical cache settings used by the project.
$RUNS = 30
$TOPK = 5
$TIME_LIMIT_S = 600
$SUBPROCESS_TIMEOUT_S = 900
$CANDIDATE_TIMEOUT_S = 14400
$BASE_SEED = 12345

# Use historical_popmusic to match the smaller-instance generation method.
# This matches the smaller-instance generation method. Set fallback to 1 only if historical mode fails and you accept a cached-candidate tour-frequency fallback.
$PRIOR_METHOD = "historical_popmusic"
$FALLBACK_CACHED_CANDIDATE = "0"

# Force rebuilding missing/old files. Usually leave candidate false and prior true for debugging/rebuild.
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
foreach ($name in $INSTANCES.Split(',')) {
    $name = $name.Trim()
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
foreach ($name in $INSTANCES.Split(',')) {
    $name = $name.Trim()
    scp "$LOCAL_TSP_INSTANCE_DIR\$name.tsp" "${REMOTE}:${REMOTE_INSTANCE_DIR}/"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "=== Preparing remote cache-build job ==="
$fallbackFlag = ""
if ($FALLBACK_CACHED_CANDIDATE -eq "1") { $fallbackFlag = "--fallback-cached-candidate" }
$forceCandFlag = ""
if ($FORCE_CANDIDATE -eq "1") { $forceCandFlag = "--force-candidate" }
$forcePriorFlag = ""
if ($FORCE_PRIOR -eq "1") { $forcePriorFlag = "--force-prior" }

$remoteCommands = @"
set -euo pipefail
mkdir -p /home/$AAI_USERNAME/workspace/TM $REMOTE_RESULTS_ROOT
cd /home/$AAI_USERNAME/workspace/TM
if [ ! -d llm-tsp-heuristics/.git ]; then
  git clone $REPO_URL
fi
cd llm-tsp-heuristics
git pull --quiet 2>/dev/null || true
sed -i 's/\r$//' server_eval/*.sh 2>/dev/null || true

bash server_eval/setup_server_env.sh
source /home/$AAI_USERNAME/data-local/TM/venvs/tsp-final-eval/bin/activate

mkdir -p $REMOTE_RESULTS_ROOT
LOG="$REMOTE_RESULTS_ROOT/build_large_tsp_caches_date +%Y%m%d_%H%M%S.log"
# The weird placeholders are replaced below to avoid PowerShell expanding bash date.
"@
$remoteCommands = $remoteCommands.Replace(([string][char]27), '$(')
$remoteCommands += @"

python -u server_eval/build_large_tsp_caches.py \
  --instances "$INSTANCES" \
  --instance-root "$REMOTE_INSTANCE_DIR" \
  --candidate-cache-dir "$REMOTE_CANDIDATE_CACHE_DIR" \
  --edge-prior-dir "$REMOTE_EDGE_PRIOR_CACHE_DIR" \
  --lkh-binary "/home/$AAI_USERNAME/data-local/TM/tools/lkh/LKH" \
  --runs $RUNS \
  --topk $TOPK \
  --base-seed $BASE_SEED \
  --time-limit-s $TIME_LIMIT_S \
  --subprocess-timeout-s $SUBPROCESS_TIMEOUT_S \
  --candidate-timeout-s $CANDIDATE_TIMEOUT_S \
  --prior-method $PRIOR_METHOD \
  $fallbackFlag \
  $forceCandFlag \
  $forcePriorFlag \
  2>&1 | tee "$LOG"
"@
$remoteCommands = $remoteCommands -replace "`r`n", "`n"

$tmp = New-TemporaryFile
Set-Content -Path $tmp -Value $remoteCommands -NoNewline -Encoding UTF8
scp $tmp "${REMOTE}:/home/$AAI_USERNAME/build_large_tsp_caches_job.sh"
Remove-Item $tmp
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Starting detached tmux session: $TMUX_SESSION ==="
ssh $REMOTE "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true; tmux new -d -s $TMUX_SESSION 'bash /home/$AAI_USERNAME/build_large_tsp_caches_job.sh'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== STARTED ==="
Write-Host "Attach/live view:"
Write-Host ("  ssh {0} 'tmux attach -t {1}'" -f $REMOTE, $TMUX_SESSION)
Write-Host "Check logs:"
Write-Host ("  ssh {0} 'ls -lh {1}; tail -80 {1}/*.log'" -f $REMOTE, $REMOTE_RESULTS_ROOT)
Write-Host "Check output files:"
Write-Host ("  ssh {0} 'ls -lh {1}/usa13509_cand-popmusic-k20-s14-sol20-nn5-tr1.cand {2}/pla7397_popmusic_edge_prior_runs30_topk5.npz {2}/usa13509_popmusic_edge_prior_runs30_topk5.npz 2>/dev/null'" -f $REMOTE, $REMOTE_CANDIDATE_CACHE_DIR, $REMOTE_EDGE_PRIOR_CACHE_DIR)

