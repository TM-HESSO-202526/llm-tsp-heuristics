# ============================================================
# TSP selected-heuristic evaluation launcher on IICT Zeus
#
# Runs from your Windows PC.
# Uses PRIVATE local TSP input files from:
#   D:\Users\antho\TM\server_eval_tsp_inputs
#
# Supports:
# - signal mode switching: distance_only / candidate_list / edge_prior / edge_prior_plus_candidate_list / all
# - split/instance filters
# - resume into an existing remote output folder
# - server-side candidate timeout enforcement
# ============================================================

# ------------------------------
# User / server settings
# ------------------------------
$AAI_USERNAME = "anthony.atallah"
$SERVER_NAME = "zeus"
$REPO_URL = "https://github.com/TM-HESSO-202526/llm-tsp-heuristics.git"

# ------------------------------
# Private local input folders on your PC
# ------------------------------
$LOCAL_INPUT_DIR = "D:\Users\antho\TM\server_eval_tsp_inputs"
$LOCAL_TSP_INSTANCE_DIR = "$LOCAL_INPUT_DIR\TSP_instances"
$LOCAL_CANDIDATE_CACHE_DIR = "$LOCAL_INPUT_DIR\LKH_candidate_cache"
$LOCAL_EDGE_PRIOR_CACHE_DIR = "$LOCAL_INPUT_DIR\LKH_edge_prior_cache"

# Results copied back here on your PC.
$LOCAL_RESULTS_DIR = "D:\Users\antho\TM\server_eval_results"

# ------------------------------
# Run settings
# SIGNAL_MODE can be:
#   distance_only
#   candidate_list
#   edge_prior
#   edge_prior_plus_candidate_list
#   all
# ------------------------------
$SIGNAL_MODE = "distance_only"
$REPS = 2
$MAX_HEURISTICS = 1000
$MAX_INSTANCES = 1000
$TIMEOUT_S = 300

# Use ALL for all seven repo instances.
# Or for example: "dsj1000,pr1002,d1291"
$INSTANCES = "ALL"

# Use all, train, val, test, or comma list like "train,val"
$SPLITS = "all"

$GLOBAL_SEED = 12345
$MAX_CANDIDATES = 20
$PRIOR_MODE = "frequency"

# Resume mode:
# Leave empty for a new run.
# To continue an interrupted run, paste the exact remote folder here.
$RESUME_REMOTE_DIR = ""

# ------------------------------
# Remote paths
# ------------------------------
$REMOTE = "$AAI_USERNAME@$SERVER_NAME.iict-heig-vd.in"
$REMOTE_INPUT_DIR = "/home/$AAI_USERNAME/data-local/TM/tsp_input"
$REMOTE_INSTANCE_DIR = "$REMOTE_INPUT_DIR/TSP_instances"
$REMOTE_CANDIDATE_CACHE_DIR = "$REMOTE_INPUT_DIR/LKH_candidate_cache"
$REMOTE_EDGE_PRIOR_CACHE_DIR = "$REMOTE_INPUT_DIR/LKH_edge_prior_cache"
$REMOTE_RESULTS_ROOT = "/home/$AAI_USERNAME/workspace/TM/final-results/tsp_eval"
$REMOTE_REPO_ROOT = "/home/$AAI_USERNAME/workspace/TM/llm-tsp-heuristics"

Write-Host "=== Local input checks ==="
if (!(Test-Path $LOCAL_TSP_INSTANCE_DIR)) {
    Write-Host "ERROR: Missing $LOCAL_TSP_INSTANCE_DIR"
    Write-Host "Expected TSPLIB .tsp files there."
    exit 1
}

if (($SIGNAL_MODE -eq "candidate_list" -or $SIGNAL_MODE -eq "edge_prior_plus_candidate_list" -or $SIGNAL_MODE -eq "all") -and !(Test-Path $LOCAL_CANDIDATE_CACHE_DIR)) {
    Write-Host "ERROR: Missing $LOCAL_CANDIDATE_CACHE_DIR"
    Write-Host "Candidate-list modes need POPMUSIC .cand files."
    exit 1
}

if (($SIGNAL_MODE -eq "edge_prior" -or $SIGNAL_MODE -eq "edge_prior_plus_candidate_list" -or $SIGNAL_MODE -eq "all") -and !(Test-Path $LOCAL_EDGE_PRIOR_CACHE_DIR)) {
    Write-Host "ERROR: Missing $LOCAL_EDGE_PRIOR_CACHE_DIR"
    Write-Host "Edge-prior modes need *_popmusic_edge_prior_runs30_topk5.npz files."
    exit 1
}

if (!($SIGNAL_MODE -eq "distance_only" -or $SIGNAL_MODE -eq "candidate_list" -or $SIGNAL_MODE -eq "edge_prior" -or $SIGNAL_MODE -eq "edge_prior_plus_candidate_list" -or $SIGNAL_MODE -eq "all")) {
    Write-Host "ERROR: SIGNAL_MODE must be one of: distance_only, candidate_list, edge_prior, edge_prior_plus_candidate_list, all"
    exit 1
}

Write-Host "=== Creating remote folders on $REMOTE ==="
ssh $REMOTE "mkdir -p $REMOTE_INPUT_DIR $REMOTE_INSTANCE_DIR $REMOTE_CANDIDATE_CACHE_DIR $REMOTE_EDGE_PRIOR_CACHE_DIR $REMOTE_RESULTS_ROOT /home/$AAI_USERNAME/workspace/TM"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Uploading TSP instances to server ==="
scp -r "$LOCAL_TSP_INSTANCE_DIR\." "${REMOTE}:${REMOTE_INSTANCE_DIR}/"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-Path $LOCAL_CANDIDATE_CACHE_DIR) {
    Write-Host "=== Uploading POPMUSIC candidate cache to server ==="
    scp -r "$LOCAL_CANDIDATE_CACHE_DIR\." "${REMOTE}:${REMOTE_CANDIDATE_CACHE_DIR}/"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (Test-Path $LOCAL_EDGE_PRIOR_CACHE_DIR) {
    Write-Host "=== Uploading edge-prior cache to server ==="
    scp -r "$LOCAL_EDGE_PRIOR_CACHE_DIR\." "${REMOTE}:${REMOTE_EDGE_PRIOR_CACHE_DIR}/"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ([string]::IsNullOrWhiteSpace($RESUME_REMOTE_DIR)) {
    $REMOTE_OUT_DIR = ""
    $RESUME_FLAG = "0"
} else {
    $REMOTE_OUT_DIR = $RESUME_REMOTE_DIR
    $RESUME_FLAG = "1"
}

Write-Host "=== Running TSP evaluation on server ==="
Write-Host "Signal mode:     $SIGNAL_MODE"
Write-Host "Repetitions:     $REPS"
Write-Host "Max heuristics:  $MAX_HEURISTICS"
Write-Host "Max instances:   $MAX_INSTANCES"
Write-Host "Instances:       $INSTANCES"
Write-Host "Splits:          $SPLITS"
Write-Host "Timeout seconds: $TIMEOUT_S"
Write-Host "Instance root:   $REMOTE_INSTANCE_DIR"
Write-Host "Candidate cache: $REMOTE_CANDIDATE_CACHE_DIR"
Write-Host "Edge-prior dir:  $REMOTE_EDGE_PRIOR_CACHE_DIR"
if ($RESUME_FLAG -eq "1") {
    Write-Host "Resume folder:   $REMOTE_OUT_DIR"
}

$remoteCommands = @"
set -euo pipefail
mkdir -p /home/$AAI_USERNAME/workspace/TM $REMOTE_INPUT_DIR $REMOTE_RESULTS_ROOT
cd /home/$AAI_USERNAME/workspace/TM
if [ ! -d llm-tsp-heuristics/.git ]; then
  git clone $REPO_URL
fi
cd llm-tsp-heuristics
git pull || true

if [ ! -f server_eval/setup_server_env.sh ]; then
  echo 'ERROR: server_eval/ is missing from the GitHub repo.'
  echo 'Fix: commit/push the TSP server_eval patch, then rerun this script.'
  exit 2
fi

sed -i 's/\r$//' server_eval/*.sh || true

bash server_eval/setup_server_env.sh
source /home/$AAI_USERNAME/data-local/TM/venvs/tsp-final-eval/bin/activate

SIGNAL_MODE=$SIGNAL_MODE \
INSTANCE_ROOT=$REMOTE_INSTANCE_DIR \
CANDIDATE_CACHE_DIR=$REMOTE_CANDIDATE_CACHE_DIR \
EDGE_PRIOR_CACHE_DIR=$REMOTE_EDGE_PRIOR_CACHE_DIR \
OUT_ROOT=$REMOTE_RESULTS_ROOT \
OUT_DIR=$REMOTE_OUT_DIR \
RESUME=$RESUME_FLAG \
REPS=$REPS \
MAX_HEURISTICS=$MAX_HEURISTICS \
MAX_INSTANCES=$MAX_INSTANCES \
TIMEOUT_S=$TIMEOUT_S \
INSTANCES=$INSTANCES \
SPLITS=$SPLITS \
GLOBAL_SEED=$GLOBAL_SEED \
MAX_CANDIDATES=$MAX_CANDIDATES \
PRIOR_MODE=$PRIOR_MODE \
bash server_eval/run_tsp_eval.sh
"@

$remoteOutput = $remoteCommands | ssh $REMOTE "bash -s"
$remoteExit = $LASTEXITCODE
$remoteOutput | ForEach-Object { Write-Host $_ }
if ($remoteExit -ne 0) { exit $remoteExit }

$latestLine = $remoteOutput | Where-Object { $_ -like "LATEST_RESULT_DIR=*" } | Select-Object -Last 1
if ($null -eq $latestLine) {
    Write-Host "ERROR: Could not determine latest remote result folder."
    exit 3
}
$LATEST_REMOTE_DIR = $latestLine.Replace("LATEST_RESULT_DIR=", "").Trim()
$LATEST_FOLDER_NAME = Split-Path $LATEST_REMOTE_DIR -Leaf

Write-Host "=== Downloading latest result folder back to local PC ==="
$LOCAL_TSP_DIR = Join-Path $LOCAL_RESULTS_DIR "tsp_eval"
New-Item -ItemType Directory -Force -Path $LOCAL_TSP_DIR | Out-Null
scp -r "${REMOTE}:${LATEST_REMOTE_DIR}" "$LOCAL_TSP_DIR\"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== DONE ==="
Write-Host "Remote latest result: $LATEST_REMOTE_DIR"
Write-Host "Local copy:           $LOCAL_TSP_DIR\$LATEST_FOLDER_NAME"
