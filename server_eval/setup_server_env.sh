#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/data-local/TM/venvs/tsp-final-eval}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$(dirname "$VENV_DIR")"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "$REPO_DIR/requirements.txt"
python -m pip install -e "$REPO_DIR"

echo "Environment ready. Activate it with:"
echo "source $VENV_DIR/bin/activate"
