#!/usr/bin/env bash
set -euo pipefail

# Always run from the project root where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="hallucination-lens"
ENV_PYTHON="/opt/anaconda3/envs/${ENV_NAME}/bin/python"
PORT="${1:-8000}"

# Verify the target conda environment exists
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "Error: Could not find Python at $ENV_PYTHON"
  echo "Check your conda environments with: conda env list"
  exit 1
fi

# Warn if another process is already holding the port
if lsof -i :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "Port $PORT is already in use. Stop the existing server first or run: ./start_studio.sh 8001"
  exit 1
fi

echo "=================================================="
echo "  Hallucination Lens Studio (100% Offline)"
echo "  Python : $ENV_PYTHON"
echo "  URL    : http://127.0.0.1:${PORT}"
echo "=================================================="

# Use exec so Ctrl+C stops uvicorn cleanly
exec "$ENV_PYTHON" -m uvicorn app.server:app --reload --port "$PORT"