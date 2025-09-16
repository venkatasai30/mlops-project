#!/usr/bin/env bash
set -euo pipefail

# Start a local MLflow Tracking Server backed by SQLite and local artifacts.
#
# Configurable via env vars (with defaults):
#   HOST         - default: 127.0.0.1
#   PORT         - default: 5000
#   BACKEND_DB   - default: sqlite:///mlflow.db (in project root)
#   ARTIFACT_DIR - default: ./mlartifacts (in project root)
#
# Examples:
#   bash scripts/run_mlflow_local.sh
#   PORT=5001 ARTIFACT_DIR=/tmp/mlartifacts bash scripts/run_mlflow_local.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"
# Use absolute paths for reliability
BACKEND_DB="${BACKEND_DB:-sqlite:///$PROJECT_ROOT/mlflow.db}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$PROJECT_ROOT/mlartifacts}"

mkdir -p "$ARTIFACT_DIR"

# Ensure we don't accidentally use S3; keep it local
unset MLFLOW_S3_ENDPOINT_URL AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION AWS_REGION || true

if ! command -v pipenv >/dev/null 2>&1; then
  echo "pipenv not found in PATH. Please install pipenv and retry." >&2
  exit 1
fi

# Verify mlflow is available in the Pipenv environment
if ! pipenv run python -c 'import mlflow' >/dev/null 2>&1; then
  echo "mlflow is not available in the pipenv environment. Run 'pipenv sync' first." >&2
  exit 1
fi

# Detect artifact serving flag supported by this mlflow version
HELP_OUTPUT="$(pipenv run mlflow server --help 2>/dev/null || true)"
if echo "$HELP_OUTPUT" | grep -q -- "--artifacts-destination"; then
  ARTIFACT_FLAGS=(--serve-artifacts --artifacts-destination "$ARTIFACT_DIR")
else
  ARTIFACT_FLAGS=(--default-artifact-root "$ARTIFACT_DIR")
fi

echo "Starting MLflow Tracking Server..."
echo " - Tracking URI: http://$HOST:$PORT"
echo " - Backend DB  : $BACKEND_DB"
echo " - Artifacts   : $ARTIFACT_DIR"
echo
echo "Tip: export MLFLOW_TRACKING_URI=\"http://$HOST:$PORT\" to log runs to this server."

exec pipenv run mlflow server \
  --host "$HOST" \
  --port "$PORT" \
  --backend-store-uri "$BACKEND_DB" \
  "${ARTIFACT_FLAGS[@]}"

