#!/usr/bin/env bash
set -euo pipefail

# Train models using Prefect flow defined in main.py
# Downloads monthly Capital Bikeshare data, prepares features, and logs to MLflow.
#
# Env/config:
#   TRAIN_FILE  - default: 202204-capitalbikeshare-tripdata.zip
#   VALID_FILE  - default: 202205-capitalbikeshare-tripdata.zip
#   TRACKING_URI - default: http://127.0.0.1:5000 (must be running)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

TRAIN_FILE="${TRAIN_FILE:-202204-capitalbikeshare-tripdata.zip}"
VALID_FILE="${VALID_FILE:-202205-capitalbikeshare-tripdata.zip}"
TRACKING_URI="${TRACKING_URI:-http://127.0.0.1:5001}"
# Use a project-local Prefect home to avoid permission/state issues
export PREFECT_HOME="${PREFECT_HOME:-$PROJECT_ROOT/.prefect}"
if [[ "${PREFECT_RESET:-0}" != "0" ]]; then
  echo "Resetting Prefect state at $PREFECT_HOME"
  rm -rf "$PREFECT_HOME"
fi
mkdir -p "$PREFECT_HOME" || true

if ! command -v pipenv >/dev/null 2>&1; then
  echo "pipenv not found in PATH. Please install pipenv and retry." >&2
  exit 1
fi

# Informational message about MLflow server
echo "Using MLflow Tracking URI: $TRACKING_URI"
echo "Prefect home: $PREFECT_HOME"
echo "Ensure your MLflow server is running (e.g., 'make mlflow')."

# Add project directory to PYTHONPATH for local imports
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT"

# Ensure main.py sees the tracking server
export MLFLOW_TRACKING_URI="$TRACKING_URI"

exec pipenv run python main.py \
  --train_file "$TRAIN_FILE" \
  --valid_file "$VALID_FILE"
