#!/usr/bin/env bash
set -euo pipefail

# Register and stage the best model for a given experiment on a given MLflow server.
# Env/config:
#   TRACKING_URI        - default: http://127.0.0.1:5001
#   EXPERIMENT_NAME     - default: citibikes-experiment-<today>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

TRACKING_URI="${TRACKING_URI:-http://127.0.0.1:5001}"
TODAY=$(date +%F 2>/dev/null || python - <<'PY'
from datetime import datetime
print(datetime.utcnow().strftime('%Y-%m-%d'))
PY
)
EXPERIMENT_NAME="${EXPERIMENT_NAME:-citibikes-experiment-$TODAY}"

echo "Staging best model from experiment: $EXPERIMENT_NAME"
echo "Tracking URI: $TRACKING_URI"

exec pipenv run python stage.py \
  --tracking_uri "$TRACKING_URI" \
  --experiment_name "$EXPERIMENT_NAME"

