#!/usr/bin/env bash
set -euo pipefail

# Serve a trained MLflow run using the current Pipenv (no pyenv/venv creation).
#
# Env/config:
#   MLFLOW_TRACKING_URI  - default: http://127.0.0.1:5001
#   RUN_ID               - if set, serve this run id; otherwise pick best today or latest overall
#   PORT                 - default: 1237

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}"
PORT="${PORT:-1237}"

if [[ -z "${RUN_ID:-}" ]]; then
  echo "Discovering RUN_ID from MLflow at $TRACKING_URI ..."
  RUN_ID=$(pipenv run python - <<'PY'
import os, mlflow
from datetime import datetime
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://127.0.0.1:5001"))
c = MlflowClient()
exp_name = "citibikes-experiment-" + datetime.utcnow().strftime("%Y-%m-%d")
exp = c.get_experiment_by_name(exp_name)
rid = ""
if exp:
    runs = c.search_runs([exp.experiment_id], run_view_type=ViewType.ACTIVE_ONLY,
                         order_by=["metrics.rmse_valid ASC","start_time DESC"], max_results=1)
    rid = runs[0].info.run_id if runs else ""
if not rid:
    exps = c.search_experiments()
    best=None
    for e in exps:
        rs = c.search_runs([e.experiment_id], run_view_type=ViewType.ACTIVE_ONLY,
                           order_by=["start_time DESC"], max_results=1)
        if rs:
            r=rs[0]
            if best is None or r.info.start_time>best.info.start_time:
                best=r
    rid = best.info.run_id if best else ""
print(rid)
PY
  )
fi

if [[ -z "$RUN_ID" ]]; then
  echo "Could not determine a RUN_ID. Ensure you have training runs in MLflow." >&2
  exit 1
fi

echo "Using RUN_ID=$RUN_ID"

# Free the port if needed
bash scripts/kill_port.sh "$PORT" >/dev/null 2>&1 || true

echo "Starting MLflow model server on port $PORT ..."
export MLFLOW_TRACKING_URI="$TRACKING_URI"
exec pipenv run mlflow models serve \
  -m "runs:/$RUN_ID/model" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --env-manager local

