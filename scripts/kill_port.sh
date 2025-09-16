#!/usr/bin/env bash
set -euo pipefail

# Kill processes listening on a TCP port (current user).
# Usage: bash scripts/kill_port.sh <port>

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <port>" >&2
  exit 2
fi

PORT="$1"

if ! command -v lsof >/dev/null 2>&1; then
  echo "lsof not found in PATH. Install it and retry." >&2
  exit 1
fi

echo "Scanning listeners on TCP port $PORT..."
PIDS=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | sort -u || true)

if [[ -z "${PIDS}" ]]; then
  echo "No listeners found on port $PORT."
  exit 0
fi

echo "Found PIDs: ${PIDS//$'\n'/, }"
echo "Sending SIGTERM..."
for pid in $PIDS; do
  kill -TERM "$pid" 2>/dev/null || true
done

sleep 1

REMAIN=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | sort -u || true)
if [[ -n "${REMAIN}" ]]; then
  echo "Still listening after SIGTERM, sending SIGKILL..."
  for pid in $REMAIN; do
    kill -KILL "$pid" 2>/dev/null || true
  done
fi

echo "Remaining listeners:"
lsof -nP -iTCP:"$PORT" -sTCP:LISTEN || true

echo "Done."

