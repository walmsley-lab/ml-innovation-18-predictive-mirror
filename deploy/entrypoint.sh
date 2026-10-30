#!/bin/sh
set -eu
# Read-only public synthetic-artifact endpoint; do not serve private data.
# Allow checkpoint downloads DURING training, not only after completion.
mkdir -p "${OUTPUT:-/app/runs/pilot}"
python -m http.server 8080 --bind 0.0.0.0 --directory /app/runs &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT HUP INT TERM
if [ "${RUN_MODE:-train}" = "sweep" ]; then
  python -m sweep --require-gpu --output "${SWEEP_OUTPUT:-/app/runs/sweep}"
else
  python -m train --require-gpu --output "${OUTPUT:-/app/runs/pilot}"
fi
wait "$server_pid"
