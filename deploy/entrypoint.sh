#!/bin/sh
set -eu
# Public endpoint serves SYNTHETIC artifacts only; never place private images here.
mode="${RUN_MODE:-train}"
case "$mode" in
  train|sweep) ;;
  campaign)
    if [ "${CAMPAIGN_APPROVED:-0}" != "1" ]; then
      echo "Full campaign requires CAMPAIGN_APPROVED=1 and explicit CAMPAIGN_MAX_RUNS" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown RUN_MODE: $mode" >&2
    exit 2
    ;;
esac
mkdir -p /app/runs
python -m http.server 8080 --bind 0.0.0.0 --directory /app/runs &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT HUP INT TERM
case "$mode" in
  sweep) python -m sweep --require-gpu --output "${SWEEP_OUTPUT:-/app/runs/sweep}" ;;
  campaign) python -m campaign --config /app/experiments/campaign.json \
              --output /app/runs/campaign --execute --require-gpu \
              --max-runs "${CAMPAIGN_MAX_RUNS:-1}" ;;
  train) python -m train --require-gpu --output "${OUTPUT:-/app/runs/pilot}" ;;
esac
wait "$server_pid"
