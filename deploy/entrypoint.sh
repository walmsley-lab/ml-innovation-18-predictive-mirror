#!/bin/sh
set -eu

mode="${RUN_MODE:-train}"
case "$mode" in
  campaign)
    if [ "${CAMPAIGN_APPROVED:-0}" != "1" ] || [ -z "${CAMPAIGN_SHA256:-}" ]; then
      echo "Campaign requires approved manifest SHA and CAMPAIGN_APPROVED=1" >&2
      exit 2
    fi
    exec python -m deploy.gpu_worker train \
      --config "${CAMPAIGN_CONFIG:-/app/experiments/campaign.json}" \
      --output "${CAMPAIGN_OUTPUT:-/app/runs/campaign}" \
      --max-runs "${CAMPAIGN_MAX_RUNS:-1}" \
      --sha256 "$CAMPAIGN_SHA256" \
      --hold-seconds "${CAMPAIGN_HOLD_SECONDS:-3600}"
    ;;
  train|sweep) ;;
  *) echo "Unknown RUN_MODE: $mode" >&2; exit 2 ;;
esac

# Legacy manual workflows can explicitly opt into public HTTP for synthetic
# artifacts. The managed GPU campaign above never starts an HTTP server.
server_pid=""
if [ "${ARTIFACT_HTTP:-0}" = "1" ]; then
  mkdir -p /app/runs
  python -m http.server 8080 --bind 0.0.0.0 --directory /app/runs &
  server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true' EXIT HUP INT TERM
fi

case "$mode" in
  train) python -m train --require-gpu --output "${OUTPUT:-/app/runs/pilot}" ;;
  sweep) python -m sweep --require-gpu --output "${SWEEP_OUTPUT:-/app/runs/sweep}" ;;
esac

# Preserve the existing opt-in manual artifact server, not a paid-run default.
if [ -n "$server_pid" ]; then
  wait "$server_pid"
fi
