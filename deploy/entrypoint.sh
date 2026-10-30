#!/bin/sh
set -eu
# Synthetic artifacts only: HTTP endpoint is public in the example Akash SDL.
python -m train --require-gpu --output "${OUTPUT:-/app/runs/pilot}"
printf 'Training complete. Retrieve synthetic artifacts, then close the lease.\n'
exec python -m http.server 8080 --bind 0.0.0.0 --directory /app/runs
