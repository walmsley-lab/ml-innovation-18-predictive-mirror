# Akash developmental evidence campaign

This extends the existing `deploy/gpu_campaign.py` lifecycle. It does not add
a second deployment/orchestration framework.

## Local validation first

```bash
python -m unittest tests.test_akash_rgb_evidence \
  tests.test_gpu_campaign tests.test_developmental_evidence -v

# Technical smoke, CPU-only and explicitly not evidence of GPU execution:
python -m experiments.developmental_evidence \
  --config experiments/akash-developmental-smoke.json \
  --out runs/akash-developmental-smoke-local --max-runs 4
```

The smoke writes four conditions, stage checkpoints, periodic checkpoints,
issued-forecast errors, object metrics, frozen-connectivity diagnostics, and
shuffled-action diagnostics. `--require-gpu` refuses to run on a CPU-only host.

The intermediate manifest is exactly 20 conditions: five seeds × four arms.
It is intentionally not launched by this change.

## Akash plan/run/recover

Use a published immutable image digest; never use a tag:

```bash
export IMAGE_DIGEST='ghcr.io/walmsley-lab/ml-innovation-18-predictive-mirror@sha256:<64-hex-digest>'
export AKT_CONSOLE_API_KEY='...'

akt context create developmental-console --deploy-via console
akt context use developmental-console

python -m deploy.gpu_campaign plan \
  --config experiments/akash-developmental-smoke.json \
  --image "$IMAGE_DIGEST" --out runs/akash-developmental-smoke \
  --max-runs 4 --hours 1

python -m deploy.gpu_campaign run \
  --config experiments/akash-developmental-smoke.json \
  --image "$IMAGE_DIGEST" --out runs/akash-developmental-smoke \
  --max-runs 4 --hours 1 --poll-seconds 20 --recovery-seconds 60 \
  --approve-paid-deployment

python -m deploy.gpu_campaign recover \
  --out runs/akash-developmental-smoke \
  --poll-seconds 20 --recovery-seconds 60
```

After the technical smoke is reviewed, substitute
`experiments/akash-developmental-pilot.json` and `--max-runs 20` for the
intermediate pilot. The `run` command requires explicit paid approval, pins
the config and image hashes, requires a JAX GPU inside the worker, periodically
recovers verified artifact snapshots, and closes/verifies the lease in a
`finally` path. `recover` is safe to repeat; completed conditions are skipped
from the developmental evidence report.

Inspect without public HTTP artifacts:

```bash
jq '{status,condition_count,conditions}' \
  runs/akash-developmental-smoke/evidence.json
find runs/akash-developmental-smoke -name 'checkpoint-*.npz' -print
akt console deployment list active -o json
git diff --check
```

An active deployment after a completed run is a failure requiring manual
closure verification; the campaign must not be treated as complete.
