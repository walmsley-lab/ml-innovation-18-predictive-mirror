# Bounded Akash GPU campaign

This is an **optional single-GPU training/evaluation workflow**, not a new
training framework. Use a laptop/CPU for tests and experiments, and rent a
GPU only for a reviewed frozen campaign. It reuses `campaign.py` /
`sweep.py` (training, validation checkpoint selection and held-out
evaluation) without changing any learned model.

## Before spending

1. Install a recent `akt` with `akt version` and create an **Akash Console
   API context** (`akt context create console --network mainnet
   --auth-method console-api --set-current`; `akt console login`).
   Console credits are required. The key stays in your local CLI context
   or in `AKT_CONSOLE_API_KEY`, never in the repository.
2. Build/publish the project's Dockerfile to GHCR, with a recorded source
   revision, and obtain a **verified immutable image digest**. The pinned
   image **must include the selected campaign JSON**; the worker refuses
   a config whose SHA-256 differs from the local approved manifest.
3. Run the local Python tests before publishing the image. The small
   `experiments/gpu-smoke.json` is intentionally 1 model, 1 seed, 3 steps,
   batch 2, 2 evaluation episodes. It is a technical smoke, not a result.

From repository root, substitute an actual reviewed digest:

```bash
IMAGE="ghcr.io/walmsley-lab/ml-innovation-18-predictive-mirror@sha256:<64-hex>"
python -m unittest tests.test_gpu_campaign -v

# No paid deployment. Validates SDL and invokes akt deploy --dry-run.
python -m deploy.gpu_campaign plan --image "$IMAGE" \
  --config experiments/gpu-smoke.json --out runs/akash-smoke-001 --hours 2

# PAID. Explicit approval, 1-GPU max-bid and Console-side 2-hour TTL.
python -m deploy.gpu_campaign run --approve-paid-deployment \
  --image "$IMAGE" --config experiments/gpu-smoke.json \
  --out runs/akash-smoke-001 --hours 2 --max-runs 1
```

`run` checks the Console context, deploys with structured `akt` JSONL,
records the deployment sequence as soon as the CLI reports it, immediately
sets Console's runtime limit, polls private worker status, downloads **all**
campaign files through authenticated `akt console shell` in verified chunks,
and attempts to close the deployment **even if training/export fails**.

Verify `runs/akash-smoke-001/state.json`:
- `state: closed`: close verified by Console;
- `state: closure_unverified`: **manual attention required**. Check
  `akt console deployment list active` and `akt close <dseq>`.
- `worker_status.status: completed`, `downloaded_files` and
  `artifacts/verified_manifest.json`: completed training and verified export.
  A successful close **does not** imply scientific success.

**Manual recovery** while a lease remains active:

```bash
python -m deploy.gpu_campaign recover --out runs/akash-smoke-001
# To stop billing immediately, even without downloading everything:
python -m deploy.gpu_campaign close --out runs/akash-smoke-001
```

If the paid deploy fails before its sequence number is captured, inspect
`akt console deployment list active` immediately; no local script can
guarantee cleanup of a deployment whose identifier was never received.

## Larger, independently reviewed runs

The legacy `experiments/campaign.json` contains **30 conditions**, not a
single pilot. It does not run the newer RGB or NCA research tickets; those
are owned by the other research PRs. After merging and reviewing their
experiment contracts, add **separately frozen campaign manifests/adapters**
rather than launching every runner in `experiments/` automatically.

For this existing 30-condition matrix, set `--config
experiments/campaign.json --max-runs 30` and choose a reviewed whole-hour
runtime. Plan the expected duration, provider price, checkpoint export volume,
and whether all conditions fit before authorizing. `--max-runs` is a
run-count guard, **not** an estimate of sufficient GPU time.

## Safety and measurement boundaries

- `akt console deployment settings <dseq> <hours>` provides an
  independent whole-hour lifetime on managed Console deployments. The
  client also enforces a polling deadline and closes in `finally`.
  Unexpected provider failures can still lose unexported artifacts.
- The SDL max bid is **2500 uact/block**. Bid rates are not hard total
  spend limits; review actual bids, block time and funded account usage.
  The program uses the cheapest qualifying provider and one NVIDIA GPU.
- `ARTIFACT_HTTP=0` is the default. SDL may still advertise the old
  port, but the campaign runs **no public artifact server**.
  Private shell transfer is slower than object storage and suited to
  modest runs. The first-run download limit is **256 MiB**; change explicitly
  with `--max-artifact-mib`. Private webcam recordings should never be
  included in public container images.
- An active `akt console shell` and the remote worker's bounded hold
  window are required to export. Closing a lease destroys its ephemeral
  storage. Verify results **before** treating the campaign as preserved.
- Immutable image and config digests, source revision, seeds, run specs,
  checkpoints, metric files and SHA-256 export manifest belong with any
  scientific claim. Model weights can be reloaded on CPU, but CPU
  inference/latency and future-camera causality require separate evaluation.
- `gpu-smoke.json` uses the original small synthetic training harness.
  It is not a validated RGB NCA campaign or a live webcam demonstration.

## References

Current `akt` deployment guide:
https://akash.network/docs/developers/deployment/akt/deployments/

Current Console integration and private shell:
https://akash.network/docs/developers/deployment/akt/console/

Previous optional GPU guide: [TRAINING.md](TRAINING.md).
