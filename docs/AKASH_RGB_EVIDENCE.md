# Akash RGB forecasting evidence campaign

**Purpose:** train current RGB CNN, transport and cellular baselines with
held-out foreground/object evidence, then run the *separate* paired-history
object-reappearance study. This is NOT the older 30-condition `campaign.py`
sweep. No performance or webcam success is asserted by deploying a container.

## What this campaign actually trains

- `experiments/akash-rgb-smoke.json`: 2 short conditions, technical smoke.
- `experiments/akash-rgb-pilot.json`: **6 conditions**: 3 seeds × CNN and
  blend-head NCA, 300 steps each. Inspect whether foreground stays localized
  before approving the larger paid matrix.
- `experiments/akash-rgb-full.json`: **18 conditions**: 3 initializations ×
  [CNN, transport, plain NCA, blend-head NCA] on mixed-world training (1,200
  updates each), plus 3 × [CNN, blend NCA] on paired-history occlusions
  (800 updates). The plain NCA is a control, not a predicted winner.
- `rgb_world_ladder_training` selects checkpoints by held-out *foreground*
  error, then evaluates unseen motion, appearances, long occlusions and
  unknown camera action. The `object_permanence_campaign` independently
  measures paired return timing/location against supplied-history linear
  extrapolation and persistence.
- Logs, best checkpoints, train-state files, per-domain metrics, PNGs and
  incremental `evidence.json` are saved **per condition**. On a later failure
  previous conditions remain available for the worker's private export.
- `evidence.json` explicitly surfaces full-image MSE, object mass, localization,
  recall and causal task-specific event rates. **Low RGB MSE is not a promotion
  gate.** Reappearance event coverage must be nonzero before interpreting a rate.
  No automatic winner is picked from untouched test results.

## Dry run (no Akash cost)

```bash
python -m unittest tests.test_akash_rgb_evidence tests.test_gpu_campaign -v
# Optional tiny local CPU wiring check (GPU flag omitted deliberately):
python -m experiments.akash_rgb_evidence \
  --config experiments/akash-rgb-smoke.json \
  --out runs/akash-rgb-local-smoke --max-runs 2
```

## Managed Akash Console

Follow [GPU_CAMPAIGN.md](GPU_CAMPAIGN.md) to establish an `akt` Console
context, publish a Docker image that includes this commit, and verify a
**real immutable GHCR digest**. Never use a tag or dummy SHA in a paid run.
The existing template requests one RTX 3090 and quotes a **maximum bid**,
not a hard spend limit. Confirm bids, account funds, export size and TTL.

```bash
IMAGE='ghcr.io/walmsley-lab/ml-innovation-18-predictive-mirror@sha256:<verified-64-hex>'
python -m deploy.gpu_campaign plan \
  --image "$IMAGE" --config experiments/akash-rgb-smoke.json \
  --out runs/akash-rgb-smoke --max-runs 2 --hours 2

# After inspecting the generated SDL, provider quote and dry-run:
python -m deploy.gpu_campaign run --approve-paid-deployment \
  --image "$IMAGE" --config experiments/akash-rgb-smoke.json \
  --out runs/akash-rgb-smoke --max-runs 2 --hours 2

# Before full spend: an intermediate 6-condition pilot.
python -m deploy.gpu_campaign plan \
  --image "$IMAGE" --config experiments/akash-rgb-pilot.json \
  --out runs/akash-rgb-pilot --max-runs 6 --hours 3
# Use 'run --approve-paid-deployment' after review and a valid image digest.

# Only after smoke + pilot recoveries and evaluation:
python -m deploy.gpu_campaign plan \
  --image "$IMAGE" --config experiments/akash-rgb-full.json \
  --out runs/akash-rgb-full --max-runs 18 --hours 8
# Paid full run uses 'run --approve-paid-deployment' with the same arguments.
```

The `run` command attempts verified private download and deployment close
even on failures. If interrupted, use:
```bash
python -m deploy.gpu_campaign recover --out runs/akash-rgb-full
python -m deploy.gpu_campaign close --out runs/akash-rgb-full
```
Check `state.json`, `artifacts/verified_manifest.json`, downloaded checkpoints,
and that the deployment was actually closed. The image digest must be pinned,
the experiment config hash must match, `JAX_PLATFORMS=cuda` fails closed, and
the private retrieval limit defaults to 256 MiB. Explicitly approve a larger
export limit if needed.

## Webcam is a distinct validation stage

A synthetic scene model is **not** a validated live-webcam predictor.
The existing live viewer currently loads the plain RGB NCA checkpoint format,
not arbitrary CNN, transport or blend variants. For now, use the retrieved
plain NCA as a *demo-only* starting point, or train a camera-specific plain NCA
with `experiments.rgb_nca fit`, then independently evaluate on a second
recording via `experiments.rgb_transfer_eval`. The live viewer verifies
forecast-issue time vs subsequent camera frames; it does not guarantee sensor
exposure timestamps. Do not upload private webcam videos to public containers.

**Promotion checklist:** no step-zero selection; non-collapsed, spatially
localized predicted foreground; genuine covered reappearances; competitive
motion-region error vs observation-only constant velocity; repeatability
over three seeds; viable GPU and local inference latency; independent camera
recording before calling a live demo an accurate predictor. Longer training
is *not* a cure for a dead gradient or an all-black/gray fixed point.

## Durability and stop conditions

The private export happens while the Akash lease is reachable. It does not
currently stream checkpoints to durable object storage during training. A
provider outage/lease expiration may lose unexported in-container outputs,
even though the runner writes per-condition results incrementally. Confirm
the provider runtime and recovery window are sufficient; recover promptly.
A 3-step smoke proves wiring, not convergence. If the 300-step pilot exhibits
all-black/gray output, missing objects, or no improvement against velocity
on relevant motion, **do not simply scale to 1,200 steps without inspecting
the failure mode first.** The full run is bounded research, not a guaranteed win.

## Expanded architecture, world-exposure and resolution sequence

All v1 configs remain frozen. The v2 configs change **one scientific axis per stage**:

| Manifest | Conditions | Question |
|---|---:|---|
| `akash-rgb-architectures.json` | 12 | Six RGB models, 2 seeds, original five training worlds, 32px |
| `akash-rgb-physics.json` | 18 | Six models, 3 seeds, all eight existing world domains, 32px |
| `akash-rgb-occlusion-all.json` | 18 | Six models on paired-history occlusion, fixed 32px |
| `akash-rgb-48px.json` | 9 | Three models at 48px, all eight domains |
| `akash-rgb-64px.json` | 6 | CNN and transport at 64px, all eight domains |

This expands **exposure to existing worlds**, not the realism of their physics engine. The current engine has constant acceleration, simple equal-mass collisions, boundary reflections, camera pan, occlusion and appearance changes. It does not model gravity, deformable bodies, lighting physics or real optics. Additional genuine physics should be a separately versioned world with a controlled train/held-out evaluation split.

The 48px and 64px conditions use the existing world ladder, which supports those sizes; the *paired occlusion* harness is locked at 32px. Train and validation seed namespaces remain separated from test. Once a domain is included in mixed training, it is **not** an unseen-domain test, even though test episodes remain unseen by seed. A model selected after looking at test results is exploratory; reserve an untouched external test for final claims.

GPU time scales with scene area (64px is four times the pixels of 32px) and unroll length, and may be dominated by compilation or model-specific kernels. Do not extrapolate the three-step smoke timing into a full run. These manifests are **not automatically launched**, and no performance/speed is claimed.

Plan the 12-condition architecture screening stage, first (no cloud spend):

```bash
python -m deploy.gpu_campaign plan \
  --image "$IMAGE" \
  --config experiments/akash-rgb-architectures.json \
  --out runs/akash-rgb-architectures-001 --max-runs 12 --hours 6
```

Before approving a paid run, review estimated provider cost and TTL. Inspect selected-step-zero frequency, +1/+3/+5/+10 foreground MSE, object localization, predicted mass, velocity controls and event coverage. If a model remains uniform/empty/step-zero, pause that family for output/loss debugging rather than buying more steps.
