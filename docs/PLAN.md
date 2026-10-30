# Predictive Mirror — research charter and PR plan

Date: 2026-10-30 · Repository: `walmsley-lab/ml-innovation-18-predictive-mirror`

## Thesis

Can a spatially organized, recurrent perceptual system develop *distinct but mutually compatible* representations from a staged sensory curriculum—and subsequently improve its predictions by actively probing its world?

The camera supplies evidence; the model predicts the next observation **before** it arrives; comparison with reality supplies a learning signal. We use brain organization as *structural priors* (locality, retinotopy, sparse pathways, different receptive fields and timescales), **not** as hardcoded labels like “object cortex.” Reconciliation means the process by which heterogeneous internal representations constrain one another; coherence is a measurable outcome. We do not claim consciousness or biological equivalence.

## Minimum demonstration and research criteria

**CIMC mirror:** actual camera frame, previously committed prediction, per-pixel error, a visualization of internal activity, and frame timestamp/latency. Do not render a camera frame as if it were a forecast.

**Research:** held-out sequences and novel combinations; identical exposure budgets for curricula; same total parameters for single-sheet and multi-sheet controls; camera action labels but no privileged object labels during training. Never treat training-loss improvement as proof of representational emergence.

**Operationalized perception:** an internal state supports predictions under occlusion and viewpoint change, and downstream behavior is causally sensitive to recurrent feedback. **Operationalized reconciliation:** predictions decoded from different populations become more compatible over recurrent inference iterations, and perturbing cross-population communication disrupts this behavior. A system can have distinct internal codes and still form coherent predictions.

## KISS architecture

The simulator generates truth but returns only observed pixels and camera action to training. A minimal cell update is

```
h_(t+1) = update(local_neighbors(h_t), pixels_t, camera_action_t)
next_pixels = readout(h_(t+1))
```

The first model is one locally connected recurrent sheet with shared update weights and no online plasticity. Later PRs add a second sheet with different timescale/receptive field and *sparse* reciprocal connections. A separate ablation introduces activity-dependent plasticity (e.g., decaying per-edge traces). The developmental world supports scripted camera motions from day one, then selectable gaze actions in a controlled comparison.

## Start gates and relationship to Neural Dynamics (#17)

This repo is a computational testbed for selected hypotheses from `ml-innovation-17-neural-dynamics`; it does not replace that repo's biological evidence, observability analysis or claims ledger. Cite the exact #17 experiment and distinguish simulated behavior from evidence about brains. In particular, #17 H2 (memory and prediction), H3 (self-generated sensation) and H4 (plasticity/stability) supply useful controls.

**Before PR #2:** Review/merge the draft bootstrap PR after CPU CI and a reproducible train/evaluate run. Treat the synthetic predictor as an untrained pilot until a meaningful comparison is run. The live mirror must make and timestamp each prediction *before* the target webcam frame arrives. Include a simple real-video persistence/velocity baseline; synthetic-to-webcam transfer is a separate empirical question, not an assumption.

**Before claiming self/world discrimination (PR #3):** Generate a crossed set of object-still/object-moving × camera-still/camera-moving episodes, including held-out rates and camera-action delays. The present bootstrap world always moves the object; it does **not** yet test discrimination between self-generated and external motion. Only pixel observations and action commands enter training; hidden truth is for evaluation.

**Before claiming reconciliation (PRs #4–5):** Compare matched-capacity single-sheet and dual-sheet models; fit diagnostic decoders *after* training, measure predictive compatibility across recurrent iterations, and compare temporary feedback disconnection with networks trained without feedback. Separate results for visible motion, occlusion and ego-motion; do not infer functional specialization from attractive activation maps.

**Before substantive paid Akash training:** Publish and pull-test the actual image by immutable digest, validate the SDL and deployment dry run, verify GPU execution with a tiny paid smoke, and establish off-provider checkpoint export/recovery. The current pilot saves only at completion and serves artifacts from a lease that continues billing: this is insufficient for long runs. Move the minimal durable-checkpoint/export gate ahead of PR #8; PR #8 can later harden automation. Review bids, set a manual spending/time cap and close the lease.

## PR sequence and indicative effort

Estimates below are **engineering effort after dependency gates**, not promises of calendar completion. A hackathon can prioritize PRs 1–2; robust emergence experiments extend beyond it.

| PR | Scope / files | Estimated effort | Acceptance gate |
|---|---|---:|---|
| **#1 Bootstrap + Akash smoke** | `src/world.py`, `src/model.py`, `train.py`, `tests/`, `Dockerfile`, `deploy/`, `docs/PLAN.md`, CI | 3–5 h | Local seeded simulation, train/evaluate and artifact output; GPU image recipe and valid SDL. No paid deployment. |
| **#2 Honest mirror MVP** | `mirror.py` plus webcam adapter and UI | 3–6 h | Causal webcam forecast with timestamps, actual/predicted/error displayed; persistence/velocity real-video controls, CPU fallback and latency logged. Synthetic-to-camera transfer is measured. |
| **#3 Evaluation + curriculum** | `evaluate.py`, `curriculum.py`, tests | 4–8 h | Persistence, constant-velocity, matched-history feedforward and recurrent controls; crossed object/camera motion cases; staged vs shuffled equal exposure; 5 seeds, CIs, fresh holdout scenes. |
| **#4 Regional scaffold** | `src/model.py`, `experiments/regions.py` | 6–10 h | Matched-parameter single-sheet vs two-sheet network, sparse feedback, timescale and receptive-field ablations. |
| **#5 Reconciliation test** | `experiments/reconciliation.py` | 6–10 h | Post-hoc decoders and compatibility across inference steps; visible/occluded/ego-motion dissociation; inference-time disconnection and feedback-free-trained control. |
| **#6 Active perception** | `src/world.py`, `experiments/gaze.py` | 6–10 h | Fixed/random/selected gaze, equal sensor and action budgets, performance on unseen scenes. |
| **#7 Cellular plasticity** | `src/plasticity.py`, `experiments/plasticity.py` | 6–12 h | Fixed vs plastic local links with otherwise identical architecture and budget, plus forgetting/stability tests. |
| **#8 Reproducibility / scale** | `deploy/`, `reports/`, updated CI | 4–8 h | Harden already-gated image and checkpoint procedures, automate recovery, repeat Akash GPU runs, preserve cost/seed manifests and negative results. |

**Calendar sketch:** Oct 30: PR 1 and, if possible, PR 2; Oct 31–Nov 1: PRs 3–4; Nov 2–4: PRs 5–7; Nov 5 onward: PR 8 and replication. Dependencies and debugging may extend this. Minimum credible mirror: PRs 1–2. Minimum *reconciliation* evidence: PRs 1, 3, 4, 5. Stronger developmental + embodied investigation: PRs 1–7.

## Synthetic priming curriculum

0. Spontaneous spatially correlated activity (optional control; no object semantics).
1. Blurred contrast, bars and edges.
2. Constant and changing motion.
3. Occlusion, reappearance and object persistence.
4. Scripted ego-motion vs object motion.
5. Restricted gaze and information-seeking actions.
6. Multiple objects, appearance changes and compositional test scenes.

Every stage contains replay from earlier stages. Compare to shuffled/mixture training with identical samples, updates and compute. Do not assert that ordering helps until replicated.

## Precommitted readouts and failure cases

- **Forecast:** foreground-weighted MSE (reported alongside plain MSE), positive-object pixel error, rollouts at 1/3/6 steps, occlusion reappearance position and timing, runtime and peak memory. Report persistence and velocity baselines.
- **Representation:** post-hoc probe decoding of motion, world position, viewpoint and appearance; no privileged labels in training; held-out combinations. Compare probes across sheets, not raw latent-vector similarity.
- **Reconciliation:** cross-sheet decoded prediction compatibility over recurrent iterations; feedback cut at evaluation and cut during training; matched single-sheet control. Measure downstream prediction change.
- **Exploration:** fixed vs random vs uncertainty-directed gaze with identical resolution, observation budget and actions; evaluate world-state inference and prediction on held-out scenes.
- **Plasticity:** online adaptation speed, forgetting, perturbation recovery, parameter/compute counts. Compare fixed, online-plastic, and shuffled-update controls.
- **Failure criteria:** no reliable advantage over appropriate baselines, no stable cross-sheet specialization, no intervention effect, or equal performance after curriculum shuffling means **the corresponding hypothesis is not supported**. Report that result.
- **Statistics:** use at least five independent seeds for exploratory runs, then more as needed; publish effect sizes and uncertainty intervals, not a single cherry-picked animation.

## Minimal repository organization

```
README.md               # run instructions
train.py                # train/evaluate/save pilot
src/world.py            # synthetic environment and action generation
src/model.py            # pure recurrent transition and readout
src/plasticity.py       # only when PR 7 needs it
tests/                  # small deterministic tests
experiments/            # self-contained studies, introduced only when needed
deploy/akash-gpu.yaml   # deployment example
deploy/entrypoint.sh    # train then serve synthetic artifacts
Dockerfile
.github/workflows/      # CPU tests and image publishing
docs/PLAN.md            # this document
```

No base classes, registries, plugins, large training frameworks or prematurely extracted abstractions. Functions accept explicit inputs and return outputs; I/O stays in scripts. Use JAX, NumPy, Python standard library. PR #1's recurrent sheet is **not** evidence of emergent specialization or plasticity.

## Akash training runbook

**Current status:** Dockerfile, CI image-build recipe, training entrypoint and SDL are in PR #1. These have **not been built on Akash**, and no GPU lease or paid job has been started. Local CPU tests and a tiny training smoke run are the initial verification. The Akash `akt` command is unavailable in the authoring environment, so run its validation locally before accepting a lease.

1. Run locally: `python -m pip install 'jax==0.9.0.1' 'numpy==2.3.5'`; then `python -m unittest discover -s tests -v` and `python -m train --steps 3 --batch-size 2 --output runs/smoke`. Python 3.11.
2. Build and smoke a **CPU** container if Docker is available (the supplied Dockerfile installs GPU-capable JAX wheels): `docker build -t predictive-mirror:pilot .` then `docker run --rm --gpus all -p 8080:8080 predictive-mirror:pilot` on an NVIDIA host; it should reject a GPU-less host rather than silently using CPU.
3. Merge the PR and allow the GitHub Actions image-publishing job to produce `ghcr.io/walmsley-lab/ml-innovation-18-predictive-mirror:sha-<commit>`. Confirm the package is publicly pullable, image exists, and capture its digest. The `bootstrap-1` image tag in the SDL is a **placeholder**, not a published image.
4. Change `services.trainer.image` in `deploy/akash-gpu.yaml` to the **published immutable `@sha256:` image**. Optionally adjust GPU model and maximum pricing cap. Install/configure `akt` as documented by Akash; run `akt sdl validate deploy/akash-gpu.yaml` then `akt deploy deploy/akash-gpu.yaml --dry-run`.
5. Inspect live provider bids interactively: `akt deploy deploy/akash-gpu.yaml` (do **not** use automatic cheapest selection or `--yes` for the first run). Verify offered GPU, driver compatibility, rate, uptime, region and lease/funding limits. Start with `STEPS=3`, `BATCH_SIZE=2` for first paid smoke; only then increase steps.
6. Confirm GPU in container logs (`device` in metrics must show GPU). Access synthetic `metrics.json`, `parameters.npz`, and `example.npz` through the public artifact server **after training**. Export them outside the lease before closing it. Only synthetic data should be exposed.
7. **Close the lease** with `akt close <dseq>` or in the Console immediately after verifying results. The container deliberately stays alive to serve artifacts; billing continues until the lease is closed. Never deploy with private webcam frames or credentials in this demo.

**Cost model:** Current SDL max bid `2500 uact/block` implies about **$1.50 per hour as an upper bid**, assuming ~6-second blocks and 1,000,000 micro-ACT per USD; actual provider lease price may be lower. This is not a spending limit. A $3 smoke budget would cover at most two hours at the full bid, **only if you manually close within that time**. Check actual quotes and fund/deposit conservatively. Akash GPU prices and availability vary by provider.

**Checkpoint durability:** the pilot saves at the end and serves synthetic results over HTTP; it is **not** yet restart-safe. Akash default storage is ephemeral; even provider-local persistent storage does not survive closing the lease or moving provider. PR #8 adds periodic restart-safe checkpoints and off-provider backups. Never run a long paid training job before that.

**References (current docs, checked 2026-10-30):**
- Akash SDL syntax: https://akash.network/docs/developers/deployment/akash-sdl/syntax-reference/
- Akash GPU configuration: https://akash.network/docs/learn/core-concepts/gpu-deployments/
- `akt` deploy / validate / close: https://akash.network/docs/developers/deployment/akt/deployments/
- Akash storage limitations: https://akash.network/docs/learn/core-concepts/persistent-storage/

## Immediate decisions

- **Implement first:** reproducible synthetic pilot, then live mirror visualization. Keep research and demo as separate entry points over the same model.
- **No premature complexity:** no transformer, RL, large network, optimizer bake-off, hand-labeled brain areas, or model-to-world hidden truth leakage.
- **Unknowns to measure:** real webcam domain gap, actual GPU pricing, update speed, memory, whether curriculum, recurrence or plasticity actually add value.
