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
h_(t+1) = update(local_neighbors(h_t), x_t, a_t)
x_hat_(t+1|t) = readout(h_(t+1))
```

Here `x_hat_(t+1|t)` is the prediction of the **next** frame, computed after observing only frame `x_t`, earlier frames, and action `a_t` known at prediction time. Its timestamp and output must be committed before the camera delivers `x_(t+1)`; frame and action alignment are checked by a no-future-leakage test. The first model is one locally connected recurrent sheet with shared update weights and no online plasticity. Later PRs add a second sheet with different timescale/receptive field and *sparse* reciprocal connections. A separate ablation introduces activity-dependent plasticity (e.g., decaying per-edge traces). The developmental world supports scripted camera motions from day one, then selectable gaze actions in a controlled comparison.

## Start gates and relationship to Neural Dynamics (#17)

This repo is a computational testbed for selected hypotheses from `ml-innovation-17-neural-dynamics`; it does not replace that repo's biological evidence, observability analysis or claims ledger. Cite the exact #17 experiment and distinguish simulated behavior from evidence about brains. In particular, #17 H2 (memory and prediction), H3 (self-generated sensation) and H4 (plasticity/stability) supply useful controls.

**Before PR #2:** Review/merge the draft bootstrap PR after CPU CI and a reproducible train/evaluate run. Treat the synthetic predictor as a *short-trained* pilot, not a validated predictor. The live mirror must make and timestamp each prediction **before** the target webcam frame arrives. PR #2 only *reports descriptive* persistence and velocity-extrapolation results on webcam frames and does not claim statistical superiority or successful transfer. PR #3 handles the formal synthetic benchmark.

**Webcam action and preprocessing contract (PR #2):** The synthetic model observes `(x_t,a_t)`, but an ordinary webcam does **not** provide camera-command telemetry. For the initial live demo set `a_t=0` as an explicit **unknown-action placeholder**, not evidence that the camera did not move; annotate the mismatch and report performance separately under a stationary camera and deliberate camera motion. Do not infer ego-motion in PR #2. PR #3 may compare action-masked training with a genuine missing-action indicator; action estimation is a later independent model. Map RGB/BGR frames to grayscale luminance using a documented channel order, center-crop to a square, area-resize to 16×16 (optionally 32×32 after separate training), normalize to [0,1], and sample at a nominal 10 Hz (100 ms per simulation tick) with actual monotonic timestamps and dropped-frame records. Make the simulator's time step explicit; evaluate a fixed-rate and a varying-rate recording separately. Do not describe simulator-to-camera performance as successful *transfer* without measuring it.

**Before running substantial paid training — PR #2.5 durability gate:** Periodically save model weights, optimizer moments, completed step, seed/sampling position, config and image/git digest atomically. Resume from the latest checkpoint; inject a forced kill mid-training and verify resumed predictions against an uninterrupted seeded reference (within specified numeric tolerance). Export checkpoints off-provider, download/verify checksum, and rehearse restore on a fresh container. This gate is **not** required to show the initial webcam demo or a tiny disposable paid smoke; it **is** required before long training. PR #8 only automates and hardens these verified procedures.

**Before claiming self/world discrimination (PR #3):** Generate a crossed set of object-still/object-moving × camera-still/camera-moving episodes, including held-out rates and camera-action delays. The present bootstrap world always moves the object; it does **not** yet test discrimination between self-generated and external motion. Only pixel observations and action commands enter training; hidden truth is for evaluation.

**Before claiming reconciliation (PRs #4–5):** Match total parameters, total hidden-state scalar count, recurrent inference/update count, sensory history, and measured compute/latency per frame as closely as possible. Report any unmatched resource explicitly, including slower-sheet effective memory. Fit diagnostic decoders **after** perceptual training on a separate probe-training split and score on untouched scenes. Track compatibility jointly with decoder correctness (agreement without accuracy does not count); compare two decoders trained on arbitrary halves of a single-sheet network as the false-reconciliation null. Include temporary feedback cuts, randomized within-sheet cuts with comparable number/magnitude of severed connections, mean-signal substitution, and a model **trained** without cross-sheet feedback. Separate visible motion, occlusion and ego-motion; no mechanistic claim follows from a pretty latent map.

**Before substantive paid Akash training:** Publish and pull-test the actual image by immutable digest, validate the SDL and deployment dry run, verify GPU execution with a tiny paid smoke, and establish off-provider checkpoint export/recovery. The current pilot saves only at completion and serves artifacts from a lease that continues billing: this is insufficient for long runs. Move the minimal durable-checkpoint/export gate ahead of PR #8; PR #8 can later harden automation. Review bids, set a manual spending/time cap and close the lease.

## PR sequence and indicative effort

Estimates below are **engineering effort after dependency gates**, not promises of calendar completion. A hackathon can prioritize PRs 1–2; robust emergence experiments extend beyond it.

| PR | Scope / files | Estimated effort | Acceptance gate |
|---|---|---:|---|
| **#1 Bootstrap + Akash smoke** | `src/world.py`, `src/model.py`, `train.py`, `tests/`, `Dockerfile`, `deploy/`, `docs/PLAN.md`, CI | 3–5 h | Local seeded simulation, train/evaluate and artifact output; GPU image recipe and valid SDL. No paid deployment. |
| **#2 Honest mirror MVP** | `mirror.py` plus webcam adapter and UI | 3–6 h | Causal timestamped webcam forecast; observed/predicted/error shown; *descriptive* webcam persistence/velocity numbers and domain gap, 16×16 grayscale/10Hz preprocessing, action-unknown policy, CPU fallback and latency. |
| **#2.5 Durability gate** | `train.py`, `deploy/`, checkpoint interruption test | 3–5 h | Atomic periodic checkpoint + complete-state resume, kill/restart equivalence, off-provider export and fresh-container restore **before substantial paid Akash training** (may follow #3 for CPU-only research). |
| **#3 Evaluation + curriculum** | `evaluate.py`, `curriculum.py`, tests | 4–8 h | Formal synthetic comparisons: persistence, velocity, matched-history feedforward and recurrent; object/camera motion crossed; staged vs shuffled with fixed replay fraction, full learning curves and final step, five seeds and fresh holdouts. |
| **#4 Regional scaffold** | `src/model.py`, `experiments/regions.py` | 6–10 h | Single vs two sheets matched for parameters, total state size, recurrent steps and per-frame compute, with sparse feedback and timescale/receptive-field ablations. |
| **#5 Reconciliation test** | `experiments/reconciliation.py` | 6–10 h | Accuracy-qualified cross-sheet compatibility over inference steps, single-sheet split-decoder null; feedback-cut, matched within-sheet-cut, mean-feedback and no-feedback-trained controls. |
| **#6 Active perception** | `src/world.py`, `experiments/gaze.py` | 6–10 h | Fixed, random, optimized-fixed and deterministic uncertainty-directed gaze (no RL), equal sensing/action budgets and held-out scene-state error. |
| **#7 Cellular plasticity** | `src/plasticity.py`, `experiments/plasticity.py` | 6–12 h | Fixed vs plastic local links with otherwise identical architecture and budget, plus forgetting/stability tests. |
| **#8 Reproducibility / scale** | `deploy/`, `reports/`, updated CI | 4–8 h | Harden already-gated image and checkpoint procedures, automate recovery, repeat Akash GPU runs, preserve cost/seed manifests and negative results. |

**Calendar sketch (targets, not commitments):** Oct 30: PR #1 review and PR #2 if viable; Oct 31–Nov 1: PRs #3–4 and #2.5 if GPU training is needed; Nov 2–4: PRs #5–7; Nov 5 onward: PR #8 and replication. PR #2.5 blocks substantial paid training, not the CPU experiments or mirror demo. Dependencies, replication and debugging can extend these dates. Minimum credible mirror: #1–2. Minimum *reconciliation* evidence: #1, #3–5. Stronger developmental + embodied investigation: #1–7.

## Synthetic priming curriculum

0. Spontaneous spatially correlated activity (optional control; no object semantics).
1. Blurred contrast, bars and edges.
2. Constant and changing motion.
3. Occlusion, reappearance and object persistence.
4. Scripted ego-motion vs object motion.
5. Restricted gaze and information-seeking actions.
6. Multiple objects, appearance changes and compositional test scenes.

Every stage contains replay from earlier stages. Compare to shuffled/mixture training with identical samples, updates and compute. Do not assert that ordering helps until replicated.

## Frozen evaluation protocol — primary outcomes before training

**Dataset discipline:** Split by entire generated trajectories and seeds, then hold out object/camera speed combinations and appearances. Synthetic ground truth (object location, occlusion, camera pose and foreground mask) is exposed to **evaluation only**, not a model input or auxiliary training target. A training loss may compare predictions with the next observed image, but must not access hidden-world metadata. In the PR #1 pilot, the image-derived bright-pixel weighting in `src/model.py` is being removed: training uses **plain pixel MSE** as an unprivileged baseline. Foreground-weighted scores are evaluation-only.

**Metric definitions (all reported in physical time or pixels):**
- Plain next-frame MSE: `mean((predicted_frame - observed_next_frame)^2)`.
- Foreground-weighted MSE: `mean((1 + 8*M_visible)*(prediction - target)^2)`, where `M_visible` is the **evaluation-only** simulator mask of *visible* object pixels (not the occluded hidden position); it must never feed training. For real video without simulator masks, report plain MSE and a separately labeled image-derived threshold proxy if used.
- Visible-object positive-pixel MSE: squared pixel error averaged **only** over ground-truth visible object pixels; undefined (report `NA`) for frames with no visible object. Do not silently count those as zero.
- Occlusion reappearance position MAE: absolute error (pixels) of decoded object center at first verified visible frame after occlusion, evaluated against hidden simulator truth; decoding method and thresholds calibrated on validation only. Track reappearance time error separately.
- Per-frame cost: median and p95 inference time after JAX synchronization, plus measured memory and number of model recurrent steps. Record compilation separately.

**One confirmatory primary metric per independent claim:** choose the following definitions and control contrasts *before training*, then freeze evaluation seeds, equal-sample conditions, 10% replay fraction (unless a pre-registered alternative experiment explicitly changes it) and analysis scripts. Other measurements are exploratory:
- **H2 / recurrence:** difference in held-out occlusion reappearance position MAE between recurrent model and **matched-history feedforward** model; persistence and velocity are mandatory supporting baselines. A smaller error is better.
- **Curriculum order:** final-step held-out occlusion reappearance MAE for staged curriculum minus shuffled full-mixture training at equal *total examples, updates, and replay fraction*. Publish all intermediate learning curves; don't select a flattering intermediate checkpoint.
- **H3 / self vs world:** balanced held-out classification accuracy for object-motion vs ego-motion cause, derived from a separately calibrated diagnostic probe; identical observation/action information to competing models and an action-history ablation.
- **Regionalization:** held-out reappearance MAE of two-sheet minus single-sheet at matched parameters, state elements, inference steps and measured latency; if compute cannot be matched, treat advantage as qualified.
- **Reconciliation:** at a predeclared final recurrent inference iteration, percentage of held-out occluded episodes where *both* independently decoded sheet predictions are within 1 pixel of ground truth **and** within 1 pixel of each other; contrast intact feedback with feedback-free **trained** control, and report the trajectory of this accuracy-qualified compatibility over inference steps. Additional nuisance controls: single-sheet half-split decoders, matched within-sheet cuts, and mean-feedback replacement. If both decoders merely collapse to a mean, the correctness criterion fails.
- **H1 / active sensing:** held-out final hidden-state estimation error at a fixed observation budget for deterministic uncertainty-directed vs *optimized-fixed* gaze (plus random, fixed and replay-yoked), with inference/policy cost reported.
- **H4 / plasticity:** held-out recovery-to-baseline duration after a controlled sensory remapping for plastic vs matched fixed-weight network, reporting retention and instability as predefined failure/safety constraints. No claim of a developmental mechanism without comparison to fixed and bounded plasticity.

**Predeclared inference:** At least five independently seeded runs for exploratory effect-size/variance estimates. Before confirmatory claims, increase the number of seeds using a precision/power plan based on pilot variance; report paired uncertainty intervals clustered by **independent trajectory and seed** (not adjacent frames as independent observations). Declare the main contrast and direction, multiplicity correction if testing multiple confirmatory claims, confidence interval, and any non-inferiority/equivalence margin *before* inspecting final holdouts. Negative and null results remain first-class outputs.

**Failure and alternative explanations:** No dependable gain over strong baselines, no region differentiation, no action benefit after computation costs, compatibility without correctness, or no specific effect beyond random connection damage do **not** support the corresponding hypothesis. A missing effect in a broken/no-power positive control is inconclusive, not falsification.

**Gaze policy in PR #6:** Use a fixed, deterministic rule selecting the admissible gaze crop with maximum model-predicted variance/ensemble disagreement over its **future** observations; it may not inspect the future target, hidden simulator state or the future prediction error. Compare to optimized-fixed at identical sensor resolution and observation count. Reinforcement learning remains out of scope.

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

**Status (Oct 30, 2026):** PR #1 contains an **unverified** GPU Dockerfile, an SDL with a placeholder image, and a training entrypoint that **requires a GPU**. No Docker/akt/GPU end-to-end check or paid lease has been run here. The CLI SDL validator checks syntax and pinned image spelling; validation does not verify that an image exists or pulls successfully.

1. **Local Python test:** Python 3.11: `python -m pip install 'jax==0.9.0.1' 'numpy==2.3.5'`; `python -m unittest discover -s tests -v`; `python -m train --steps 3 --batch-size 2 --output runs/smoke`. Report baseline numbers, including any failures.
2. **2a — CPU container import/training check (Docker required, no GPU):** `docker build -t predictive-mirror:pilot .`; `docker run --rm -e JAX_PLATFORMS=cpu --entrypoint python predictive-mirror:pilot -c 'import jax; print(jax.devices())'`; `docker run --rm -e JAX_PLATFORMS=cpu --entrypoint python predictive-mirror:pilot -m train --steps 2 --batch-size 2 --output /tmp/cpu-smoke`. These **override the GPU-only default entrypoint**. Tests run locally (the image currently does not bundle `tests/`). No Docker check has yet been executed.
3. **2b — NVIDIA GPU container check (requires NVIDIA runtime):** `docker run --rm --gpus all --entrypoint python predictive-mirror:pilot -m train --require-gpu --steps 2 --batch-size 2 --output /tmp/gpu-smoke`. It **must fail** if JAX cannot see an NVIDIA GPU. Report driver/CUDA compatibility and device in metrics. A CPU-only check is not GPU validation.
4. **Publish and pin:** After review/merge, publish the GHCR image, confirm pullability from a clean environment, record exact image digest and **replace** `services.trainer.image` in `deploy/akash-gpu.yaml` with `ghcr.io/...@sha256:<actual digest>`. This is mandatory: the `:bootstrap-1` placeholder is not a real published release, even if the validator accepts a syntactically valid tag. Never use `:latest` or a mutable release tag for experiments.
5. **Select billing context before spending:** Record the active `akt` context and whether auth is **keyring** (on-chain escrow/deposit + transaction fees) or **console-api** (Console credits; **no deployment deposit**). Escrow is prepayment, **not a spending ceiling**; Console credits are also not a per-job limit. Record a human-controlled time budget and maximum bid, and set a calendar/alarm external to the job for closing the lease.
6. **Validate and inspect:** `akt sdl validate deploy/akash-gpu.yaml`; `akt deploy deploy/akash-gpu.yaml --dry-run`. A dry run plans the deployment (and may resolve signer/deposit) but **does not** prove provider bids, available GPUs, adequate usable funding, ability to pull/start the image, or that training runs. Inspect provider GPU, price, region, drivers, storage and contract details interactively before accepting a bid.
7. **Tiny paid deployment only after preflight:** `akt deploy deploy/akash-gpu.yaml` (interactive selection, no `--yes` or cheapest auto-selection). For first paid smoke explicitly set `STEPS=3`, `BATCH_SIZE=2` in the SDL; verify those values before submitting. If any step fails **after** a paid deployment is created, retain its dseq and deliberately close or recover it—`akt` does **not** auto-close paid state on failure.
8. **Read artifacts and terminate billing:** Verify JAX GPU in logs and compare the 3-step output. Download synthetic `metrics.json`, `parameters.npz` and `example.npz` and verify integrity **before** closing. The public HTTP endpoint exposes synthetic output only; never put webcam recordings, credentials or patient data there. Run `akt close <dseq>`, then `akt query deployment <dseq>` and verify closed state; for Console, use the equivalent Console status query as well. Record closure time, actual cost, and final state. A running HTTP artifact server continues to incur lease costs until closure.
9. **PR #2.5 durability gate before large jobs:** Periodic atomic checkpointing, full optimizer/RNG/config resume, controlled mid-run interruption, external export with checksum, and fresh-container restore are mandatory before spending on longer training. PR #8 adds automation and replication, not the first durable checkpoint.

**Rate estimate, not spending cap:** 2500 `uact` per ~6-second block × ~600 blocks/hour ≈ 1,500,000 micro-ACT ≈ **$1.50/hour upper bid** at the current pricing convention. Provider bid may be lower; additional storage/fees or context behavior must be checked. Fund/deposit is not a hard cap, and `akt close` plus verified closed-state readback is required.

**References (checked Oct 30, 2026):**
- https://akash.network/docs/developers/deployment/akt/deployments/
- https://akash.network/docs/developers/deployment/akash-sdl/syntax-reference/
- https://akash.network/docs/learn/core-concepts/persistent-storage/
- https://akash.network/docs/learn/core-concepts/gpu-deployments/

## Immediate decisions

- **Implement first:** reproducible synthetic pilot, then live mirror visualization. Keep research and demo as separate entry points over the same model.
- **No premature complexity:** no transformer, RL, large network, optimizer bake-off, hand-labeled brain areas, or model-to-world hidden truth leakage. Do not interpret increasing latent agreement as perceptual reconciliation unless independent accuracy rises too.
- **Unknowns to measure:** real webcam domain gap, actual GPU pricing, update speed, memory, whether curriculum, recurrence or plasticity actually add value.
