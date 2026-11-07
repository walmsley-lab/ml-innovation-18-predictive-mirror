# Compact predictive mirror: a matched CPU pilot

Question: can a smaller recurrent visual state retain useful forecasting
ability while reducing inference cost? This experiment connects a compact
predictive model to a shared training, evaluation and gallery workflow.
It is an implemented pilot, not evidence of general performance superiority.

## Model

`src.compact_predictive_mirror` reuses the existing RGB NCA cell update and
nonwrapping motion candidates. It average-pools sensory input and prediction
error to a half-height, half-width hidden sheet. During observation it corrects
that sheet against the RGB forecast committed before the observation arrived.
During open-loop forecasting it advances the sheet without further observations.

The hidden state predicts local motion-routing logits and an appearance
residual. Those fields are interpolated to the original image resolution;
transport moves the full-resolution RGB pixels, then adds the bounded residual.
This preserves fine image detail that would be lost by reconstructing the
whole image from pooled RGB. Output values stay in `[0,1]`. The first output is
one frame after the last observation; `assimilate`, `observe` and `dream` also
expose the streaming state explicitly.

Half-resolution state has one quarter as many hidden values. The full-resolution
RGB buffer, routing weights and transport candidates remain dense. This is not
sparse execution, a fourfold total-memory reduction, a calibrated uncertainty
model, or a biological simulation. Prediction-error heatmaps measure observed
discrepancies only.

## Run

Use the existing environment from [Getting started](GETTING_STARTED.md).
Pillow is optional and needed only for `--gallery`.

```bash
# Short integration smoke: every learned arm, not a performance study.
python -m experiments.compact_mirror_comparison \
  --steps 2 --seeds 11 --width 4 --ticks 1 \
  --validation-scenes 1 --test-scenes 1 --repeats 3 \
  --out runs/compact-mirror-smoke

# Longer exploratory comparison; tune budgets on validation, not this test.
python -m experiments.compact_mirror_comparison \
  --steps 100 --seeds 11 23 41 --batch-scenes 2 \
  --validation-scenes 8 --test-scenes 12 --repeats 15 --gallery \
  --out runs/compact-mirror-pilot
```

Open `runs/compact-mirror-pilot/index.html` for the score table and galleries.
Each panel shows the last observation, its issued forecast, the actual future,
and absolute error on a fixed `[0,1]` scale. Leads are synthetic frames, not
seconds. This is offline replay, not a claim about live webcam latency.

The output directory must be new or empty. Select fewer learned arms with
`--models compact full_resolution`; all three classical controls still run.
Default inference geometry is 32×32, eight context frames and ten future
frames. The paired scene generator requires context ≥8, horizon ≥6, size ≥32;
the compact model additionally requires even image dimensions. Timing is on
CPU for both JAX and NumPy references.

## Comparison and information boundary

The learned arms are compact, the same hybrid with full-resolution state,
the existing RGB NCA, the minimal ConvLSTM, and learned transport without
recurrent memory. Compact and full-resolution arms start with identical
parameter values for a given seed. All learned arms receive identical batches,
update counts, widths, optimizer settings and RGB-derived motion-weighted loss.
Width and update matching do **not** match parameter count, FLOPs, training
time or optimization sufficiency; this is not a reference-paper reproduction.

Controls are last-frame persistence, image-derived global constant velocity,
and the existing RGB history-linear predictor. The latter assumes a static
background and one saturated moving blob; it estimates appearance and motion
from past RGB without known palette IDs, simulator positions or future inputs.
Its renderer and constant-velocity dynamics have limitations on bouncing scenes.

Every scene ID provides one episode with a wall bounce in the forecast interval
and two opposing histories with exactly the same final observation but different
reappearance futures. Rejection sampling uses truth only to construct the bounce
task. Appearance varies by deterministic per-scene RGB gain and bias, identically
across the paired histories. The occlusion generator has a small set of shared
motion templates: this tests interpolation within that family, not new physics.

Training, validation and test scene IDs occupy disjoint ranges. Before training,
`protocol.json` freezes their IDs, configurations, source hashes, dependency
versions, CPU description and saved validation/test arrays. Future targets enter
the training loss and evaluation only. Each model receives only its RGB prefix
at inference. Lowest validation RGB MSE selects a checkpoint, including step
zero; ties retain the earlier checkpoint. All selections are written and hashed
before any model is scored on the test set. Prediction arrays are saved before
scoring, and each gallery records its exact anchor and target indices.

## Read the results

- `metrics.json`: per-scene RGB MSE; bounce error by lead, including changed-pixel
  MSE; paired reappearance timing/localization success; latency samples; state
  and parameter payload sizes; separate MSE/latency and event-success/latency
  frontiers. Overall MSE weights all clips equally (two occlusion, one bounce).
- `selection.json`: initialization seed, validation curve, selected step and
  checkpoint file/hash. Step zero means no trained checkpoint beat initialization.
- `*-predictions.npz`: exact scored forecasts in `[B,T,H,W,3]` order. Every group
  of three is bounce, rightward occlusion history, leftward occlusion history.
- `*-weights.npz`: selected flat parameter dictionaries, compatible with
  `src.experiment_harness.load_checkpoint`. Reuse the saved protocol's model,
  width, ticks and geometry; these arrays are not generic webcam checkpoints.
- Compact diagnostics compare intact inference to a repeated-last-frame history
  and to disabled sensory-error feedback on the **same** checkpoint. Sensitivity
  is not proof of useful memory; compare correctness and use separately trained
  ablations for stronger mechanism claims.

Latency covers one actual test context through its full forecast horizon,
including NumPy/JAX input and output materialization. Every test clip is timed
with batch size one, one warmup, and repeated synchronized calls. Compilation,
camera capture, scoring and PNG rendering are excluded. Do not run competing
CPU jobs while collecting timings. Record hardware and rerun on the intended
deployment device before making performance promises.

`state_payload_bytes` counts the recurrent carry arrays, including RGB buffers,
from their shapes/dtypes. It excludes input history, outputs, intermediate
activations and allocator overhead. `peak_memory_mib` and `energy_j` are null
because this runner does not measure them. Parameter and state payloads must
not be presented as measured process/device peaks.

Model seeds stay separate in the report, and pair members are averaged before
reporting per-scene success. Raw scene values support later cluster-aware
analysis; this pilot makes no statistical significance claim and has no
automatic promotion gate. Do not select a model or tune its budget on these
test outcomes and then reuse them as independent confirmation. Successful
reappearance alone does not establish object identity, calibrated uncertainty,
or transfer to an independently recorded camera scene.

## Validation

A local 30-update pilot ran all five learned arms with seeds 11, 23 and 41,
four validation scenes, six test scenes (18 clips), and nine timing repeats per
clip on an Intel Core i7-8750H CPU. The [checked-in measurement summary](compact_mirror_pilot.json)
includes the exact command, source-file hashes, verified source commit,
per-seed/per-scene results and artifact hashes. Full arrays, checkpoints and
galleries remain in the local `runs/compact-mirror-pr-pilot/` directory; they
are not included in the summary or the repository.

| Model | Mean RGB MSE across model seeds | Mean of per-seed median latency | State payload per inference |
|---|---:|---:|---:|
| Compact hybrid | 0.002677 | 5.35 ms | 20 KiB |
| Same hybrid, full-resolution state | 0.002699 | 14.72 ms | 44 KiB |
| Classical RGB history-linear control | 0.001780 | 1.93 ms | Not reported |

The compact hybrid took about 2.75× less time than its full-resolution control
in this pilot, with similar RGB error. Neither hybrid, nor any other learned
arm, passed the reappearance criterion at this budget; the history-linear
control passed all six scene pairs. This supports investigating the compact
representation's cost, **not** a claim that it outperforms competent classical
forecasting or has learned object permanence. The small training and test
budgets do not establish optimization sufficiency or statistical superiority.

```bash
python -m unittest tests.test_compact_predictive_mirror \
  tests.test_compact_mirror_comparison -v
```

Tests cover retained history, prediction-error correction, gradient flow,
full-resolution detail, nonwrapping transport, exact first-lead alignment,
disjoint scene IDs, oracle event scoring, step-zero checkpoint selection,
context-only predictor input, timing on actual clips, gallery pairing and
an artifact-checked end-to-end training/evaluation smoke.
