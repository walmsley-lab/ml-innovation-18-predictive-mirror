# Circus Curriculum: learn unfamiliar worlds without losing ordinary prediction

The research question is whether training on unusual but measurable sensory
worlds improves generalization, including back to ordinary physical scenes.
Surreal imagery is inspiration for controlled environments, not evidence about
external realms, intelligence, or understanding.

## First experiment: rotating 4D objects

The independent `experiments/circus_curriculum/` package implements a genuine
4D geometric generator and a small RGB recurrent predictor. It adds no framework
and does not change the existing Predictive Mirror models.

The generator rotates scaled tesseract vertices in two 4D coordinate planes,
projects them perspectively from 4D to 3D, applies a 3D camera rotation, and
projects to a soft 2D RGB wireframe with visible vertex markers. Rotation
matrices are orthogonal and preserve 4D distances. The model receives only RGB;
vertices, phases, rotation planes, camera matrices and other generator truth
are evaluator metadata. No skeletal coordinates or known projection matrices
enter prediction.

The model has an explicit recurrent vector `z_t` with configurable dimension
(`--latent-dim 16`, `64`, `256`, etc.). A small visual encoder updates this state
from the observation history. Open-loop recurrent transitions then predict
coarse RGB residuals that are added to the full-resolution image. The final
observation anchors lead zero; the first forecast is lead one.

Latent width is representational dimension, not inferred physical dimension.
This first model is a deterministic point predictor. It does not approximate a
posterior over explanations or sample calibrated alternative futures. Actions,
multimodal observations and stochastic belief states are future experiments.

## Matched curricula and holdouts

Three curricula start from identical weights for each initialization seed:

- `ordinary`: only the existing moving-disc RGB world, including wall bounces.
- `hypergeometry`: only projected rotating tesseracts.
- `mixed`: replaces half the ordinary training clips with hypergeometry clips.

All three use the same number of unique clips, total batch size, optimizer,
updates, latent width, training index schedule and motion-weighted RGB loss.
Each mixed batch contains exactly half of each domain. Ordinary exposure is
therefore lower in the mixed arm; this is an equal-total-budget test, not a
claim that ordinary exposure and additional training are both held constant.

Checkpoint selection uses one common validation set with equal numbers of
ordinary and in-distribution geometric scenes. It minimizes mean RGB MSE, with
step zero eligible and earlier ties retained. This selection rule targets both
domains and is not an ordinary-only optimization study. Every selection is
saved before generating or evaluating the final test frames.

| Test case | What changes relative to geometric training? |
|---|---|
| `iid` | Independent episode seeds only |
| `rotation` | Unseen coordinate-plane pairs; angular-speed range stays fixed |
| `object` | An eight-vertex 16-cell replaces the sixteen-vertex tesseract |
| `viewpoint` | Camera-angle magnitudes move from `[0, 0.3]` to `[0.9, 1.2]` radians |
| `combined` | All three interventions together |
| `ordinary` | Independent physical moving-disc episodes |

The geometric test cases reuse scene seeds, initial phases and angular speeds
to pair interventions. The combined case is harder to interpret causally, so
individual-factor cases remain separate. The ordinary result directly reports
mixed-minus-ordinary MSE, including paired differences for every episode and
separate model seeds. Negative differences favor mixed training. No automatic
significance or superiority label is assigned.

Classical references include persistence, image-derived constant velocity,
and one history-ridge predictor for each curriculum. The ridge references see
the same clip pools and predict future coarse residuals from the full past RGB
history; their closed-form training and capacity are not compute-matched to
the neural learner. They use a declared 4×4 image grid.

## Representation diagnostics, not a claim of recovered dimensions

After freezing each forecaster, separate linear probes fit four periodic
generator targets at the last observed frame: the sine and cosine of each
rotation phase. These labels never train or select the visual forecaster.
Probe-training episode seeds are disjoint from training, validation and test.

Compare probes of trained recurrent state, the same network at random
initialization, the last pooled image and the pooled observation history.
All use the same probe episodes, labels and fixed ridge penalty. Feature
dimensions are reported because their capacities differ. Metrics include
coordinate-wise MSE and R²; a constant target has undefined R², represented as
null rather than a misleading score.

These targets use the simulator's coordinate convention. Projection ambiguity
and object symmetries can make phases unidentifiable, especially with a new
view or object. Successful decoding can reflect visual correlations, and poor
decoding need not mean useless memory. Forecast gains plus good probes would
motivate further interventions and identifiability tests; they would not by
themselves establish recovery of a uniquely correct 4D latent geometry.

## Run and inspect

Use the repository's existing JAX/NumPy environment. Pillow is optional for
`--gallery` only. All computations in this first runner use CPU.

```bash
# Integration smoke, not evidence of a useful predictor.
python -m experiments.circus_curriculum \
  --steps 2 --seeds 11 --latent-dim 16 --size 16 \
  --context 4 --horizon 3 --train-episodes 8 \
  --validation-episodes 2 --test-episodes 2 --probe-episodes 8 \
  --out runs/circus-smoke

# Exploratory comparison. Choose a fresh output directory for every run.
python -m experiments.circus_curriculum \
  --steps 100 --seeds 11 23 41 --latent-dim 64 \
  --train-episodes 64 --validation-episodes 16 \
  --test-episodes 16 --probe-episodes 64 --gallery \
  --out runs/circus-hypergeometry
```

Inspect `metrics.json` and the `*-lead-*.png` files. Gallery panels show last
observation, forecast, true future and absolute error on a fixed scale. Exact
anchor and target indices are stored with each gallery file. Frame counts are
synthetic time, not verified wall-clock seconds.

`protocol.json` records source hashes, environment, episode IDs, shared batch
schedule, training arrays and selection rules before training. `selection.json`
records validation curves and checkpoint hashes before test evaluation.
Predictions are saved before scoring; test arrays and probe weights are saved
too. Local output directories are not overwritten. `source_revision` is HEAD
at launch; hashes identify the executed files even when the checkout is dirty.

Latency is a small warmed-up CPU measurement on one actual IID context, from
context to the entire forecast horizon, including RGB materialization. It
excludes capture and rendering. `latent_state_bytes` counts only the vector;
it is neither total state memory nor peak process/device memory.

```bash
python -m unittest tests.test_circus_hypergeometry -v
```

Tests check 4D topology and distance preservation, prefix-invariant generation,
isolated held-out factors, image bounds, latent dimensions, gradient flow,
causal forecasting, equal curriculum sizes, frozen probe fitting, disjoint
splits, checkpoint selection and a complete artifact-checked smoke.

## Later worlds: deliberately not implemented in this first study

| World | Proposed next experiment | Essential control |
|---|---|---|
| A: Hypergeometry | Hypersphere slices and additional projections | Separate unseen geometry from unseen renderer/appearance |
| B: The Circus | Transformations, rule switches and ambiguous reappearances | Contradiction relative to explicit model assumptions, versus hidden regime change or insufficient evidence |
| C: Visible Language | Synchronized geometry, rhythm and actions | Paired versus shuffled modalities; held-out mappings and actions |
| D: Fractal Time | Nested deterministic/stochastic rhythms | Longer horizons and unfamiliar time scales, versus short-history baselines |

Finite observations do not generally prove that no globally consistent
explanation exists: a richer hidden-state model can often explain them. A
contradiction detector therefore needs a declared hypothesis class and noise
model. Multiple plausible explanations need proper probabilistic scores and
calibration, not just diverse-looking predictions. The existing multifuture
and identity/uncertainty experiments provide useful controls for that stage.

Only after individual generators are validated should the proposed four-world
mixture be compared with ordinary-only, geometry-augmented and
impossible-world/cross-modal curricula under matched budgets.

## Research context

[CRONOS (2026)](https://arxiv.org/abs/2605.23699) tests counterfactual physical
consistency under interventions including viewpoint and appearance, motivating
separate intervention holdouts here. It is not a benchmark reproduction.
[How Far is Video Generation from World Model (2024 preprint; ICML 2025)](https://arxiv.org/abs/2411.02385)
reports that strong in-distribution physical prediction need not imply
out-of-distribution generalization, motivating the ordinary-transfer test.
[MIT CSAIL's MultiModal Action Conditioned Video Generation (ICCV 2025)](https://people.csail.mit.edu/yichenl/projects/multimodalvideo/)
studies shared visual, auditory and tactile representations; it motivates the
later cross-modal track, which this RGB-only experiment does not implement.
