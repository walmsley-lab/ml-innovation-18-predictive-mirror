# Predictive Mirror

**Can simple recurrent cells learn to anticipate what they will see next?**

A small, CPU-friendly JAX research lab for predictive perception. Our models observe
a few frames, update their internal state, then predict future frames without
new observations. We test what locality, memory, communication, oscillations,
and active sensing contribute to that ability.

Two complementary outcomes:

- **Predictive Mirror:** a live RGB camera demonstration that compares a
  previously issued forecast against the observation that actually arrived.
- **Perceptual Systems Lab:** independently runnable experiments, checkpoints,
  visualizations, and reproducible evidence. Models need not share a representation.

**Evidence so far:** in one synthetic three-seed comparison, distribution-trained
RGB NCA achieved +10-frame MSE **0.00800**, compared with last-frame persistence
**0.01207** and image-derived constant velocity **0.00496**. It learned beyond
persistence, but **did not beat constant velocity**. A +10-frame forecast is
approximately one second only at a verified 10 fps; synthetic time is nominal.
These results do **not** establish real-webcam prediction or object permanence.

**Illustrated research note:** [Read *From Pixels to Persistent Worlds?* (PDF)](paper/From_Pixels_to_Persistent_Worlds.pdf) · [editable Markdown](paper/From_Pixels_to_Persistent_Worlds.md). The figures distinguish conceptual schematics from reported measurements.

## Start here

Python 3.11 and a CPU are enough for the synthetic experiments:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'jax==0.9.0.1' 'numpy==2.3.5'
python -m unittest discover -s tests -v
```

To see the RGB NCA, first train a small model on the same synthetic-world\nfamily used by the visual gallery:

```bash
python -m experiments.rgb_curriculum \
  --seeds 11 --steps 30 --batch-size 2 \
  --context 8 --horizon 10 --out runs/rgb-intro
```

Then render its predictions and actual future frames side by side.
The gallery needs Pillow (not needed for CPU training/tests):

```bash
python -m pip install pillow
python -m experiments.rgb_prediction_gallery \
  --parameters runs/rgb-intro/seed-11-curriculum.npz \
  --context 8 --horizon 10 --seed 73000 --episodes 3 \
  --out runs/rgb-intro-gallery
```

Open `runs/rgb-intro-gallery/episode-00.png`. Inspect the actual images
alongside persistence, constant velocity, and the NCA; **a short smoke run
is not expected to produce strong forecasts**. Never treat a small validation
gain as an independent test result.

See **[Training and demos](docs/GETTING_STARTED.md)** for a full multi-seed
run, independent benchmark, optional RGB webcam recording, and *verified*
live viewer. No cloud GPU is required for the entry-level demos.

## Find an experiment

For a shared comparison of compact recurrent state, full-resolution recurrence,
ConvLSTM and motion predictors, see the **[compact predictive mirror pilot](docs/COMPACT_PREDICTIVE_MIRROR.md)**.
It trains on matched RGB clips and saves held-out forecasts, CPU timings and
optional galleries; the protocol distinguishes measured results from hypotheses.

**[Experiment map](docs/EXPERIMENTS.md)** — start with a question, then open
one model, one runner, its evidence, and its tests. The models remain independent:

- Image prediction: grayscale recurrent model and RGB pixel NCAs.
- Cellular connectivity: local, fixed references, learned soft offsets,
  graph-connected joints, and interacting particles.
- Time and memory: fast/slow sheets, oscillations, phase entrainment,
  hidden-state interventions, and local sensory error traces.
- Learning and action: trajectory diversity, staged curricula, occlusion,
  camera transfer, and active gaze.

**[Research charter (historical)](docs/PLAN.md)** records the original
hypotheses, constraints, and planned experiments. It is not a current
task list; consult this README and the experiment map for the present state.

## How the code is organized

```text
src/          Small, mostly pure models, synthetic worlds, shared baselines
experiments/  Independent training, evaluation, and visual-demo scripts
tests/        Deterministic regression and no-future-leakage checks
docs/         Experiment-specific protocols and historical research notes
runs/         Local checkpoints and metrics (gitignored)
recordings/   Local webcam captures (gitignored)
deploy/       Optional cloud/GPU exploration; not required to get started
```

**Code conventions:** functions and explicit arrays, not model registries
or class hierarchies; keep experiments self-contained until shared code is
actually reused; keep simulator truth out of inference; retain step-zero
checkpoints as eligible; report failures, seeds, splits, and strong controls.
Avoid combining scores from RGB pixels with those from provided skeleton
joints or tracked particles.

This is a set of **small experimental programs**, not a claim of biological
equivalence or consciousness. Research evidence precedes architectural
complexity.
