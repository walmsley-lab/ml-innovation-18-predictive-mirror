# Predictive Mirror

This repo aims to implement a video stream that anticipates the future state of the video. It implements this with both traditional cnn, nca, and other methodologies in an attempt to quantify performance differences. We leverage abstractions and toy with various techniques such as locality, memory, communication, osselation, and active sensing. Training leverages synthetic data, 3d simulations as well as benchmarks. The goal is to develop a pipeline to train and benchmark a strong, compact, efficient prediction and perception model. It is implemented in JAX to make the underlying mathematics transparent and facilitate high performance inference across varied hardware. 

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'jax==0.9.0.1' 'numpy==2.3.5'

python -m unittest discover -s tests -v

python -m experiments.rgb_curriculum \
  --seeds 11 --steps 30 --batch-size 2 \
  --context 8 --horizon 10 --out runs/rgb-intro

python -m pip install pillow
python -m experiments.rgb_prediction_gallery \
  --parameters runs/rgb-intro/seed-11-curriculum.npz \
  --context 8 --horizon 10 --seed 73000 --episodes 3 \
  --out runs/rgb-intro-gallery
```

[Experiment map](docs/EXPERIMENTS.md) — start with a question, then open
one model, one runner, its evidence, and its tests. The models remain independent:

- Image prediction: grayscale recurrent model and RGB pixel NCAs.
- Cellular connectivity: local, fixed references, learned soft offsets,
  graph-connected joints, and interacting particles.
- Time and memory: fast/slow sheets, oscillations, phase entrainment,
  hidden-state interventions, and local sensory error traces.
- Learning and action: trajectory diversity, staged curricula, occlusion,
  camera transfer, and active gaze.

```text
src/          Small, mostly pure models, synthetic worlds, shared baselines
experiments/  Independent training, evaluation, and visual-demo scripts
tests/        Deterministic regression and no-future-leakage checks
docs/         Experiment-specific protocols and historical research notes
runs/         Local checkpoints and metrics (gitignored)
recordings/   Local webcam captures (gitignored)
deploy/       Optional cloud/GPU exploration; not required to get started
```

Code conventions: functions and explicit arrays, not model registries
or class hierarchies; keep experiments self-contained until shared code is
actually reused; keep simulator truth out of inference; retain step-zero
checkpoints as eligible; report failures, seeds, splits, and strong controls.
Avoid combining scores from RGB pixels with those from provided skeleton
joints or tracked particles. This is a set of **small experimental programs**, 
not a claim of biological equivalence or consciousness. Research evidence 
precedes architectural complexity.
