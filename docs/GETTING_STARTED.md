# Training and demonstrations

This is the shortest path from **a small trained RGB cellular predictor** to
**a visible forecast**. Run commands from the repository root on Python 3.11.
GPU and cloud deployment are optional. `runs/` and `recordings/` are local
and ignored by Git.

## First: CPU synthetic training and a visual result

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'jax==0.9.0.1' 'numpy==2.3.5' pillow
python -m unittest discover -s tests -v

# A quick smoke, one random initialization, NOT an evidence-quality campaign
python -m experiments.rgb_curriculum \
  --seeds 11 --steps 30 --batch-size 2 \
  --context 8 --horizon 10 --ticks 2 --size 32 \
  --out runs/rgb-intro
```

This creates `runs/rgb-intro/seed-11-single.npz`,
`runs/rgb-intro/seed-11-curriculum.npz` and `metrics.json`.
The two models start from the same weights; the first repeatedly sees a
single trajectory, and the other sees progressively varied RGB motion.
The output summarizes validation-selected checkpoints and unseen test
performance. A checkpoint selected at **step zero** is untrained and
predicts persistence; that is an important negative result, not a bug.

To inspect a frozen model on a **fresh episode from the same synthetic
world family**, use a seed outside its training/validation/test ranges:

```bash
python -m experiments.rgb_prediction_gallery \
  --parameters runs/rgb-intro/seed-11-curriculum.npz \
  --seed 73000 --episodes 3 --context 8 --horizon 10 --size 32 \
  --ticks 2 --out runs/rgb-intro-gallery
```

Open `runs/rgb-intro-gallery/episode-00.png`. Each row is a different
future lead; columns include last observation, NCA forecast, persistence,
constant velocity, future ground truth, absolute error and cell activity.
This is a **visual diagnostic** on a separate deterministic test episode,
not proof of generalization from a single smoke run.

Evaluate the same checkpoint on additional frozen episodes:

```bash
python -m experiments.rgb_benchmark \
  --checkpoint runs/rgb-intro/seed-11-curriculum.npz \
  --seed 73000 --episodes 12 --context 8 --horizon 10 \
  --out runs/rgb-intro-benchmark/metrics.json
```

### For a more meaningful synthetic comparison

```bash
python -m experiments.rgb_curriculum \
  --seeds 11 23 41 --steps 100 --batch-size 2 \
  --context 8 --horizon 10 --ticks 2 \
  --out runs/rgb-curriculum-study
```

Inspect `runs/rgb-curriculum-study/metrics.json` and compare the model's
+10-frame MSE against persistence and constant velocity. Run the gallery
again with `runs/rgb-curriculum-study/seed-11-curriculum.npz`.
Do not confuse this protocol with the separately frozen
`experiments.rgb_motion_distribution` benchmark: they use **different
synthetic generators**, so their raw MSE values are not interchangeable.

## Optional: train and view a real RGB webcam

The synthetic model is **not** expected to transfer automatically to a
webcam. Use the RGB recording and training path below for a camera-specific
checkpoint. On Fedora/Linux, use the color video device (often camera 0)
rather than an IR or grayscale device.

```bash
python -m pip install opencv-python

# Move an object across the camera view, vary direction, include occlusions.
python -m experiments.rgb_nca capture \
  --camera 0 --seconds 120 --fps 10 --size 32 \
  --out recordings/rgb-webcam-train.npz

python -m experiments.rgb_nca fit \
  --input recordings/rgb-webcam-train.npz \
  --out runs/rgb-webcam-1s \
  --steps 100 --batch-size 2 --context 8 --horizon 10 --ticks 2
```

Read `runs/rgb-webcam-1s/metrics.json` **before** demonstrating:

- `best_validation_step: 0` or `selection_status` beginning with
  `failed_` means the run has **not** shown learned anticipation.
- A validation improvement is **not** independent real-video evidence;
  the fit command uses a temporal holdout from the same recording.
- Ten frames is nominally a second at 10 fps, but inspect recorded
  timestamps and measured live horizon before calling it a wall-clock second.

The verified viewer pairs predictions with subsequently observed frames:

```bash
python -m experiments.rgb_verified_live \
  --parameters runs/rgb-webcam-1s/parameters.npz \
  --camera 0 --fps 10 --size 32 --context 8 --lead 10 --ticks 2 \
  --out runs/rgb-live-verified/metrics.json
```

The viewer compares the original observed frame, the **already-issued**
forecast, actual future camera frame, and error. Press `q` or Escape
to quit. If the camera is busy, close other processes using that device.
For headless checking, append `--no-display --max-frames 100`.

For an independent recording-based evaluation, capture a **different**
recording and use `experiments.rgb_transfer_eval --help`. Supply the
training recording to `--exclude-training` to reject identical files.
Make the movement deliberate; mostly static footage can make a persistence
model appear deceptively good.

## What success looks like

**Technical demo:** predictions are issued before truth arrives; images
and time alignment are observable; the live display runs on your machine.

**Scientific progress:** held-out data from independent episodes/recordings,
multiple seeds, better performance than meaningful controls, and specific
tests of occlusion and internal state. Either outcome is informative.
We keep negative results and independent model variants instead of silently
switching architectures.

See [experiment map](EXPERIMENTS.md) for alternative selectors, oscillators,
graphs, particles, and causal probes. For GPU/hosting notes, see
[optional training infrastructure](TRAINING.md).
