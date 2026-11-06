# Perceptual Atlas: authored audiovisual scene sketches

**Question:** Can reproducible synthetic audiovisual scenes help people communicate
an imagined visual experience, and provide controlled observations for learning
appearance-invariant hidden structure?

This is a hand-authored **synthetic scene tool**, not mind-reading, a clinical
instrument, a validated perceptual model, or a record of private experience.
It requires no camera or microphone and performs no network requests.

## 1. Explore and share a scene

Open [the standalone browser demo](../demos/perceptual_atlas.html) directly
in a browser. No installation, server, account or GPU is required.

Controls: four spatial fields (spiral, folded, lattice, chamber); simulated
illumination; complexity; temporal flow; tone pitch; audio/geometry coupling;
and a reproducible seed. The optional low-volume sine tone is **off by
default**. The demo uses gradual motion without deliberate flashing. A
reduced-motion preference starts playback paused.

Select **Record**, change the controls, then **Record** again to finish.
Use **Save JSON** to export the time-stamped parameter sequence, and **Load
JSON** or **Replay** to reproduce it. The file is a voluntarily authored
description, **not** brain data or a recording of consciousness.

Example sequence:

```json
{
  "format": "perceptual-atlas-v1",
  "initial": {
    "seed": 7, "mode": "spiral", "light": 0.15,
    "detail": 0.55, "flow": 0.42, "pitch": 0.42, "coupling": 0.65
  },
  "events": [],
  "duration": 4
}
```

## 2. Generate a CPU dataset for model experiments

With the repository's NumPy environment:

```bash
python -m experiments.perceptual_atlas_dataset \
  --seed 41000 --episodes 8 --frames 16 --size 32 \
  --out runs/perceptual-atlas/dataset.npz
```

The companion `dataset.json` records the protocol, seed, geometry modes,
shape, exposure settings, and SHA-256 of the NPZ. Output paths are not
overwritten. Each scene is rendered twice using **the same underlying
geometric field** under simulated dark and bright presentation.

NPZ arrays:

| Field | Shape | Role |
|---|---|---|
| `frames_dark` | (episodes, frames, H, W, 3) | RGB observation |
| `frames_bright` | same | Paired RGB observation |
| `cue_hz` | (episodes, frames) | Synchronized synthetic pitch cue |
| `latent_intensity` | (episodes, frames, H, W) | Generator truth, evaluator-only |

The optional pitch cue is an explicitly supplied, synchronized control and
cannot be inferred from pixels by fiat. It modulates the field through a
known coupling function. Set up held-out seeds, geometries, and cue
relationships before using these scenes for prediction benchmarks.

The Python dataset and browser use the same analytic field family and
parameters. They are not guaranteed pixel-identical at different aspect
ratios or temporal sampling. The browser is an interactive sketch; the
dataset is a fixed-resolution research stimulus.

## 3. Regression checks

```bash
python -m unittest tests.test_perceptual_atlas_dataset -v
node --test tests/test_perceptual_atlas.cjs
```

Python tests cover reproducibility, paired RGB changes with identical
generator truth, pitch coupling, bounds, and artifact no-overwrite behavior.
JavaScript tests cover scene normalization, reproducibility, field/presentation
separation, coupling, input ranges, replay, and the self-contained viewer.
The browser rendering still deserves manual inspection on the target device.

## 4. What this does not demonstrate

The light slider models *display conditions* rather than measuring the
lighting of a real room. The generated geometry reflects rules explicitly
written into the simulator; it is not evidence for inaccessible dimensions.
The audiovisual correspondence is scripted, not learned. Exported control
sequences are an attempt to communicate subjective imagery, not a literal
view into someone's mind.

No model is trained in this PR, so it makes **no prediction, decoding,
perception-transfer or representation-learning performance claim**.

## 5. Research gates

1. **Renderer invariance** — show that changing simulated illumination
   preserves the underlying field while changing RGB. This is a generator
   property, verified by deterministic tests.
2. **Learnability** — fit a small predictor to a tiny fixed training sample,
   then verify genuine forecast improvement over persistence and an
   image-history baseline. No external labels at inference time.
3. **Disentanglement** — train on paired illumination conditions and test
   on held-out seeds, modes and illumination. Compare with equally budgeted
   ordinary-world-only and mixed training.
4. **Cross-modal relevance** — compare synchronized audio cues with
   shuffled cues, zero coupling, and video-only controls on new combinations.
5. **Human communication** — if volunteers choose to author sketches,
   evaluate whether another person can reconstruct *reported motifs* from
   them. Use informed consent, avoid identifying information, and make no
   claims about thought decoding.

Keep this work independent of the existing 4D-projection curriculum.
Only consider combining their results after each individual experiment
shows learnability against strong controls.
