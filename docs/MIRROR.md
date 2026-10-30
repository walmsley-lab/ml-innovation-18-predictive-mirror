# Local predictive webcam mirror (PR 13)

Run **after** reviewing a trained checkpoint and installing an optional local GUI dependency:

```bash
python -m pip install opencv-python
python -m mirror --parameters runs/pilot/parameters.npz --architecture recurrent --camera 0 --fps 10
```

The viewer shows **observed / predicted / absolute error** for a 16×16 grayscale image upscaled to readable pixels. `q` or Escape closes it. Only a local JSON time/error report is saved; camera images are not uploaded or persisted. No camera is needed to run unit tests, and no webcam was available to test the GUI here.

**Causality:** The pipeline predicts after `frame_t`, commits prediction/timestamp, then requests `frame_(t+1)`. OpenCV capture-driver buffering means this establishes application-order causality, **not** independently verified sensor exposure timing. The camera has no action telemetry: the action channel is set to the documented unknown/zero placeholder. Camera-motion performance is an out-of-distribution test, not proof of learned ego-motion.

**Scope:** The adapter reads recurrent or multisheet checkpoints, uses center-cropped BGR→grayscale INTER_AREA normalization to [0,1], and samples nominally at 10 Hz (actual arrival timestamps recorded). The simple "linear frame difference" baseline is not object velocity extrapolation; these webcam metrics are **descriptive only**. Synthetic-to-webcam quality is not guaranteed. The expensive GPU training image does not package OpenCV; this entry point is intended for your local laptop after training.
