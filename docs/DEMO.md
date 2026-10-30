# PR 23 — Live predicted mirror with internal-activity panels

Uses the **same pretrained checkpoint** as the research code. Local launch (OpenCV optional and only needed for live webcam):

```bash
python -m pip install opencv-python
python -m mirror --parameters runs/pilot/parameters.npz --architecture recurrent --fps 10
# Or use a multisheet checkpoint with --architecture multisheet
```

The view shows observed frame, prediction committed **before** the next camera read, absolute error and two internal population maps. For a recurrent single sheet, the population maps are arbitrary channel halves; they are **not** two proven brain regions. For a dual-sheet network, the maps are its fast and slow populations. Activity is computed *before* the target observation arrives and displayed with the same [0,1] fixed scale rather than per-frame normalization. Press q to exit. Report JSON preserves per-frame latency, committed/received timestamps, prediction and persistence errors and population-activity means. No webcam frames are saved or uploaded.

**Important:** OpenCV driver buffering means prediction-before-read is proven in application order, but sensor exposure timestamp causality is not independently guaranteed. Cameras have no action labels; a zero placeholder is passed as **unknown** action. RGB/BGR is converted to normalized grayscale 16×16 at nominal 10Hz, whose observed cadence may vary. An appealing live video is not evidence of occlusion memory or developmental emergence; those claims depend on offline controlled experiments.
