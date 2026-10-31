# Causal prediction-horizon experiment

The webcam model is trained for one next-frame step. This evaluator unrolls it without future observations: warm up recurrent state on 8 **observed** frames, predict one frame, feed that **prediction** back, then repeat. It does **not** assume the model was trained for multi-step stability.

Run on the previously recorded local video and the better 100-step checkpoint:

```bash
python -m experiments.video_horizons --input recordings/webcam.npz \
  --parameters runs/webcam-100/parameters.npz --horizons 1 3 5 10 \
  --out runs/video-horizons.json
```

Each horizon reports overall pixel MSE against last-frame persistence and constant-velocity pixel extrapolation, plus MSE on *changed pixels* defined relative to the final observed frame. Changed-pixel regions use **future truth only for evaluation**. Histories and prediction targets remain entirely within the final 20% of the recording, with nonoverlapping windows. No future frame is ever fed to the predictor during its forecast.

**Interpretation:** a 10-frame forecast is only nominally 1 second if frames were captured at a stable 10 Hz. The evaluator uses frame offsets, not timestamps. Motion-region scoring highlights a failure mode hidden by background pixels, but the mask is an oracle for scoring and is not an available input at inference. Avoid cherry-picking horizons or calling this independent validation; collect a second separate recording later.
