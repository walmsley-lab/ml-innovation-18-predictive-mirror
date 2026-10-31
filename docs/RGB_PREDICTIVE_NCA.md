# Independent experiment: RGB predictive neural cellular automaton

This is a **separate architecture** from the grayscale recurrent sheet, fast/slow multisheet, phase oscillators and skeleton graph NCA. Their code and checkpoints remain unchanged. The new model has a spatial cellular state (12 channels by default at every 32x32 RGB pixel), a **shared local 3x3 selection/update rule**, repeated internal updates, explicit observed-prediction correction and an unobserved future "dream" rollout. There is no 3D depth model or emergent object identity.

## NCA components

- **Select:** for now each cell selects its fixed 3x3 spatial neighborhood via a shared convolution. **Relative-coordinate selectors, explicit references/pointers and learned routing are not in this PR.** Test them as separately labeled implementations rather than conflating them with this local model.
- **Update:** each cell uses the same small gated rule and maintains its own latent state, advancing for `--ticks` iterations for each input or forecast.
- **Sense / correct:** before new observation `x_t`, the model had dreamed its prediction `x_hat_t`. The correction stage receives `x_t - x_hat_t`, then clamps its RGB input to `x_t`. We explicitly dream between observed context frames so this is an actual prediction error rather than simple temporal difference.
- **Dream / predict:** no new observations are supplied. Iterate the NCA, read out a three-channel RGB residual, then feed that predicted RGB state into the next forecast; score +1..+H against withheld future labels only.

The initial zero RGB readout makes the first forecasts exactly last-frame persistence. Training must beat this baseline on independent held-out RGB sequences before asserting predictive ability.

## Color camera and quick CPU experiment

On this HP Fedora laptop, `/dev/video0` advertises YUYV/MJPG, while `/dev/video2` advertises `GREY` at 340x340 (likely IR). Use **camera 0**, and stop any stale Python process holding it first. The capture code preserves **three RGB channels** instead of calling `COLOR_BGR2GRAY`.

```bash
source .venv/bin/activate
python -m pip install opencv-python
python -m experiments.rgb_nca capture --camera 0 --seconds 60 --fps 10 --size 32
python -m experiments.rgb_nca fit --steps 30 --batch-size 2 --horizon 5 --context 6 --ticks 2
cat runs/rgb-nca/metrics.json
```

If a short run is stable and results are meaningful:

```bash
python -m experiments.rgb_nca fit --steps 100 --batch-size 4 \
  --horizon 5 --context 6 --out runs/rgb-nca-100
```

Keep independent checkpoint directories. Multi-tick dream training may take longer than the prior small one-step grayscale model.

## RGB live / replay demo

```bash
python -m experiments.rgb_nca_viewer --source replay --size 32 \
  --input recordings/rgb-webcam.npz --parameters runs/rgb-nca/parameters.npz
python -m experiments.rgb_nca_viewer --source live --camera 0 --size 32 \
  --parameters runs/rgb-nca/parameters.npz
```

The display shows actual future RGB (offline replay only), RGB persistence, RGB prediction, absolute RGB error (replay) and pseudocolored neural activity. Live errors are scored **after** corresponding future observations arrive. Requested FPS isn't an independently verified sensor exposure rate.

## Evidence and limits

- Per-horizon full-frame MSE, changed-region MSE and changing-pixel counts versus persistence and pixel extrapolation. No hidden simulator truth enters inference.
- The 70/10/20 temporal train/embargo/validation split selects the best saved checkpoint; resulting scores are **validation only**, not unbiased test estimates. Capture another independent recording for honest evaluation.
- Never directly compare RGB MSE with the earlier grayscale models on different video; they require a common color/luma benchmark on the same scenes.
- No cell growth/topology evolution, learned selection, reference-based routing, object permanency or 3D reconstruction is claimed.
