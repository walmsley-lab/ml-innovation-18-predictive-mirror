# Evidence-gallery bridge for recorded and live mirror demos

The repo already supports [verified live delayed scoring](RGB_VERIFIED_LIVE.md) and [independent recording transfer](RGB_WEBCAM_TRANSFER.md). This PR does not replace those runners. It produces **offline held-out replay** contact sheets that a viewer or paper author can inspect, with input/target pairing, checkpoint/recording hashes, and actual capture interval when timestamps exist.

```bash
python -m experiments.verified_replay_gallery \
  --checkpoint runs/rgb-curriculum/seed-11-curriculum.npz \
  --recording recordings/heldout-rgb.npz --horizon 10
```

Output: observed anchor, committed-model offline forecast, later actual frame, and absolute error in `forecast-NN.png`, with `manifest.json` containing scene/window and file provenance. Do not imply this replay was performed on-device in real time, or that it demonstrates camera-world generalization without the independent recording check.

For a genuinely **live** result, use `experiments.rgb_verified_live`; it issues predictions before future capture and measures real elapsed time. The recorded-video gallery instead makes credible performance visuals for the paper once a useful checkpoint is available.
