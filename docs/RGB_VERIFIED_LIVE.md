# Live mirror, with delayed target verification

`python -m experiments.rgb_verified_live --parameters runs/rgb-curriculum/seed-11-curriculum.npz --camera 0 --fps 10 --lead 10`

The viewer displays the observed frame at issue time, its **already-issued**
+10 prediction, the actual frame when that target finally arrives, and
their absolute error. A live camera view alone is not a predictive mirror.

All scores are from matched **earlier** predictions, never predictions
made using the current target. Report measured elapsed horizon and
inference latency; nominal +10 frames is not necessarily one wall-clock
second on a busy CPU. Run headless with `--no-display --max-frames 100`.
This is a measurement/demo utility, not proof the checkpoint learned motion.
