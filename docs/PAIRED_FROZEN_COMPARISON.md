# Frozen paired-history performance comparison

Use one deterministic evaluator on the same scene-pair seeds and return events.

```bash
python -m experiments.paired_frozen_comparison --seeds 71000 71001 71002
python -m experiments.paired_frozen_comparison \
  --forecast convlstm:rgb_only:bthwc:runs/convlstm-test-predictions.npz \
  --forecast simvp:rgb_only:btchw:runs/simvp-test-predictions.npz
```

The scorer does **not** train models; the NCA, classical, ConvLSTM and SimVP runners must export frozen predictions of identical test episodes. This experiment reports per-scene-pair return success, MSE by lead, and a paired-future contrast error. The final image of each opposite-history pair is byte-identical. Reference labels with known simulator geometry are **excluded** from the fair RGB-only ranking.

Only add a reference when it actually completed evaluation. Keep source revision, checkpoint hash, training schedule, input privileges and inference costs in the accompanying evidence archive. This is not yet object identity, probabilistic calibration, or a demonstration of NCA superiority.

Prefer a performance gallery with *intact history vs opposing history* rather than whole-image MSE alone. Train on disjoint scene seeds; never optimize on the frozen test leaderboard.
