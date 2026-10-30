# PR 10 — Observation-derived contrast objective (optional)

The 120-update recurrent CPU run with **plain MSE** improved whole-frame MSE relative to persistence (0.01152 vs 0.01387), but **missed all 30 held-out reappearance events** in the moving-camera evaluation (16-pixel no-detection penalty each). A sparse foreground pixel is easy for plain MSE to ignore.

For an explicitly different training-objective experiment, `--loss-mode contrast` multiplies pixel squared error by `1 + 32*(observed_next_pixel > 0.9)`. The highlight comes solely from the **observed next frame** available as an ordinary supervised training target. It is *not* a simulator-provided object mask, world position or hidden state. Ground-truth simulator masks remain evaluation-only.

```bash
python -m train --architecture recurrent --curriculum staged --steps 240 --batch-size 8 --loss-mode contrast --output runs/contrast
python -m evaluate --parameters runs/contrast/parameters.npz --output runs/contrast/evaluation.json
```

The 240-update CPU pilot with this objective reduced reappearance MAE to **7.1 pixels** (30 events), but worsened plain moving-camera MSE to **0.0220**, versus **0.0117** for the 120-update plain-loss run. Because training steps differ, this is a motivating diagnostic, **not a fair objective ablation**. On Akash, compare plain versus contrast with identical seeds, curriculum, steps, compute and holdouts. The threshold 0.9 and positive weight 32 are frozen; other objectives are future research questions. Never claim successful perceptual tracking while detection accuracy remains poor.

Checkpoints now pin loss mode; `sweep.py` reads `LOSS_MODE=plain|contrast` and keeps it in each row and summary key. There is no simulator-label leakage into the loss.
