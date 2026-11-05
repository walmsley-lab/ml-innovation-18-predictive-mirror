# History-required object permanence: frozen research protocol

The hypothesis is **not** that any recurrent NCA will necessarily win.
The test asks whether a predictor commits to the correct reappearance
location and time when its last visible input is identical across
two histories with opposite approach trajectories.

Run matched CNN/transport/hierarchical/plain NCA/blend NCA/routing
architectures with identical RGB inputs, training pairs, contrast-aware
losses, validation selection and held-out test pairs. Models see
only observed image history. Baselines are last-frame persistence and
a **strong RGB-history linear extrapolator** with declared knowledge of
the synthetic disc/occluder. Oracle future truth is used only as a
scorer golden case, never as a benchmark entrant.

```bash
# Deterministic CPU smoke:
python -m unittest tests.test_object_permanence_campaign -v
# Substantive experiment; deliberately not launched by this PR:
python -m experiments.object_permanence_campaign \
  --models cnn transport hierarchical nca nca_blend routing \
  --seeds 11 23 41 --steps 150 \
  --train-pairs 24 --val-pairs 8 --test-pairs 12
```

Gates: report one seeded checkpoint per model and validation-selected
step, paired history-contrast MSE, framewise object mass/location/timing,
**per-event success** (within 3px / ±1 frame, no premature visibility)
and a paired bootstrap confidence interval against both simple
controls. Each run writes SHA-256-linked checkpoint and a two-direction
forecast gallery. Report failures and undefined metrics as such.

The paired episode bootstrap assumes episode pairs are independent
within this simulator; two directions from one seed are not independent.
Train/validation/test seed namespaces are disjoint. Equal SGD steps
are not equal FLOPs, and this protocol does not claim equal throughput.

A successful NCA must beat **both** persistence and the strong
history-based extrapolator with stable localization on held-out
trajectories. Low image loss or perfect bright-mask recall is not a
substitute for a real object return. No result is claimed until
the substantive campaign has actually been run.
