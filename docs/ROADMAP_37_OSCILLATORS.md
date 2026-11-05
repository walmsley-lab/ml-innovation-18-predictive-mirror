# Roadmap 37 — matched RGB oscillator training

`python -m experiments.roadmap_37_oscillatory --steps 30`. Three models train separately from the **same NCA parameter initialization**, schedule and episodes: plain, uncoupled phase, coupled phase. Holdout includes unseen period phase, chirps and irregular motion. Record validation-selected step (including zero), frozen test MSE at multiple leads, and wall-clock including compilation.

Extra phase state and spatial coupling require extra computation despite equal trainable parameters; this is not equal-FLOP evidence. A periodic-world win alone neither implies phase coordination in neural tissue nor improvement on real video. Negative results are preserved; do not discard the oscillator family after a short pilot.
