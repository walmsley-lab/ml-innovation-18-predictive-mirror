# Shared cellular transition: signal -> visual prediction

**Difference from the earlier transfer pilot:** the trainable 3x3
hidden-to-hidden convolution (`params["core"]`) is now precisely the same
parameter tensor used by both signal transmission and RGB prediction.
Sensor/readout adapters remain domain-specific. A known destination cue
is used only in the signal task, never in the visual task.

```bash
python -m experiments.shared_core_transfer --pre-steps 60 --rgb-steps 60
python -m unittest tests.test_shared_core_transfer -v
```

Compare scratch, transmission pretraining, and spatially shifted
pretraining on held-out RGB sequences. This *tests* the developmental
hypothesis; it does not establish it. Scratch has fewer total updates;
a matched unrelated-task pretraining control and checkpoint selection
are still needed to isolate the effect of transmission learning.
