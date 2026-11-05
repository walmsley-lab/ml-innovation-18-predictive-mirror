# Roadmap 42 — hardware-constrained cellular comparisons

Run `python -m experiments.roadmap_42_hardware`. Evaluate the **existing** portable two-state local cell, a 4-bit quantized state variant, a no-coupling and no-slow-memory control against persistence and observed-image velocity on identical RGB episodes. Measure CPU rollout latency per size/horizon and separately report an *optimistic theoretical* state-bit floor, input-DAC/output-ADC transactions and site updates.

This is not a chip model. The circuit uses NumPy float32 even when numerically quantized; no demonstrated bytes, joules, ADC cost or physical I/O advantage follows. The readout is per RGB pixel/channel and may dominate a hardware substrate. Device contracts from Hardware-Informed Model / Developmental Physical Computing should eventually supply noise, precision, bandwidth, energy and peripheral constraints without silently extrapolating.
