# Progressive spatial oscillators: matched RGB NCA ablation

**Three arms:** plain RGB NCA, uncoupled oscillators, and locally coupled
oscillators (0.25 Kuramoto-like gain). Phase has heterogeneous spatial
initialization, observation-responsive rotation and optional neighbor
alignment. The cellular RGB model weights and optimizer are identical
across arms; phase adds compute and state but **no trainable parameters**.

Run `python -m experiments.rgb_phase_coupling_ablation --horizon 10 --steps 60`.

The phase mechanism is a deliberately simple *hypothesis*, not evidence of
brain-equivalent complex-valued computation, learned velocity entrainment,
traveling waves, or successful one-second anticipation. Multi-seed held-out
+10 RGB/motion error must beat the plain and velocity controls before
promoting oscillator complexity. Keep prior grayscale oscillator models.
