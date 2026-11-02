# PR 51 — RGB phase entrainment and traveling-phase differences

An **independent** phase-dependent RGB NCA experiment extending, not
overwriting, the earlier plain/uncoupled/coupled phase ablation.

The four controls are: plain RGB NCA, spatially heterogeneous free-running
phase, nearest-neighbor coupled phase, and phase *entrained by observed
sensory prediction error*. Each mode uses identical trainable base model
weights and equal training scenes, optimization steps, and three seeds.
Oscillator phase adds compute and nontrainable state. The initial
zero-readout persistence model competes as step-zero checkpoint in all arms.

Sensory error influences phase only when a new observed frame actually
arrives. During open-loop rollout, phase continues on intrinsic and
neighbor dynamics, without reading future pixels. Heterogeneous natural
frequencies create evolving relative phase patterns; coupling can form
spatial phase gradients. No claim of literal biological traveling waves
or neuronal binding without additional spatial-wave analysis.

    python -m experiments.rgb_phase_entrainment --steps 40 --horizon 10
    python -m unittest tests.test_rgb_phase_entrainment -v

Reports held-out +1/+3/+5/+10 RGB MSE, global Kuramoto-like coherence and
batch evaluation time (including first-call compilation). Original phase
and RGB NCA implementations/checkpoints are not modified.
