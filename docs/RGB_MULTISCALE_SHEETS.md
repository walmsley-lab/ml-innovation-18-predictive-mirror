# PR 49 — Independent fast/slow RGB neural cellular sheets

Two distinct modes use the same parameter dictionary and RGB episodes:
single fast local recurrent sheet versus fast and half-rate slow sheets with
cross-sheet feedback. Both use prediction-error correction and ten-step
open-loop RGB forecasts. Weights are initialized identically, checkpoint
step zero is eligible, and each arm receives the same data and AdamW updates.

    python -m experiments.rgb_multiscale_sheets --steps 50 --horizon 10
    python -m unittest tests.test_rgb_multiscale_sheets -v

Outputs: separate weights for each seed/arm and frozen held-out
+1/+3/+5/+10 MSE versus persistence and observed-image constant velocity.
The single mode allocates the same dictionary but does not *use* the slow
weights. Thus active capacity and recurrent compute are **not** matched;
the study explores whether extra timescales help, not a fair efficiency
winner. The original RGB NCA is unchanged; there is no claimed benefit
before measured, multi-seed tests.
