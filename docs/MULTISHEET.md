# PR 5 — Two recurrent sheets

Two locally recurrent sheets (6 fast channels + 2 slow channels) use different integration constants and receptive-field dilation. Both have the **same 8 hidden scalars per pixel and exactly the same 729 trainable parameters** as the single-sheet model. This is a spatial/temporal prior, not preassigned object or motion cortex. Cross-talk is bandwidth-limited by the two-channel slow population; there is no explicit graph pruning yet.

```bash
python -m train --architecture multisheet --curriculum staged --steps 120 --batch-size 8 --output runs/multisheet
python -m evaluate --architecture multisheet --parameters runs/multisheet/parameters.npz --output runs/multisheet/evaluation.json
```

Use `src.multisheet.trace` for post-hoc probes and `predict(..., feedback_strength=0.0)` for a test-time feedback intervention. Strict causal claims require matched measured FLOPs/latency, training without feedback, random within-sheet cuts and decoder-accuracy-qualified compatibility; none is implemented or claimed by this PR. Do not label sheet function until evaluated.
