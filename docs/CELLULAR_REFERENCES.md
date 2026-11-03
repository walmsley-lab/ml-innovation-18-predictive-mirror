# Co-learn content-addressed cellular communication

This independent experiment tests a small content-addressed **soft reference**
router. Query, key, sender message, receiver update and readout parameters
are learned jointly from signed-pulse transmission. Compare uniform routing,
learned routing, and a shuffled-address control.

```bash
python -m experiments.cellular_references --steps 60 --seeds 11 23 41
python -m unittest tests.test_cellular_references -v
```

The destination is an explicit query mask. Only the source pulse carries
the hidden sign label; the loss supervises its delivery at the destination.
Training and validation distances are 2..4 Manhattan cells; test distances
are 5..7. Report sign accuracy, MSE, routing entropy and selected checkpoint.

**Limitations:** All-to-all attention uses O((HW)^2) storage and is intended
only for 8–10 pixel grids. It is not a scalable implementation of global
object references. The source/destination relationship comes from the task,
not RGB object discovery. The uniform control allocates but does not use
query/key weights, so active parameter counts differ. This is a mechanism
probe, not evidence of emergent binding or learned visual attention.
