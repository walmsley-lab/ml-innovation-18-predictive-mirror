# Separate cellular message-modulation experiment

Synthetic task: two signed source pulses are emitted at distinct grid
locations, and one source is marked relevant using a *given* cue. A destination
is given as a query. Can a shared cellular rule deliver the relevant sign
in the presence of the interfering pulse?

```bash
python -m experiments.cellular_message_modulation --steps 60 --seeds 11 23 41
python -m unittest tests.test_cellular_message_modulation -v
```

Models share initialization, training examples and optimizer steps.
The three arms are:
- `plain`: ungated spatial messages;
- `gated`: a learned convolutional receiver gate multiplies messages;
- `shuffled_gate`: spatially shifted receiver gates break location alignment.

The gate controls **receiver-side message strength**, not sender-specific
connections or dynamic graph rewiring. A cue identifies the relevant source;
this experiment does not discover relevance autonomously. The plain arm
allocates inactive gate parameters, so active capacity and runtime are
different. Report the null control and failures honestly.

This is a separate skill probe; combining staged transmission training
with these message rules is reserved for the developmental transfer
experiment. No original NCA or freeze-frame benchmark is modified.
