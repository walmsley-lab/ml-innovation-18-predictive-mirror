# Does transmission pretraining transfer into RGB prediction?

This experiment **depends on the cellular transmission model and checkpoint**
from the preceding PR. It compares three RGB NCA initializations under
the same RGB training episodes, optimizer and fine-tune update budgets:

- scratch initialized;
- partially initialized with a trained communication core;
- same transferred core spatially shifted to disrupt its learned geometry.

A shape-checked mapping transfers the hidden-to-proposal convolution
channels and proposal mixing block; the sensor-specific connections,
RGB readout and gate are *not* transferred. Because the two architectures
are different, this is a **heuristic initialization experiment** and does
not preserve all pretrained computational semantics.

```bash
python -m experiments.cellular_transmission --steps 80 --seeds 11
python -m experiments.transmission_rgb_transfer \
  --communication-checkpoint runs/cellular-transmission/seed-11-local_messages.npz \
  --steps 60 --seeds 11 23 41
python -m unittest tests.test_transmission_rgb_transfer -v
```

Report validation-selected step (including step zero), multi-lead test
RGB MSE, persistence/velocity controls, and seed-wise outcomes. Do not
claim the communication curriculum helps merely because the weights can
be loaded. A stronger future experiment should share *exact* communication
weights architecturally and match total pretraining+fine-tuning compute.
The current independent RGB NCA and communication experiments remain intact.
