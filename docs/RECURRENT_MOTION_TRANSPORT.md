# Motion transport with persistent recurrent cellular state

The old RGB transport experiment uses a feedforward residual. This new
candidate explicitly combines motion-based image transport with an
NCA that maintains hidden state and corrects it after observations.

```bash
python -m experiments.recurrent_transport_nca --seeds 11 23 41 --steps 80
python -m unittest tests.test_recurrent_transport_nca -v
```

Arms use the same images, initialization and updates: NCA-only,
transport-only, and transport+NCA correction. The NCA receives the
actual *committed* prediction at the next observation; withheld future
frames are never supplied during dreaming. Local transports cover only
nine candidate displacements and cannot resolve all multiple-object
motions. The branches have unequal active computation. No performance
advantage is claimed without held-out event/generalization results.
