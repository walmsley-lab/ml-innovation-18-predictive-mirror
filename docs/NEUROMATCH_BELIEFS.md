# Neuromatch bridge: beliefs, memory, active sensing, replay

Four executable reference investigations complement the existing RGB NCA
rather than duplicating its training pipeline:

- **Belief filtering:** causal constant-velocity Kalman estimates with missing
  observations, explicitly compared to last-seen position persistence.
- **Structured object memory:** a *two-object, 1D, appearance-tagged detection*
  tracker through crossing and temporary occlusion. Detections are supplied;
  this is **not RGB object recognition**.
- **Active sensing:** expected entropy reduction from an explicit known
  observation likelihood. Compare the selected view against a uniform-action
  expectation; **not a learned sensor policy**.
- **Continual replay:** parse actual RGB NCA replay experiment metrics
  produced by `experiments/roadmap_33_rgb_replay.py`; report A→B forgetting,
  return-to-A recovery, and paired seed comparisons. No extra training or
  claim that replay improved results.

```bash
python -m unittest discover -s tests -p 'test_neuromatch_beliefs.py' -v
python -m experiments.neuromatch_beliefs
python -m experiments.roadmap_33_rgb_replay --out runs/replay-comparison
python -m experiments.neuromatch_beliefs --replay-json runs/replay-comparison/metrics.json
```

Simulator truth is used by the runner solely for scoring; filtering and active
action selection have no access to target future labels. Validate assumptions
using independently held-out RGB recordings before promoting them to a
predictive representation or camera agent.

Related lessons: Neuromatch Hidden Dynamics / Kalman filtering, optimal
control, macrolearning / replay, and modularity.
