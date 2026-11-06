# Neuromatch-inspired research matrix

These are **twelve runnable research hypotheses**, not twelve validated
performance improvements. The baseline implementation already contains
related event-update, RGB homeostasis, harmonic, replay, and active-gaze
studies. This matrix identifies what the new small comparison probes add.

| ID | Hypothesis / measurement | Runner / module | Review |
|---|---|---|---|
| 01 | Sparse changes vs dense observation update counts | `experiments.neuromatch_vision` / `sparse_observe` | [Mirror #211](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/211) |
| 02 | Fixed divisive normalization under lighting gain | `experiments.neuromatch_vision` / `normalize_rgb` | Mirror #211 |
| 03 | Hidden-state trajectory velocity similarity vs time shuffle | `experiments.neuromatch_vision` / `velocity_cka` | Mirror #211 |
| 04 | Top-activity channel lesions vs random-energy null | `experiments.neuromatch_vision` / `targeted_channel_lesions` | Mirror #211 |
| 05 | Local inhibitory feedback vs shuffled/none | `run_neuromatch_circuits` | [NCA #9](https://github.com/walmsley-lab/ml-innovation-19-neuro-cellular-autonoma/pull/9) |
| 06 | Kalman hidden-state tracking vs persistence | `experiments.neuromatch_beliefs` | [Mirror #212](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/212) |
| 07 | Local supervised delta vs global loss gradient and Oja | `run_neuromatch_circuits` | NCA #9 |
| 08 | Paired A→B→A replay forgetting and recovery metrics | `experiments.neuromatch_beliefs --replay-json` | Mirror #212 |
| 09 | Active view information gain vs uniform-action reference | `experiments.neuromatch_beliefs` | Mirror #212 |
| 10 | Two-slot object memory under crossing/occlusion | `experiments.neuromatch_beliefs` | Mirror #212 |
| 11 | Spectral-targeted feedback vs wrong-mode equal-energy null | `examples.neuromatch_control_run` | [Dynamics #1](https://github.com/walmsley-lab/ml-innovation-8-learned-dynamics-control/pull/1) |
| 12 | Fitted-model MPC vs nominal MPC / PD under parameter shift | `examples.neuromatch_control_run` | Dynamics #1 |

## Review and smoke checks

From **Predictive Mirror**:

```bash
python -m unittest discover -s tests -p 'test_neuromatch_vision.py' -v
python -m unittest discover -s tests -p 'test_neuromatch_beliefs.py' -v
python -m experiments.neuromatch_vision --out runs/neuromatch-vision/metrics.json
python -m experiments.neuromatch_beliefs --out runs/neuromatch-beliefs/metrics.json
```

Use a genuine frozen checkpoint for meaningful latent interventions:

```bash
python -m experiments.neuromatch_vision --parameters runs/my-trained.npz
```

For replay retention, first run the *existing* RGB replay experiment:

```bash
python -m experiments.roadmap_33_rgb_replay --out runs/roadmap-33-rgb
python -m experiments.neuromatch_beliefs --replay-json runs/roadmap-33-rgb/metrics.json
```

In **Neuro Cellular Automata**:

```bash
python -m pip install numpy
python -m unittest discover -s tests -v
python -m run_neuromatch_circuits
```

In **Learned Dynamics Control**:

```bash
python -m pip install numpy
python -m unittest discover -s tests -v
python -m examples.neuromatch_control_run
```

## Interpret evidence conservatively

The first runs are **synthetic reference studies**. In particular:

- Sparse observation assimilation is not sparse learned convolution or a
  measured reduction in GPU energy.
- Photometric normalization has not been paired with equal-budget RGB
  retraining. The velocity-CKA diagnostic is not full dynamical similarity
  analysis (DSA).
- Hidden-state lesion ranking is chosen from **past context**; future frames
  are evaluator-only. A random checkpoint is merely a smoke control.
- Kalman filtering, two-target memory, and information-gain selection use
  simplified provided observations or known likelihoods, not learned RGB
  detection or attention.
- The E/I, local learning, wave feedback, and model MPC comparisons use tiny
  independent NumPy systems and are not deployed into real devices, trained
  mirror policies, or cardiac treatment planning.
- The RGB replay diagnostic consumes the existing NCA training experiment.
  It summarizes changes but does not train or improve replay itself.

**Promotion gate:** run three or more seeds and a genuine held-out benchmark,
retain persistence/constant-velocity or established task baselines, report
wall time and memory alongside error, and reject an added mechanism when it
does not improve the selected capability. Keep the original experiments
independent and the test data out of state updates and action selection.

## Study next

Read Neuromatch's lessons on Hidden Dynamics / Kalman Filtering,
NeuroAI Microcircuits, Comparing Networks, Causal Inference, Macrolearning,
and Optimal Control. Work through one exercise per hypothesis *after*
measuring an interesting outcome in this matrix.
