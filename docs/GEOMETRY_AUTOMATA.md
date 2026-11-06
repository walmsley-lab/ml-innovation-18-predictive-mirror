# From arbitrary geometry to perceptual automata

**Question:** What minimal dynamics let a structure register an interaction,
propagate it, remember partial observations, and predict what comes next?

This is a first-principles mechanistic experiment, **not** a demonstration of
subjective feeling, sentience, or a physical body.

## Physics before categories

Define a geometry as a graph \(G=(V,E,X)\): rest coordinates, undirected edges,
and local state \((q_i,v_i)\). There is no grid, object-recognition head,
supervised object label, or privileged future state.

\[
F_{ij}=k_{ij}(q_j-q_i),\qquad F_{ji}=-F_{ij}
\]
\[
v_i^{t+1}=v_i^t+\Delta t\left(\sum_{j\in N(i)}F_{ij}
-\gamma v_i-\kappa q_i+u_i\right),\quad
q_i^{t+1}=q_i^t+\Delta t\,v_i^{t+1}.
\]

A localized contact \(u_i\) causes motion; neighboring sites exchange equal
and opposite forces. Arbitrary connected graph topology is supplied. Physics
is a small linear displacement-spring approximation, not a general material
simulator. Geometry validation and relabeling tests protect basic invariants.

The model receives only sparse local displacement sensors, the known outgoing
motor impulse, and the graph. A single shared recurrent cellular update
moves information one edge per step and issues the **next** sensor prediction.
Missing sensors use the cell's own previous prediction. Future displacements
and hidden velocities are evaluator-only.

## Learning and falsification

Compare (a) learned one-hop graph messages and (b) the **same weights/size**
with messages disabled. Pair random weights, episode indices, optimizer updates,
and observation budgets. Training includes irregular connected meshes,
branching trees, and rings with 10/13/16 sites.

Hold out **19/21/23-site graphs** and whole episodes. For each, report sensor
MSE against persistence, then perturb the causal channels: misalign motor
commands, remove sensory observations, or cut all message links. These are
diagnostic interventions on the learner, not independent evidence that its
internal representation corresponds to physical cause.

The key progression is:

1. State transition: contact changes local physical state.
2. Coupling: the perturbation reaches other sites via edges.
3. Sensing: partial sensors register consequences of an unseen local action.
4. Prediction: a shared learner forecasts future sensor values.
5. Agency (future): use learned uncertainty to choose where to touch.

The action sequence is externally supplied, not a learned active policy.
The sensor locations and connectivity are known, not inferred from raw input.
A next experiment should demand longer propagation across **unsensed**
regions, followed by changing connectivity while preserving coordinates.
Then compare models on the correct intervention outcome.

## Run from the repo root

```bash
python -m unittest discover -s tests -p 'test_geometry_automata.py' -v
python -m experiments.geometry_automata --seeds 11 --steps 8 \
  --channels 8 --out runs/geometry-automata-smoke
python -m experiments.geometry_automata --seeds 11 23 41 --steps 50 \
  --channels 12 --out runs/geometry-automata
```

Outputs: metrics.json and per-seed, per-arm parameter checkpoints. CPU-first;
do not rent cloud GPUs until the local test earns it. This is deliberately
separate from the existing sand-dollar and spring training curriculum.
Picbreeder-style evolving geometries can be added **after** this randomly
generated geometry baseline is understood.

## Initial local diagnostic (not replicated)

A one-seed, 40-update CPU pilot selected step 40 for both models. Sensor MSE
on held-out irregular/branch/ring geometries was approximately
1.15e-5 / 0.983e-5 / 1.121e-5 with messages; persistence was
1.233e-5 / 1.132e-5 / 1.145e-5. Disabling messages changed the scores only
slightly: 1.155e-5 / 0.991e-5 / 1.122e-5.

**Conclusion:** The simulator and training path work, and a small predictor
sometimes beats persistence. This **does not show** a benefit from cellular
communication. We should report that negative control transparently and
increase the causal difficulty before making a stronger claim.

The system has neither real-world sensor data, a self-model, autonomous
experimentation, nor any evidence of subjective perception.
