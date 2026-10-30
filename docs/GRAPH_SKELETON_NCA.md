# Independent experiment: graph neural cellular automata on articulated skeletons

**Hypothesis:** messages sent along known kinematic edges may improve one-step joint-location predictions and recovery after temporary occlusion compared with identical recurrent cells whose messages are disabled.

This is a **separate implementation**: it neither modifies the live pixel mirror nor combines features from prior models. Never equate the joint-coordinate MAE in this experiment with the pixel MSE in other experiments.

## Representation and method

- 12 named 2D joints; 11 fixed undirected skeleton edges.
- Hand-coded deterministic synthetic actor with changing morphology, gait phase and speed.
- Each node has 16 recurrent channels and shares the same local neural update rule.
- Input is **supplied visible 2D joint coordinates**, a visibility bit, graph edges and its own recurrent state. Occluded coordinates are zeroed before input.
- Simulator's hidden joint positions are never used as training targets; they are evaluation-only. The model is trained only to predict *future visible* joints.
- **Not a 3D model, not a real-video pose estimator, not learned skeleton discovery**. The graph is prior knowledge.

## Paired controls and metrics

One run trains a graph-message model and a local-only model on identical episodes and optimizer budgets, plus last-seen coordinate persistence. Score held-out actors for visible-joint error, hidden-joint error and reappearance error (at a declared 32px coordinate scale), reporting event counts and visibility counts.

```bash
python -m unittest tests.test_graph_skeleton_nca -v
python -m experiments.graph_skeleton_nca --steps 100 --occlusion-frames 3
cat runs/graph-skeleton-nca/metrics.json
```

For a three-panel evaluation GIF (requires Pillow):

```bash
python -m pip install Pillow
python -m experiments.graph_skeleton_demo --parameters runs/graph-skeleton-nca/graph.npz
```

Repeat across `--occlusion-frames 1`, 3 and 5 and multiple seeds. Future requirements: varying topology, perturbation/zero-shot movement, explicit kinematic-motion baselines, independent 2D pose extraction, wall-clock latency and eventual tests of 3D structure. A coupled system should be a **new preregistered experiment**, not a change to this model.
