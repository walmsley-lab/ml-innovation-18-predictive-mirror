# PR 47 — Graph NCA, ten-frame open-loop skeleton forecasting

A distinct extension of the existing `graph_skeleton_nca` model. The
one-step baseline and its GIF demo remain unchanged.

**Question:** Can messages among supplied 2D skeleton joints help maintain
a predictive state across ten future steps with no observations?

Each training sample supplies **visible 2D joint coordinates and visibility
flags**, not camera RGB, depth, or discovered joint topology. The known
12-joint graph is a structural prior. The model absorbs eight observed
skeleton poses, then recursively evolves internal state and predicts ten
future poses without additional observations. Only future **visible** joint
coordinates are used as training targets; hidden truth is evaluation-only.

Matched arms: graph messages enabled, graph messages disabled; plus
last-seen and observed-joint constant-velocity controls. Report joint error
on visible, hidden, and reappearance events at +1/+3/+5/+10 frames,
including event counts. Three independent initialization seeds and fixed
disjoint held-out synthetic actors; save each arm's checkpoint and report.

    python -m experiments.graph_skeleton_horizons \
      --steps 100 --seeds 11 23 37 --context 8 --horizon 10 \
      --out runs/graph-skeleton-horizons

    python -m unittest tests.test_graph_skeleton_horizons -v

This 2D coordinate MAE is **not directly comparable** to RGB NCA image MSE.
One frame is one synthetic timestep, equivalent to one second only when a
10-Hz convention is explicitly imposed. This is a test of graph-local
dynamics, not RGB perception or real human pose detection.
