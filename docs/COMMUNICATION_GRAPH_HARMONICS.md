# Geometry versus communication graph harmonics

Earlier `rgb_harmonics.lattice_modes` analyzes fixed rectangular grid
modes on image-aligned recurrent states. A **communication graph**
instead specifies the actual edges that transmit messages. Its
combinatorial Laplacian has different eigenvectors/eigenvalues,
even if node states are *displayed* on an arbitrary rectangular layout.

This independent reference uses 12 supplied skeleton joints,
the known `graph_skeleton_nca.EDGES`, and a 3×4 display grid as a
**different adjacency control**. A degree-preserving edge-rewired
communication graph tests whether an observation depends on
connectivity beyond its node degrees.

Positive intervention: suppress low nonconstant eigenmodes of the
communication graph. Matched null: remove equal L2 state energy
projected onto a random spatial subspace. Forecast consequences are
measured in a **known graph diffusion reference**; they do not
claim a trained graph NCA utilizes the same modes or that these are
physical brain oscillations.

```bash
python -m experiments.communication_graph_harmonics
python -m unittest tests.test_communication_graph_harmonics -v
```

To promote the biological communication hypothesis, repeat against
**frozen trained graph-NCA checkpoints** on held-out motion tasks,
preserve a true topology intervention, and show improvement relative
to rewired and energy-matched random controls. Correlational spectra
alone are descriptive, not evidence of computational necessity.
