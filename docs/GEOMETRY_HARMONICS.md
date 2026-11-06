# Harmonic cooperation from arbitrary geometry

## Scientific question

Do geometry and local interactions support collective modes that can be
identified from noisy sensor signals, combined, disrupted, and eventually
exploited by a learned graph-local perceiver?

This is a **mass-spring graph-wave analog**, not a solver for a thin elastic
Chladni plate. A real plate's bending modes involve a biharmonic operator,
physical boundary constraints, stiffness, thickness and mass distribution.
The graph Laplacian here characterizes our **chosen discrete coupling**.

## First principles

Each site has a displacement and velocity. Symmetric edge weights produce
a graph Laplacian L. For a scalar displacement q and externally controlled
drive f, the numerical simulator integrates:

\[
  \ddot q = -\omega_0^2 q - cLq -\gamma\dot q-\beta q^3+f(t).
\]

For beta=0, eigenvectors of L are normal mode shapes. Their **angular**
frequencies obey:

\[
  \Omega_k = \sqrt{\omega_0^2+c\lambda_k}.
\]

These generally **are not integer harmonics** of one fundamental frequency.
Even a visually elaborate combination of modes does not by itself establish
nonlinearity or emergent intelligence.

## Executable progression

1. **Find the modes:** Diagonalize L *only for evaluator analysis*. Drive one
   node at a predicted resonance and at a detuned frequency. Measure the
   target modal RMS on an equal-length late window.
2. **Scan broadband forcing:** Excite one site with deterministic white
   noise. Estimate the spectral peak in a mode-coordinate time series. Report
   the error rather than assuming a noisy finite sample recovers an exact
   eigenfrequency. This noise is injected externally, not ambient vacuum
   noise or intrinsic noise of space.
3. **Combine frequencies:** Sum two independent drives. A linear medium must
   reproduce the sum of their separate responses, up to numerical error.
   Introduce a cubic local nonlinearity, then measure the break from
   superposition. That is a test for physical nonlinearity, **not yet**
   evidence of useful neural cooperation.
4. **Perturb/recover:** Under continuous periodic forcing, knock one site
   away from its nominal motion. Quantify how fast a damped system
   reconverges to the unperturbed forced orbit. This is ordinary driven
   relaxation, **not self-correction or autonomy**.
5. **Phase-space progression:** Project displacement and velocity onto a
   mode to obtain a circular phase plane. Add a normalized progress axis
   to inspect a **constructed helix**. The helix is not assumed to be a
   learned higher-dimensional latent or a subjective state.
6. **Train a neural local rule:** Feed sparse broadband-driven observations
   into the existing JAX graph perceiver. Compare the same initial weights
   with neighbor messages on/off, across unseen graph shapes and sizes.
   Evaluation keeps the full physical state outside the model.

## Run

```bash
python -m unittest discover -s tests -p 'test_geometry_harmonics.py' -v
python -m experiments.geometry_harmonics \
  --seeds 11 23 41 --out runs/geometry-harmonics

# Matching recurrent learner, now using harmonic input streams
python -m experiments.geometry_automata \
  --medium harmonic --seeds 11 23 41 --steps 50 \
  --out runs/geometry-harmonic-learning
```

The physics diagnostic writes metrics.json plus per-seed NPZ arrays with
geometry, modal basis, response trajectories and helical chart. The learning
campaign writes its own paired checkpoints and validation/test metrics.

## Falsification and dimensional progression

- Does a true mode resonance exceed an equal-energy detuned drive?
- Does measured linear superposition agree with the computed one?
- Do we see genuine nonlinear mode mixing **above linear controls**, rather
  than mislabeling a sum of modes as nonlinear interaction?
- When information must cross unsensed graph sites, does the learned
  communication mechanism outpredict a no-message model? If not, report
  the null result.
- Can a finite, noisy sensor history identify a mode or topology better
  than a straightforward spectral estimator?
- Can a learned 2D phase representation retain direction and winding without
  being handed polar coordinates? Only then consider slowly drifting phase
  and higher-dimensional helical structure.

**Order of complexity:** 1D scalar oscillation → 2D phase plane →
graph-spatial coordination → slow parameter drift → controlled geometry
changes → learned active tactile sensing. Avoid assuming extra dimensions
create intelligence; compare against simpler reference representations.

Do not call damped oscillation 'persistent life', periodicity 'memory', or
a constructed phase-space helix 'phenomenology'. Show causal necessity and
out-of-distribution performance before promoting any mechanism.

## References

- University of Colorado Boulder, Chladni Figures demonstration:
  https://physicslabs.colorado.edu/demos/oscillations-and-waves/instruments/resonance-in-plates-bars-solids/chladni-figures/
- Exploration of Resonant Modes for Circular and Polygonal Chladni Plates
  (Entropy, 2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC10969725/
- Thin plate physical equations and boundary conditions:
  https://doi.org/10.3390/e24020215
- Graph spectral mode decomposition and dynamical transfer:
  https://link.springer.com/article/10.1140/epjs/s11734-026-02636-w

**Status:** Implemented simulation and training adapter; held-out
quantitative benefits of graph communication or emergent self-maintenance
are not claimed in advance.
