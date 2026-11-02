# PR 52 — Causal interventions on a frozen RGB NCA

This experiment changes *only* internal hidden-state content. The same
checkpoint and RGB observation prefix are used in every arm:
intact, zeroed hidden state, and spatially displaced hidden state.
Perturbations happen immediately after observation context or after a
specified number of open-loop dream steps, **never** by changing future
inputs. The most important test is whether predicted RGB differs after
the intervention, and whether any difference helps or hurts on unseen
scenes.

    python -m experiments.rgb_hidden_interventions \
      --checkpoint runs/rgb-nca-1s/parameters.npz \
      --context 10 --horizon 10 --intervene-at 3
    python -m unittest tests.test_rgb_hidden_interventions -v

The report includes forecast perturbation magnitude, per-horizon RGB
error against frozen persistence and constant-velocity controls, and
sample forecasts for subsequent visualization. All future scene truth is
scoring-only. Initial zero-readout persistence models should show no
output difference; no causal memory claim is warranted from them.

This probe cannot by itself establish that hidden state encodes objects,
that memory improves prediction, or that the model is biologically
equivalent to cortical recurrence.
