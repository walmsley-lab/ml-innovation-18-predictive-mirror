# Diagnosing the zero-effect experiments

Several experiments report that an intervention changes the open-loop RGB
forecast by ~0. A ~0 delta has two very different causes, and they must not be
reported the same way:

- **Genuine null** — the forecast still depends on its inputs and *does* move
  when a known positive-control perturbation is applied, yet the specific factor
  under test (a transferred core, a suppressed harmonic subspace, a feedback
  term, a camera action) truly does not change held-out predictions.
- **Dead measurement** — the open-loop forecast has collapsed to an
  input-independent fixed point. For this model family that fixed point is an
  **all-black frame pinned against the `clip(x, 0, 1)` lower rail**. Once the
  forecast is railed, *no* upstream intervention can change it, so a ~0 delta is
  uninformative rather than evidence of a null.

This note records the root cause, the evidence, and the positive controls that
were added so each result can be read correctly. The shared helpers live in
[`src/forecast_diagnostics.py`](../src/forecast_diagnostics.py); every runner
now writes a `measurement` block with three diagnostics:

- `clip_rail_saturation` — fraction of forecast pixels pinned to the clip rails;
- `input_independence` — mean spread of the forecast across different scenes
  (≈0 means every scene maps to the *same* frame: a collapsed fixed point);
- `positive_control_response` + `measurement_verdict` — by how much a *known*
  perturbation moves the forecast, and a per-lead label:
  `live` (control moves it → a ~0 result is a real null),
  `dead` (collapsed and unmovable → ~0 is uninformative),
  `insensitive` (forecast is input-dependent but even the control barely moves
  it → the control is too weak to certify that lead).

## Root cause shared by the frozen RGB NCA (`predictive_rgb_nca`)

`dream()` commits `future = clip(image + 0.20·tanh(hidden @ rgb_readout + bias), 0, 1)`
and feeds that frame back as the next input. On the frozen campaign checkpoint
(`seed-11-nca.npz`) the readout pre-activation already has magnitude ≈2.4, so
`tanh` is saturated and the correction is a nearly state-independent ≈−0.2 per
channel. The synthetic world is sparse (mean pixel intensity ≈0.04), so a few
dream steps drive every pixel to the lower clip rail and the rollout reaches an
**absorbing all-black fixed point**:

| lead | clip-rail saturation | forecast spread across scenes |
|---|---|---|
| 1 | 0.994 | 0.0079 |
| 5 | 1.000 | 0.0000 |
| 10 | 1.000 | 0.0000 |

At leads ≥5, different scenes produce *bit-identical* black frames; even wiping
the entire hidden state, or forcing the fed-back image to mid-gray, decays back
to black within two steps. The forecast "beats persistence" only because
predicting darkness is near-optimal on a mostly-dark world — not because it
anticipates motion. This single mechanism explains three of the four results.

## The four experiments

### 1. Harmonic mode interventions — **broken measurement at long leads**

*Verdict (frozen checkpoint): leads 1,3,5 `live`; lead 10 `dead`.*

Suppressing the low spatial modes of the hidden state changes the forecast by
2.7e-7 (lead 1), decaying to exactly 0 at leads 5 and 10 — even though the
harmonic microscope shows ~49% of the hidden spatial energy lives in those
modes. The positive control (wiping all hidden memory) moves the forecast by
0.026 at lead 1 but by 0.000 at lead 10. So:

- At **lead 10** the measurement is dead (railed, input-independent, unmovable):
  the ~0 suppression delta says nothing about harmonics.
- At **early leads** the measurement is live, yet low-mode suppression moves the
  forecast ~10⁵× less than a real state change does. The honest reading is that
  **this checkpoint barely uses its hidden state for forecasting** (it predicts
  near-static darkness), so it is a poor substrate for a harmonic-causality
  claim. This matches the protocol's own hedge that "a persistence-dominated
  model is an expected negative control."

**Not a clean scientific null.** Re-run on a checkpoint whose forecast is
*not* collapsed (`measurement.input_independence` clearly > 0 and
`clip_rail_saturation` well below 1 at the target leads) before interpreting the
low-mode deltas at all.

### 2. Transmission → RGB transfer — **broken: degenerate shared solution**

*Verdict: leads 1,3 `insensitive`; leads 5,10 `dead`.*

Scratch, transferred-core, and spatially-shifted-core arms produce forecasts
that are **bit-identical at leads 5 and 10** and differ by <1e-4 at leads 1–3.
The forecasts are 98–100% railed. The positive control (scaling the scratch
readout ×3) also fails to move the railed forecast. All three arms fine-tune
into the *same* collapsed black-frame predictor, so the initialization cannot
matter — this is a degenerate task solution, not an informative transfer null.
The `init_effect_on_forecast` field (transmission/shifted vs scratch) is
reported next to the positive control so the two near-zero magnitudes can be
compared directly.

**Not evidence about transfer.** A meaningful test needs a fine-tuned model that
actually forecasts motion (non-collapsed `measurement` block).

### 3. Homeostatic recovery — **genuine prediction null (pathway is live)**

*Verdict: `live` at every lead.*

Here the mechanism is demonstrably connected to predictions. On the frozen
selected `plain` weights, dialing the recovery gain up moves the forecast
monotonically — strength 0.15 → 0.0007–0.005, strength 1.0 → 0.011–0.013 across
leads 1–10 — so the recovery→forecast pathway is live everywhere. Internally the
feedback also halves `mean_abs_hidden` and cuts spatial variance ~10×. Yet when
each arm is *independently retrained*, held-out MSE is equal across plain,
local-recovery, and shuffled-recovery to within seed noise: the retrained
readout simply absorbs the regulariser.

**This is a real null:** the regulariser changes the internal representation but
neither helps nor hurts generalization. The positive control rules out the
"disconnected feedback term" explanation.

### 4. 3D action conditioning — **genuine but weak/underpowered effect**

*Verdict: leads 1,3 `insensitive`; lead 5 `live`.*

Unlike the RGB NCA, the action model's forecast is healthy: it is **not** railed
(saturation <0.5%) and differs across scenes (`input_independence` ≈0.035). The
action pathway is live — feeding the trained action-aware model counterfactual
plans moves its forecast monotonically and sign-sensitively (zeroed actions:
1.8e-4 → 8.8e-4; negated: 3.6e-4 → 1.8e-3 across leads 1→5). So actions are not a
dead wire.

But the learned action response (~1e-3 at lead 5) is ~10× smaller than the
world's *true* action dependence (`world_action_dependence` ≈0.016–0.018, the
ceiling on any learnable effect). The model under-uses actions, and this small
effect is far below the aggregate test-MSE floor, which is why two
separately-trained aware/blind arms score within ~1e-6 of each other.

**A real but weak effect, reported as underpowered rather than absent.** To
resolve it, score an action-dependent region or compare one model under
counterfactual action plans (as the positive control does) instead of comparing
the aggregate MSE of two independently trained arms.

## Summary

| Experiment | Zero-effect cause | Classification |
|---|---|---|
| Harmonic mode interventions | Frozen NCA collapses to railed black frame; hidden state barely drives the forecast | **Broken** at long leads; uninformative substrate at short leads |
| Transmission → RGB transfer | All arms fine-tune to the same collapsed black predictor | **Broken** (degenerate shared solution) |
| Homeostatic recovery | Feedback reshapes internals; retrained readout absorbs it | **Genuine null** (pathway verified live) |
| 3D action conditioning | Real, ordered action response ~10× below the world's action signal and below the MSE floor | **Genuine but underpowered** (effect verified live) |

**Rule going forward:** before interpreting any ~0 intervention result, check the
`measurement` block. A `dead` lead means fix the model (escape the clip-rail
collapse) before drawing conclusions; a `live` lead with a ~0 scientific delta is
a real null; an `insensitive` lead needs a stronger positive control or a
targeted metric.
