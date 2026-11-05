# Training-budget sweep: undertrained vs collapsed vs learning

**Question.** When a model produces useless forecasts, is it *undertrained*
(needs more steps), *collapsed* (stuck at an input-independent fixed point that
more steps cannot escape), or actually *learning* (localizing the moving
object)? Training longer only helps the first. This sweep trains several model
families at increasing step budgets, saves a checkpoint at every budget, and
scores each one on **foreground and localization**, not just MSE, so the three
cases are distinguishable instead of blurred into one low MSE number.

Runner: `experiments/training_budget_sweep.py`
(`python -m experiments.training_budget_sweep --help`).
Anti-collapse model variants: `src/rgb_grounded_nca.py`.

## Design

* **Five independent families** on identical worlds: `cnn`, `transport`,
  `nca` (the collapsing baseline, **kept unchanged as a control**), `routing`
  (learned soft references), `hierarchical` (two-scale surprise).
* **Three anti-collapse levers** applied to the NCA cellular core, as separate
  arms so each lever is attributable:
  * `nca_fg` — foreground-sensitive loss (extra weight on true object pixels);
  * `nca_blend` — non-saturating convex-blend output head
    `future = (1-gate)·image + gate·color`, a convex blend of two in-range
    quantities that **cannot accumulate past the clip rail**;
  * `nca_grounded` — blend head + foreground loss + balanced scene sampling.
* **Budgets** 300 / 600 / 1200 steps, **3 seeds** (11, 23, 41). Each arm is
  trained once to the largest budget; a checkpoint is snapshotted at every
  budget, so a later, larger run can re-score the same trajectory.
* **Validation tracks foreground, not just MSE**: foreground recall / IoU,
  soft-centroid localization error, predicted foreground mass, plus the collapse
  diagnostics (clip-rail saturation, cross-scene input independence). The
  predictor only ever sees RGB context; palette/visibility truth is read by the
  evaluator *after* each forecast is committed.
* **Status label** per checkpoint (heuristic, not a significance test):
  * `collapsed` — saturation ≥ 0.9 **and** input-independence ≤ 1e-3 **and**
    predicted mass < 0.05;
  * `learning` — predicted mass ≥ 0.3 **and** foreground recall ≥ 0.2;
  * `undertrained` — otherwise.

## Results (10-frame horizon, mean over 3 seeds)

| arm | budget | status | MSE | mass | recall | soft-centroid px | clip-sat | input-indep |
|---|---|---|---|---|---|---|---|---|
| cnn | 300/600/1200 | undertrained | 0.0073–0.0078 | 1.6–2.1 | 0.00 | 9–10 | 0.00 | ~5e-3 |
| transport | 300/600/1200 | undertrained | 0.0071–0.0073 | 2.5–7.4 | 0.00 | 8–10 | 0.00 | ~5e-3 |
| hierarchical | 300/600/1200 | undertrained | 0.0074–0.0084 | 3–21 | 0.00 | 9–10 | 0.01–0.23 | ~7e-3 |
| **nca** (control) | 300/600/1200 | **collapsed** | 0.0104 | 0.00 | 0.00 | — | **1.00** | **0.0** |
| **routing** | 300/600/1200 | **collapsed** | 0.0104 | 0.00 | 0.00 | — | **1.00** | **0.0** |
| **nca_fg** | 300/600/1200 | **collapsed** | 0.0104 | 0.00 | 0.00 | — | **1.00** | **0.0** |
| nca_blend | 300/600/1200 | undertrained | 0.0087–0.0090 | 0.0–1.4 | 0.00 | ~8.7 | **0.00** | ~4e-4 |
| nca_grounded | 300/600/1200 | undertrained | 0.011–0.012 | 0.0–0.36 | 0.00 | ~8.7 | **0.00** | ~6e-4 |

(`—` soft-centroid: no object was localized because the forecast is all black.
`mass` is predicted foreground mass ÷ true mass; values ≫ 1 mean the model
paints diffuse brightness everywhere, not a localized object.)

## What the sweep establishes

1. **Collapse is an architecture problem, not a budget problem.** The `nca`
   control is collapsed identically at 300, 600 and 1200 steps — saturation
   exactly 1.0, input-independence exactly 0, MSE pinned at 0.0104 (the value a
   constant all-black forecast scores). More steps never move it: the all-black
   clip-rail fixed point is absorbing, and once there the `tanh` and the clip
   both zero the gradient, so there is no signal to escape. Training the
   collapsing NCA longer is futile, exactly as cautioned.

2. **Foreground-sensitive loss alone does NOT prevent collapse** — a genuine
   negative result for that lever in isolation. `nca_fg` carries 6× extra loss
   weight on true object pixels and still collapses to the identical black fixed
   point at every budget. The loss *wants* foreground, but it cannot reach a
   model that the output head has already driven to the rail with vanished
   gradients. A loss term cannot fix an absorbing fixed point the architecture
   creates.

3. **`routing` collapses for the same reason as the baseline** — it inherits the
   additive residual-clip head, so learned soft references buy nothing once the
   rail is reached.

4. **The non-saturating blend head is what actually prevents collapse** — the
   decisive positive result. `nca_blend` and `nca_grounded` never saturate
   (clip-sat 0.00) and stay input-dependent (nonzero independence); their
   forecasts are no longer pinned to black. This isolates **output dynamics**,
   not the loss or the sampler, as the lever that removes the absorbing fixed
   point.

5. **Preventing collapse is necessary but not sufficient — and at these budgets
   nobody reaches "learning".** No arm attains foreground recall; soft-centroid
   error stays near 8–10 px on a 32-px world (≈ a third of the frame), i.e. no
   real localization. The non-collapsed baselines (`cnn`/`transport`/
   `hierarchical`) emit diffuse, over-bright, weakly scene-specific forecasts
   (mass ≫ 1, recall 0); the blend NCAs emit dim, only weakly differentiated
   forecasts (independence ~5e-4). "Eventual learning" was **not observed** in
   the 300–1200 CPU-step range. That is the honest finding, and the per-budget
   checkpoints are precisely what lets a larger GPU run re-score this trajectory
   and detect learning if and when it appears.

## How to reproduce

```bash
python -m experiments.training_budget_sweep \
  --out runs/training-budget-sweep --seeds 11 23 41 --budgets 300 600 1200
```

Writes per-arm/seed/budget checkpoints `{arm}-seed{seed}-step{budget}.npz` and a
`metrics.json` carrying every per-budget validation report and status label.

## Limits

CPU-scale budgets and small (32-px, 2-ball) worlds. Validation foreground and
localization come from palette/visibility truth read only after each forecast.
The status labels are heuristic thresholds on collapse diagnostics and
foreground recall, not a significance test. No result here is promoted to the
README; the checkpoints exist so the same trajectory can be re-scored at larger
scale. See also [zero-effect diagnosis](ZERO_EFFECT_DIAGNOSIS.md) for the
collapse/dead/insensitive verdict machinery and
[event benchmark](RGB_EVENT_BENCHMARK.md) for the localization metrics reused
here.
