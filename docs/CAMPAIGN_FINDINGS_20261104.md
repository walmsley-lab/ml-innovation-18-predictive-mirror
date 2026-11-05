# First diverse-world / RGB-inferred Gaussian campaigns — findings

**Status:** preliminary, reported 2026-11-04. These are results from local
campaign runs as relayed by the experiment operator, **not** independently
replicated repository CI results. The local raw manifests/checkpoints have
**not** been committed or independently inspected for this note. No new model
is promoted on these findings.

## Questions and protocol

Two recently merged runners were exercised:

- `experiments.rgb_world_ladder_training` — six RGB families
  (`cnn`, `transport`, `hierarchical`, `nca`, `nca_blend`, `routing`)
  × two curricula (linear / mixed) × three seeds, **150 update steps**.
  Held-out tests cover seven domains: linear, accelerating, collision,
  long occlusion, camera pan, appearance and combined shifts. All these
  arms are **RGB-only, action-blind**. Reports include MSE, foreground
  metrics, color-sensitive soft object mass/localization, and per-arm
  parameter counts and training times.
- `experiments.rgb_gaussian_nca` — three seeded RGB-inferred Gaussian
  predictors, **80 update steps**, held-out `ood_camera`, with shared
  inferred slots and three dynamics modes: static, velocity and cellular.
  Separate image persistence and pixel-velocity baselines are also scored.
  This is *not* oracle-state 3D prediction; the model sees past RGB, camera
  intrinsics and planned actions, not simulator object geometry.

The analysis also reported an inference-timing probe. Hardware, batching,
warm-up/synchronization details, sample counts and variance are **not yet
included in the provided report**, so timing ratios are descriptive only.

## Reported cost (one run configuration)

| model | parameters | training wall time | inference per rollout |
|---|---:|---:|---:|
| CNN | 1,035 | 3.9 s | 7.0 ms |
| Transport | 540 | 4.2 s | 11.5 ms |
| Hierarchical | 2,288 | 9.7 s | 16.2 ms |
| Plain NCA | 2,315 | 39.3 s | 34.4 ms |
| Blend-head NCA | 2,324 | 39.1 s | 35.4 ms |
| Learned routing | 617 | 28.3 s | 73.0 ms |

The tested structured/recurrent alternatives incurred roughly 2–10× the
CNN's inference latency, without an established object-prediction win.
**Equal training steps are not equal FLOPs, and these timings are not
portable hardware benchmarks.** Preserve the exact host/JAX/runtime and
timing method in the next evidence export.

## Held-out viewpoint results: Gaussian predictor

Lower is better. All values below were reported as aggregate results; no
per-seed intervals or artifact hashes were supplied with this summary.

| predictor | image MSE | visible-geometry-region MSE |
|---|---:|---:|
| Image persistence | **0.0113** | **0.1051** |
| Gaussian, static slots | 0.0177 | 0.1821 |
| Gaussian, velocity slots | 0.0201 | 0.2048 |
| Gaussian, cellular dynamics | 0.0200 | 0.2040 |
| Image constant velocity | 0.0209 | 0.1868 |

Image persistence was better than any tested Gaussian mode on both
reported errors. Within the shared-slot Gaussian representation, static
outperformed velocity and cellular modes. One seed reportedly selected
training step zero, so learning was not reliably improving validation
performance at this budget. This result **does not isolate** a defect in
Gaussian representation versus scene-state inference, visual rendering,
or dynamics. Test these components separately before scaling training.

## Perceptual results and two different collapse modes

- **Held-out camera pan:** real RGB-only learned arms had approximately
  zero brightness-threshold foreground recall. Their future camera actions
  were *unknown*, so this is not a fair test of action conditioning.
- **Unseen motion / long occlusion:** all real arms missed many objects.
  Reported average hard misses ranged approximately 2.5–7.3 out of 7.3
  visible objects (depending on family) and location error was about
  22–45 px in a 32-px world. The routing and hierarchical arms had modest
  metric-dependent localization differences, but no robust performance
  or cost advantage has been established.
- **Original NCA:** prior experiments established a lower clip-rail,
  input-independent **black** open-loop forecast fixed point.
- **Blend-head NCA:** apparent `foreground_recall=1.000` and low
  foreground MSE were **not** object localization. The operator found an
  almost uniform-brightness prediction with color-specific
  `predicted_mass_fraction=0.0`, precision ≈0.011,
  `missing_visible_predictions=4/4`, and little/no scene dependence.
  This is a different **degenerate uniform-output solution**, not a
  successful anti-collapse perceptual model. Brightness-based recall and
  hue-specific mass answer different questions and must be interpreted
  together.

The earlier blend treatment prevented *black rail saturation* in the
budget sweep; this new result shows that avoiding one bad fixed point
is **insufficient** to prevent other constant-output failures.

## Reappearance: unmeasured, not failed

The reported dedicated re-entry metric found **zero scored events** in
all tested domains. The current `rgb_event_metrics.score` marks re-entry
only when an object is visible at a sampled forecast lead and invisible
at the **immediately preceding** frame. By default the sweep scores
leads 1, 3, 5 and 10. Consequently, a trajectory may contain genuine
reappearance yet receive no re-entry score when that transition is
between scored leads. The world generator's nominal `occlusion_long`
domain does not itself certify a fully occluded and reappearing object
within the evaluated interval.

No conclusion about long-horizon **object permanence** follows from this
run. Occlusion-domain foreground recall is *not* a substitute for
measured re-entry.

## Interpretation

**Supported within these small runs:** no convincing advantage of
cellular/persistent Gaussian models over the simplest predictors,
inconsistent curriculum benefit, substantial preliminary compute
cost for recurrent/routing arms, and a new diffuse constant-output
degeneracy in the blend NCA.

**Not supported:** that CNN or persistence already *solves* object
permanence; that NCAs cannot learn motion at larger budgets; that
Gaussian-state dynamics are inherently unhelpful; or that the
`occlusion_long` campaign tested successful or unsuccessful
reappearance. Three seeds, 150/80 updates, 32/24-pixel images and
very small synthetic worlds strongly limit inference.

## Evidence and next gates

1. **Archive** sanitized `metrics.json`, run commands/configurations,
   commit SHA, JAX/host versions, timing procedure, seeds, per-seed
   distributions, benchmark episodes, checkpoint hashes and selected
   steps. The operator reported local artifacts under
   `scratchpad/campaign/{perceptual,gaussian}/`; **these paths were
   not inspected or imported into this PR**. Do not imply that the
   metrics or model weights are currently reproducible solely from
   this prose.
2. **Fix event coverage before more training:** build deterministic
   visible → fully barrier-occluded → reappeared trajectories, verify
   that the evaluation window contains each transition, and score
   every frame/event. Separate barrier occlusion from camera exit.
   Include perfect/missing/shifted/dim uniform-output controls.
3. **Add collapse checks:** spatial variance, across-scene variance,
   saturation, hard foreground precision/recall, hue-specific mass,
   soft localization, and visual galleries. A single recall or MSE
   number must never promote a constant output.
4. **Disentangle Gaussian failures:** present-time reconstruction,
   slot assignment stability, approximate depth/view consistency,
   oracle-versus-RGB-inferred state, and matched static/velocity/
   cellular head comparisons. Evaluate action sensitivity under
   paired known future camera plans.
5. **Only then scale** matched training budgets and report per-seed
   uncertainty alongside real inference time/compute; keep all
   architecture variants and failed runs.

## Reproduction entrypoints

```bash
python -m experiments.rgb_world_ladder_training \
  --models cnn transport nca routing hierarchical nca_blend \
  --curricula linear mixed --seeds 11 23 41 --steps 150

python -m experiments.rgb_gaussian_nca \
  --seeds 11 23 41 --steps 80
```

These CLI recipes target the same stated runner families/budgets;
**they are not verified byte-for-byte launch manifests** of the
operator's completed campaigns. Both runners save results under
`runs/` by default; original artifacts must be recovered from
the operator for independent replication. See
[perceptual protocol](RGB_WORLD_LADDER_TRAINING.md),
[Gaussian predictor](RGB_GAUSSIAN_NCA.md),
[prior budget sweep](TRAINING_BUDGET_SWEEP.md), and
[object evaluator](RGB_EVENT_BENCHMARK.md).
