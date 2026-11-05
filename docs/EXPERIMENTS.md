# Experiment map

**Start with a question, open its model, run its experiment, then inspect the
output.** Each line below is a separate technique or control. Historical
experiments and specialized inputs are intentionally retained.

## Pixel-based predictive models

| Question | Model / runner | Input and output |
|---|---|---|
| Can a recurrent sheet forecast changes? | `src/model.py`, `train.py` | Synthetic grayscale; early minimal baseline |
| Does a residual video model anticipate a webcam? | `src/video_residual.py`, `video_train.py` | Recorded grayscale video, held-out forecast |
| Can local cells predict RGB motion? | `src/predictive_rgb_nca.py`, `experiments/rgb_motion_distribution.py` | 32×32 synthetic RGB, +1 to +10 frames; [protocol](RGB_MOTION_DISTRIBUTION.md) |
| Does training on diverse motion help? | `experiments/rgb_curriculum.py`, `experiments/rgb_motion_distribution.py` | Fixed synthetic train/validation/test scenes; compare to persistence and velocity |
| Does emphasizing the distant future matter? | `experiments/rgb_horizon_objective.py` | Same RGB NCA; alternative loss targets |
| Do cell neighborhoods or remote references matter? | `src/selectable_rgb_nca.py`, `experiments/rgb_selector_benchmark.py` | Local, relative, fixed-reference arms; [protocol](RGB_SELECTOR_BENCHMARK.md) |
| Can cells learn which offset to read? | `src/rgb_soft_references.py`, `experiments/rgb_soft_references.py` | Soft learned spatial offset routing; [protocol](RGB_SOFT_REFERENCES.md) |
| Does local recovery stabilize and improve a cellular predictor? | `src/rgb_homeostatic_nca.py`, `experiments/rgb_homeostatic_recovery.py` | Plain vs local vs shuffled nontrainable activity regulation; [protocol](RGB_HOMEOSTATIC_RECOVERY.md) |
| Do fast and slow sheets help? | `src/rgb_multiscale_sheets.py`, `experiments/rgb_multiscale_sheets.py` | Dual-rate recurrent state; [protocol](RGB_MULTISCALE_SHEETS.md) |
| Can phase coupling/entrainment contribute? | `src/rgb_phase_nca.py`, `src/rgb_phase_entrainment.py`; corresponding `experiments/` runners | Independent oscillatory controls; [protocol](RGB_PHASE_ENTRAINMENT.md) |
| Are spatial harmonics causally useful for forecasting? | `src/rgb_mode_interventions.py`, `experiments/rgb_mode_interventions.py` | Frozen low-mode suppression vs per-channel energy-matched random subspace; [protocol](RGB_MODE_INTERVENTIONS.md) |
| Do recurrent states prefer particular lattice harmonics? | `src/rgb_harmonics.py`, `experiments/rgb_harmonic_microscope.py` | Frozen trained/untrained/shuffled hidden-state spectra; [protocol](RGB_HARMONIC_MICROSCOPE.md) |
| Does prediction depend on hidden memory? | `src/rgb_hidden_interventions.py`, `experiments/rgb_hidden_interventions.py` | Perturb a frozen checkpoint, then forecast; [protocol](RGB_HIDDEN_INTERVENTIONS.md) |
| Can a minimal circuit-style cell forecast RGB? | `src/rgb_circuit_cell.py`, `experiments/circuit_rgb_comparison.py` | Independent fixed two-state primitive, one-step ridge head, held-out open-loop RGB; [protocol](CIRCUIT_RGB_COMPARISON.md) |
| Can transient local error traces adapt predictions? | `src/rgb_error_trace.py`, `experiments/rgb_error_trace.py` | Frozen weights; state-only adaptation; [protocol](RGB_ERROR_TRACE.md) |

## Structured representations (not image recognition)

| Question | Model / runner | What is supplied? |
|---|---|---|
| Do graph messages help with articulated motion? | `src/graph_skeleton_nca.py`, `experiments/graph_skeleton_horizons.py` | **Given 2D joints and known skeleton edges**, not detected RGB poses; [protocol](GRAPH_SKELETON_HORIZONS.md) |
| Can moving cells interact predictively? | `src/particle_nca.py`, `experiments/particle_nca_motion.py` | **Given tracked 2D positions and identities**, not extracted objects; [protocol](PARTICLE_NCA_MOTION.md) |
| Does choosing where to look change predictions? | `src/active_gaze_dynamics.py`, `experiments/active_gaze_dynamics.py` | Controlled 1D moving target, known belief dynamics; **not a trained neural gaze policy**; [protocol](ACTIVE_GAZE_DYNAMICS.md) |

## Observe and evaluate

| Goal | Command / guide |
|---|---|
| Inspect model forecasts, future truth, and activity | `python -m experiments.rgb_prediction_gallery --help`; [gallery](RGB_PREDICTION_GALLERY.md) |
| Score a frozen RGB NCA on new synthetic seeds | `python -m experiments.rgb_benchmark --help`; [benchmark contract](RGB_BENCHMARK_CONTRACT.md) |
| Localize objects (hard/segmentation/soft) and see why a model misses them | `python -m experiments.rgb_event_benchmark --help`, `python -m experiments.rgb_event_gallery --help`; [event benchmark](RGB_EVENT_BENCHMARK.md) |
| Capture RGB webcam frames; fit a video-specific NCA | `python -m experiments.rgb_nca --help`; [guide](GETTING_STARTED.md) |
| Compare committed forecasts with later webcam frames | `python -m experiments.rgb_verified_live --help`; [verified live](RGB_VERIFIED_LIVE.md) |
| Compare independently recorded webcam footage | `python -m experiments.rgb_transfer_eval --help`; [protocol](RGB_WEBCAM_TRANSFER.md) |
| Report experiments without invalid cross-representation rankings | `python -m experiments.evidence_report --help`; [evidence guide](CROSS_TECHNIQUE_EVIDENCE.md) |
| Tell a genuine null from a collapsed/dead forecast measurement | every runner's `metrics.json` `measurement` block; [zero-effect diagnosis](ZERO_EFFECT_DIAGNOSIS.md) |
| Tell undertraining from collapse across step budgets, and which anti-collapse lever works | `python -m experiments.training_budget_sweep --help`; [budget sweep](TRAINING_BUDGET_SWEEP.md) |
| Review the first diverse-world and RGB-inferred Gaussian campaign results | [findings and evidence gaps](CAMPAIGN_FINDINGS_20261104.md) |

**Status and evidence:** Source code and a runnable CLI mean *implemented*, not
*experimentally proven*. The synthetic RGB NCA has beaten persistence on one
three-seed distribution sweep but not constant velocity. Other architectural
variants need their own trained runs and evaluation before claiming benefits.
For graph and particle tasks, compare within their own units and inputs.
Independent webcam transfer and robust one-second anticipation remain open.

## Current research evidence and future milestones

The Gaussian-state NCA [roadmap](RESEARCH_TIMELINE.md) now has **implemented** renderer, oracle-state dynamics, RGB-inferred dynamics, and active-view comparison runners. Their presence does not establish learned 3D scene understanding or superiority to simpler predictors. Gaussian splatting is a representation/rendering technique, not a predictor by itself.

The [November 2026 campaign findings](CAMPAIGN_FINDINGS_20261104.md) record reported matched perceptual and RGB-inferred Gaussian results, including a diffuse-output NCA failure and the absence of scored reappearance events. Those metrics have not yet been archived as inspectable raw artifacts in this repository; no model is promoted on the summary alone.

## Where to read next

- [Training and demos](GETTING_STARTED.md) — minimal CPU run, visualization,
  full comparison, and optional live webcam.
- [Original charter](PLAN.md) — historical research hypotheses and design
  rationale, not current status.
- [GPU notes](TRAINING.md) — optional infrastructure and costs; skip for the
  local examples.

**Rule for adding a technique:** put its independent question, runnable
entry point, representation, strongest control, and evidence status here.
Keep the experiment's detailed protocol beside it in a short document;
do not make the README into a chronological change log.
