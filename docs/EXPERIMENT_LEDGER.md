# Experiment ledger

What each experiment **is designed to establish** and **why it matters** to the
broader investigation of emergence, structure–dynamics–function mapping,
perception, adaptation, robustness, communication and substrate efficiency.

This is an **experiment-intent and evidence-scope catalog**, not a claim that
every hypothesis was experimentally demonstrated. [`EXPERIMENTS.md`](EXPERIMENTS.md)
is the smaller *what/how* entry map (question → model → input). This ledger
catalogs all **147 Python runners** under `experiments/` at this revision;
`field_perturbation_probe.py` appears under two roles but is counted once.
Implementation, smoke-test success, held-out results and independent biological
measurements are distinct levels of evidence. Consult each runner's actual
metrics, checkpoint provenance and protocol before making scientific claims.

**Filename notation:** arrows such as `roadmap_35_rgb_gates.py` →
**`rgb_promotion_gate.py`** are **proposals only**; no files are renamed by
this PR. Existing imports and documented commands remain authoritative.

## How to read "kind"

Every runner plays one of these roles. The umbrella word "experiment" blurs
them, so each entry is tagged:

- **world** — constructs a synthetic environment and its evaluator-only
  ground-truth / validity controls. Produces data, scores no model.
- **training-study** — trains one or more models and compares to baselines.
- **probe** — intervenes on a *frozen/trained* model (lesion, reset, perturb)
  to test what it causally depends on.
- **baseline** — a classical / null / privileged-oracle method that sets the
  bar a learned model must clear.
- **evaluation** — scores frozen models and observation-only baselines on
  held-out data with no intervention (go/no-go harnesses, galleries).
- **calibration** — calibrated positive controls, conformal intervals, proper
  scoring of uncertainty.
- **audit** — cost / provenance / evidence accounting and guardrails.
- **demo** — live or rendered visualization; produces no held-out metric.
- **stability** _(one case)_ — a rule-stability simulation, not a perception result.

> A directory split into `experiments/worlds/`, `experiments/probes/`, … is the
> natural follow-up once the incoming-PR stream quiets; it is deferred now
> because renaming `experiments/` would collide with PRs still landing there.

---

## worlds

| experiment | establishes | why it matters |
|---|---|---|
| `world3d_generate.py` | Generates 3D perspective RGB episodes + evaluator-only truth (depth, instance IDs, poses, centers) via analytic ray-sphere or MuJoCo. | Supplies the controlled 3D environment with withheld truth the anticipation experiments depend on. |
| `layered_advection_benchmark.py` → **`layered_advection_validate.py`** | Generates layered-advection episodes and validates ground-truth flow, instance IDs, depth ordering, visibility. | A validated multi-layer occlusion world with evaluator-only truth ("benchmark" misnames it — it scores no model). |
| `camera_robustness.py` | Builds synthetic sensor corruptions (exposure/WB/rolling-shutter/drops) with evaluator-only capture truth; stratified persistence error. | Characterizes how sensor artifacts distort the benchmark before any model claim. |
| `rgb_world_ladder.py` | Freezes a ladder of perceptual-difficulty RGB worlds; per-domain visibility/collision stats + baseline event-metrics. | Defines graded observation-only benchmarks (with validity controls) before model selection. |
| `rgb_ood_world.py` | Deterministic family of harder OOD RGB worlds; manifest marks model-visible vs evaluator-only fields. | OOD stress environments with explicit input-privilege controls for transfer tests. |
| `physical_field_worlds.py` | Verifies finite-difference diffusion/wave simulators; exposes state shapes/parameters/initial mass. | Validated physical-dynamics environments for later predictive comparisons. |

## training-studies

| experiment | establishes | why it matters |
|---|---|---|
| `rgb_nca.py` | Trains a predictive RGB NCA on real webcam frames; multi-lead forecast vs persistence/pixel-velocity on chronological holdout. | Direct test of whether a cellular predictor beats trivial baselines at real camera motion. |
| `horizon_phase.py` | Plain vs uncoupled-phase vs coupled-phase recurrent video models; multi-horizon MSE on grayscale webcam holdout. | Does persistent phase-coupled recurrence help longer-horizon anticipation vs plain recurrence. |
| `roadmap_35_rgb_gates.py` → **`rgb_promotion_gate.py`** | Validation-gated curriculum promotion vs fixed vs mixed sampling of one RGB NCA across motion/occlusion/camera proxies. | Do competence-gated curricula help a cellular predictor generalize across harder regimes. |
| `rgb_homeostatic_recovery.py` | Matched-seed ablation of local activity recovery vs shuffled vs plain RGB NCA, + frozen positive-control gain sweep. | Does a homeostatic regulariser improve prediction, and is the null genuine not a dead wire. |
| `rgb_motion_nca.py` | RGB NCA on rendered ball trajectories; overfit vs held-out generalization vs persistence/constant-velocity. | Does the cellular model learn transferable motion rather than memorizing. |
| `roadmap_33_rgb_replay.py` → **`rgb_replay_rehearsal.py`** | Bounded reservoir replay vs recent-only (update-matched) on A→B→A; forgetting via prequential + held-out MSE. | Does cheap replay preserve earlier perceptual competence under task shift. |
| `roadmap_37_oscillatory.py` → **`rgb_oscillatory_motion.py`** | RGB phase-oscillator NCA vs matched plain NCA on periodic/chirp/nonperiodic domains vs classical controls. | Does oscillatory coupling buy anticipation on (ir)regular motion beyond a plain cellular baseline. |
| `action_world3d_nca.py` | Action-aware vs action-blind RGB NCAs on 3D worlds, OOD, + counterfactual-action positive control + action-dependence ceiling. | Does conditioning on ego-motion improve 3D anticipation, bounding the learnable effect. |
| `rgb_horizon_objective.py` | Uniform vs late-weighted horizon loss on the RGB NCA, matched weights/data, same +10 validation selection. | Does emphasizing the far horizon improve one-second anticipation without changing the model. |
| `rgb_phase_entrainment.py` | Four-arm phase-entrainment RGB NCA; held-out MSE + global phase order + wall-time. | Does sensory-error phase entrainment aid prediction (phase order reported as non-evidence of binding). |
| `cellular_references.py` | Co-trains sender/router/receiver content-addressed pointers; distance generalization (train 2-4, test 5-7) vs shuffled/uniform. | Can cells learn content-based long-range references — a binding mechanism for distant predictors. |
| `synthetic_to_video.py` → **`synthetic_to_real_transfer.py`** | Synthetic-pretrained vs scratch grayscale video model fine-tuned on identical webcam batches; transfer gain vs persistence. | Does synthetic-world pretraining transfer to real perception (extra compute made explicit). |
| `rgb_phase_coupling_ablation.py` | Plain vs uncoupled vs coupled (Kuramoto-like) phase RGB NCA on fixed synthetic splits, step-0 eligible. | Does local phase coupling add anticipation value at zero extra trainable parameters. |
| `shared_core_transfer.py` | Pretrains an exactly-shared cellular conv on 1D routing, fine-tunes on RGB vs scratch and weight-shifted controls. | Does a communication-learned cellular core transfer to visual prediction (shared-substrate claim). |
| `physical_field_prediction.py` | Learned stencil/CNN/shared-cellular field predictors vs a numerical reference and fitted linear stencil on diffusion/wave. | Do learned cellular dynamics match classical PDE solvers at field forecasting. |
| `field_multiscale_comparison.py` | Matched-init fine/extra-ticks/dilated/encoder-decoder/bidirectional field architectures + frozen direction ablations. | Which multiscale cellular structure best predicts fields; are top-down/bottom-up flows load-bearing. |
| `rgb_motion_distribution.py` | RGB NCA trained on one trajectory vs a scene distribution; unseen test scenes vs persistence/centroid-velocity. | Does diverse-world training let the cellular predictor beat baselines at motion anticipation. |
| `rgb_curriculum.py` | Paired single-scene vs progressive-curriculum RGB NCA, matched init/budget, all-scenario test vs velocity/persistence. | Isolates the curriculum effect (not architecture) on one-second forecasting. |
| `roadmap_32_rgb_adaptation.py` → **`prequential_rgb_adaptation.py`** | Prequential online-adapted vs frozen RGB NCA on A/B/A shifts, each frame forecast before updating. | Does persistent online adaptation beat a frozen model when dynamics change. |
| `cellular_transmission.py` | Shared 3x3 cellular rule with/without local messages delivering/retaining a signed pulse; distance generalization. | Is local inter-cell communication necessary to transmit and persist signals. |
| `gaussian_oracle_dynamics.py` | Local cellular Gaussian dynamics on privileged observed centers/velocities vs static/velocity; 3D position + rendered RGB. | Upper bound: do learned cellular dynamics beat trivial extrapolation given clean state. |
| `rgb_gaussian_nca.py` | RGB-inferred Gaussian-state NCA (no oracle poses); cellular/static/velocity on OOD-camera test vs RGB baselines. | Do cellular dynamics over an RGB-inferred object representation beat pixel baselines. |
| `phase_ablation.py` → **`rgb_phase_coupling_webcam.py`** _(name clashes with the synthetic ablation — disambiguate)_ | Equal-parameter webcam training of plain vs uncoupled vs coupled phase oscillators (K=0 vs K>0) on temporal holdout. | Does oscillatory phase coupling add value over an equally-sized ordinary recurrent video model. |
| `nca_selector_ablation.py` → **`rgb_nca_selector_ablation.py`** | Three RGB NCA selector/routing modes (incl. fixed mirror-pixel references) on webcam holdout; MSE vs persistence. | Do learned vs fixed cell-routing rules change forecasting. |
| `rgb_motion_transport.py` | Residual-only vs transport-only vs transport+correction RGB modes vs velocity/persistence, matched budget. | Does an explicit motion-transport inductive bias help vs plain residual learning. |
| `roadmap_33_replay.py` → **`continual_replay.py`** | Compute-matched reservoir replay vs recent-only across A/B/A; forgetting deltas. | Does persistent memory (replay) reduce catastrophic forgetting vs recent-only. |
| `roadmap_32_adaptation.py` → **`prequential_adaptation.py`** | Prequential online regression through A/B/A: adaptive vs frozen vs stale-feedback, validation-tuned rate. | Does prediction-first continual adaptation beat a frozen model under shift. |
| `development_23_curriculum.py` → **`staged_vs_mixed_curriculum.py`** | Tiny RGB NCA staged vs mixed, identical multiset/replay/budget; held-out MSE. | Isolates curriculum ordering holding data/compute constant. |
| `roadmap_36_latent_dynamics.py` → **`latent_linear_dynamics.py`** | Train-only PCA + stable ridge latent transition (validation-tuned); multi-domain forecasts vs classical controls. | Does compact learned linear latent dynamics beat constant-velocity/persistence. |
| `structured_pretraining_transfer.py` | cnn/nca pretrained on various synthetic sources vs scratch at equal updates; held-out OOD scores. | Does structured pretraining transfer to better anticipation than scratch. |
| `development_28_action.py` → **`action_conditioned_motion.py`** | Linear action-conditioned motion head vs action-blind/shuffled-action/coasting on camera-pan worlds. | Does conditioning on known camera actions improve motion anticipation. |
| `roadmap_41_transfer.py` → **`cellular_core_transfer.py`** | Transfers exact shared cellular-core from synthetic pulse to RGB vs shuffled-label/scratch-equal/scratch-finetune across OOD. | Transfer specificity of a cellular representation, separating update vs exposure parity. |
| `object_permanence_campaign.py` | Six model families on paired-history occlusion worlds; held-out reappearance success with paired bootstrap CIs vs persistence/linear-history and reset/shuffle controls. | Directly tests whether learned predictors beat simple baselines at object reappearance (core claim). |
| `graph_skeleton_horizons.py` | 10-step open-loop graph NCA with/without message passing vs last-seen/constant-velocity on occluded 2D joints. | Does cellular message passing improve multi-horizon anticipation of hidden/reappearing joints. |
| `roadmap_35_gates.py` → **`capability_gate_curriculum.py`** | Small observed-state regressor under gated/fixed/mixed curricula with validation-only promotion gates. | Do validation-gated developmental curricula beat fixed/random schedules. |
| `training_budget_sweep.py` | 300/600/1200-step budgets across 8 families (incl. 3 anti-collapse NCA levers); undertrained/collapsed/learning labels. | Does cellular NCA genuinely learn (vs collapse) relative to CNN/transport given more compute. |
| `rgb_world_ladder_training.py` | 5 families on linear-only vs mixed curricula over disjoint domains; per-domain incl. OOD test metrics. | Does curriculum breadth help cellular prediction generalize beyond baselines. |
| `rgb_developmental_curriculum.py` | Presentation order (staged vs shuffled) on identical NCA multiset/budget; frozen test scores per stage. | Does developmental ordering alone aid cellular perception vs shuffled control. |
| `frozen_rgb_campaign.py` | Paired NCA/CNN/transport on identical worlds, frozen eval, per-lead episode-bootstrap CIs vs constant-velocity. | Does cellular prediction beat CNN/transport and velocity, with uncertainty. |
| `roadmap_39_particle_vision.py` | One particle NCA on RGB-palette centroids vs privileged simulator positions (×message on/off) vs motion controls. | Does cellular particle prediction work from observed pixels, not just privileged state. |
| `rgb_selector_benchmark.py` | NCA neighborhood-selection modes (local/relative/reference) at matched +10 budget, step-0 eligible, vs controls. | Does structured cellular message selection improve prediction over local-only. |
| `roadmap_38_graph_topology.py` | Graph-skeleton NCA with true vs degree-matched-rewired vs no-message edges vs last-seen/velocity. | Does correct communication topology (not just messaging) help cellular joint prediction. |
| `rgb_multiscale_sheets.py` | Single fast sheet vs fast+slow-feedback sheet at matched params/budget, step-0 eligible, vs controls. | Does multi-timescale cellular state improve one-second RGB anticipation. |
| `recurrent_transport_nca.py` | Recurrent NCA vs transport-only vs recurrent+transport-correction on occluded 2-ball worlds vs controls. | Does combining persistent cellular state with explicit transport beat either alone. |
| `particle_nca_motion.py` | Message-passing vs isolated particle-NCA on oracle 2D positions; held-out error vs persistence/constant-velocity. | Does cellular inter-cell messaging improve motion anticipation in a structured world. |
| `graph_skeleton_nca.py` | Graph-NCA on oracle 2D skeletons with/without messages; visible/hidden/reappearance joint error vs last-seen. | Does cellular message passing yield object-permanence-like recovery of occluded joints. |
| `transmission_rgb_transfer.py` | Scratch vs cellular-transmission vs spatially-shifted-transfer RGB inits, fixed budget; +positive-control +collapse check. | Do pretrained cellular-communication weights transfer to improve RGB forecasting. |
| `cellular_message_modulation.py` | Receiver-side message-gating vs plain on a two-pulse relevant-signal task; MSE + sign accuracy. | Does cellular message modulation select the relevant signal over an ungated baseline. |
| `rgb_soft_references.py` | Learned soft-offset reference routing vs fixed/broken-correspondence; test MSE + routing entropy, step-0 eligible. | Does learned cellular reference routing beat fixed/null correspondence. |
| `rgb_hierarchical_surprise.py` | Flat recurrence vs fast/slow sheets vs latent/surprise auxiliary losses on occluded RGB; held-out MSE vs baselines. | Which hierarchical/predictive-coding components improve anticipation over simpler recurrence. |
| `rgb_motion_nca_sweep.py` | Multi-seed robustness sweep over the Step-5 RGB motion NCA; aggregate overfit/generalization + beats-baseline counts. | Does any NCA advantage replicate across seeds, not one lucky run. |
| `roadmap_40_irregular_time.py` → **`timed_motion_ridge.py`** | Time-normalized vs time-blind ridge motion vs scheduled/per-frame-velocity/persistence across jitter/dropped/accel cadences. | Does knowing the scheduled inter-frame interval improve motion anticipation under irregular timing. |

## probes

| experiment | establishes | why it matters |
|---|---|---|
| `reconciliation.py` → **`multisheet_reconciliation_probe.py`** | Linear ridge decoders on frozen fast/slow sheet activity test joint/compatible encoding of occluded position, + feedback-cut control. | Do persistent multi-sheet representations hold a coherent hidden-state estimate through occlusion. |
| `world3d_action_pairs.py` | Whether a frozen forecaster's prediction changes under paired counterfactual (positive vs negated) camera-action plans. | Is a trained model actually sensitive to ego-motion (prerequisite for 3D anticipation). |
| `field_perturbation_probe.py` _(calibration-flavored)_ | Calibrates physical impulse-response controls to a numerical reference; optionally probes a frozen field checkpoint. | Measures model perturbation-response claims against known physics. |
| `paired_occlusion_histories.py` | Frozen model (or persistence) on bit-identical-last-frame paired histories + reset/shuffle; future-contrast MSE + reappearance. | Does a model use motion history (not just the last frame) to anticipate the correct reappearance side. |
| `rgb_harmonic_microscope.py` | Frozen RGB NCA observed/dreamed fields → lattice harmonic spectra vs random-init and site-shuffle; forecast MSE. | What spatial structure a trained cellular predictor captures. |
| `world3d_causal.py` | Frozen state-lesion and action-disruption interventions across OOD 3D domains; paired-CI contrasts vs intact. | Do predictions causally depend on persistent state and actions, not surface texture. |
| `cellular_routing_probe.py` | Frozen routing lesions (learned/uniform/shuffled/key-knockout) on a destination-query task; routing attention metrics. | Does a trained cellular model route information to the correct source (binding mechanism). |
| `guaranteed_reappearance.py` | Scores a frozen model (default oracle-truth positive control) on guaranteed barrier/camera-exit reappearance. | A clean pass/fail anticipation benchmark where a correct predictor must forecast a guaranteed return. |
| `persistent_gaussian_observer.py` | Intervenes on a frozen Gaussian-observer (context-length, resets, shuffled history); occluded 3D center drift vs oracle. | Does a persistent memory filter aid object permanence under occlusion vs reset controls. |
| `rgb_mode_interventions.py` | Causally suppresses low harmonic modes in a frozen NCA vs energy-matched random null + memory-wipe control + collapse check. | Do specific internal modes causally carry the forecast (real vs dead nulls). |
| `frozen_state_descriptions.py` | Sparse (SINDy-like) descriptors fit to a frozen NCA's hidden trajectories; tested under held-out and perturbed states. | Do internal dynamics admit a compact intervenable mechanistic description. |
| `rgb_hidden_interventions.py` | Hidden-state reset/spatial-permutation at a rollout point on a frozen RGB NCA; forecast change vs intact + held-out. | Does the cellular hidden state causally carry predictive memory. |
| `rgb_error_trace.py` | Frozen RGB NCA with a within-episode decaying local error trace (several modes); forecast difference vs frozen rollout. | Does a short-lived local error signal (not weight plasticity) improve anticipation. |
| `rgb_occlusion_probe.py` | Frozen NCA across clear/occluded/two-ball scenarios vs controls; post-occlusion reappearance counts. | Does a trained cellular model sustain predictions through occlusion (object-permanence proxy). |
| `gaussian_reconstruction_diagnostics.py` | Decomposes frozen last-frame reconstruction error into renderer/primitive gap (oracle) vs encoder error (inferred). | Is a trained RGB-only Gaussian observer's error perception or just the shared renderer. |
| `gaussian_correspondence.py` | Permutation-aware slot-to-object correspondence stability across train/OOD-camera/partial-occlusion on a frozen checkpoint. | Does a trained cellular Gaussian observer maintain object identity without oracle correspondence. |
| `reconciliation_controls.py` | Linear probes then intact vs feedback-cut/mean/within-sheet-lesion/single-split/trained-without-feedback controls under occlusion. | Is cross-sheet feedback (not raw capacity) what keeps compatible occluded-position estimates. |
| `active_field.py` → **`active_plasticity.py`** | Frozen plastic-field model under fixed/random/model gaze × fixed/plastic adaptation; MSE, visible fraction, gain. | Do active sensing + within-episode plasticity reduce error vs passive/fixed controls. |
| `gaussian_active_view.py` | Paired camera policies (active vs fixed/random) on frozen RGB-Gaussian ensembles across OOD depth/camera. | Does active viewpoint selection improve the persistent Gaussian observer under shift. |
| `active_world3d.py` → **`active_world3d_eval.py`** _(name clashes with `src.active_world3d`)_ | Active vs fixed/random gaze on identical hidden 3D worlds for a frozen action-RGB-NCA ensemble. | Does active sensing improve 3D prediction over passive controls on matched environments. |

## baselines

| experiment | establishes | why it matters |
|---|---|---|
| `classical_transport_controls.py` | Causal classical forecasts (persistence, pixel-velocity, flow-gather, coast, reid) on occluded episodes; joint failure overlap. | Sets the simpler-baseline bar and failure correlations the cellular predictor must beat. |
| `world3d_benchmark.py` | Frozen OOD 3D worlds scored for RGB persistence + privileged oracle depth-reprojection at multiple leads. | The frozen 3D benchmark and privileged/non-privileged floor for forecasting claims. |
| `gaussian_oracle_baselines.py` | RGB/geometry errors for privileged static/velocity/future-oracle Gaussian scenes. | The privileged-information ceiling against which RGB-only perception is judged. |
| `latent_forward_backward.py` → **`hmm_filter_smoother.py`** | Analytic 3-state HMM: causal filtering (full history) beats last-obs/shuffled-past/unconditional on next-obs NLL. | A clean oracle control proving history genuinely informs anticipation. |
| `development_30_feedback.py` → **`heuristic_feedback_control.py`** | Heuristic pixel-to-state feedback vs no-feedback/shuffled-context on an occlusion world, equal update budgets. | An interpretable non-neural reference for whether spatial feedback aids hidden-object tracking. |
| `interpretable_sindy.py` | SINDy vs affine vs DMD on privileged state trajectories vs a coast baseline across domains/noise. | Interpretable classical-dynamics baselines (privileged-state upper reference). |
| `roadmap_34_crossmodal.py` → **`sensor_fusion_baseline.py`** | Ridge vision/inertial/fused/robust predictors; fused beats single-modality and coast/zero, degrades gracefully under dropout. | A classical fusion floor the cellular cross-modal claims must clear. |
| `development_29_active_sampling.py` | Information-gain gaze lowers belief entropy/position-MAE vs fixed/random at equal observation count in a 1D POMDP. | A classical active-sensing floor for where-to-look claims. |
| `development_25_completion.py` | Linear self-supervised masked reconstruction beats visible-only on partial occlusion; fails on fully-hidden frames. | A null linear floor bounding single-frame completion before invoking cellular memory. |
| `development_27_egomotion.py` | Known-action or registration compensation recovers world-frame velocity; no-action/shuffled-action controls fail. | A classical observer-vs-object-motion floor for egomotion separation. |
| `development_26_depth.py` | Calibrated two-view block-matching disparity beats monocular-constant/shuffled-right on textured bands. | A classical stereo-depth floor distinguishing geometry from learned perception. |
| `development_22_adaptation.py` | Causal normalized-appearance + persistent-EMA encoder retains contrast under exposure swings where raw/temporal-diff fail. | A classical demonstration that persistent state buys exposure invariance. |
| `development_24_correspondence.py` _(no companion doc)_ | Hand-built connected-components tracker (coast vs nearest) keeps two identical blobs' identity through crossing/dropouts. | A classical object-permanence/identity floor learned predictors must beat. |
| `communication_graph_harmonics.py` | Harmonic spectra + causal interventions on the true skeleton graph vs display grid and degree-preserving rewire (diffusion ref). | Message-graph topology (not pixel layout) governs signal propagation. |
| `circuit_rgb_comparison.py` | 4-feature linear head per ablation of an idealized two-state circuit cell; RGB scores vs persistence/velocity + optional NCA. | An analog-circuit-primitive comparison point bounding a minimal physical cell. |
| `gaze.py` → **`bayesian_gaze.py`** | Exact Bayesian filter on a 1D world comparing fixed/random/optimized/greedy info-seeking gaze on posterior error. | An optimal-inference active-sensing floor for judging learned gaze policies. |
| `active_gaze_dynamics.py` → **`active_gaze_dynamics_eval.py`** _(clashes with `src.active_gaze_dynamics`)_ | Fixed/random/info-seeking gaze policies on a moving 1D target. | A floor for whether info-seeking gaze reduces tracking error vs naive policies. |

## evaluations

| experiment | establishes | why it matters |
|---|---|---|
| `rgb_nca_viewer.py` | Frozen RGB NCA forecasts on replay-holdout and live camera vs persistence, with a 5-column visual mirror. | Does trained anticipation hold on real/held-out frames. |
| `rgb_transfer_eval.py` | Frozen RGB NCA on independent real camera recordings, hash-excluding training files, true capture intervals. | The key external-validity test: does the predictor transfer to real cameras never trained on. |
| `developmental_transfer.py` | Saved staged vs shuffled-curriculum milestone snapshots on identical held-out tasks across stages/architectures. | Does ordered developmental training yield better capability trajectories than shuffled. |
| `compare_horizons.py` → **`synchronized_comparison.py`** | Synchronized live/replay forecasts of three horizon-phase arms + persistence, committed before each next frame. | Do persistent/phase-coupled predictors beat persistence under a no-future-leak protocol. |
| `rgb_event_gallery.py` | Per-episode/lead/object soft+hard localization numbers + optional PNG panels (observed/truth/model). | Makes event-localization scores interpretable (why a predictor wins/loses tracking through occlusion). |
| `video_horizons.py` | Autoregressive rollout of the real-video recurrent predictor over horizons on chronological holdout vs persistence/velocity. | Does the persistent recurrent predictor beat baselines on real webcam video. |
| `rgb_prediction_gallery.py` | Frozen RGB-NCA forecasts, baselines, truth, error, cell-activity panels per lead + per-lead metrics. | Visual+quantitative evidence for whether the cellular predictor anticipates better. |
| `rgb_event_benchmark.py` | Frozen RGB models + observation-only baselines on identical motion-world episodes via palette event/localization metrics. | Quantifies whether the cellular predictor beats baselines at object-level event anticipation. |
| `occlusion_holdout.py` | Recurrent/feedforward/multisheet nets vs persistence/velocity on held-out barrier-occlusion; MSE + reappearance error by wall width. | Directly tests the central object-permanence question. |
| `rgb_benchmark.py` → **`rgb_benchmark_eval.py`** _(clashes with `src.rgb_benchmark`)_ | A single frozen RGB-NCA checkpoint on held-out scenes vs persistence/velocity (+optional flow); wins at horizon. | The baseline go/no-go check for whether the cellular predictor beats simple baselines at all. |

## calibration

| experiment | establishes | why it matters |
|---|---|---|
| `roadmap_31_calibration.py` → **`conformal_calibration.py`** | Split-conformal intervals on a scalar position proxy; marginal/conditional coverage + interval score vs fixed-width. | Prediction can carry calibrated uncertainty simpler point baselines lack. |
| `roadmap_31_rgb_calibration.py` → **`conformal_rgb_calibration.py`** | Split-conformal per-lead max-channel RGB pixel intervals on frozen forecasts hit target coverage vs fixed-radius. | Adds calibrated uncertainty to already-issued forecasts. |
| `frozen_visual_bridge.py` | Frozen nca/cnn/transport episode-level conformal intervals + coverage across domains/corruptions and optional recordings. | Calibrated uncertainty-aware comparison of cellular vs baselines under shift. |
| `multifuture_calibration.py` | A two-branch predictor beats a deterministic coast floor on energy score + branch calibration (Brier) on ambiguous futures. | A calibrated multimodal baseline the cellular model must beat to claim it captures ambiguity. |
| `field_perturbation_probe.py` _(listed under probes)_ | Calibrated physical impulse-response positive controls. | Grounds any field perturbation-response claim in known physics. |

## audits

| experiment | establishes | why it matters |
|---|---|---|
| `evidence_report.py` | Aggregates metric JSONs into provenance-checked, protocol-matched groups with SHA256; refuses cross-protocol ranking. | Keeps the project honest: compares like-with-like, preventing false winner claims. |
| `comparable_compute.py` | Separated JIT-compile, steady-step, warmed-rollout wall times per family with distribution stats + host fingerprint. | Quantifies the compute-cost side of the persistent/cellular tradeoff on equal workloads. |
| `communication_reach_audit.py` | Per-pathway communication reach/operation budgets + tracemalloc host-memory proxy across unroll horizons. | The locality/cost envelope of cellular message passing (scalability case). |
| `roadmap_42_hardware.py` → **`hardware_cost_audit.py`** | Fixed circuit rule (full/quantized/local) on shared RGB; CPU latency + lower-bound physical state/ADC/DAC budget. | Scopes real hardware cost without overclaiming neuromorphic energy wins. |
| `roadmap_43_evidence.py` | Ingests roadmap metrics JSONs; only within-protocol/same-input paired bootstrap comparisons, refusing a global leaderboard. | Guards the central claim against invalid cross-task/privilege comparisons. |
| `campaign_archive.py` → **`campaign_evidence_archive.py`** | Tamper-evident SHA256-hash-linked archive of metrics/checkpoints + one regenerated held-out forecast per arm. | Defensible provenance so predictive-vs-baseline results can be audited not trusted. |
| `developmental_capabilities.py` → **`capability_charter_validator.py`** | Validates the capability-charter schema; rejects any capability whose model observations include future/oracle inputs. | Enforces the validity guardrails behind every capability claim. |

## demos

| experiment | establishes | why it matters |
|---|---|---|
| `horizon_phase_live.py` | Runs a trained multi-horizon phase/recurrent checkpoint live on a webcam; observed/persistence/forecast panes (no metrics). | Qualitative demonstration of the long-horizon predictive mirror. |
| `graph_skeleton_demo.py` | Renders an animated GIF of a frozen graph-skeleton NCA's observed/predicted/truth poses under occlusion. | A visual sanity check that skeleton-graph prediction tracks motion through occlusion. |

## stability

| experiment | establishes | why it matters |
|---|---|---|
| `plasticity.py` → **`plasticity_homeostasis.py`** | Homeostatic decay keeps recurrent weights bounded and avoids saturation where plain Hebbian diverges, vs a fixed-weight control. | Tests whether persistent local plasticity stays stable — a prerequisite for cellular memory, not itself a perception result. |

---

## Added since first catalog (roadmap 44–70, bridge, late additions)

These 28 runners landed after the initial pass; they complete coverage of all
147 experiments. Grouped by kind here; a later edit can fold them into the
tables above. (`roadmap 46–55` share `docs/ROADMAP_46_55_ATTENTION.md`;
`roadmap 62–70` share `docs/ROADMAP_62_70_BIOLOGY.md`.)

**training-studies**

| experiment | establishes | why it matters |
|---|---|---|
| `layered_output_comparison.py` | Layered output-head variants, validation-selected; held-out RGB/motion MSE + sharpness + a 2×2 oracle-flow/visibility substitution and persistence/velocity controls. | Do learned output heads beat classical persistence/velocity at pixel+motion anticipation. |
| `rgb_tiny_cnn_baseline.py` | Tiny CNN residual forecaster across seeds, validation-selected; lead-wise test MSE with episode-bootstrap CIs vs constant-velocity, optional frozen-NCA compare. | A learned non-cellular reference to confirm the NCA's edge is real, not only vs trivial controls. |
| `roadmap_64_graph_dynamics.py` | Held-out MSE for 3-coefficient graph-message dynamics fit on a synthetic graph teacher vs row-shuffled and no-message null graphs. | Identifiability test of whether supplied graph structure aids prediction in-distribution. |
| `roadmap_68_substrates.py` | Held-out MSE + edge-communication-cost proxies across connectome / grid / particle-proximity / uncoupled substrates, matched states and 3 coefficients each. | Does connectome topology beat generic substrates — cross-substrate efficiency framing. |

**probes**

| experiment | establishes | why it matters |
|---|---|---|
| `roadmap_53_interventions.py` | Accuracy-damage + causal output-delta from cutting the top-priority vs a matched-random attention window vs intact/none. | Causal importance of surprise-selected attention windows for forecast accuracy. |
| `roadmap_65_lesions.py` | Accuracy-damage + output-effect-L2 from node lesions vs degree-matched/random/sham on a predictor frozen on the intact graph. | Counterfactual model-dependency test (explicitly not biological) of which nodes matter. |
| `roadmap_67_harmonics.py` | Accuracy-damage + forecast-delta from removing low graph-harmonic modes vs energy-matched random vs sham on a frozen predictor. | Do low-frequency graph modes causally drive the predictor. |
| `visual_memory_bridge.py` | Prequential A/B/A forecast MSE for frozen / recent-only / replay output-bias adaptation on frozen NCA/CNN/transport + known-action control + validation stage gate. | Does cheap online bias adaptation + bounded rehearsal on frozen forecasters improve anticipation under shift. |

**baselines**

| experiment | establishes | why it matters |
|---|---|---|
| `roadmap_48_working_memory.py` | Hidden-object MAE / recall / capacity under recency/novelty/random bounded-memory policies with a fixed-color sensor. | Non-learned persistent-memory reference for object permanence under occlusion. |
| `roadmap_52_surprise.py` | Pre/event/recovery/glitch-window MSE for fixed / full-correction / bounded-innovation / delayed-feedback scalar estimators. | Classical surprise-recovery reference for how prediction should handle sudden change vs noise. |

**evaluations**

| experiment | establishes | why it matters |
|---|---|---|
| `active_gaze_comparison.py` _(collides with `src.active_gaze_comparison`)_ | Paired fixed/random/active 1D-gaze belief-tracking across seeds. | Non-neural active-sensing reference: does belief-driven gaze anticipate motion better than fixed/random. |
| `roadmap_20_recording_eval.py` | Strict-causal offline replay of a timestamped webcam recording, frozen model vs persistence/constant-velocity at real inter-frame horizons. | Does the frozen cellular forecaster beat simple controls on real camera footage. |
| `roadmap_44_live_mirror.py` | Live validation-selected frozen forecaster vs persistence/velocity, predictions timestamped before future frames, + deterministic replay harness. | Test-only live verification the selected model beats controls without on-stream tuning. |
| `roadmap_46_attention.py` | Event-recall + observed-pixel-fraction for surprise/salience/center/random window selection (prior-RGB only) vs evaluator-only future motion. | Does prediction-error (surprise) attention cover upcoming motion better than null — precursor to cellular attention. |
| `roadmap_47_budgeted_attention.py` | Masked constant-velocity-vs-persistence MSE + unique-updated-pixel counts across attention modes incl. full references. | Does a spatial attention budget buy accuracy per updated pixel. |
| `roadmap_49_lifecycle.py` | False-retirement / missed-visible / reappearance-transition counts for threshold-based object-lifecycle inference from blobs. | Operational object-permanence (birth/death/occlusion) from visible pixels alone. |
| `roadmap_50_correspondence.py` | Visible/hidden/post-crossing MAE for beam-search identity correspondence vs nearest/coast on same-color detections. | Does multi-hypothesis tracking preserve identity through crossings better than greedy. |
| `roadmap_51_irregular_memory.py` | False-death / unsupported-presence / missed-object counts under regular/jitter/burst timing with time- vs frame-based expiry. | Robustness of persistent memory to irregular sampling (live-relevant). |
| `roadmap_54_active_gaze.py` | Position MAE / sighting rate / gaze-travel for active-Bayesian vs trained-fixed vs random gaze (one 3×3 window/step). | Does belief-driven active sensing beat fixed/random — non-neural active-perception reference. |
| `roadmap_63_retinotopy.py` | Temporal-change-energy + layout-effect MSE of graph-signal rasterization under true vs permutation-shuffled registration. | Node order alone isn't retinotopy; layout must be externally supplied — guards mapping claims. |
| `roadmap_66_neural_agreement.py` | Held-out neural-response MSE + per-neuron Pearson for a lag-selected linear decoder vs shuffled/constant-mean nulls, with a noise-ceiling slot. | Scores stimulus→neural agreement with controls (synthetic by default — protocol, not biological support). |
| `roadmap_69_topology_transfer.py` | Held-out MSE for frozen source-pretrained coefficients on a new target graph vs stale-source / shuffled-target / target-refit arms. | Do topology-conditioned cellular coefficients transfer across graphs (not biological transfer). |

**audits**

| experiment | establishes | why it matters |
|---|---|---|
| `roadmap_45_release.py` | Deterministic release zip with per-source SHA256, train/val/test byte+session disjointness, privacy omission of raw footage. | Provenance/reproducibility so results can't silently reuse data across splits. |
| `roadmap_55_scene_campaign.py` | Provenance-linked manifest bundling 46–54 with per-module SHA256 + a checkpoint-gated trained-RGB comparison (never faked when absent). | Aggregates attention evidence while barring random-init from masquerading as trained. |
| `roadmap_61_selection.py` | Validation+latency-gated frozen-model selection with fail-closed provenance (ckpt/session/device sha, finite metrics) + optional test attach. | Chooses the winning model without test leakage, meeting a real-time latency budget. |
| `roadmap_62_connectome_contract.py` | Validated SHA256-provenance-gated connectome-adjacency loader (directed/undirected preserved) + labeled synthetic toy fallback. | Prevents graph studies from passing synthetic graphs off as measured biological connectomes. |
| `roadmap_70_biology.py` | Provenance manifest aggregating 62–69 with per-module SHA256 + explicit `not_demonstrated` status for four biological-evidence claims. | Prevents synthetic graph studies from being read as measured biological/neural evidence. |

**demos**

| experiment | establishes | why it matters |
|---|---|---|
| `rgb_verified_live.py` | Live webcam 4-panel observed/forecast-+10/actual/abs-error view with delayed-scoring metrics for pre-committed forecasts. | Live demonstration that a frozen cellular forecaster's early predictions match reality (no baselines). |

---

## Cleanup backlog (drives the code-simplicity pass)

Collected across all 147 runners; these are **future opportunities**, not a
checklist that must be completed before this incremental PR can merge.
Seven RGB-family training runners already use the shared clipped-AdamW step;
other migrations, renames and reorganization remain deferred. Any migration
must preserve the original rate, gradient clipping and update semantics, with
its own tests. Behavior-shifting candidates require separate review.

1. **Shared `train_and_select` helper.** The jitted grad-norm-clip + `adamw`
   `update`/`make_update` closure plus the per-arm `{params, m, v, best, step,
   val}` validate-and-select loop (step-zero-eligible) is copy-pasted near-verbatim
   across **~25 training-studies**. Several define the `@jax.jit` closure *inside*
   a seed/mode loop, forcing recompilation. `shared_core_transfer.py`'s
   `optimizer()/fit()` is the model to generalize. Single biggest win.
2. **Shared frozen-checkpoint harness.** `np.load(allow_pickle=False) →
   {k: jnp.asarray(f[k])}`, the `hashlib.sha256(read_bytes())` provenance digest,
   `benchmark.controls`, and `leads = tuple(sorted({horizon, *(h for h in LEADS
   if h<=horizon)}))` recur across **~12 eval/probe files**. `rgb_hidden_interventions.py`
   and `rgb_error_trace.py` are nearly the same harness and should share one.
3. **Small duplicated utilities to extract:** the dependency-free `_png` writer
   (several gaussian experiments + galleries), `tile()` (`rgb_event_gallery` /
   `rgb_prediction_gallery`), `shift()` (`development_26` / `development_27`), the
   `dynamics_`-prefix param flatten/unflatten (`campaign_archive`,
   `gaussian_active_view`, `active_world3d`), and the even/odd paired-contrast math
   (`object_permanence_campaign` / `paired_occlusion_histories`).
4. **Filename collisions with imported `src` modules** (confusing): `rgb_benchmark.py`,
   `active_world3d.py`, `active_gaze_dynamics.py` → add an `_eval` suffix.
5. **Proposed renames** (numbered `roadmap_*`/vague names → descriptive): see the
   bolded arrows in the tables above. Two name clashes to resolve when renaming:
   `phase_ablation.py` vs the existing `rgb_phase_coupling_ablation.py`, and the
   `_eval` collisions in (4).
6. **Formatting normalization.** Many `roadmap_*`, gaussian, and dense RGB files omit
   PEP8 spacing and use one-letter names; clean exemplars are `reconciliation.py`,
   `rgb_nca.py`, `graph_skeleton_horizons.py`, `evidence_report.py`. A formatter +
   light commenting pass over the dense files.
7. **Behavior-shifting candidates (flag before touching):** `roadmap_33_replay.py`'s
   computed-but-unused `_committed_prediction`; signed `extra_encoder_error_mse` in
   `gaussian_reconstruction_diagnostics.py` (can read negative/misleading);
   per-step re-rendering in `roadmap_39_particle_vision.py` and `action_world3d_nca.py`
   (cacheable); the `nca_blend` dead/fragile cross-PR import in `rgb_world_ladder_training.py`.
8. **Missing doc:** `development_24_correspondence.py` has no companion `docs/*.md`.
