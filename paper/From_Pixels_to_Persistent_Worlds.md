# From Pixels to Persistent Worlds?
## Reproducing, Improving, and Understanding Distributed Visual Prediction

*Predictive Mirror — illustrated research note · November 2026 · Research prototype*

> **Scope.** This report distinguishes original experimental measurements, operator-reported results, established literature, and capabilities proposed for future development. Its diagrams are conceptual unless labeled as outputs from a checkpoint.

### Abstract

Can a network of small, locally interacting computational cells learn to anticipate visual events? Predictive Mirror explores this question with neural cellular automata (NCAs), conventional visual predictors, controllable synthetic environments, and interventions on recurrent state. We use established NCA and video-prediction research as reference capabilities to reproduce and improve, not as ideas we must avoid because they are no longer novel.

In one previously reported three-seed synthetic benchmark, an RGB cellular predictor reduced ten-frame image MSE relative to last-frame persistence (approximately 0.0080 versus 0.0121), but did not match image-derived constant-velocity extrapolation (approximately 0.0050). Separate campaigns exposed degenerate image outputs, uninformative interventions, and cases in which object information survived a hard detection threshold before disappearing at later horizons. A non-saturating readout removed one clipping pathology, but has not yet established accurate object tracking. These results motivate our next phase: reproduce strong reference results, make small models train reliably, compare them against capable alternatives, and test which internal mechanisms produce useful behavior under partial observability and communication constraints.

## 1. The predictive mirror

Imagine a mirror that shows what it expects to happen next. A moving ball is easy to extrapolate while visible, but once it passes behind an obstacle the current image no longer reveals its motion. An observer may still anticipate its return by using earlier observations and a model of the environment.

Our computational challenge is deliberately smaller. A predictor receives a short sequence of RGB images, updates its internal state, and commits future frames **before** those frames are observed. The future observation is available only for training targets or later evaluation, never as a forecasting input.

A useful predictive mirror must do more than render plausible colors. It should anticipate what moves, where and when it moves, when something is hidden, and what earlier observations imply about future events. Some outcomes are intrinsically ambiguous: if two worlds produce exactly the same complete observation history but different futures, no deterministic RGB-only model can know which hidden outcome occurred.

**Figure 1 — Prediction is a testable commitment.** Illustrate *observed frames → committed forecast → subsequent truth → error*. For published figures use real checkpoint outputs and identical visual scales. A ball behind an occluder is an explanatory example, **not** evidence that the present NCA achieves object permanence.

## 2. A simple local computational substrate

Our primary RGB NCA is a grid of cells with persistent hidden vectors. Each cell repeatedly receives information from a 3×3 neighborhood, applies a shared learned update, and changes its state. During observation, a sensory discrepancy influences the recurrent computation; during an open-loop forecast, no new sensory information arrives. A learned readout converts hidden activity into changes in the predicted image.

This is intentionally a small, interpretable experimental substrate, not a model of individual biological neurons. Distributed and recurrent processing in visual systems motivates asking about local receptive fields, feedback, temporal information, and long-range communication. A biological analogy does not establish that these mechanisms arise in our model or confer any advantage.

The laboratory maintains independent comparisons rather than one monolithic architecture: direct RGB models, image transport, hierarchical processing, soft spatial references, oscillator and harmonic probes, graph and particle systems with explicitly different inputs, active sensing, and controlled physical worlds. A source file or runnable experiment establishes implementation, **not** empirical success.

**Figure 2 — Local cell, global forecast.** Show a neighborhood, hidden vector, shared update rule, and repeated ticks; pair it with a clearly labeled diagnostic hidden-state activity map if a verified checkpoint is available. A small cortex feedback schematic can convey motivation without inventing brain-activation measurements.

## 3. Early experimental evidence

Small synthetic scenes let us control motion, color, collisions, occlusion, and viewpoint changes, while keeping ground truth for scoring. We compare an NCA with persistence and a constant-velocity baseline estimated from earlier RGB frames; simulator object positions and velocities are not supplied to those predictors.

A reported three-seed benchmark with eight context frames and a ten-frame horizon produced the following mean RGB errors:

| Method | Ten-frame RGB MSE (lower is better) |
| --- | ---: |
| Last-frame persistence | 0.0121 |
| Recurrent RGB NCA | 0.0080 |
| Image-derived constant velocity | **0.0050** |

These rounded numbers are **historical operator-reported results from one synthetic generator**. They are not independently re-executed estimates with confidence intervals, and they do not demonstrate object tracking, webcam generalization, or an advantage over modern learned predictors.

Later exploratory CPU studies encountered more severe problems. A residual-plus-clip forecast head could drive almost every pixel to black and become independent of the input; additional optimization steps did not rescue that saturated solution. Replacing the readout with a non-saturating blend prevented the specific clip-rail collapse in the reported training sweep, but did **not** yield reliable foreground detection or localization at the tested budgets. This is an engineering improvement in trainability, not yet a performance breakthrough.

The key development lesson is to confirm that a model can learn a simple known task—such as overfitting a handful of moving-object clips—before investing in larger architectures or interpretation experiments. Then increase environmental and horizon difficulty in measured stages.

**Figure 3 — Forecasts, not a leaderboard alone.** The principal result image should compare last observation, NCA, persistence, RGB-derived velocity, and truth at the **same** horizon, with an error heatmap. Accompany it with per-horizon performance and model-seed variation from one frozen run. Do not mix incompatible synthetic campaigns.

## 4. Understanding failures as a route to better models

An image metric can reward the wrong behavior. In one exploratory campaign, a nearly uniform bright prediction achieved reported foreground recall 1.0 while foreground precision was only about 0.011 and the model failed to localize objects. In a different frozen-checkpoint diagnostic, an object contributed detectable color mass at short horizons even though a hard threshold declared it absent; the long-horizon rollout then lost that information and collapsed toward black.

These are distinct cases: **metric exploitation**, **detection sensitivity**, and **genuine loss of predicted information**. The distinction matters for training. A stronger threshold does not repair missing dynamics; a better dynamics model does not cure an invalid metric.

A similar issue appears in mechanistic interpretation. If a forecast is already independent of hidden state, removing or perturbing that state may have no visible effect. Such a measurement is uninformative about the mechanism's usefulness. A controlled positive intervention must first show that the assay responds when it should. Then targeted interventions can distinguish a descriptive activity pattern from a mechanism that contributes to prediction.

The repository now contains diagnostic measures for foreground location and mass, clip-rail saturation, input dependence, event timing, and intervention sensitivity. They should guide development without becoming a burden on every small PR.

**Figure 4 — Three different failures.** Use verified, independently sourced examples of bright-frame collapse, a soft object rejected by a hard detector, and genuine late-horizon black-frame collapse. Always mark the separate run provenance. A small inset can show *intact / targeted lesion / known-live positive control* as a **method**, not an accomplished causal discovery.

## 5. Reproduce what works, then exceed it

We do not need to invent a principle to benefit from understanding it. Mordvintsev and colleagues showed that NCAs can grow and regenerate target patterns. Richardson and colleagues showed that NCAs can reproduce evolving reaction–diffusion and other spatiotemporal patterns. These are excellent capability references: a faithful reproduction gives us something real to optimize and extend.

The video-prediction literature supplies equally important competitors. ConvLSTM introduced spatially organized recurrent memory; PhyDNet incorporates physically structured dynamics; and SimVP shows how capable a relatively simple convolutional predictor can be without recurrence. The OpenSTL ecosystem offers implementations and evaluation conventions, but published benchmark numbers from different datasets cannot be compared directly with our tiny RGB world.

Our planned performance progression is deliberately practical:

1. **Reproduce:** Pattern growth/regeneration, spatiotemporal PDE dynamics, and strong conventional video-prediction references under their own published settings.
2. **Learn reliably:** Repair the present NCA forecast dynamics, demonstrate short-sequence overfitting, then increase horizon and scene complexity.
3. **Compete:** Compare direct RGB, image transport, and hybrid dynamics-plus-rendering models on common RGB-only history tasks, with meaningful training budgets.
4. **Adapt:** Test occlusion, identity, partial observability, uncertain futures, active observation, and transfer to recorded or live video.
5. **Explain and optimize:** Study causal state/communication roles, actual information reach, latency, memory traffic, and measured physical cost when available.

Reproducing a published capability is progress even if it is not novel. Beating a reference is a valuable performance milestone even without a new explanatory theory. Explaining why a mechanism works is a separate scientific achievement.

**Figure 5 — Capability ladder.** Draw a clear progression from *pattern growth and recovery → evolving spatial fields → observed video motion → history-dependent reappearance → adaptable prediction*. Use only actual replication artifacts for achieved stages; mark others as planned. Place RGB prediction, motion transport, and persistent structured-state models as competing or complementary alternatives rather than claiming one finished hybrid.

## 6. Toward adaptable predictive computation

A useful perceptual foundation must jointly address **dynamics** (what changes), **memory** (what remains relevant when information is missing), and **observation** (what is visible and what evidence might be acquired). These functions need not belong to one monolithic network or share one internal representation.

Our next discriminating test uses paired histories whose last visible image is identical but whose earlier trajectories differ. A successful predictor should produce the appropriate different futures when the earlier evidence is informative. It must compete with history-aware classical filters and learned convolutional predictors; merely beating last-frame persistence would not establish a special NCA capability. Reappearance, identity preservation, and irreducible future ambiguity require distinct evaluations.

Once functioning models perform useful tasks, we can study the consequences of local communication, longer-range references, recurrent state, oscillations, and spatial harmonics. Traveling-wave-like visualizations can be compelling, but a spatial spectrum alone does not prove a propagating functional wave. Perturbation experiments must first demonstrate measurement sensitivity and then a task-specific consequence.

Our longer-term interest includes physical computing. Locality and shared rules could someday be useful on distributed or in-memory substrates, but present CPU experiments have not demonstrated an efficiency advantage. We therefore measure computation and memory/communication costs alongside predictive quality rather than treating biological inspiration as a hardware-efficiency result.

**Figure 6 — Research architecture, not product claim.** Show three interacting functions—dynamics, memory, observation—feeding a future renderer, with evaluation feeding back into training. A small conceptual grid of multiple plausible futures may illustrate uncertainty, but should not be presented as samples from a trained probabilistic model.

### Conclusion

Predictive Mirror began as an effort to anticipate future camera frames using small networks of recurrent, locally interacting cells. Our initial studies exposed both a modest aggregate-image improvement over persistence and clear failures of training, forecasting, and measurement. They do not yet establish object permanence, cellular superiority, biological equivalence, or practical efficiency.

The next stage emphasizes reproducible **capability**: match established demonstrations, improve stable learning, compare competent predictors, and investigate why specific mechanisms help in particular environments. The broader aim is not to prove that NCAs are uniquely intelligent. It is to build and understand adaptable computational systems that predict, remember, and respond effectively.

**Reproduce what works. Improve what does not. Understand why.**

### Selected references

- Mordvintsev, A., Randazzo, E., Niklasson, E., and Levin, M. (2020). *Growing Neural Cellular Automata*. Distill. https://distill.pub/2020/growing-ca/
- Richardson, A. et al. (2024). *Learning spatio-temporal patterns with Neural Cellular Automata*. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1011589
- Shi, X. et al. (2015). *Convolutional LSTM Network*. https://arxiv.org/abs/1506.04214
- Le Guen, V., and Thome, N. (2020). *Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction*. CVPR. https://openaccess.thecvf.com/content_CVPR_2020/html/Le_Guen_Disentangling_Physical_Dynamics_From_Unknown_Factors_for_Unsupervised_Video_Prediction_CVPR_2020_paper.html
- Gao, Z. et al. (2022). *SimVP: Simpler Yet Better Video Prediction*. CVPR. https://openaccess.thecvf.com/content/CVPR2022/html/Gao_SimVP_Simpler_Yet_Better_Video_Prediction_CVPR_2022_paper.html
- Tan, C. et al. (2023). *A Comprehensive Benchmark of Spatio-Temporal Predictive Learning / OpenSTL*. https://arxiv.org/abs/2306.11249
- Rao, R. P. N., and Ballard, D. H. (1999). *Predictive coding in the visual cortex*. https://doi.org/10.1038/4580

### Evidence and image provenance

Source repository: https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror

Experimental images should identify the model checkpoint, source revision, generator and scene seeds, training/evaluation protocol, and forecast horizon. Operator-reported historical numbers remain marked as such until reproduced. This report intentionally uses conceptual vector diagrams for ideas whose experimental image artifacts are not bundled; **no conceptual drawing should be mistaken for a measured result**.
