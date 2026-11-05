# Prior art and novelty gates — Predictive Mirror

Research position as of 2026-11-05. This is a **living, cited critical map**, not a proof that a novelty search is exhaustive. Sources include the uploaded Perplexity research pass (cut off during Section 7.3 and lacking its full numbered reference list) and independently checked primary sources.

## Already demonstrated elsewhere

| Established area | Closest primary source | What it means for us |
| --- | --- | --- |
| Differentiable local NCA rules and regeneration | Mordvintsev et al., [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) (2020) | Shared local update rules, hidden channels and emergent pattern growth are not new |
| NCAs learning spatiotemporal and PDE trajectories, including unseen initial conditions | Richardson et al., [PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1011589) (2024), [code](https://github.com/AlexDR1998/NCA) | "An NCA learns moving spatial patterns" is **not** a novelty claim; compare mechanistic dynamics and symmetry controls |
| NCAs predicting traffic/geospatial fields | Chen et al., [arXiv:2009.09347](https://arxiv.org/abs/2009.09347) (2020, preprint) | NCA-based forecasting exists outside morphogenesis |
| Rotational bias and isotropic NCA variants | Mordvintsev et al., [arXiv:2205.01681](https://arxiv.org/abs/2205.01681) (2022, preprint) | Kernel orientation and symmetry are explicit confounders for locality comparisons |
| High-resolution NCA through coarse states and local decoders | Pajouheshgar et al., [SIGGRAPH 2026](https://doi.org/10.1145/3799902.3811053), [project](https://cells2pixels.github.io/) | Scaling pixel outputs and locally decoded appearance are existing architectural advances |
| Local asynchronous NCAs capable of visual reasoning and OOD generalization | Etcheverry et al., [arXiv:2609.36126](https://arxiv.org/abs/2609.36126) (Sept 2026, **preprint**), [code](https://github.com/paradigms-of-intelligence/reasoning-nca) | Locality, recurrent depth, self-organization, robustness, and dynamic compute are **not** by themselves original claims; tasks differ |
| Neural particle automata for self-organizing dynamics | Pajouheshgar et al., [SIGGRAPH 2026](https://doi.org/10.1145/3799902.3811052), [project](https://selforg-npa.github.io/) | Relevant to separate particle-NCA investigations; not novel simply to move learned cells as particles |
| Competitive recurrent-free forecasting | Gao et al., [SimVP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Gao_SimVP_Simpler_Yet_Better_Video_Prediction_CVPR_2022_paper.html), [code](https://github.com/A4Bio/SimVP) | A small NCA must compete against competent spatial/temporal convolutional learning |
| Multi-family video-prediction evaluation and efficiency metrics | Tan et al., [OpenSTL, NeurIPS Datasets & Benchmarks 2023](https://arxiv.org/abs/2306.11249), [code](https://github.com/chengtan9907/OpenSTL) | Baseline selection, normalization, protocols and cost accounting need a published comparator; OpenSTL is Apache 2.0 |
| Physics/appearance disentanglement for video prediction | Le Guen and Thome, [PhyDNet CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Le_Guen_Disentangling_Physical_Dynamics_From_Unknown_Factors_for_Unsupervised_Video_Prediction_CVPR_2020_paper.html) | Structured dynamics and recurrent correction are established; not an invention of our pipeline |
| Stochastic futures | Babaeizadeh et al., [SV2P](https://arxiv.org/abs/1710.11252); Voleti et al., [MCVD](https://arxiv.org/abs/2205.09853) | Ambiguous futures require probabilistic evaluation; building a stochastic video model is not intrinsically novel |

**Status:** preprints are not called peer-reviewed merely because they are popular; a SIGGRAPH 2026 conference publication must not be mislabeled as only a preprint. The report's numerical Moving-MNIST MSE/FLOPs are *not* comparable to our normalized 32x32 RGB numbers.

## Conditional hypotheses, not proven novelty

1. **Paired-history partial observability:** Evaluate two different histories sharing a byte-identical last observation and opposing futures. This is a useful controlled test, but **not proven new**; continue searching memory/object-permanence and paired counterfactual benchmarks.
2. **Causal usefulness of local communication:** Intervene on a *responsive* trained predictor with calibrated positive controls and matched-null perturbations, then compare actual forecast capability. Simply drawing hidden-state heatmaps, modes or waves is descriptive, not mechanistic proof. Prior work on computational causal interventions must be investigated before any novelty claim.
3. **Constrained accuracy–cost advantages:** Find whether an NCA sits on the accuracy-vs-compute/communication frontier, compared on identical tasks and hardware. FLOPs and CPU latency alone cannot establish energy or hardware-substrate advantage.

## Baseline promotion decisions

- **Must have before NCA advantage claims:** persistence, RGB-derived motion, history-aware linear/Kalman controls, a competent recurrent CNN (e.g. ConvLSTM), and a competent recurrent-free CNN (SimVP or reference-faithful equivalent). Keep matched-resource and adequately trained competence regimes separate.
- **Later conditional:** PredRNN/PhyDNet for richer temporal/physical worlds; a classical probabilistic tracker before SV2P; MCVD only if multimodal high-fidelity video becomes central and compute supports it.
- **Never:** silently reinterpret a "SimVP-like" small implementation as the published reference, compare 8->10 synthetic RGB directly against a Moving-MNIST 10->10 leaderboard, or assume non-recurrence means absence of temporal context.
- **Causality:** CD-NOD is an observational discovery method, not our direct-intervention positive control. An impulse injected into the network is an intervention on that network, but only a task-specific effect can establish functional relevance.

## What would change the research direction

- If the **known-prior** baseline wins, label it separately rather than making a false "RGB only" claim.
- If the strongest fair history/ConvLSTM/SimVP predictor dominates the NCA on success and cost, pause cellular architecture expansion and publish honest failure diagnostics.
- If a cellular model has an advantage, test whether it persists under identifiability, symmetry, scene shift, more seeds and matched computational budgets before attaching a mechanism claim.
- Identity persistence, velocity memory, calibrated uncertainty and long-range communication require different experiments. A successful one-object reappearance does not establish the others.
- Biological relevance requires explicit observable correspondences (receptive fields, feedback, temporal propagation) and competing null models; anatomical analogy alone is not evidence.

## Immediate next action

Complete evidence audit and existing pair-history tests, then export the same dataset to standard learned-model baselines under one manifest. Archive all result/checkpoint hashes. On the first frozen report distinguish **measured evidence**, **historically reported outcomes**, **source-code implementation**, **proposed research**, and **external literature**.

**Open literature question:** find prior work that explicitly constructs *same full observed history, different possible futures* (irreducible uncertainty) versus *same last observed image, different informative histories* (memory), and directly tests causal dependence on local communication. The attached research report did not establish priority over all such work.
