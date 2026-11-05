# Predictive Mirror: pre-evaluation research contract (proposal v1)

**Status: proposed, not yet frozen.** This is a preregistration *candidate*, not a claim that any study has run or met its gates. Freeze the versioned JSON and record its SHA256 before inspecting a new final test set. Any later edits require a new version and a clear post-hoc label.

## Question and estimand

On RGB-only paired histories with exactly the same last observed image, can a model use prior observations to predict which side of an occluder an object will reappear on? Is a cellular NCA better than a competent non-privileged history predictor at an acceptable cost?

**Primary outcome:** event success on paired-history barrier returns. A return counts only if predicted object mass is present, its soft centroid is within 3 px of the returned object, onset is within 1 frame, and there are no premature visible predictions. Average the two opposing directions within each independent scene seed **before** estimating uncertainty. This metric does **not** establish identity preservation.

**History contrast:** compare predictions for two different histories sharing exactly the same last image. Last-image-only systems should not distinguish them. Quantify paired future-contrast error; do not conflate predicted difference with correctness.

## Hypotheses, tests and null outcomes

| Hypothesis | Contrast | Falsifier / inconclusive case |
| --- | --- | --- |
| H1: previous observations affect predictions usefully | intact history vs same-last-frame history-reset and history-shuffle; history-aware classical reference | no meaningful improvement or a dead/invariant predictor; a reset contrast alone is not enough |
| H2: learned cellular recurrence gives additional task value | frozen NCA vs strongest **non-privileged** classical or learned history model | NCA does not improve primary outcome by a prespecified practical margin or is dominated at stated resource budget |
| H3: particular internal-state / communication mechanisms contribute | intact vs targeted manipulation vs energy-matched null; calibrated known-live positive control | targeted intervention not distinguishable from null; if positive control fails, label **unmeasurable**, not null |
| H4: cellular model is viable at its intended cost | accuracy–cost frontier vs conventional alternatives on the same device | dominated or exceeds prespecified cost ceiling without an outcome advantage; no physical-hardware inference from CPU timing |

Separate *scientific null*, *inconclusive evidence*, and *uninformative assay*. An initial study that finds no NCA advantage is a valid stopping outcome, not a mandate to add more modules.

## Split and information rules

- Observation: past RGB frames, context length 8. Horizon: 10 synthetic frames, **not** verified real-time seconds.
- Keep pair members in one statistical cluster; preserve train/validation/final test scene-seed disjointness and freeze final test before selection.
- No truth positions, color IDs, visibility flags, future frames, or generator state at inference. Known-color/known-geometry trackers and simulator oracles must be labeled **privileged reference**; never rank them as fair RGB-only competitors.
- Baselines: persistence, observed-frame velocity, unprivileged history regression, an RGB-derived Kalman-style estimator, CNN/transport, and later a conventional recurrent reference.
- Validate scoring using perfect oracle predictions, missing/shifted/bright outputs, paired histories, camera exit vs barrier occlusion and genuinely ambiguous/indistinguishable observations.

## Decision and statistical design

The JSON specifies a *proposed* minimal practical advantage of 0.10 absolute event-success probability and a proposed maximum 3x inference-time premium when arguing efficiency. **These are policy choices, not derived effect estimates or statistical facts.** Review them with pilot variance, commit a frozen contract, then evaluate untouched test episodes.

Use independent scene pairs as units within each model seed and model seeds as an additional level of uncertainty. Report model seed variation, the paired effect (absolute percentage-point change), a hierarchical cluster/bootstrap interval and the number of independent scene pairs. A confidence interval crossing the practical margin is **inconclusive**, not proof of no effect. Do not call a fixed 5–10 seeds intrinsically adequate; estimate precision and feasible power from pilot variance.

Decision labels: supported task advantage / no demonstrated advantage / inconclusive / measurement dead. Pause **architecture expansion** if established predictors match or dominate cellular models on accuracy–cost; continue diagnostic/failure-method work without claiming superiority.

## Reproducibility and changes

Archive exact source revision, command, seed roots, dependency versions, candidate checkpoints, checkpoint-selection rule, test results, plotting inputs and source hashes. Record original-run / fresh-reproduction / operator-reported-only statuses separately. Label all changes after test inspection as exploratory and increment the contract version.

Use `python -m src.research_contract --contract experiments/research_contract_v1.json` to schema-check and print the contract SHA256. Do not interpret schema validation as experimental preregistration, data verification, or user approval.
