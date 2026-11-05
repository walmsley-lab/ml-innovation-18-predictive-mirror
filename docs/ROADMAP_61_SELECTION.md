# Roadmap 61: validation-only model selection and hardware gate

Prepare a JSON object `{"candidates":[...]}` containing checkpoint SHA256, model name, validation MSE measured under a shared protocol/session, target-device measured inference latency, device identifier, and evidence flag. Selector refuses missing latency, random-init smoke, incomparable hardware and uncalibrated test reuse.

Run `python -m experiments.roadmap_61_selection --validation-manifest runs/validation.json --max-latency-ms 100`. The selection and validation-input hash are frozen before optionally opening a separately produced test manifest. Failure to meet the hardware budget is a meaningful `no_eligible_model` result, not a silent fallback. Test data must not change the choice. Correctness of claimed recording/session independence still requires operator review. This is a model-selection controller, not automatic checkpoint training.
