# Frozen visual integration bridge (pre-architectural expansion)

Run `python -m experiments.frozen_visual_bridge --nca-checkpoint FILE --cnn-checkpoint FILE --transport-checkpoint FILE`.
No checkpoint can be replaced by random initialization unless `--smoke` is explicit. 
Use separately captured `--calibration-recording` and `--test-recording` for normalized RGB episode NPZ files (`frames` with shape `[N,T,H,W,3]`). A different SHA does not prove separate acquisition sessions.

Every model consumes **exactly the same RGB history and horizon**. Persistence and image-derived constant velocity accompany each. Domains include motion, occlusion, ego-motion and combinations, with optional independent exposure/white-balance/dropout stressors. Model labels and parameters are checked against architecture keys. Calibration uses independent full episodes; intervals bound *all* RGB pixel/channel errors at each lead when held-out episodes are exchangeable. Do not interpret this as conditional or shifted-domain coverage, and do not confuse clean-scene targets with future sensor artifacts.

Research gate: run multiple seeds with *actual trained* checkpoints from the same training domain and record checkpoint SHAs, exposure, compute, numerical performance and failure galleries. This bridge is deliberately additive; it does **not** merge or claim success for roadmap 31–35 proxy tasks, train any model, certify test-recording independence, or prove real-world superiority.
