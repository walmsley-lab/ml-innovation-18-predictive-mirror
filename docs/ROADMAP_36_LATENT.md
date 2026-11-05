# Roadmap 36 — observation-derived visual latent dynamics

Run: `python -m experiments.roadmap_36_latent_dynamics`, and `python -m unittest tests.test_roadmap_36_latent_dynamics -v`.

A compact PCA representation of observed RGB is fitted on train-only episodes. Ridge regression learns an autoregressive latent transition; validation episodes select rank and shrinkage; held-out linear, accelerated, occluded and camera-motion scenes assess open-loop performance alongside persistence and image-derived constant velocity. No hidden object or camera state enters inference. The latent model sees only the final observed frame; note the baseline's greater history budget. Training checkpoint and per-domain scores are saved. No nonlinear learned state, object permanence, or real-camera superiority is claimed. Future comparison should freeze identical RGB history adapters and train capacity-matched recurrent latent dynamics.
