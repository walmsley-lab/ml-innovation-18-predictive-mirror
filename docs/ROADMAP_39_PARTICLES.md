# Roadmap 39 — particle representations and observation privilege

Train the existing particle NCA on two representations: synthetic **RGB-derived centroids** using known colors, versus **privileged tracked simulator positions**. Each arm has a message-passing and no-messages control, paired optimization budgets, causal last-seen handling of missing observations and independent held-out occlusion scenes. Compare each representation's particle forecast against its own constant-velocity and persistence baselines. Keep the model family and all negative results.

**Critical boundary:** this is not RGB-to-object discovery. The image decoder assumes known unique marker colors and the privileged arm has information unavailable in a real camera. Results cannot be combined into one undifferentiated leaderboard. Run `python -m experiments.roadmap_39_particle_vision`.
