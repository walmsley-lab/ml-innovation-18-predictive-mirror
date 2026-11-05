# Roadmap 41 — explicit shared cellular operator transfer

`python -m experiments.roadmap_41_transfer`. Preserve the existing signal experiment. Use the exact shared 3×3 hidden-to-hidden core in both pulse pretraining and RGB prediction. The source is an **engineered pulse signal**, not observed neurons.

Compare: signal-pretrained and shuffled-pulse-label pretraining (both pretrain+RGB fine-tune), scratch with the **same total updates** (more direct RGB exposure), and scratch with the same RGB fine-tune updates (lower total compute). None dominates the others on all resource axes. Evaluate independent occluded, camera-moving and interaction scenes; keep checkpoint hashes, updates and all negative transfer results. Do not conflate transfer with biological neural agreement; this requires neural recordings and the parallel biological roadmap.
