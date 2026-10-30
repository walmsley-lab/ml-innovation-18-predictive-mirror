# PR 21 — Developmental retention and transfer diagnostics

Save the weights at fixed 25%, 50%, 75%, and 100% training milestones. Evaluate **every** snapshot against fresh, fixed stage-0 through stage-3 observational sequences; the same held-out episodes are used for both staged and shuffled curricula. Training keeps the same observations, total updates, 10% replay fraction and model architecture.

```bash
python -m train --architecture recurrent --curriculum staged --steps 240 \
 --batch-size 8 --seed 31 --save-milestones --output runs/staged
python -m train --architecture recurrent --curriculum shuffled --steps 240 \
 --batch-size 8 --seed 31 --save-milestones --output runs/shuffled
python -m experiments.developmental_transfer --staged runs/staged \
 --shuffled runs/shuffled --architecture recurrent --out runs/development.json
```

The resulting 2×4×4 matrix reports stage-specific next-frame MSE and image-derived visible-object error. Snapshot export does not give the network simulator position/occlusion labels. Keep seeds out of the training stream. Repeated milestone snapshots preserve history for forgetting checks, but the current experiment alone is **not** a causal test that pretraining produces transfer; that requires matched reset/pretrain/fine-tune controls. Report both learning curves and the final checkpoint; do not choose a flattering midpoint. Reappearance localization and energy/compute comparisons remain separate.
