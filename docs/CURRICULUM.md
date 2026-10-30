# PR 3 — Curriculum and strong image-only baselines

Seeded four-stage progression: stationary dot/no camera; moving dot/no occluder; occlusion with a fixed camera; then ego-motion. Each episode retains hidden world truth **only for evaluation**. `staged` and `shuffled` modes use *exactly the same (seed, stage) pairs* in different orders and a predeclared 10% replay fraction.

```bash
python -m train --steps 120 --batch-size 8 --curriculum staged --output runs/staged
python -m train --steps 120 --batch-size 8 --curriculum shuffled --output runs/shuffled
python -m evaluate --parameters runs/staged/parameters.npz --episodes 32 --output runs/staged/evaluation.json
```

Compare full learning curves and final checkpoints under matched compute/seeds. CLI `--steps` controls **training updates**, not simulation frames. Resume a curriculum run with its original step budget: changing the budget changes the schedule. PR #3 implements descriptive pixel and visible-object scores and persistence/velocity baselines. The pre-registered matched-history feedforward, reappearance timing/location statistics, multi-seed intervals and confirmatory tests remain to be implemented; do **not** claim H2 or curriculum superiority from these reports.
