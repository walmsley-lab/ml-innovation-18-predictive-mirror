# Contrastive future recognition (experimental)

**Question:** Can a small state/future two-tower scorer discriminate plausible visual continuations better than untrained embeddings and a supervised pairwise MLP? Can it compete with observation-only persistence and classical constant velocity?

```bash
python -m unittest discover -s tests -p 'test_contrastive_future_recognition.py' -v
python -m experiments.contrastive_future_recognition --steps 100 --seeds 11 23 41
cat runs/contrastive-future/metrics.json
```

The task has eight observed RGB frames and **four supplied**, same-renderer, four-frame future candidates (cardinal motion directions). The correct future is **guaranteed to be among the candidates**; this measures *recognition/ranking*, **not** future generation or an active sensing policy. Scene center, speed, direction, and candidate order vary by seed. Candidate generation and positive labels are dataset/evaluator operations. Model input contains RGB observations and RGB candidate futures only, not velocities, coordinates, labels, or simulator truth. The state tower only receives past frames.

Training is on scenes seeded 31000+, while final evaluations use disjoint scenes seeded 91000+. First/middle/last RGB frame features are spatially downsampled to an 8×8 grid; both towers use independent small tanh projections and cosine similarity. InfoNCE scores one correct pair against all candidates across the minibatch, including the *three same-scene* hard negatives. The supervised MLP scores the same candidate set using concatenated features. An untrained two-tower baseline, last-frame persistence, and image-derived constant velocity are controls. All are scored on the same sealed test set. The JSON reports accuracy, correct-vs-best-wrong margin, model parameter counts, seeds, and wall time. No checkpoint is claimed from a smoke test.

**Interpretation:** A random or untrained scorer can obtain about 25% accuracy; constant velocity is a strong control in this deterministic world. If the tower does not beat that control, do **not** claim any predictive benefit. If the tower does well, the next step is to rank candidate futures from existing, frozen RGB predictor checkpoints, including occlusions and genuinely ambiguous futures. For ambiguous histories where multiple different futures are equally valid, do not assign a unique contrastive positive; evaluate a distribution or multi-positive objective. Softmax candidate scores are relative rankings, **not calibrated uncertainty**. This is an isolated CPU-friendly JAX research probe, not an architectural change to the live mirror or a clinical model.
