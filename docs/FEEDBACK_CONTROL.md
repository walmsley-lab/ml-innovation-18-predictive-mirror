# Feedback-free trained control

Compare **three** conditions, training the first two from scratch using identical seeds, updates and observed input episodes:

```bash
python -m train --architecture multisheet --curriculum staged --steps 120 --batch-size 8 --seed 31 --output runs/intact
python -m train --architecture multisheet_no_feedback --curriculum staged --steps 120 --batch-size 8 --seed 31 --output runs/no-feedback
python -m evaluate --architecture multisheet --parameters runs/intact/parameters.npz --episodes 32 --output runs/intact/evaluation.json
python -m evaluate --architecture multisheet_no_feedback --parameters runs/no-feedback/parameters.npz --episodes 32 --output runs/no-feedback/evaluation.json
```

The existing `multisheet.predict(params, frames, actions, feedback_strength=0.0)` provides the third condition: **intact-trained, feedback cut at test time**. It is *not* equivalent to training without feedback. The baseline uses the exact same initial parameter tensors and 8 persistent channels per pixel. Cross-sheet connection weights are **inactive** and therefore the number of *effective trainable parameters* is smaller despite the equal nominal parameter count. Report that limitation alongside results and use the PR #15 true-occlusion event counts. Neither this control nor any short smoke run proves emergence, functional specialization, or biological equivalence. Compare measured latency and compute, and add random within-sheet lesions before causal claims.
