# Controlled structured pretraining and transfer

```bash
python -m experiments.structured_pretraining_transfer \
  --architectures cnn nca --seeds 11 23 41 --steps 30
python -m unittest tests.test_structured_pretraining -v
```

Train the *same* CNN or recurrent NCA from a fixed seed on a small
source curriculum: physical waves, diffusion, occluded sprites,
ordinary linear motion, or temporally shuffled controls. Then
fine-tune on the same downstream RGB occlusion task and evaluate
held-out long occlusion, acceleration and combined motion.

The scratch arm trains on the target distribution for **the same
total optimizer updates**, not just the fine-tuning portion.
Pretrained arms receive extra task-diverse synthetic input but fewer
target-only updates. Seed, source, target train step, checkpoint
hash, elapsed time and independent test errors are recorded.
Comparisons are within architecture; parameter count is not matched
between CNN and NCA. No winning curriculum is claimed from smoke
runs, and uncontrolled source-pixel entropy is a serious confound.

"Retinal wave" is a mammalian developmental analogy; the tested
wave fields are mathematical training stimuli **not biologically
validated species-specific retinal activity**. Claims about fly
development, real webcam transfer or long-term retention require
separate studies.
