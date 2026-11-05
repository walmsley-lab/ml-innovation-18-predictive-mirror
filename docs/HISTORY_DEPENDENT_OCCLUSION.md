# Identifiable history: same final observation, different future

Pairs share the exact last observed RGB frame (fully occluded) but
show **opposite incoming trajectories** earlier in the context.
Their held-out future images differ by direction and object-return
position. No predictor operating only on the final frame can
disambiguate this pair.

```bash
# Frozen last-frame persistence, analyst readout and oracle controls:
python -m experiments.paired_occlusion_histories --seeds 11 23 41
# An independently trained RGB-only checkpoint:
python -m experiments.paired_occlusion_histories \
  --checkpoint runs/rgb-world-ladder-training/cnn-seed11-mixed-best.npz --model cnn
python -m unittest tests.test_paired_occlusion_histories -v
```

The runner compares intact, reset (repeat final frame), and
shuffled (reverse earlier frames, preserve final) inputs. It scores
**every** future frame via the guarantee-reappearance evaluator,
including the object-return side, location, timing, and color mass.
The global pair contrast reports whether two different histories
produce different forecasts. Oracle-truth output is explicitly
a scorer sanity check, not a learned-model win.

The non-neural **observed-history motion fit** detects the benchmark's
known colored disc and fixed occluder from RGB. It uses no future
truth or simulator positions, but is intentionally a strong
environment-specific prior.

The repeated-frame reset destroys velocity while preserving final RGB;
shuffling tests sensitivity to ordering but may create unnatural
sequences. These are diagnostics, not guaranteed causal proof that
one specific memory pathway drives the forecast.
