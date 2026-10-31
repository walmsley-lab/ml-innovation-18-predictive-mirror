# Distinct follow-up: make the NCA selection function explicit

**Goal:** Separate *which cells each cell reads* from *how the shared cellular
update rule processes their information*. This follows the independent
RGB predictive NCA experiment in PR #34, but does not modify or replace it.

The new `src/selectable_rgb_nca.py` uses the same cellular update and RGB
dreaming contract for **three distinct selectors**:

- **Local:** read a 3x3 pixel neighborhood via uniform local pooling, including
  zero-padded edges. This is a selector-only comparison arm, **not exactly the
  learned 3x3 convolution** in the base RGB NCA.
- **Relative:** read specified displacement offsets, initially cardinal
  neighbors two pixels away. The offsets are fixed before training.
- **Reference:** read a directly specified distant cell through an explicit
  `[H,W,2]` integer pointer map. The pilot uses **diagonally mirrored
  coordinates solely as deterministic example references**. This is NOT
  learned routing, semantic object identity, pose extraction, or a graph
  inferred from perception.

All three arms have identical trainable parameter structures and start from
the same weights. Their update budgets and real-data batches are paired, while
their number of selected neighbors, distance, memory bandwidth and computation
are **not** matched. These pilot controls answer whether selection changes the
forecast, not whether a type is universally superior.

## CPU pilot using separate RGB footage

```bash
python -m experiments.nca_selector_ablation \
  --input recordings/rgb-webcam.npz --steps 30 --context 6 --horizon 5 \
  --out runs/nca-selectors
cat runs/nca-selectors/metrics.json
```

This creates independent `local.npz`, `relative.npz`, `reference.npz`,
and `metrics.json`. The RGB data stays local. No previous model/checkpoint
is modified. Do **not** use grayscale `recordings/webcam.npz` for these
three-channel selectors.

## Next experimental controls

1. Use a second recording and multiple seeds; no success claim from
   validation-selected or single-recording results.
2. Equate selected-neighbor count, effective receptive field and measured
   latency to isolate selection rather than compute.
3. Design **learned** reference pointers from current/past observations; test
   against fixed, random and shuffled-pointer controls.
4. Compare reference-based selection on *object/skeleton* graphs only when
   the origin of those references is recorded (provided vs learned).
5. Keep RGB NCA, selectors, oscillators, graph NCA and original grayscale
   recurrent models as separate implementations until common benchmark
   evidence justifies reconciliation.

**No result in this branch establishes perception of 3D, object identity,
neuronal binding, biological equivalence, or performance superiority.**
