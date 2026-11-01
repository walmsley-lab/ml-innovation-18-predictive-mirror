# Neighborhood selection comparison on the common +10 RGB benchmark

Independent **selectable RGB NCA** variant: local 3x3, fixed relative
displacements, and externally provided reference pointers. Run three
initialization seeds with equal optimizer, parameter counts, update budget,
train/validation/test scenes and observation-only baselines.

`python -m experiments.rgb_selector_benchmark --steps 60 --horizon 10`

Reference pointers are a deterministic **diagonal mirror map**, not learned
attention, actual object correspondences or a biologically equivalent selector.
Neighborhood density and compute differ across arms. Report per-seed +10 MSE,
motion MSE and inference cost before interpreting any differences. No selector
is assumed superior. Preserve the original selector/webcam experiment.
