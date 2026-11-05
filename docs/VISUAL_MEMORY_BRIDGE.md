# Visual prequential adapter: roadmap 32–35 integration

After reviewing the frozen checkpoint evaluation bridge, run `python -m experiments.visual_memory_bridge --nca-checkpoint X --cnn-checkpoint Y --transport-checkpoint Z`. `--smoke` is an explicitly **untrained** functional test. This runner is stacked on the frozen evaluation PR.

The **model weights stay frozen**. A simple per-channel visual output-bias adapter is adjusted after the real target is observed and the previously committed prediction scored. Compare frozen, recent-only (two updates), and replay (two updates, one prior labeled sample). Every mode receives the same RGB observation stream and each adaptive mode has equal gradient count. Report model parameter count separately from adapter size and memory bytes. The staged conditions A (ordinary motion), B (appearance plus exposure drift), A-return test short-run recovery. This *does not* establish useful end-to-end NCA plasticity or robust lighting invariance.

A separate known camera-action postprocessing control checks whether supplied commands help a frozen RGB forecaster; unknown actions cannot be set to zero and called a stationary camera. Validation-only `visual_gate` accepts predeclared per-domain visual error thresholds; it never sees test labels. Future follow-up needs genuine independent validation experiments to drive the gate, not only toy thresholds.

For confirmatory studies: multiple checkpoint seeds, independent recording sessions, matched total optimization/compute, prequential per-frame exports, observability flags for delayed/missing sensors, and task-specific occlusion metrics. Keep negative results. No performance win is claimed for this small integration experiment.
