# Performance-first research progression (November 2026)

**Principle:** Reproduce what works, improve what does not, understand why. Previous work not being novel is **not** a reason to avoid reproducing it. Keep models independent and run the smallest informative experiment first.

This map tracks **reviewable implementation PRs**, not demonstrated scientific outcomes. The baseline comparison and original reference-parity reproductions require training budgets and verified result artifacts after the code merges.

| Phase | PR | Capability / engineering question | What constitutes a substantive result? |
| --- | --- | --- | --- |
| A | [#197](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/197) | Can RGB NCA overfit one short moving-object sequence without collapse? | Actual in-sample motion forecast better than initialization/persistence |
| B1 | [#198](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/198) | Can local NCAs learn growth and repair? | Working small JAX exercise, followed by faithful Mordvintsev settings and behavioral parity |
| B2 | [#199](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/199) | Can local rules learn time-varying PDE fields? | Working Gray-Scott experiment, followed by reference-faithful Richardson reproduction |
| C | [#200](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/200) | Does +1->+3->+10 curriculum help training? | Held-out gain over equal-budget direct +10 training |
| D | [#201](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/201) | What does conventional ConvLSTM recurrence achieve? | Trained shared-task forecasts; separately run faithful SimVP via [#193](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/193) |
| E | [#202](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/202) | Does history improve reappearance prediction? | Fair frozen RGB-only event comparison including strong classical/learned references |
| F | [#203](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/203) | Is transport + synthesis better than either separately? | Improved location/mass/appearance at a controlled cost, on independent scenes |
| G | [#204](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/204) | Does recurrent state causally improve an actual task? | Responsive positive control and task-specific difference on trained, non-collapsed checkpoint |
| H | [#205](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/205) | When is identity or future ambiguity not observable? | Verified information regimes, then classical and learned prediction/active-sensing comparisons |
| I | [#206](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/206) | Can actual held-out camera forecasts be inspected? | Artifact-linked offline replay gallery; live timing verified separately |
| J | [#207](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/207) | Which approach lies on the accuracy/cost frontier? | Same-device, same-task quality/latency/memory and measured energy where available |
| K | [#208](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/208) | Can the paper use real prediction figures? | Artifact-linked, hash-checked publication galleries and corrected claims |

**Paper:** [#195](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/195) adds the illustrated manuscript and PDF; it uses original conceptual vectors, not fabricated model outputs.

## Merge/review order

1. **Build ability:** A -> B1/B2 -> C. Parallel reference baselines: D after external bridge #193 is reviewed.
2. **Measure actual forecasting:** E after evaluator/classical baseline work #187–194, plus trained D predictions.
3. **Improve mechanism and representation:** F is independent of G; use a working checkpoint from A/C/E when interpreting G.
4. **Extend capability:** H for identity/ambiguity, I for independent real recordings, J for measured costs.
5. **Publish evidence:** K after the paper merges and real figure arrays exist.

## Stop conditions that aid performance rather than block research

- If a fixed-clip NCA cannot learn, repair gradients/readout before running long generalization campaigns.
- If a fair conventional method dominates on one task, **learn from it**, retain it as the reference, and revise the specific NCA hypothesis; do not delete cellular work wholesale.
- If an intervention assay has a dead positive control, improve the assay/model; don't interpret a no-change result as proving a mechanism unnecessary.
- If compute cost overwhelms accuracy gains, report the tradeoff and explore different state/transport representations or sparse updates before adding more dense recurrence.
- Don't demand p-values or a broad literature review in every small PR. Use lightweight smoke tests for development, and robust replication/uncertainty only when making general scientific claims.

**Result status:** All listed performance steps are implementation branches with tests and CLI entry points. No large reference-compatibility reproduction, complete model parity campaign, or new performance win is asserted merely by creating these PRs.
