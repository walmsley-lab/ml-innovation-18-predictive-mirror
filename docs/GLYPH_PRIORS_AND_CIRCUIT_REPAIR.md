# Future capability: hand-authored glyph priors and circuit interventions

**Motivation.** A camera predictor can learn local motion yet fail to identify
a stable symbol when its appearance changes. Recognizing letters and numbers
would add a testable representation capability, but it must not be confused
with multi-horizon pixel prediction.

**Order of work (not implemented in this PR):**

1. **Recognize before repairing.** Use simple generated glyphs (0–9, A–Z)
   with identity labels, plus held-out fonts, stroke widths, rotations, scale,
   backgrounds, noise, partial occlusions, and moving glyphs. Split by
   rendering family AND glyph instances, not random frames.
2. **Prior comparison.** Compare (a) no prior, (b) explicit fixed stroke/edge/
   topology filters, (c) hand-authored low-resolution glyph templates, and
   (d) matched-size learned filters or randomly initialized templates.
   All arms get equal downstream training data, compute, and representation
   capacity. Label hand-provided identity explicitly.
3. **Prediction coupling.** Report *both* identity accuracy and actual
   frame-prediction skill (+1/+3/+5/+10); require improvements on unseen
   instances, not only digits resembling the hand-authored prototypes.
   Test erasure/ablation of the prior after training and identity swaps.
4. **Later: targeted circuit repair.** Freeze a proven model with a reproducible
   perception failure. Compare a bounded weight-space edit with gradient
   finetuning, no edit, matched random edit, and retraining. Probe collateral
   regressions, seeds, agent tokens/cost, and scene generalization.
   Do not implement direct weight editing before discovering a reusable
   mechanism and measuring its causal necessity.

This is **developmental scaffolding**, not an innate alphabet claim or a
biology-equivalence claim. Glyph classification may be a useful side task,
but never a substitute for forecasting unseen frames.

Related exploratory work: https://rikiyat.github.io/blog/weight-space-programming/
