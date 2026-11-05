# Source-linked figures for the illustrated research note

The [paper PR](../paper/README.md) initially contains conceptual vector diagrams, **not invented checkpoint output**. Before publication-quality model comparisons, generate real panels with the SAME dataset and forecast horizon.

```bash
python -m experiments.paper_figure_pack \
  --data runs/frozen-comparison/figure-input.npz \
  --provenance runs/frozen-comparison/figure-provenance.json \
  --out runs/paper-figures
```

The input NPZ must contain:
- `observed`: [N,H,W,3], final observed RGB.
- `nca`, `persistence`, `velocity`, `truth`: [N,horizon,H,W,3] normalized RGB.

The provenance JSON must contain checkpoint SHA256, code revision, test scene seeds, generator and protocol. This tool **refuses** missing or inconsistent input rather than inventing placeholder pictures. It writes matched forecast/truth/error panels, lead-wise comparable MSE, hashes and an artifact manifest. The run must still be audited to ensure that its model never saw future truth at inference and the historical checkpoint was selected without final-test leakage.

The app's original RGB galleries and event failure gallery remain the canonical sources for their respective campaigns; don't fuse bright-collapse and black-collapse images from separate checkpoints into a single fake sequence.

After the manuscript is merged, add verified PNGs and their manifest under `paper/figures/` in a follow-on evidence PR, then regenerate the PDF. The report should remain easy for a reader to reproduce rather than accumulating decorative charts.
