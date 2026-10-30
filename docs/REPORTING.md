PR 24 — Evidence-only report generation

Run after actual training and evaluation:

```
python -m report --sweep runs/campaign --out reports/results.md
python -m report --sweep runs/campaign --feedback runs/feedback-control.json --out reports/results.md --figure reports/feedback.svg
```

The generator refuses empty or duplicate results, treats zero qualifying reappearance events as NA, and computes seed-paired descriptive curriculum differences without claiming statistical significance. Optional SVG uses distinct scales for visible-motion MSE and occlusion-reappearance MAE; it requires real feedback-control JSON, never fabricated test numbers. Full conclusions require confidence intervals across independent seeds, image/source provenance, all negative results and confirmation that interventions have matched active capacity and compute. Neural Dynamics #17 retains the separate biological evidence ledger. No actual training outcomes are included in this PR.
