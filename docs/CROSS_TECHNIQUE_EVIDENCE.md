# PR 57 — Cross-technique evidence without a false leaderboard

This is an **evidence audit**, not a new neural architecture.

The CLI reads a manifest of saved local result JSON files. Each entry
declares its exact nested metric path, technique, representation, benchmark,
unit, horizon, held-out split, seeds and checkpoint-selection protocol.
It verifies metric files by SHA256 and hashes checkpoints when supplied.
Missing artifacts are explicit, not represented with fabricated zeroes.

Only entries sharing ALL comparability fields are grouped together.
For example, a provided-2D-skeleton joint error in pixels is never
ranked against RGB whole-image MSE. Neither can train-set metrics be
mistaken for independent test results.

An example manifest is included for the Step 6 RGB trajectory sweep:

    python -m experiments.evidence_report \
      --manifest experiments/evidence_manifest.example.json \
      --out runs/evidence-report

    python -m unittest tests.test_evidence_report -v

The source artifacts must already exist under runs/. Use --allow-missing
to report which artifacts are absent instead of aborting. Outputs are
evidence.json and report.md, including provenance warnings. The sample
manifest is **schema and command guidance**, not a claim that experiments
have been rerun or that checkpoint contents are verified.

There is no best-model selection or cross-technique universal score.
Maintain separate visual demonstrations and exact experiment protocols.
