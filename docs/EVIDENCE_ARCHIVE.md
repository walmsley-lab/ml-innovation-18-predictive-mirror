# Evidence archiving: files, not assertions

Use this utility to package a **new** measured result with exact locally available files, SHA256 integrity checks and environment metadata. A recorded Git SHA does not prove the checkout was clean; the report also records a dirty-tree flag when available. A successful hash check is **not** a scientific replication or an audit of an earlier operator report.

Example (do not silently substitute a smoke run for the historical benchmark):

```bash
python -m experiments.rgb_motion_distribution --out runs/fresh-motion
python -m experiments.archive_evidence \
  --metrics runs/fresh-motion/metrics.json \
  --checkpoint runs/fresh-motion/seed-11-distribution.npz \
  --out runs/evidence-motion-20261105 \
  --label fresh-reproduction \
  --command 'python -m experiments.rgb_motion_distribution --out runs/fresh-motion' \
  --selection 'best validation MSE, including step zero'
python -m experiments.archive_evidence --verify runs/evidence-motion-20261105
```

Include **all** checkpoints used in a displayed aggregate, the exact training environment, final-test seed ranges, source revision and plotting inputs. The utility copies declared files; it does not infer unlisted checkpoints, capture every environment variable, rerun commands, validate evaluation splits or prove that original measurements exist.

Labels:

- `fresh-reproduction`: newly generated result with declared files (not certified to match historical results).
- `historical-unverified`: locally available historical files without certified original execution provenance.
- Missing historical runs must remain **reported-only / artifact unavailable** in prose and the campaign provenance audit. Never populate an archive with simulated numbers.

Treat command strings as potentially sensitive: review for embedded tokens before sharing an archive. The existing `experiments/evidence_report.py` groups only protocol-comparable metrics; this tool answers the separate question of *whether bytes and environment are traceable*.
