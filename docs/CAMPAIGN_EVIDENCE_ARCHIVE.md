# Campaign evidence archive

This is a **strict importer**, not an assertion that the two past campaigns
have already been archived or replicated. Obtain the original local
`scratchpad/campaign/perceptual` and `scratchpad/campaign/gaussian`
directories from the operator; each must contain its `metrics.json`
and all referenced `*.npz` best checkpoints. Never regenerate missing
historical checkpoints and present them as the originals.

```bash
python -m experiments.campaign_archive archive \
  --kind perceptual --source scratchpad/campaign/perceptual \
  --out runs/evidence-archive/perceptual \
  --source-commit <EXACT_ORIGINAL_40_HEX_COMMIT>
python -m experiments.campaign_archive archive \
  --kind gaussian --source scratchpad/campaign/gaussian \
  --out runs/evidence-archive/gaussian \
  --source-commit <EXACT_ORIGINAL_40_HEX_COMMIT>
python -m experiments.campaign_archive verify \
  --archive runs/evidence-archive/perceptual
```

Each archival result row links to a copied, SHA-256-verified checkpoint;
the original `metrics.json` is hashed and preserved. A deterministic
representative RGB context, forecast and later truth are generated from
one checkpoint per arm/curriculum (Gaussian: one seed, all three modes)
and stored in `examples/*.npz` with hashes. These example forecasts
are **replayed from the current installed source**, not claimed to be
the original campaign forecasts. The complete per-seed metrics remain
in the hash-linked source `metrics.json`.

**Important provenance limits:** exact launch command, original source
commit, package versions, hardware, original wall-clock protocol and
original checkpoint-saving policy cannot be verified from the old
metric files alone. `source_commit_claimed` is an operator attestation
until separately corroborated by immutable run logs. The archive
explicitly sets `commit_provenance_verified=false`. Width/tick
settings used for replay must be supplied when nondefault. A run
without all original checkpoint files fails closed; `--no-examples`
is for metadata-only troubleshooting and does not satisfy the
representative-forecast gate.

All materials reside in `runs/` (gitignored), not embedded in a PR.
For external review, publish a sanitized read-only archive with its
`manifest.json`, along with a permanent URL and checksum. This
repository does not contain the local operator's historical files.
