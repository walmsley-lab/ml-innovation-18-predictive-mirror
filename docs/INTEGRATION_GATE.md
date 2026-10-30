# PR 18 — CPU integration gate before renting GPU

Run the complete, tiny cross-model integration suite before building an Akash image:

```bash
python -m unittest discover -s tests -v
python -m sweep --steps 2 --batch-size 2 --checkpoint-every 1 \
  --seeds 11 --architectures recurrent feedforward multisheet \
  --curricula staged shuffled --episodes 2 --output runs/integration
python -m sweep --steps 2 --batch-size 2 --checkpoint-every 1 \
  --seeds 11 --architectures recurrent feedforward multisheet \
  --curricula staged shuffled --episodes 2 --output runs/integration --resume
python -m deploy.verify_matrix runs/integration --seeds 11 \
  --architectures recurrent feedforward multisheet --curricula staged shuffled
```

CI also validates imports and these six conditions before publishing images on main. Missing/duplicated run cells, mismatched spec/metrics, or absent checkpoints fail the gate. These checks prove only **CPU integration**, not learning quality, Docker CUDA compatibility, live Akash deployment, public GHCR visibility, off-provider recovery or provider billing.

**Operator-only gates still required**: verify immutable `@sha256` public image, NVIDIA/CUDA launch, `akt sdl validate`, `akt deploy --dry-run`, interactive bids and funding context; perform a **3-step disposable GPU smoke**, export and checksum artifacts, close deployment and read back closed status. Longer paid runs require a fresh-container restore test. Do not launch paid compute from CI or this PR.

PR #17's long-occlusion benchmark may be merged separately before running the substantive campaign.
