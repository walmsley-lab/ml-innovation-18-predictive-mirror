# PR 2 — Durable Akash training

Run a local interruption/restart check:

```bash
python -m unittest discover -s tests -v
python -m train --steps 2 --batch-size 2 --checkpoint-every 1 --output runs/test
python -m train --steps 4 --batch-size 2 --checkpoint-every 1 --resume --output runs/test
```

The periodic `checkpoint.npz` includes model weights, AdamW first/second moments, completed step, seed and batch size. Since world batches derive deterministically from `seed + step`, no mutable data-sampler PRNG state is required. A resumed run must match an uninterrupted run to test tolerance. A GPU/CPU change may introduce small numeric differences.

Akash serves the synthetic `runs/` directory **during training** on port 8080; periodically download `checkpoint.npz` to trusted storage using its displayed URL, save a SHA-256 checksum, and verify restore on a different container before a long paid job. Serving is not itself off-provider backup. Treat the endpoint as public and never put private data there. Close and confirm the lease closed after exporting results.

Image build, digest pinning, GPU driver validation and live Akash bids remain pending, and no paid deployment has been performed.
