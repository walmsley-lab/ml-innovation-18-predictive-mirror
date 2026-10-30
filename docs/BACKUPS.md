# Checksum-verified Akash exports (manual, before closing any lease)

This helper runs **on your computer**, not inside the paid training container. It only pulls explicitly selected **synthetic** artifact files from the public HTTP endpoint. A published checkpoint is atomic, with an accompanying `.sha256` sidecar; the exporter validates the remote digest before saving an archive with its own manifest.

```bash
# Single training run: replace SERVER with actual Akash service URL
python -m deploy.artifacts export --url "https://SERVER/" --out backups/smoke \
  --path pilot/checkpoint.npz --path pilot/metrics.json --path pilot/parameters.npz

# In-progress sweep checkpoint: replace path with the actual active run
python -m deploy.artifacts export --url "https://SERVER/" --out backups/sweep1 \
  --path sweep/recurrent-staged-seed31/checkpoint.npz \
  --path sweep/recurrent-staged-seed31/spec.json

# Verify local backup and restore into a fresh runs/ tree
python -m deploy.artifacts restore --archive backups/sweep1 --dest runs/
```

For a new Akash container, mount/copy the restored `runs/` tree before starting with `RESUME=1`; changing `SOURCE_REVISION`, model configuration or checkpoint budget will be rejected by the sweep. Backup must be downloaded **off-provider** before lease termination. These commands have been tested using a local `file://` source in unit tests; real Akash HTTP and GPUs have **not yet been tested**.

The exporter requires exact paths; it does not scan an active sweep automatically. Backup while training at meaningful intervals, not only at completion. After verifying exports, close and read back the Akash deployment state. No credentials, webcam video, or private data should be put on the public synthetic-artifact server.
