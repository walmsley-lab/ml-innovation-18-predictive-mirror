# Akash image preflight — one disposable GPU smoke

**No paid deployment is created by these helpers.** The current `deploy/akash-gpu.yaml` uses a deliberately unpublished placeholder tag. A successful GitHub Actions push build publishes a `sha-<commit>` tag, but a successful image build alone does not prove *public* pullability or its immutable digest.

1. After your reviewing agent merges PR #14, identify the published image for the relevant `main` commit and inspect it from a clean Docker environment:

   ```bash
   docker buildx imagetools inspect ghcr.io/walmsley-lab/ml-innovation-18-predictive-mirror:sha-<full-main-commit>
   ```

   Copy the **top-level digest** printed by the inspection command (a 64-digit SHA-256). If the package requires login or cannot be pulled anonymously, fix GHCR package visibility first. For an actual NVIDIA host, also validate CUDA/JAX driver compatibility.

2. Generate the **GPU-only 3-step smoke** locally without changing the checked-in SDL:

   ```bash
   python -m deploy.akash_preflight \
      --image ghcr.io/walmsley-lab/ml-innovation-18-predictive-mirror@sha256:<digest> \
      --out runs/akash-smoke.yaml
   akt sdl validate runs/akash-smoke.yaml
   akt deploy runs/akash-smoke.yaml --dry-run
   ```

   The renderer refuses mutable/placeholder images, `STEPS > 3`, `BATCH_SIZE > 2`, price bids over 2500 uact/block, non-GPU manifests and ambiguous fields. It sets `RUN_MODE=train`, `LOSS_MODE=plain`, `CHECKPOINT_EVERY=1`, `REQUIRE_GPU=1`, and `RESUME=0` where available. A dry run does **not** verify funding, live availability, provider drivers or image startup.

3. Optional before cloud: run the manual GitHub Actions workflow **Published GPU image CPU preflight** with the same `@sha256:` image string. It anonymously pulls the image and verifies that CPU mode works and GPU-required mode rejects a CPU-only host. This checks the published image, **not** NVIDIA GPU execution.

4. **Only then:** `akt deploy runs/akash-smoke.yaml`, review bids interactively, inspect price and GPU model, note deployment dseq and provider. Stream logs with `akt provider lease-logs <dseq> --follow` (or `akt console logs <dseq> trainer --follow` for managed Console). Confirm `metrics.json` identifies JAX GPU and artifacts are available. Export/checkpoint off-provider. **Close immediately** with `akt close <dseq>` and verify closed state using `akt query deployment <dseq>` (or Console deployment state). An endpoint that keeps serving files keeps incurring costs until the lease is closed. A failed deployment after creating paid state must also be closed.

5. Before a substantial sweep, review PRs #11 and #12 for interruption-safe training and artifact checksum export, then run a small benchmark with actual GPU, one seed/architecture at a time. No sustained training until the backup/restore gate passes.

The bid at 2500 uact/block corresponds to approximately $1.50/hr under ~6-second blocks; a provider's actual bid may be lower. This is an upper bid, **not** an automatic cost cap. Console credits and on-chain escrow have different funding mechanics; record which context you use. Do not place secrets or private webcam footage on the public artifact server.

Source: [Akash deployment CLI documentation](https://akash.network/docs/developers/deployment/akt/deployments/).
