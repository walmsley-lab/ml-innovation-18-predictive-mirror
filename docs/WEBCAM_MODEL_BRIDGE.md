# Inspect cloud-trained RGB models through the verified webcam viewer

The synthetic RGB campaign may select different model families. Specify a
**matching model family and checkpoint**, rather than passing CNN/transport or
blend-head weights to the old plain-NCA viewer. Supported:
`nca`, `nca_blend`, `cnn`, `transport`.

All models see only observed RGB history. The application matches each
already-issued forecast to a subsequent camera observation and measures timing.
It does **not** establish that synthetic training transferred to real video.

## Run locally (from repository root)

```bash
python -m pip install opencv-python
# Replace this path with a *verified* recovered cloud checkpoint:
MODEL=runs/akash-rgb-full/artifacts/conditions/ladder-nca_blend-mixed-s11/nca_blend-seed11-mixed-best.npz

env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda \
 python -m experiments.rgb_verified_live \
  --model nca_blend --parameters "$MODEL" \
  --camera 0 --fps 10 --size 32 --context 8 --lead 10 --ticks 2 \
  --out runs/webcam-cloud-blend/metrics.json
```

For a standalone recording, capture **two independent** local videos with
`python -m experiments.rgb_nca capture`; independently score a frozen model:

```bash
python -m experiments.rgb_transfer_eval \
  --model nca_blend --checkpoint "$MODEL" \
  --recordings recordings/webcam-test.npz \
  --exclude-training recordings/webcam-train.npz \
  --out runs/webcam-transfer/metrics.json
```

The exclusion hash rejects literally identical source recordings. The user is
responsible for ensuring the model has never trained on the separate test
recording or near-duplicates. Cloud **synthetic-only** models have not trained
on either local video. Current webcam `rgb_nca fit` trains only the plain NCA
and is not a fine-tuner for the selected cloud model.

The live verifier's time horizon is measured from a *committed* forecast to
later camera frames; a +10 frame forecast is not inherently one wall-clock
second. Check median inference and measured horizon duration. Benchmarks on
synthetic scenes and on independent real videos must be reported separately.
