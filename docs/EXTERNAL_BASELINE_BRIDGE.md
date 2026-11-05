# External video-predictor benchmark bridge

Export exactly the paired-history synthetic world into **framework-neutral NPZ files** for credible recurrent (ConvLSTM) and recurrent-free (SimVP/OpenSTL) baselines, without introducing PyTorch dependencies into the JAX core.

```bash
python -m experiments.external_video_bridge --out runs/paired-external
python -m experiments.external_video_bridge --out runs/paired-external --verify
```

The exported `train.npz`, `validation.npz`, and `test.npz` contain **only** two float32 arrays:

- `inputs`: [N,8,3,32,32], observed RGB context, values in [0,1].
- `targets`: [N,10,3,32,32], supervised future targets, values in [0,1].

These are conventional [batch,time,channel,height,width] arrays suitable for external loaders. The `manifest.json` lists exact split roots, scene-pair IDs, scene counts, protocol and SHA256 of each frozen file.

**No future leakage:** The on-disk file contains future targets because supervised training and scoring require them. **Inference adapters must forward only `inputs` to the model**. Keep `targets` out of the model call. The experiment cannot technically prevent an external script from cheating; inspect external model inference code.

External model output should be written as `predictions.npz` with exactly one array `predictions` of shape [N,10,3,32,32], in the original dataset order. To score:

```bash
python -m experiments.external_video_bridge --out runs/paired-external \
  --score path/to/predictions.npz --split test
```

The scorer re-creates evaluator-only events, verifies the input dataset hash, and reports reappearance success, paired future-contrast error and RGB MSE. Perfect future output must score 100% and zero MSE. An external reference's training code, checkpoint selection, test-set access, capacity and hardware costs remain its own provenance obligations, not validated by this bridge.

**Comparator guardrails:**

- A 32x32 RGB world score is not interchangeable with published Moving-MNIST benchmark values. Do not reuse leaderboard MSE, PSNR or FPS without the full normalization and protocol.
- SimVP is not the same as a generic small CNN. Run a reference-faithful official model or explicitly label the approximation `SimVP-like`.
- Separate matched-resource comparisons from adequately trained competence references.
- Group both opposing histories of a given scene seed into **one** cluster for intervals. Reappearing position is not identity preservation; probabilistic ambiguity remains a separate task.
- Do not claim an external ConvLSTM/SimVP benchmark has been **run** merely because this exporter works.
