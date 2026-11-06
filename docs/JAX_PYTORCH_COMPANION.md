# JAX vs PyTorch: one predictive RGB NCA

## Purpose

Compare one actual model, not two research programs. The primary implementation remains src/predictive_rgb_nca.py (JAX). The optional companion at experiments/torch_rgb_nca.py is an idiomatic PyTorch nn.Module using nn.Conv2d. It does not change default training, runtime dependencies, or the live mirror.

Both implement the same 3x3 shared cell update, gated hidden state, observed prediction-error correction, autoregressive forecasting, motion-weighted loss, and optional cell activity. Public arrays use [batch, time, height, width, RGB]. Torch uses NCHW internally and transposes JAX HWIO weights for parity. The Torch model can load JAX weights, including trained NPZ dictionaries, but is *not* a full checkpoint/optimizer importer.

## Reproduce

Install the repository JAX/NumPy dependencies first, then separately install a compatible CPU PyTorch wheel (PyTorch is optional for the rest of this repository).

    python -m unittest tests.test_torch_rgb_nca -v
    python -m experiments.compare_torch_jax_rgb_nca --repeats 10
    python -m experiments.compare_torch_jax_rgb_nca --size 32 --context 8 --horizon 10 --ticks 2 --channels 12 --repeats 10 --out runs/jax-torch/metrics.json

Run the two benchmarks on an otherwise idle machine. The latter configuration is substantially more expensive, especially JAX compilation. JAX compile time is excluded; PyTorch uses eager mode, not torch.compile. This is a comparison of selected execution paths, NOT an intrinsic framework speed ranking.

The script reports max absolute output error, median and minimum milliseconds for warmed inference and loss+gradient, framework versions, device labels, shape, and repeat count. It passes runtime data/parameters as JIT arguments to avoid accidental whole-computation constant folding, uses float32 and matched weights, synchronizes JAX before stopping the clock, and fixes Torch intra-op threads. It does not benchmark data-loading, optimizer steps, compilation cost, GPU execution, or an identical lower-level convolution kernel. For a throughput claim, add a matched GPU comparison and compile-time/steady-state accounting first.

## Readability comparison

- JAX functions make the mathematical model and transformations explicit. jax.lax.scan is a concise recurrent rollout, at the price of less familiar control flow.
- PyTorch names layers with nn.Conv2d and makes observation/dream iterations regular Python loops. Its parameter ownership and standard training ecosystem are familiar, at the price of module boilerplate, layout changes, and more implicit mutable parameter state.
- Lines of code are descriptive, not a quality metric; compare whether a student can identify the observation boundary, prediction-error input, hidden update, and future-leakage constraints.
- This is deliberately one optional companion experiment, not a PyTorch migration.

## Retrospective (November 2026)

The original exercise was a small, demonstrable predictive mirror. The current project has become a broad and useful *perceptual systems laboratory*, but research branches grew faster than a repeatable, compelling forecast demo. The most valuable lessons to date are about strong classical controls, output-collapse mechanisms, local information propagation, causal probes, and the danger of measuring the wrong behavior. In particular, the published synthetic NCA +10 forecast beats persistence but not image-derived constant velocity. Later, separately reported campaigns found no demonstrated advantage from more complex cellular/Gaussian approaches at the limited budgets tested; those runs require original artifact and provenance checks before numerical claims are promoted.

The next allocation of effort should be: (1) a visually convincing held-out RGB forecast with fixed data split and an independent camera replay, (2) reproduction/competitive training of compact CNN and ConvLSTM or transport references, (3) only then carefully budgeted NCA/topology/harmonic interventions on trained models that actually have the tested capability, and (4) optional deployment/hardware evaluation after a quality target is achieved. Keep experiments separate and preserve negative results, but avoid making every hypothesis a prerequisite for the demo.

Relevance beyond vision: this develops habits for studying observability, recurrence, state estimation, information routing, stability and controlled interventions. Those habits transfer to neural/connectome modeling, neuromorphic substrate design and cardiac-state estimation. A pixel-grid NCA is NOT evidence about biological connectomes or patient-specific electrophysiology; those applications require their own data, models, constraints and validation.
