"""Compare matched-weight JAX-jit and PyTorch-eager RGB NCA on CPU.

Only times warmed forward inference and loss+backward, *not* data loading,
optimizer updates or compilation. Performance is hardware/backend dependent.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from experiments.torch_rgb_nca import RGBNCA
from src import predictive_rgb_nca as jax_nca


def measure(fn, repeats):
    for _ in range(2):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(samples)), "min_ms": float(np.min(samples))}


def run(size=16, batch=1, context=4, horizon=3, ticks=1, channels=8,
        repeats=5, threads=1):
    torch.set_num_threads(threads)
    params = jax_nca.init(jax.random.PRNGKey(19), channels)
    # A nonzero readout prevents the persistence initialization from obscuring
    # forward parity and makes the benchmark exercise learned corrections.
    params = dict(params, rgb_readout=jnp.full((channels, 3), 0.013, dtype=jnp.float32))
    model = RGBNCA(channels).load_jax_params(params)
    model.eval()
    examples = np.random.default_rng(19).random(
        (batch, context + horizon, size, size, 3), dtype=np.float32)
    context_np = examples[:, :context]
    j_context, j_examples = jnp.asarray(context_np), jnp.asarray(examples)
    t_context, t_examples = torch.from_numpy(context_np), torch.from_numpy(examples)
    # Pass runtime arrays to jit; capturing them as constants can make timings invalid.
    compiled_forward = jax.jit(lambda p, x: jax_nca.rollout(p, x, horizon, ticks))
    compiled_grad = jax.jit(lambda p, x: jax.value_and_grad(jax_nca.loss)(
        p, x, context, horizon, ticks))

    def jax_forward():
        jax.block_until_ready(compiled_forward(params, j_context))

    def jax_backward():
        value, grads = compiled_grad(params, j_examples)
        jax.block_until_ready((value, grads))

    def torch_forward():
        with torch.no_grad():
            model(t_context, horizon, ticks)

    def torch_backward():
        model.zero_grad(set_to_none=True)
        model.loss(t_examples, context, horizon, ticks).backward()

    parity = float(np.max(np.abs(
        np.asarray(compiled_forward(params, j_context)) -
        model(t_context, horizon, ticks).detach().numpy())))
    report = {
        "shape": [batch, context, size, size, 3], "horizon": horizon,
        "ticks": ticks, "channels": channels, "repeats": repeats,
        "torch_threads": threads, "max_abs_forecast_error": parity,
        "frameworks": {"jax": jax.__version__, "torch": torch.__version__},
        "hardware": {"cpu": platform.processor() or platform.machine(),
                     "jax_devices": [str(d) for d in jax.devices()], "torch_device": "cpu"},
        "comparison": {"jax_jit_forward": measure(jax_forward, repeats),
                       "torch_eager_forward": measure(torch_forward, repeats),
                       "jax_jit_loss_grad": measure(jax_backward, repeats),
                       "torch_eager_loss_backward": measure(torch_backward, repeats)},
        "limits": "CPU only, warm JAX JIT vs eager PyTorch, no compiler/optimizer/data loading; "
                  "not evidence of universal framework superiority",
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--ticks", type=int, default=1)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    report = run(args.size, args.batch, args.context, args.horizon,
                 args.ticks, args.channels, args.repeats, args.threads)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
