"""Paired particle-NCA pilot: message-passing versus isolated particle cells.

This is a structured-position experiment, not an RGB object detector. Future
positions are never model inputs, only labels and evaluation truth.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import particle_nca as nca, rgb_motion_world
from train import adamw


def dataset(seeds, context=8, horizon=10, size=32, particles=2):
    """Generate trajectories; return observed position tensor, not RGB."""
    return np.stack([
        rgb_motion_world.episode(int(seed), steps=context+horizon,
                                 size=size, balls=particles)["positions"] / (size+16)
        for seed in seeds]).astype(np.float32)


def controls(context, horizon):
    """No privileged velocity; estimate from the two last observed positions."""
    last = context[:, -1:]
    persistence = np.repeat(last, horizon, axis=1)
    step = context[:, -1]-context[:, -2]
    velocity = np.stack([np.clip(last[:, 0]+(lead+1)*step, 0, 1)
                         for lead in range(horizon)], axis=1)
    return {"persistence": persistence, "constant_velocity": velocity}


def score(predictions, truth, leads, world_pixels=48):
    result = {}
    for lead in leads:
        offset = predictions[:, lead-1]-truth[:, lead-1]
        error = np.linalg.norm(offset, axis=-1)*world_pixels
        result[str(lead)] = {
            "position_mae_px": float(error.mean()),
            "position_mse_px2": float(np.mean(error**2)),
            "particle_targets": int(error.size)}
    return result


def render_gif(context, predictions, truth, output, world_pixels=48):
    """Optional position-only animation. Ground truth is display-only."""
    from PIL import Image, ImageDraw
    labels = ("Last observed", "Particle NCA", "Velocity", "Actual future")
    last = context[-1]
    velocity = controls(context[None], len(truth))["constant_velocity"][0]
    colors = [(237, 109, 91), (106, 196, 232), (244, 202, 110),
              (163, 135, 218)]
    frames = []
    for t in range(len(truth)):
        canvas = Image.new("RGB", (4*184, 204), (18, 27, 39))
        draw = ImageDraw.Draw(canvas)
        positions = (last, predictions[t], velocity[t], truth[t])
        for col, (label, coordinates) in enumerate(zip(labels, positions)):
            x0 = col*184
            draw.text((x0+8, 8), label, fill=(232, 238, 240))
            draw.rectangle((x0+8, 28, x0+176, 196), outline=(72, 93, 107))
            for particle, (x, y) in enumerate(coordinates):
                px = x0+8+int(np.clip(x, 0, 1)*168)
                py = 28+int(np.clip(y, 0, 1)*168)
                draw.ellipse((px-5, py-5, px+5, py+5),
                             fill=colors[particle % len(colors)])
        frames.append(canvas)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(destination, save_all=True, append_images=frames[1:],
                   duration=130, loop=0)
    return str(destination)


def run(output="runs/particle-nca-motion", seeds=(11, 23, 37), steps=100,
        batch_size=4, heldout=16, context=8, horizon=10,
        size=32, particles=2, channels=12, preview=False):
    if (not seeds or steps < 1 or batch_size < 1 or heldout < 1
            or context < 2 or horizon < 1 or size < 16 or particles < 1):
        raise ValueError("Invalid particle or training settings")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    leads = tuple(sorted({h for h in (1, 3, 5, 10, horizon) if h <= horizon}))
    report_rows = []
    for seed in seeds:
        initial = nca.init(jax.random.PRNGKey(seed), channels)
        arms = {"messages": initial, "no_messages": initial}
        first = {name: jax.tree.map(jnp.zeros_like, initial) for name in arms}
        second = {name: jax.tree.map(jnp.zeros_like, initial) for name in arms}

        def make_update(use_messages):
            @jax.jit
            def update(params, m, v, frames, i):
                error, grad = jax.value_and_grad(nca.loss)(
                    params, frames, context, horizon, use_messages)
                norm = jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(grad)))
                grad = jax.tree.map(lambda g: g/jnp.maximum(norm, 1.), grad)
                params, m, v = adamw(params, grad, m, v, i)
                return params, m, v, error
            return update
        updates = {"messages": make_update(True),
                   "no_messages": make_update(False)}

        for i in range(1, steps+1):
            sample_seeds = 100_000+seed*100_000+i*batch_size+np.arange(batch_size)
            batch = jnp.asarray(dataset(sample_seeds, context, horizon, size, particles))
            for name in arms:
                arms[name], first[name], second[name], loss = updates[name](
                    arms[name], first[name], second[name], batch, i)
            if i == steps or i % max(1, steps//5) == 0:
                print(f"seed={seed} step={i} particle_loss={float(loss):.6f}",
                      flush=True)

        eval_seeds = 60_000_000+seed*1000+np.arange(heldout)
        heldout_frames = dataset(eval_seeds, context, horizon, size, particles)
        prefix = heldout_frames[:, :context]
        future = heldout_frames[:, context:context+horizon]
        predictions = controls(prefix, horizon)
        for name, enabled in (("messages", True), ("no_messages", False)):
            predictions[name] = np.asarray(nca.rollout(
                arms[name], jnp.asarray(prefix), horizon, enabled))
            np.savez_compressed(folder/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray, arms[name]))
        metrics = {name: score(forecast, future, leads, size+16)
                   for name, forecast in predictions.items()}
        case = {"seed": int(seed), "metrics": metrics}
        if preview:
            case["preview"] = render_gif(
                prefix[0], predictions["messages"][0], future[0],
                folder/f"seed-{seed}-preview.gif", size+16)
        report_rows.append(case)
    result = {
        "experiment": "structured_position_particle_nca",
        "representation": "provided tracked particle positions, not RGB",
        "seeds": list(map(int, seeds)), "train_steps": steps,
        "context": context, "horizon": horizon, "particles": particles,
        "size": size, "leads": list(leads), "runs": report_rows,
        "caveats": "Independent structured-input pilot. Particle identities and "
                   "2D coordinates are supplied by synthetic world. "
                   "Model never sees RGB pixels or future coordinates during inference. "
                   "The distance-based interaction includes additional compute; "
                   "not a controlled parameter-count-versus-runtime comparison. "
                   "Position error cannot be directly equated with RGB image MSE."}
    (folder/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/particle-nca-motion")
    p.add_argument("--seeds", nargs="+", type=int, default=(11, 23, 37))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--heldout", type=int, default=16)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--particles", type=int, default=2)
    p.add_argument("--channels", type=int, default=12)
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()
    report = run(a.out, a.seeds, a.steps, a.batch_size, a.heldout, a.context,
                 a.horizon, a.size, a.particles, a.channels, a.preview)
    print(json.dumps({"output": str(Path(a.out)/"metrics.json"),
                      "runs": len(report["runs"])}, indent=2))


if __name__ == "__main__":
    main()
