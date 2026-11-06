"""A compact recurrent neural sensorimotor model with compositional commands."""
import jax
import jax.numpy as jnp

from src.embodied_arithmetic_world import NODES


def init(key, size=8, width=24):
    if size < 8 or width < 4:
        raise ValueError("Invalid model dimensions")
    dim = size * size * 3 + 4
    a, b, c = jax.random.split(key, 3)
    feature_dim = dim + 4 + 2 + 2 + 1 + width
    return {"w": .05 * jax.random.normal(a, (feature_dim, width)),
            "b": jnp.zeros((width,)),
            "head": .01 * jax.random.normal(b, (width, dim)),
            "out_bias": jnp.zeros((dim,)),
            "shape_head": .04 * jax.random.normal(c, (width, size * size)),
            "shape_bias": jnp.zeros((size * size,)),
            # Learned algebra: zero, successor offset/gain,
            # negation gain/offset, addition left/right/offset.
            "ops": jnp.asarray([0., .5, .8, -.8, 0., .65, .65, 0.])}


def command_value(ops, tree):
    """Bottom-up differentiable interpretation of a bounded syntax tree."""
    def step(i, values):
        op, left, right = tree[i]
        a = values[jnp.clip(left, 0, NODES - 1)]
        b = values[jnp.clip(right, 0, NODES - 1)]
        value = jnp.where(op == 1, ops[0],
                jnp.where(op == 2, ops[1] + ops[2] * a,
                jnp.where(op == 3, ops[3] * a + ops[4],
                          ops[5] * a + ops[6] * b + ops[7])))
        return values.at[i].set(jnp.where(op == 0, 0., value))
    memory = jax.lax.fori_loop(0, NODES, step, jnp.zeros((NODES,)))
    index = jnp.maximum(0, jnp.sum(tree[:, 0] > 0) - 1)
    return jnp.clip(memory[index], -8., 8.)


def sensor_gates(mask, size):
    """Four modality flags: RGB, proprioception(2), touch, vibration."""
    vision = jnp.broadcast_to(mask[..., :1],
                               (*mask.shape[:-1], size * size * 3))
    fingers = jnp.broadcast_to(mask[..., 1:2],
                                (*mask.shape[:-1], 2))
    return jnp.concatenate((vision, fingers, mask[..., 2:]), axis=-1)


def rollout(params, observed, tree, axes, stage, mask, size=8):
    """Only current/past observations and current command inform next forecast."""
    if observed.ndim != 3 or observed.shape[-1] != size * size * 3 + 4:
        raise ValueError("Expected [batch,time,RGB+4] observations")
    batch, time, _ = observed.shape
    if (tree.shape != (batch, time, NODES, 3) or
            axes.shape != stage.shape != (batch, time, 2) or
            mask.shape != (batch, time, 4)):
        raise ValueError("Incompatible sensory and program tensors")
    commands = jax.vmap(jax.vmap(lambda x: command_value(
        params["ops"], x)))(tree)
    gated = observed * sensor_gates(mask, size)
    width = params["b"].shape[0]

    def update(hidden, inputs):
        sensor, flags, axis, domain, command = inputs
        features = jnp.concatenate((sensor, flags, axis, domain,
                                    command[:, None], hidden), axis=-1)
        hidden = .58 * hidden + .42 * jnp.tanh(
            features @ params["w"] + params["b"])
        predicted = sensor + hidden @ params["head"] + params["out_bias"]
        contour = jax.nn.sigmoid(hidden @ params["shape_head"] +
                                 params["shape_bias"])
        return hidden, (predicted, contour)

    _, (forecast, shape) = jax.lax.scan(
        update, jnp.zeros((batch, width)),
        (jnp.swapaxes(gated, 0, 1), jnp.swapaxes(mask, 0, 1),
         jnp.swapaxes(axes, 0, 1), jnp.swapaxes(stage, 0, 1),
         jnp.swapaxes(commands, 0, 1)))
    return jnp.swapaxes(forecast, 0, 1), \
        jnp.swapaxes(shape, 0, 1).reshape(batch, time, size, size), commands


def loss(params, observed, tree, axes, stage, mask, target, shape,
         motor, size=8):
    forecast, contour, commands = rollout(
        params, observed, tree, axes, stage, mask, size)
    visual = size * size * 3
    rgb = jnp.mean((forecast[..., :visual] - target[..., :visual]) ** 2)
    proprio = jnp.mean((forecast[..., visual:visual+2] -
                        target[..., visual:visual+2]) ** 2)
    touch = jnp.mean((forecast[..., visual+2] - target[..., visual+2]) ** 2)
    vibration = jnp.mean((forecast[..., visual+3] - target[..., visual+3]) ** 2)
    touch_domain = stage[..., 1]
    shape_loss = jnp.sum(((contour - shape) ** 2).mean(axis=(-1, -2)) *
                         touch_domain) / jnp.maximum(1., touch_domain.sum())
    grammar = jnp.mean((commands - motor) ** 2)
    return rgb + .5 * proprio + .5 * touch + .3 * vibration + \
        .4 * shape_loss + .1 * grammar
