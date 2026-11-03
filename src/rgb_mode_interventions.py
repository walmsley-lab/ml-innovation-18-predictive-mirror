"""Matched-energy causal interventions on fixed RGB NCA spatial modes.

No future observations, parameter updates or hard-coded temporal oscillators.
"""
import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as nca
from src.rgb_hidden_interventions import warm_state

MODES = ("intact", "suppress_low_modes", "matched_random")


def random_modes(height, width, count, seed):
    """Zero-mean, orthonormal random spatial subspace of equal rank."""
    if count < 1 or count >= height * width:
        raise ValueError("Need 0 < count < number of sites")
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(height * width, count))
    raw -= raw.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(raw, mode="reduced")
    return q.T.reshape(count, height, width).astype(np.float32)


def component(hidden, modes):
    """Orthogonal projection of a channelwise centered spatial field."""
    basis = jnp.asarray(modes, dtype=hidden.dtype)
    centered = hidden - hidden.mean(axis=(1, 2), keepdims=True)
    coeff = jnp.einsum("khw,bhwc->bkc", basis, centered)
    return jnp.einsum("khw,bkc->bhwc", basis, coeff)


def perturb(hidden, low_modes, null_modes, mode, strength=1.0):
    """Remove low-mode content or equal-L2-energy random-mode content.

    Match perturbation L2 per sample and hidden channel. When low-mode
    activity is absent, both interventions are exactly zero.
    """
    if mode not in MODES or not 0 <= strength <= 1:
        raise ValueError("Unknown mode or invalid suppression fraction")
    if mode == "intact":
        return hidden
    target = component(hidden, low_modes)
    if mode == "suppress_low_modes":
        return hidden - strength * target
    random_component = component(hidden, null_modes)
    target_norm = jnp.sqrt(jnp.sum(jnp.square(target), axis=(1, 2),
                                   keepdims=True))
    null_norm = jnp.sqrt(jnp.sum(jnp.square(random_component),
                                 axis=(1, 2), keepdims=True))
    # Extremely rare exact zero random projection: use the target direction
    # as the fallback to keep the energy matching honest.
    matched = jnp.where(null_norm > 1e-9,
        random_component * (target_norm / jnp.maximum(null_norm, 1e-9)),
        target)
    return hidden - strength * matched


def forecast(params, context, low_modes, null_modes, horizon=10, ticks=2,
             mode="intact", intervene_at=0, strength=1.0):
    """Intervene before lead intervene_at+1 and forecast without new input."""
    if (mode not in MODES or horizon < 1 or ticks < 1
            or not 0 <= intervene_at < horizon or not 0 <= strength <= 1):
        raise ValueError("Invalid forecast/intervention settings")
    if (context.ndim != 5 or context.shape[1] < 2
            or context.shape[-1] != 3):
        raise ValueError("Expected RGB [B,C>=2,H,W,3]")
    if (low_modes.shape != null_modes.shape
            or tuple(low_modes.shape[1:]) != tuple(context.shape[2:4])):
        raise ValueError("Intervention bases must have matching rank and image size")
    state = warm_state(params, context, ticks)

    def predict(s, index):
        def intervene(current):
            hidden, image = current
            return (perturb(hidden, low_modes, null_modes, mode, strength), image)
        s = jax.lax.cond(index == intervene_at, intervene, lambda s: s, s)
        return nca.dream(params, s, ticks)

    _, images = jax.lax.scan(predict, state, jnp.arange(horizon))
    return jnp.swapaxes(images, 0, 1)
