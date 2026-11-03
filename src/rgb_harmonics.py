"""Fixed, non-periodic grid-Laplacian modes and RGB-NCA hidden-state tracing.

Modes are spatial eigenvectors (not temporal oscillations or representations).
The analytic cosine basis equals the four-neighbor path-grid graph Laplacian.
"""
import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as nca


def lattice_modes(height, width, count=16):
    """Return (eigenvalues [K], spatial modes [K,H,W]); omit constant mode.

    Include the entire eigenvalue tie at the cutoff so that a degenerate
    eigenspace is not arbitrarily split. No periodic boundary conditions.
    """
    if min(height, width) < 2 or count < 1:
        raise ValueError("Need dimensions >=2 and at least one nonconstant mode")

    def path_basis(n):
        coordinate = np.arange(n)[:, None] + 0.5
        k = np.arange(n)
        vectors = np.cos(np.pi * coordinate * k / n)
        vectors[:, 0] /= np.sqrt(n)
        vectors[:, 1:] *= np.sqrt(2 / n)
        return 2 - 2 * np.cos(np.pi * k / n), vectors

    ey, vy = path_basis(height)
    ex, vx = path_basis(width)
    eigenvalues = (ey[:, None] + ex[None, :]).reshape(-1)
    modes = np.einsum("yk,xl->klyx", vy, vx).reshape(-1, height, width)
    ordered = np.argsort(eigenvalues, kind="stable")
    nonconstant = ordered[eigenvalues[ordered] > 1e-10]
    cutoff = eigenvalues[nonconstant[min(count, len(nonconstant)) - 1]]
    selected = nonconstant[eigenvalues[nonconstant] <= cutoff + 1e-9]
    return eigenvalues[selected], modes[selected]


def trace(params, observed_context, horizon=10, ticks=2):
    """Record post-observation and open-loop recurrent states; no future input."""
    if (observed_context.ndim != 5 or observed_context.shape[1] < 2
            or observed_context.shape[-1] != 3 or horizon < 1 or ticks < 1):
        raise ValueError("Expected RGB [B,C>=2,H,W,3], positive horizon/ticks")
    batch, _, height, width, _ = observed_context.shape
    state = nca.initial_state(params, batch, height, width)

    def absorb(s, frame):
        corrected = nca.observe(params, s, frame, ticks)
        next_state, _ = nca.dream(params, corrected, ticks)
        return next_state, corrected[0]

    state, seen = jax.lax.scan(
        absorb, state, jnp.swapaxes(observed_context[:, :-1], 0, 1))
    state = nca.observe(params, state, observed_context[:, -1], ticks)
    seen = jnp.concatenate((seen, state[0][None]), axis=0)

    def predict(s, _):
        s, frame = nca.dream(params, s, ticks)
        return s, (s[0], frame)

    _, (dreamed, predicted) = jax.lax.scan(predict, state, None, length=horizon)
    return (jnp.swapaxes(seen, 0, 1), jnp.swapaxes(dreamed, 0, 1),
            jnp.swapaxes(predicted, 0, 1))


def shuffle_sites(hidden, seed=0):
    """One fixed spatial permutation for all times, episodes and channels."""
    values = np.asarray(hidden)
    b, t, h, w, c = values.shape
    permutation = np.random.default_rng(seed).permutation(h * w)
    return values.reshape(b, t, h * w, c)[:, :, permutation].reshape(values.shape)


def spectrum(hidden, modes):
    """Spatially demean each frame/channel and report energy in fixed modes.

    Fractions use ALL nonconstant spatial energy as denominator, not just
    the selected modes; hence their sum can be <1 when truncated.
    """
    values = np.asarray(hidden)
    spatial_modes = np.asarray(modes)
    if (values.ndim != 5 or spatial_modes.ndim != 3
            or values.shape[2:4] != spatial_modes.shape[1:3]):
        raise ValueError("Hidden [B,T,H,W,C] and spatial modes [K,H,W] required")
    centered = values - values.mean(axis=(2, 3), keepdims=True)
    coeff = np.einsum("khw,bthwc->btkc", spatial_modes, centered, optimize=True)
    power = np.square(coeff).mean(axis=(0, 1, 3))
    total = float(np.square(centered).sum(axis=(2, 3)).mean())
    fraction = power / total if total > 1e-15 else np.zeros_like(power)
    return {"fractions": fraction.tolist(),
            "selected_fraction": float(fraction.sum()),
            "total_nonconstant_energy": total}
