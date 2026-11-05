"""Checked 2D diffusion and damped wave worlds with explicit boundaries.

State is [u] for diffusion, [u, velocity] for the wave equation.
Grid spacing dx and time step dt are physical, not update-count aliases.
"""
import numpy as np

BOUNDARIES = ("reflect", "zero")
KINDS = ("diffusion", "wave")


def laplacian(u, dx=1., boundary="reflect"):
    """Five-point Laplacian. 'reflect' means zero normal finite-volume flux."""
    u = np.asarray(u)
    if u.ndim != 2 or min(u.shape) < 3 or dx <= 0:
        raise ValueError("Expected [H,W] field with H,W>=3 and dx>0")
    if boundary not in BOUNDARIES:
        raise ValueError("Unknown boundary")
    p = np.pad(u, 1, mode="edge" if boundary == "reflect" else "constant")
    return (p[2:, 1:-1] + p[:-2, 1:-1] +
            p[1:-1, 2:] + p[1:-1, :-2] - 4 * u) / dx**2


def check(kind, dt, dx, diffusivity=0.15, speed=0.5, damping=0.):
    if kind not in KINDS or dt <= 0 or dx <= 0 or damping < 0:
        raise ValueError("Invalid physical parameters")
    if kind == "diffusion":
        if diffusivity <= 0 or diffusivity * dt / dx**2 > .25 + 1e-12:
            raise ValueError("Explicit 2D diffusion violates CFL r<=1/4")
    else:
        if speed <= 0 or speed * dt / dx > 1 / np.sqrt(2) + 1e-12:
            raise ValueError("Explicit 2D wave violates CFL c*dt/dx<=1/sqrt(2)")
        if damping * dt >= 2:
            raise ValueError("Damping update unstable at dt*damping>=2")


def initial(seed=11, size=24, kind="diffusion", dx=1.):
    if size < 8 or kind not in KINDS or dx <= 0:
        raise ValueError("Invalid world")
    rng = np.random.default_rng(int(seed))
    y, x = np.mgrid[:size, :size]
    center = rng.uniform(.25 * size, .75 * size, 2)
    spread = rng.uniform(1.2, 2.8)
    u = np.exp(-.5 * ((x - center[0])**2 +
                        (y - center[1])**2) / spread**2)
    u *= rng.choice((-1., 1.))
    if kind == "diffusion":
        return np.stack([u]).astype(np.float64)
    # A stationary initial displacement requires v=0 (not a second image).
    return np.stack([u, np.zeros_like(u)]).astype(np.float64)


def step(state, kind="diffusion", dt=.2, dx=1., diffusivity=.15,
         speed=.5, damping=.02, boundary="reflect"):
    check(kind, dt, dx, diffusivity, speed, damping)
    if boundary not in BOUNDARIES:
        raise ValueError("Invalid boundary")
    state = np.asarray(state, dtype=np.float64)
    channels = 1 if kind == "diffusion" else 2
    if state.ndim != 3 or state.shape[0] != channels:
        raise ValueError(f"{kind} needs {channels} state channel(s)")
    u = state[0]
    if kind == "diffusion":
        return np.stack([u + dt * diffusivity * laplacian(u, dx, boundary)])
    # Leapfrog-like staggered acceleration with explicit velocity storage:
    # v(t+dt) = v(t) + dt [c² Δu - gamma v(t)]
    # u(t+dt) = u(t) + dt v(t+dt).
    velocity = (1 - damping * dt) * state[1] + (
        speed**2 * dt) * laplacian(u, dx, boundary)
    next_u = u + dt * velocity
    if boundary == "zero":
        next_u[[0, -1], :] = 0
        next_u[:, [0, -1]] = 0
        velocity[[0, -1], :] = 0
        velocity[:, [0, -1]] = 0
    return np.stack([next_u, velocity])


def episode(seed=11, kind="diffusion", size=24, steps=24, dt=.2, dx=1.,
            diffusivity=.15, speed=.5, damping=.02, boundary="reflect"):
    if steps < 2:
        raise ValueError("Need at least two frames")
    kwargs = dict(kind=kind, dt=dt, dx=dx, diffusivity=diffusivity,
                  speed=speed, damping=damping, boundary=boundary)
    check(kind, dt, dx, diffusivity, speed, damping)
    state = initial(seed, size, kind, dx)
    history = [state.copy()]
    for _ in range(steps - 1):
        state = step(state, **kwargs)
        history.append(state.copy())
    states = np.asarray(history)
    return {"states": states, "observed": states[:, 0],
            "kind": kind, "seed": int(seed), "parameters": kwargs,
            "integration": "explicit finite-difference physical fields v1"}


def verify(seed=11, size=24, steps=24):
    d = episode(seed, "diffusion", size, steps)
    w = episode(seed, "wave", size, steps)
    diffusion_mass = d["states"][:, 0].sum(axis=(1, 2))
    # With zero-flux boundaries mass is preserved by the exact stencil.
    mass_error = float(np.abs(diffusion_mass - diffusion_mass[0]).max())
    wave_amplitude = float(np.max(np.abs(w["observed"])))
    if mass_error > 1e-8 or not np.isfinite(w["states"]).all():
        raise AssertionError("Physical reference invariant violated")
    return {"diffusion_mass_max_error": mass_error,
            "wave_max_abs_amplitude": wave_amplitude,
            "diffusion_state_shape": list(d["states"].shape),
            "wave_state_shape": list(w["states"].shape)}
