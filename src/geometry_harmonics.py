"""Geometry-constrained oscillators, frequency sweeps, and modal interventions.

A mass-spring GRAPH wave analog, not a thin-plate bending solver: physical
Chladni plates use flexural (biharmonic) dynamics and boundary conditions.
Modes are an EVALUATOR basis, never an input to a perceptual learner.
"""
import numpy as np

from src.geometry_automata import geometry, generate


def laplacian(coords, edges):
    """Unnormalized symmetric positive-semidefinite weighted graph Laplacian."""
    xy, pairs = geometry(coords, edges)
    weights = np.zeros((len(xy), len(xy)), np.float64)
    for a, b in pairs:
        w = 1. / (.2 + float(np.linalg.norm(xy[a] - xy[b])))
        weights[a, b] = weights[b, a] = w
    return np.diag(weights.sum(axis=1)) - weights


def modes(coords, edges, grounding=.35, coupling=.8):
    """Normal frequencies and spatial mode shapes for the linear toy medium."""
    if grounding <= 0 or coupling <= 0:
        raise ValueError("Require positive grounding and coupling")
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian(coords, edges))
    frequencies = np.sqrt(grounding + coupling * np.maximum(eigenvalues, 0.))
    return frequencies, eigenvectors


def tone(steps, dt, frequency, profile, amplitude=.32):
    """Local/specified spatial drive, no access to future physical state."""
    if steps < 3 or dt <= 0 or frequency <= 0:
        raise ValueError("Invalid harmonic drive")
    return amplitude * np.sin(frequency * dt * np.arange(steps))[:, None] * profile[None]


def simulate(coords, edges, force, dt=.055, grounding=.35, coupling=.8,
             damping=.16, cubic=0., shock=None):
    """Semi-implicit local spring wave: q'=v; v'=-gq-cLq-dv-bq^3+f.

    Inputs force[t] are external drives at time t. The optional shock adds
    a displacement at a nominated time; it is an evaluator intervention.
    """
    medium = laplacian(coords, edges)
    drive = np.asarray(force, np.float64)
    if (drive.ndim != 2 or drive.shape[1] != len(medium) or
            not np.isfinite(drive).all() or dt <= 0 or dt > .15 or
            grounding <= 0 or coupling <= 0 or damping < 0 or cubic < 0):
        raise ValueError("Invalid oscillator simulation")
    if shock is not None:
        at, perturb = shock
        perturb = np.asarray(perturb, np.float64)
        if not 0 <= at < len(drive) or perturb.shape != (len(medium),):
            raise ValueError("Invalid displacement perturbation")
    q = np.zeros(len(medium), np.float64)
    v = q.copy()
    displacement = [q.copy()]
    velocity = [v.copy()]
    for t, forcing in enumerate(drive):
        if shock is not None and t == shock[0]:
            q += perturb
        acceleration = (-grounding*q - coupling*(medium@q) -
                        damping*v - cubic*q**3 + forcing)
        v += dt * acceleration
        q += dt * v
        displacement.append(q.copy())
        velocity.append(v.copy())
    return np.stack(displacement), np.stack(velocity)


def modal_power(displacement, basis, early=0):
    """Energy proxy in the linear structural mode basis (not total energy)."""
    coefficients = displacement[early:] @ basis
    return np.mean(coefficients**2, axis=0)


def phase_trajectory(displacement, velocity, frequency, basis_vector):
    """A constructed helix chart: mode's (q,-v/omega) plus elapsed time.

    The phase plane is physically meaningful for linear modal motion.
    Appending a progress coordinate makes a helix visualization, but does
    not prove that a latent manifold was learned or has higher dimension.
    """
    q = displacement @ basis_vector
    p = -(velocity @ basis_vector) / frequency
    theta = np.unwrap(np.arctan2(p, q))
    time = np.linspace(0., 1., len(q))
    return np.stack((q, p, time), axis=-1), theta


def broadband(seed, steps, nodes, amplitude=.10):
    """Externally applied deterministic white noise; not vacuum noise."""
    if steps < 3 or nodes < 1 or amplitude <= 0:
        raise ValueError("Invalid broadband excitation")
    rng = np.random.default_rng(int(seed))
    force = np.zeros((steps, nodes), np.float64)
    force[:, int(rng.integers(nodes))] = amplitude * rng.standard_normal(steps)
    return force


def sensor_record(seed, nodes=16, kind="irregular", steps=28,
                  dt=.055, sensor_fraction=.35):
    """Drop-in input contract for geometry_perceiver: sparse q, local force.

    Oscillatory q is scalar; the prior graph perceiver expects [N,2],
    so the second coordinate is explicitly zero. Structural eigenvalues
    and unobserved q/v are excluded from model inputs.
    """
    if not 0 < sensor_fraction <= 1:
        raise ValueError("Sensor fraction must be in (0,1]")
    from src.geometry_automata import adjacency
    xy, edges = generate(seed, nodes, kind)
    force = broadband(seed + 100, steps, nodes, amplitude=.28)
    q, _ = simulate(xy, edges, force, dt=dt, damping=.10)
    rng = np.random.default_rng(seed + 901)
    sensing = np.zeros((nodes, 1), np.float32)
    chosen = rng.choice(nodes, max(2, int(np.ceil(nodes*sensor_fraction))),
                        replace=False)
    sensing[chosen] = 1.
    truth = np.stack((q, np.zeros_like(q)), axis=-1).astype(np.float32)
    act = np.stack((force, np.zeros_like(force)), axis=-1).astype(np.float32)
    return {"coords": xy, "edges": edges,
            "adjacency": adjacency(xy, edges),
            "observed": truth * sensing[None],
            "sensor_mask": np.broadcast_to(sensing, (steps+1, nodes, 1)).copy(),
            "forces": act, "truth": truth,
            "geometry_kind": kind, "seed": int(seed)}
