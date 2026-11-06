"""First-principles graph mechanics and localized, causal sensory observations.

The body is arbitrary supplied 2D nodes + undirected springs. There are no
object labels or image features: state is displacement/velocity at each site.
Contact forces are applied externally; only sparse displacement sensors and
known outgoing motor forces are available to the learning automaton.
"""
import numpy as np


def geometry(coords, edges):
    """Validate arbitrary connected geometry; no grid/skeleton assumption."""
    xy = np.asarray(coords, np.float32)
    pairs = np.asarray(edges, np.int32)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError("Need at least three finite planar sites")
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs) or
            (pairs < 0).any() or (pairs >= len(xy)).any() or
            np.any(pairs[:, 0] == pairs[:, 1])):
        raise ValueError("Need valid, non-self undirected edges")
    edges_sorted = np.sort(pairs, axis=1)
    if len(np.unique(edges_sorted, axis=0)) != len(pairs):
        raise ValueError("Duplicate undirected edge")
    seen = {0}
    for _ in range(len(xy)):
        for a, b in edges_sorted:
            if int(a) in seen or int(b) in seen:
                seen.update((int(a), int(b)))
    if len(seen) != len(xy):
        raise ValueError("Geometry graph must be connected")
    return xy, edges_sorted


def generate(seed, nodes=12, kind="irregular"):
    """Seeded representative geometries; geometry() accepts arbitrary meshes."""
    if nodes < 5 or kind not in ("irregular", "ring", "branch"):
        raise ValueError("Invalid procedural geometry")
    rng = np.random.default_rng(int(seed))
    if kind == "ring":
        angles = np.linspace(0, 2*np.pi, nodes, endpoint=False)
        radii = rng.uniform(.26, .47, nodes)
        xy = np.stack((radii*np.cos(angles), radii*np.sin(angles)), axis=-1)
        edges = [(i, (i + 1) % nodes) for i in range(nodes)]
    elif kind == "branch":
        xy = rng.uniform(-.42, .42, (nodes, 2))
        edges = [(i, int(rng.integers(0, max(1, (i + 1) // 2)))) for i in range(1, nodes)]
    else:
        xy = rng.uniform(-.45, .45, (nodes, 2))
        edges = []
        connected = {0}
        while len(connected) < nodes:
            a, b = min(((i, j) for i in connected
                        for j in range(nodes) if j not in connected),
                       key=lambda pair: float(np.sum((xy[pair[0]]-xy[pair[1]])**2)))
            edges.append((a, b))
            connected.add(b)
        for i in range(nodes):
            distances = np.linalg.norm(xy - xy[i], axis=1)
            neighbors = np.argsort(distances)[1:3]
            for j in neighbors:
                if rng.random() < .25 and int(j) != i:
                    candidate = tuple(sorted((i, int(j))))
                    if candidate not in {tuple(sorted(edge)) for edge in edges}:
                        edges.append(candidate)
    return geometry(xy, np.asarray(edges, np.int32))


def adjacency(coords, edges):
    """One-hop, degree-normalized graph for local cellular communication."""
    xy, pairs = geometry(coords, edges)
    weights = np.zeros((len(xy), len(xy)), np.float32)
    for a, b in pairs:
        distance = float(np.linalg.norm(xy[a] - xy[b]))
        weight = 1. / (distance + .20)
        weights[a, b] = weight
        weights[b, a] = weight
    return weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-6)


def contact_force(coords, center, push, radius=.18):
    """Known outgoing motor impulse, locally distributed over the body sites."""
    xy = np.asarray(coords, np.float32)
    center = np.asarray(center, np.float32)
    push = np.asarray(push, np.float32)
    if center.shape != (2,) or push.shape != (2,) or radius <= 0:
        raise ValueError("Invalid motor command")
    response = np.exp(-np.sum((xy-center)**2, axis=1) / (2*radius**2))
    return (response[:, None] * push).astype(np.float32)


def step(coords, edges, displacement, velocity, force, dt=.12,
         stiffness=.65, damping=.13, tether=.12):
    """Symmetric internal forces: pairwise action/reaction across any graph."""
    xy, pairs = geometry(coords, edges)
    q = np.asarray(displacement, np.float32)
    v = np.asarray(velocity, np.float32)
    drive = np.asarray(force, np.float32)
    if q.shape != xy.shape or v.shape != xy.shape or drive.shape != xy.shape:
        raise ValueError("State and force must be [N,2]")
    if not 0 < dt <= .3 or min(stiffness, damping, tether) < 0:
        raise ValueError("Invalid physical coefficients")
    inner = np.zeros_like(q)
    lengths = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    flux = (stiffness / (1. + lengths))[:, None] * (q[pairs[:, 1]] - q[pairs[:, 0]])
    np.add.at(inner, pairs[:, 0], flux)
    np.add.at(inner, pairs[:, 1], -flux)
    acceleration = inner - tether * q - damping * v + drive
    new_velocity = v + dt * acceleration
    new_displacement = q + dt * new_velocity
    return new_displacement, new_velocity


def episode(seed, nodes=12, steps=18, kind="irregular", sensors=.35,
            action=True):
    """Causal t-to-t+1 rollout; q,v are evaluator truth, not input features."""
    if steps < 3 or nodes < 5 or not 0 < sensors <= 1:
        raise ValueError("Invalid episode settings")
    xy, edges = generate(int(seed), nodes, kind)
    rng = np.random.default_rng(int(seed) + 19011)
    q = np.zeros((nodes, 2), np.float32)
    v = np.zeros_like(q)
    sensors_n = max(2, int(np.ceil(sensors*nodes)))
    observed_nodes = rng.choice(nodes, size=sensors_n, replace=False)
    sensor_mask = np.zeros((nodes, 1), np.float32)
    sensor_mask[observed_nodes] = 1.
    q_history = [q.copy()]
    forces = []
    for t in range(steps):
        if action and t % 4 == 0:
            target = xy[int(rng.integers(nodes))]
            direction = rng.normal(size=2)
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            force = contact_force(xy, target, (.6 * direction).astype(np.float32))
        else:
            force = np.zeros_like(q)
        forces.append(force)
        q, v = step(xy, edges, q, v, force)
        q_history.append(q.copy())
    truth = np.stack(q_history)
    return {"coords": xy, "edges": edges,
            "adjacency": adjacency(xy, edges),
            "observed": truth * sensor_mask[None],
            "sensor_mask": np.broadcast_to(sensor_mask, (steps+1, nodes, 1)).copy(),
            "forces": np.stack(forces),
            "truth": truth, "geometry_kind": kind,
            "seed": int(seed)}
