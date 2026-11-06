"""Deterministic 4D rotations observed only through rendered 2D RGB projections.

World coordinates, phases, plane IDs and cameras are evaluator metadata.
The first Circus experiment uses consistent geometry, not impossible worlds.
"""

from itertools import combinations, product

import numpy as np

CASES = ("iid", "rotation", "object", "viewpoint", "combined")
TRAIN_PLANES = ((0, 1), (0, 3))
HELD_OUT_PLANES = ((0, 2), (1, 3))


def rotation(dimension, plane, angle):
    first, second = plane
    if first == second or min(plane) < 0 or max(plane) >= dimension:
        raise ValueError("Need two distinct axes within the rotation dimension")
    matrix = np.eye(dimension)
    c, s = np.cos(angle), np.sin(angle)
    matrix[first, first] = matrix[second, second] = c
    matrix[first, second], matrix[second, first] = -s, s
    return matrix


def shape(kind, axis_scales):
    scales = np.asarray(axis_scales, float)
    if scales.shape != (4,) or np.any(scales <= 0):
        raise ValueError("Need four positive axis scales")
    if kind == "tesseract":
        vertices = np.array(list(product((-1., 1.), repeat=4)))
        edges = [(i, j) for i, j in combinations(range(16), 2)
                 if np.count_nonzero(vertices[i] != vertices[j]) == 1]
    elif kind == "16cell":
        vertices = np.concatenate((np.eye(4), -np.eye(4)))
        edges = [(i, j) for i, j in combinations(range(8), 2)
                 if not np.array_equal(vertices[i], -vertices[j])]
    else:
        raise ValueError("Unknown 4D shape")
    vertices *= scales
    vertices /= np.max(np.linalg.norm(vertices, axis=1))
    return vertices, np.asarray(edges, int)


def project(points, camera, distance4=4., distance3=4.):
    """4D perspective to 3D, camera rotation, then perspective to 2D."""
    points = np.asarray(points)
    if points.shape[-1] != 4 or np.shape(camera) != (3, 3):
        raise ValueError("Expected 4D points and a 3D camera rotation")
    denominator4 = distance4 - points[..., 3:4]
    if np.any(denominator4 <= 0):
        raise ValueError("Point behind 4D projection camera")
    xyz = points[..., :3] * distance4 / denominator4
    xyz = xyz @ np.asarray(camera).T
    denominator3 = distance3 - xyz[..., 2:3]
    if np.any(denominator3 <= 0):
        raise ValueError("Point behind 3D projection camera")
    return xyz[..., :2] * distance3 / denominator3


def render(points, edges, size=32):
    """Soft wireframe plus visible vertex markers; no depth or ID tensor input."""
    if size < 16:
        raise ValueError("Need at least 16 image pixels per axis")
    points = np.asarray(points) * (size * .28) + (size - 1) / 2
    yy, xx = np.mgrid[:size, :size]
    grid = np.stack((xx, yy), axis=-1)
    strength = np.zeros((size, size))
    for first, second in edges:
        start, end = points[first], points[second]
        line = end - start
        fraction = np.clip(np.sum((grid - start) * line, axis=-1)
                           / max(float(line @ line), 1e-12), 0, 1)
        distance2 = np.sum((grid - start - fraction[..., None] * line)**2, axis=-1)
        strength = np.maximum(strength, np.exp(-distance2 / (2 * .45**2)))
    rgb = .035 + strength[..., None] * np.array([.55, .65, .8])
    # Markers make this a visually structured task; identities are not supplied
    # separately. Symmetries and projection ambiguity can still remain.
    for i, point in enumerate(points):
        blob = np.exp(-np.sum((grid - point)**2, axis=-1) / (2 * .65**2))
        color = np.array([.65 + .25*np.sin(i*2.4), .55 + .3*np.cos(i*1.3), .7])
        rgb = rgb * (1 - blob[..., None]) + blob[..., None] * color
    return np.clip(rgb, 0, 1).astype(np.float32)


def episode(seed, steps=14, size=32, case="iid"):
    if case not in CASES or steps < 2 or size < 16:
        raise ValueError("Invalid hypergeometry case, duration or image size")
    rng = np.random.default_rng(int(seed))
    scales = rng.uniform(.65, 1., 4)
    initial = rng.uniform(-np.pi, np.pi, 2)
    speeds = rng.uniform(.04, .12, 2) * rng.choice((-1, 1), 2)
    view = rng.uniform(-.3, .3, 2)
    if case in ("viewpoint", "combined"):
        view = np.where(view >= 0, 1., -1.) * (.9 + np.abs(view))
    camera = rotation(3, (0, 2), view[0]) @ rotation(3, (1, 2), view[1])
    kind = "16cell" if case in ("object", "combined") else "tesseract"
    planes = HELD_OUT_PLANES if case in ("rotation", "combined") else TRAIN_PLANES
    vertices, edges = shape(kind, scales)
    phases = initial[None] + np.arange(steps)[:, None] * speeds[None]
    coordinates, projections, frames = [], [], []
    for angles in phases:
        transform = rotation(4, planes[1], angles[1]) @ rotation(4, planes[0], angles[0])
        points = vertices @ transform.T
        projected = project(points, camera)
        coordinates.append(points)
        projections.append(projected)
        frames.append(render(projected, edges, size))
    return {
        "frames": np.stack(frames),
        "truth": {"coordinates4": np.stack(coordinates),
                  "projected2": np.stack(projections), "phases": phases,
                  "speeds": speeds, "camera": camera, "view_angles": view,
                  "planes": np.asarray(planes), "shape": kind, "edges": edges},
        "seed": int(seed), "case": case,
    }


def batch(seeds, steps=14, size=32, case="iid"):
    return np.stack([episode(seed, steps, size, case)["frames"] for seed in seeds])


def phase_targets(seeds, context, size=32, case="iid"):
    """Evaluator-only periodic generator coordinates; not uniquely identifiable."""
    angles = np.stack([episode(s, context, size, case)["truth"]["phases"][-1]
                       for s in seeds])
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=-1)
