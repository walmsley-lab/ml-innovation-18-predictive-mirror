"""Small controllable RGB world for testing learned motion prediction.

The renderer returns evaluation truth alongside the frames. Training code should
pass only ``episode["frames"]`` to a model; positions, velocities, visibility
and camera offsets are deliberately separate diagnostic fields.
"""

import numpy as np


COLORS = np.asarray(((0.95, 0.18, 0.12), (0.12, 0.75, 0.98),
                     (0.98, 0.78, 0.12), (0.55, 0.25, 0.95)), np.float32)


def _initial_objects(rng, count, world_size, view_margin, view_size):
    objects = []
    for index in range(count):
        radius = int(rng.integers(max(2, world_size // 18),
                                  max(3, world_size // 10)))
        # Start inside the initial camera view. The surrounding world exists
        # for later camera motion, but an episode must not begin as blank data.
        low = view_margin + radius + 2
        high = view_margin + view_size - radius - 2
        x = float(rng.integers(low, max(low + 1, high)))
        y = float(rng.integers(low, max(low + 1, high)))
        speed = float(rng.uniform(0.7, 2.0))
        angle = float(rng.uniform(-np.pi, np.pi))
        objects.append({"position": np.array((x, y), np.float32),
                        "velocity": np.array((speed * np.cos(angle),
                                               speed * np.sin(angle)), np.float32),
                        "radius": radius, "color": COLORS[index % len(COLORS)]})
    return objects


def _advance(objects, world_size):
    for obj in objects:
        obj["position"] += obj["velocity"]
        radius = obj["radius"]
        for axis in (0, 1):
            low, high = float(radius), float(world_size - radius - 1)
            if obj["position"][axis] < low or obj["position"][axis] > high:
                obj["position"][axis] = np.clip(obj["position"][axis], low, high)
                obj["velocity"][axis] *= -1

    # Equal-mass elastic collision approximation: exchange velocities when
    # two discs overlap and are moving toward one another.
    for left in range(len(objects)):
        for right in range(left + 1, len(objects)):
            first, second = objects[left], objects[right]
            delta = second["position"] - first["position"]
            distance = float(np.linalg.norm(delta))
            minimum = first["radius"] + second["radius"]
            if distance == 0 or distance > minimum:
                continue
            normal = delta / distance
            relative = second["velocity"] - first["velocity"]
            if float(relative @ normal) < 0:
                first["velocity"], second["velocity"] = (
                    second["velocity"].copy(), first["velocity"].copy())
            correction = (minimum - distance + 1e-3) * normal / 2
            first["position"] -= correction
            second["position"] += correction


def _camera_path(rng, steps, world_size, size, enabled):
    offsets = np.zeros((steps, 2), np.float32)
    if not enabled:
        return offsets
    limit = max(0, (world_size - size) // 2)
    for index in range(1, steps):
        offsets[index] = np.clip(offsets[index - 1] + rng.choice(
            (-1.0, 0.0, 1.0), size=2), -limit, limit)
    return offsets


def _render(objects, camera, size, world_size, occlusion):
    frame = np.full((size, size, 3), 0.035, np.float32)
    yy, xx = np.ogrid[:size, :size]
    visible = np.zeros(len(objects), bool)
    for index, obj in enumerate(objects):
        # Camera coordinates are centered on the middle of the larger world;
        # subtract the initial view margin to obtain pixel coordinates.
        x, y = obj["position"] - camera - (world_size - size) / 2
        disc = (xx - x) ** 2 + (yy - y) ** 2 <= obj["radius"] ** 2
        in_view = bool(disc.any())
        visible[index] = in_view
        frame[disc] = obj["color"]
    if occlusion:
        left, right = size // 2 - max(1, size // 18), size // 2 + max(1, size // 18)
        frame[:, left:right] = np.asarray((0.12, 0.12, 0.15), np.float32)
        # After rendering the foreground stripe, distinguish objects actually
        # visible in the returned image from merely being inside the camera.
        for index, obj in enumerate(objects):
            x, y = obj["position"] - camera - (world_size - size) / 2
            disc = (xx - x) ** 2 + (yy - y) ** 2 <= obj["radius"] ** 2
            visible[index] = bool(np.any(disc & ~(
                (xx >= left) & (xx < right))))
    return frame, visible


def episode(seed, steps=20, size=32, balls=1, occlusion=False,
            camera_motion=False, world_margin=8):
    """Generate ``steps`` RGB observations and evaluation-only world truth.

    ``frames`` has shape ``[T,H,W,3]``. Positions are in world coordinates;
    ``camera_offsets`` are the camera's world-coordinate offsets. The model
    should receive only the returned ``frames`` array.
    """
    if steps < 2 or size < 16 or balls < 1 or world_margin < 0:
        raise ValueError("steps >=2, size >=16, balls >=1 and nonnegative margin required")
    world_size = size + 2 * int(world_margin)
    rng = np.random.default_rng(int(seed))
    objects = _initial_objects(rng, int(balls), world_size, int(world_margin), size)
    camera_offsets = _camera_path(rng, steps, world_size, size, camera_motion)
    frames, positions, velocities, visible = [], [], [], []
    for time_index in range(steps):
        positions.append(np.stack([obj["position"].copy() for obj in objects]))
        velocities.append(np.stack([obj["velocity"].copy() for obj in objects]))
        frame, in_view = _render(objects, camera_offsets[time_index], size,
                                 world_size, occlusion)
        frames.append(frame)
        visible.append(in_view)
        if time_index + 1 < steps:
            _advance(objects, world_size)
    return {
        "frames": np.stack(frames).astype(np.float32),
        "positions": np.stack(positions).astype(np.float32),
        "velocities": np.stack(velocities).astype(np.float32),
        "visible": np.stack(visible),
        "camera_offsets": camera_offsets,
        "occlusion": bool(occlusion),
        "camera_motion": bool(camera_motion),
    }


def batch(seeds, steps=20, size=32, balls=1, occlusion=False,
          camera_motion=False):
    """Return frame-only training data with shape ``[B,T,H,W,3]``."""
    episodes = [episode(seed, steps, size, balls, occlusion, camera_motion)
                for seed in seeds]
    return np.stack([item["frames"] for item in episodes])
