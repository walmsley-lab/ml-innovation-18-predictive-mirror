"""Tiny deterministic camera world. Hidden truth is never a training target."""

import numpy as np


def sequence(seed, steps=12, size=16, occlusion=True, camera=True, object_motion=True, wall_width=1):
    """Return observable frames [T+1,H,W,1], camera deltas [T,1], hidden truth.

    A dot moves in world coordinates, sometimes behind a fixed vertical wall.
    The sensor can translate horizontally; only its movement command is observed.
    """
    if not 1 <= wall_width <= size // 2:
        raise ValueError('wall_width must fit within half a frame')
    rng = np.random.default_rng(seed)
    width = size + 6
    x = int(rng.integers(3, width - 3))
    direction = int(rng.choice([-1, 1])) if object_motion else 0
    y = int(rng.integers(3, size - 3))
    view = 3
    wall_x = width // 2
    wall_left = wall_x - (wall_width - 1) // 2
    wall_right = wall_left + wall_width
    positions, views, frames, actions, visible_masks, occluded, in_view = [], [], [], [], [], [], []
    for t in range(steps + 1):
        frame = np.zeros((size, size, 1), np.float32)
        visible_mask = np.zeros((size, size, 1), bool)
        if occlusion:
            left, right = max(0, wall_left - view), min(size, wall_right - view)
            if left < right:
                frame[:, left:right, 0] = 0.35
        px = x - view
        behind_wall = occlusion and wall_left <= x < wall_right
        on_sensor = 0 <= px < size
        # Truth fields below are never returned by batch() to training.
        occluded.append(bool(behind_wall and on_sensor))
        in_view.append(bool(on_sensor))
        if on_sensor and not behind_wall:
            frame[y, px, 0] = 1.0
            visible_mask[y, px, 0] = True
        frames.append(frame)
        visible_masks.append(visible_mask)
        positions.append(x)
        views.append(view)
        if t < steps:
            delta = int(rng.choice([-1, 0, 1])) if camera else 0
            delta = int(np.clip(view + delta, 0, width - size) - view)
            actions.append([delta])
            view += delta
            x += direction
            if x < 1 or x >= width - 1:
                direction = -direction
                x += 2 * direction
    return (np.stack(frames), np.asarray(actions, np.float32),
            {"world_x": np.asarray(positions), "camera_x": np.asarray(views),
             "visible_mask": np.stack(visible_masks),
             "occluded": np.asarray(occluded, bool),
             "in_view": np.asarray(in_view, bool)})


def batch(seeds, steps=12, size=16, occlusion=True, camera=True):
    runs = [sequence(int(s), steps, size, occlusion, camera) for s in seeds]
    return (np.stack([r[0] for r in runs]),
            np.stack([r[1] for r in runs]))

def curriculum_batch(seeds, stages, steps=12, size=16):
    """A curriculum restricts experience, never supplies hidden state to a model."""
    settings = (
        dict(object_motion=False, occlusion=False, camera=False),
        dict(object_motion=True, occlusion=False, camera=False),
        dict(object_motion=True, occlusion=True, camera=False),
        dict(object_motion=True, occlusion=True, camera=True),
    )
    episodes = [sequence(int(seed), steps=steps, size=size, **settings[int(stage)])
                for seed, stage in zip(seeds, stages)]
    return np.stack([episode[0] for episode in episodes]), np.stack([episode[1] for episode in episodes])
