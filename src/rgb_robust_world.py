"""Small deterministic 2D worlds with deliberately held-out dynamics/appearance.

This is a training-data generator, NOT a 3D simulator. Rendered RGB and
planned camera actions are model-eligible; simulator state is evaluator-only.
"""
import numpy as np

DOMAINS = ("train", "ood_motion", "ood_appearance", "ood_camera", "ood_combined")


def episode(seed, steps=18, size=32, objects=2, domain="train", occluder=True):
    if steps < 3 or size < 16 or not 1 <= objects <= 4 or domain not in DOMAINS:
        raise ValueError("Invalid episode configuration")
    rng = np.random.default_rng(int(seed))
    motion = domain in ("ood_motion", "ood_combined")
    appearance = domain in ("ood_appearance", "ood_combined")
    moving_camera = domain in ("ood_camera", "ood_combined")
    world_size = 2 * size
    radius = rng.uniform(1.8, 3.0, size=objects)
    pos = rng.uniform(size*0.7, size*1.3, size=(objects,2))
    angle = rng.uniform(-np.pi, np.pi, size=objects)
    speed = rng.uniform(1.8, 2.6, size=objects) if motion else rng.uniform(.4, 1.15, size=objects)
    vel = np.column_stack((np.cos(angle)*speed, np.sin(angle)*speed))
    accel = rng.uniform(-.13, .13, size=(objects,2)) if motion else rng.uniform(-.015, .015, size=(objects,2))
    drag = 0.001 if motion else 0.02
    palette = np.array([[.92,.19,.12],[.12,.70,.94],[.96,.75,.11],[.53,.27,.94]], np.float32)
    if appearance:
        colors = rng.uniform(.12, 1., size=(objects,3)).astype(np.float32)
        for color in colors:
            color[np.argmax(color)] = 1.0
        background = rng.uniform(.04,.18,size=3).astype(np.float32)
    else:
        colors = palette[:objects].copy()
        background = np.array([.035,.035,.035],np.float32)
    camera = np.zeros((steps,2),np.float32)
    actions = np.zeros((steps-1,2),np.float32)
    if moving_camera:
        actions = rng.choice(np.array([-2.,-1.,0.,1.,2.],np.float32), size=(steps-1,2))
        for t in range(1,steps):
            camera[t] = np.clip(camera[t-1]+actions[t-1], -.35*size, .35*size)
            actions[t-1] = camera[t]-camera[t-1]
    yy, xx = np.mgrid[:size,:size]
    frames, positions, velocities, visible, in_view, hidden = [], [], [], [], [], []
    for t in range(steps):
        frame = np.broadcast_to(background,(size,size,3)).copy()
        mask_by_object = []
        for k in range(objects):
            x, y = pos[k] - camera[t] - size/2
            disc = ((xx-x)**2+(yy-y)**2) <= radius[k]**2
            mask_by_object.append(disc)
            frame[disc] = colors[k]
        stripe = np.zeros((size,size),bool)
        if occluder:
            center = size//2 + (int(np.rint(3*np.sin(t/5))) if appearance else 0)
            stripe = np.abs(xx-center) <= 2
            frame[stripe] = [.11,.11,.13]
        present = np.asarray([bool(m.any()) for m in mask_by_object],bool)
        seen = np.asarray([bool(np.any(m & ~stripe)) for m in mask_by_object],bool)
        positions.append(pos.copy())
        velocities.append(vel.copy())
        visible.append(seen)
        in_view.append(present)
        hidden.append(present & ~seen)
        frames.append(frame.astype(np.float32))
        if t < steps-1:
            vel = (vel+accel)*(1.0-drag)
            pos = pos+vel
            for k in range(objects):
                for axis in range(2):
                    lower,upper = radius[k],world_size-radius[k]
                    if pos[k,axis] < lower or pos[k,axis] > upper:
                        pos[k,axis] = np.clip(pos[k,axis],lower,upper)
                        vel[k,axis] *= -1.0
    return {
        "frames": np.stack(frames), "camera_actions": actions,
        "camera_xy": camera, "positions": np.stack(positions),
        "velocities": np.stack(velocities), "visible": np.stack(visible),
        "in_view": np.stack(in_view), "occluded": np.stack(hidden),
        "colors": colors, "radii": radius, "domain": domain,
        "units": "pixels_2d_not_metric_3d",
    }


def model_inputs(record, include_planned_actions=False):
    """Only RGB and optionally explicit commanded camera displacements.

    The action at t specifies the transition t -> t+1 and is known to an
    action-conditioned predictor before the corresponding future observation.
    """
    data = {"frames": np.asarray(record["frames"])}
    if include_planned_actions:
        data["camera_actions"] = np.asarray(record["camera_actions"])
    return data
