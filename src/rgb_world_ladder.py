"""Deterministic object-motion worlds for perceptual generalization.

Separate from the frozen v1 RGB benchmark: **do not change old test
distributions**. The model receives RGB; positions/events stay evaluator-only.
"""
import numpy as np
from src.rgb_motion_world import COLORS

DOMAINS = ("linear", "accelerating", "collision", "occlusion_short",
           "occlusion_long", "camera_pan", "appearance", "combined")


def _reflect(pos, velocity, radius, size):
    for axis in range(2):
        if pos[axis] < radius:
            pos[axis] = radius + radius - pos[axis]
            velocity[axis] = abs(velocity[axis])
        elif pos[axis] > size - radius - 1:
            pos[axis] = 2 * (size - radius - 1) - pos[axis]
            velocity[axis] = -abs(velocity[axis])


def episode(seed, steps=20, size=32, balls=2, domain="linear"):
    """Return frames and independent evaluation truth.

    Model-eligible: frames and, only for action-conditioned experiments,
    the known camera_actions[t] for the t -> t+1 transition.
    """
    if (domain not in DOMAINS or steps < 4 or size < 20
            or not 1 <= balls <= 4):
        raise ValueError("Invalid world dimensions/domain")
    rng = np.random.default_rng(int(seed))
    radius = np.full(balls, max(2.0, size / 13), np.float32)
    pos = rng.uniform(size * .25, size * .75, (balls, 2)).astype(np.float32)
    angle = rng.uniform(-np.pi, np.pi, balls)
    speed = rng.uniform(.65, 1.8, balls)
    vel = np.stack((np.cos(angle) * speed, np.sin(angle) * speed), axis=-1).astype(np.float32)
    acceleration = np.zeros_like(vel)
    if domain in ("accelerating", "combined"):
        acceleration = rng.uniform(-.10, .10, (balls, 2)).astype(np.float32)

    # Scripted paths make reappearance and collision episodes repeatable;
    # random secondary objects still vary the task.
    if domain.startswith("occlusion_") or domain == "combined":
        pos[0] = (size * .13, size * .53)
        vel[0] = (1.9, 0.)
        acceleration[0] = 0.
    if domain == "collision" and balls >= 2:
        pos[0], pos[1] = (size*.24,size*.5), (size*.76,size*.5)
        vel[0], vel[1] = (1.5,0), (-1.5,0)
    colors = COLORS[:balls].copy()
    if domain in ("appearance","combined"):
        # Preserve diagnostic palette identity for per-object comparisons.
        # Change background brightness and sizes, NOT object palette.
        radius *= rng.uniform(.75,1.4,balls).astype(np.float32)
    background = np.array([.035]*3,np.float32)
    if domain in ("appearance","combined"):
        background[:] = float(rng.uniform(.015,.10))

    actions = np.zeros((steps-1,2),np.float32)
    if domain in ("camera_pan","combined"):
        actions = rng.choice((-1.,0.,1.),(steps-1,2)).astype(np.float32)
    camera = np.zeros((steps,2),np.float32)
    for t in range(steps-1):
        camera[t+1] = np.clip(camera[t]+actions[t],-size*.15,size*.15)
        actions[t] = camera[t+1]-camera[t]

    is_occlusion = domain.startswith("occlusion_") or domain == "combined"
    width = size*.20 if domain=="occlusion_short" else size*.36
    if domain=="combined":
        width = size*.28
    yy,xx = np.mgrid[:size,:size]
    output,positions,velocities,visible,reason,overlaps = [],[],[],[],[],[]
    for t in range(steps):
        picture = np.broadcast_to(background,(size,size,3)).copy()
        discs = []
        for j in range(balls):
            sx,sy = pos[j]-camera[t]
            mask = (xx-sx)**2+(yy-sy)**2 <= radius[j]**2
            picture[mask] = colors[j]
            discs.append(mask)
        stripe = np.zeros((size,size),bool)
        if is_occlusion:
            stripe = np.abs(xx-size//2)<=width/2
            picture[stripe] = [.13,.13,.15]
        state,causes=[],[]
        for j,mask in enumerate(discs):
            seen = bool(np.any(mask & ~stripe))
            in_view=bool(np.any(mask))
            state.append(seen)
            causes.append("visible" if seen else
                          "barrier" if in_view and is_occlusion else "out_of_view")
        positions.append(pos.copy())
        velocities.append(vel.copy())
        visible.append(state)
        reason.append(causes)
        output.append(picture.astype(np.float32))
        if t == steps-1:
            break
        pos += vel
        vel += acceleration
        for j in range(balls):
            _reflect(pos[j],vel[j],radius[j],size)
        # Simple equal-mass discs: a deliberate, deterministic collision test.
        count = 0
        for j in range(balls):
            for k in range(j+1,balls):
                delta=pos[k]-pos[j]
                d=float(np.linalg.norm(delta))
                minimum=radius[j]+radius[k]
                if d<minimum and d>1e-6:
                    normal=delta/d
                    if float((vel[k]-vel[j])@normal)<0:
                        vj=float(vel[j]@normal);vk=float(vel[k]@normal)
                        vel[j]+=(vk-vj)*normal
                        vel[k]+=(vj-vk)*normal
                        count += 1
                    correction=.5*(minimum-d+1e-4)*normal
                    pos[j]-=correction
                    pos[k]+=correction
        overlaps.append(count)
    return {"frames":np.stack(output),"positions":np.stack(positions),
            "velocities":np.stack(velocities),"visible":np.asarray(visible,bool),
            "visibility_reason":np.asarray(reason),"camera_offsets":camera,
            "camera_actions":actions,"collision_counts":np.asarray(overlaps,int),
            "colors":colors,"domain":domain,"seed":int(seed),
            "occlusion":is_occlusion}


def model_inputs(record, action_conditioned=False):
    result={"frames":record["frames"]}
    if action_conditioned:
        result["camera_actions"]=record["camera_actions"]
    return result


def split(name,count,context=8,horizon=10,size=32,balls=2,domain="linear"):
    """Disjoint, fixed training/validation/test seed namespaces."""
    roots={"train":31000,"validation":61000,"test":91000}
    if name not in roots or count<1:
        raise ValueError("Invalid split")
    return [episode(roots[name]+i,context+horizon,size,balls,domain)
            for i in range(count)]
