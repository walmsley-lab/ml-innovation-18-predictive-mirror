"""Embodied sensor world: arithmetic programs, spring mechanics, blind palpation.

Programs command motor actions. The learning interface exposes syntax, current
sensory observations, known motor axis and modality masks; never the evaluated
program value, geometry, stiffness, or latent physical state at inference.
"""
import re

import numpy as np

NODES = 16
OPS = {"Z": 1, "S": 2, "NEG": 3, "ADD": 4}
SMALL = ("Z", "S(Z)", "NEG(S(Z))", "S(S(Z))", "NEG(S(S(Z)))",
         "ADD(S(Z),S(Z))", "ADD(S(Z),NEG(S(Z)))")
COMPOSED = ("ADD(S(S(Z)),S(Z))", "NEG(ADD(S(Z),S(S(Z))))",
            "S(ADD(S(Z),S(Z)))", "ADD(NEG(S(Z)),NEG(S(S(Z))))",
            "NEG(S(ADD(S(Z),S(Z))))")


def parse(expression):
    """Bounded postorder syntax: each row [operator, left index, right index]."""
    raw = "".join(expression.split())
    words = re.findall(r"ADD|NEG|S|Z|[(),]", raw)
    if "".join(words) != raw or not words:
        raise ValueError("Invalid program tokens")
    tree = []

    def read(pos):
        if pos >= len(words) or words[pos] not in OPS:
            raise ValueError("Expected program operator")
        op = words[pos]
        pos += 1
        left = right = 0
        if op != "Z":
            if pos >= len(words) or words[pos] != "(":
                raise ValueError("Expected '('")
            left, pos = read(pos + 1)
            if op == "ADD":
                if pos >= len(words) or words[pos] != ",":
                    raise ValueError("Expected ','")
                right, pos = read(pos + 1)
            if pos >= len(words) or words[pos] != ")":
                raise ValueError("Expected ')'")
            pos += 1
        tree.append((OPS[op], left, right))
        if len(tree) > NODES:
            raise ValueError("Program too large")
        return len(tree) - 1, pos

    _, consumed = read(0)
    if consumed != len(words):
        raise ValueError("Unexpected trailing tokens")
    return np.pad(np.asarray(tree, np.int32),
                  ((0, NODES - len(tree)), (0, 0)))


def interpret(expression):
    """The physics engine's oracle; used as a training LABEL, never input."""
    values = []
    for op, left, right in parse(expression):
        if op == 0:
            break
        value = (0 if op == 1 else
                 values[left] + 1 if op == 2 else
                 -values[left] if op == 3 else
                 values[left] + values[right])
        values.append(value)
    return values[-1]


def disc_shape(size, radius, scallop, phase, center):
    """Smooth sand-dollar-like fivefold contour, no claim of biological fidelity."""
    yy, xx = np.mgrid[:size, :size]
    x = (xx / (size - 1) * 2 - 1) - center[0]
    y = (yy / (size - 1) * 2 - 1) - center[1]
    theta = np.arctan2(y, x)
    distance = np.hypot(x, y)
    edge = radius + scallop * np.cos(5 * theta + phase)
    return (distance <= edge).astype(np.float32)


def touch_at(x, y, radius, scallop, phase, center):
    offset = np.asarray([x, y]) - center
    angle = np.arctan2(offset[1], offset[0])
    edge = radius + scallop * np.cos(5 * angle + phase)
    # A local indentation force, bounded to [0,1].
    return float(np.clip(12 * (edge - np.linalg.norm(offset)), 0, 1))


def _render_spring(position, size):
    yy, xx = np.mgrid[:size, :size]
    cx = (size - 1) / 2 + .35 * size * position
    cy = (size - 1) / 2
    bob = np.exp(-.5 * (((xx - cx) / 1.0) ** 2 +
                        ((yy - cy) / 1.3) ** 2))
    rgb = np.full((size, size, 3), .025, np.float32)
    rgb[..., 0] += .85 * bob
    rgb[..., 2] += .09
    return np.clip(rgb, 0, 1)


def _render_disc(shape, finger, size):
    yy, xx = np.mgrid[:size, :size]
    px = (finger[0] + 1) * .5 * (size - 1)
    py = (finger[1] + 1) * .5 * (size - 1)
    tip = np.exp(-.5 * (((xx - px) / .8) ** 2 +
                        ((yy - py) / .8) ** 2))
    rgb = np.full((size, size, 3), .025, np.float32)
    rgb += shape[..., None] * np.asarray([.25, .31, .33], np.float32)
    rgb[..., 0] += .65 * tip
    return np.clip(rgb, 0, 1)


def episode(seed, stage="spring", steps=20, size=8, grammar="small",
            stiffness="usual", blind=False):
    """Observation[t], action[t] -> observation[t+1]; truth is for scoring."""
    if (stage not in ("spring", "touch") or steps < 4 or size < 8
            or grammar not in ("small", "composed")
            or stiffness not in ("usual", "stiff")):
        raise ValueError("Invalid embodied episode")
    rng = np.random.default_rng(int(seed))
    choices = SMALL if grammar == "small" else COMPOSED
    programs = []
    for t in range(steps):
        if t % 3 == 0:
            command = choices[int(rng.integers(len(choices)))]
        programs.append(command)
    tree = np.stack([parse(p) for p in programs])
    motor = np.asarray([interpret(p) for p in programs], np.float32)
    axes = np.eye(2, dtype=np.float32)[
        np.zeros(steps, np.int32) if stage == "spring" else
        rng.integers(0, 2, steps)]
    observations = []
    if stage == "spring":
        stiffness_k = float(rng.uniform(.7, 1.25) if stiffness == "usual"
                            else rng.uniform(1.85, 2.45))
        mass = float(rng.uniform(.85, 1.15))
        damping = float(rng.uniform(.12, .22))
        x, velocity, act, act_v = float(rng.uniform(-.12, .12)), 0., 0., 0.
        for t in range(steps + 1):
            compression = max(0., abs(x) - .76)
            contact = min(1., 3 * compression)
            sound = (.20 * velocity + 1.2 * compression * abs(velocity)
                     + rng.normal(0, .004))
            observed = np.concatenate((_render_spring(x, size).ravel(),
                                      [act, act_v, contact, sound]))
            observations.append(observed.astype(np.float32))
            if t == steps:
                break
            force = float(motor[t]) * .42
            accel = (-stiffness_k * x - damping * velocity + force -
                     13 * np.sign(x) * compression) / mass
            velocity += .12 * accel
            x += .12 * velocity
            act_v = .6 * act_v + .4 * (force - act)
            act += .12 * act_v
        shape = np.zeros((size, size), np.float32)
    else:
        radius = float(rng.uniform(.42, .62))
        scallop = float(rng.uniform(.03, .09))
        phase = float(rng.uniform(-np.pi, np.pi))
        center = rng.uniform(-.09, .09, 2)
        shape = disc_shape(size, radius, scallop, phase, center)
        angle = rng.uniform(-np.pi, np.pi)
        finger = center + rng.uniform(radius - .14, radius + .13) * np.asarray(
            [np.cos(angle), np.sin(angle)])
        previous_touch = 0.
        for t in range(steps + 1):
            contact = touch_at(*finger, radius, scallop, phase, center)
            # Vibration is correlated with *change* in contact, not a label.
            vibration = .6 * abs(contact - previous_touch) + rng.normal(0, .004)
            observation = np.concatenate((_render_disc(shape, finger, size).ravel(),
                                          [*finger, contact, vibration]))
            observations.append(observation.astype(np.float32))
            previous_touch = contact
            if t == steps:
                break
            axis = int(np.argmax(axes[t]))
            finger[axis] = np.clip(finger[axis] + .05 * motor[t], -.95, .95)
    obs = np.stack(observations)
    stage_flag = np.tile(np.asarray([1., 0.] if stage == "spring"
                                    else [0., 1.], np.float32), (steps, 1))
    return {
        "observed": obs[:-1], "target": obs[1:], "tree": tree, "axes": axes,
        "stage": stage_flag, "shape": np.tile(shape, (steps, 1, 1)),
        "motor_label": motor, "blind": bool(blind),
    }


def dataset(start, count, stage="spring", steps=20, size=8, grammar="small",
            stiffness="usual", blind="mixed"):
    if count < 1 or blind not in ("mixed", "yes", "no"):
        raise ValueError("Invalid dataset parameters")
    records = [episode(start + i, stage, steps, size, grammar, stiffness,
                       blind=(blind == "yes" or (blind == "mixed" and i % 2 == 0)))
               for i in range(count)]
    arrays = {k: np.stack([rec[k] for rec in records])
              for k in ("observed", "target", "tree", "axes",
                        "stage", "shape", "motor_label")}
    arrays["blind"] = np.asarray([rec["blind"] for rec in records], bool)
    return arrays
