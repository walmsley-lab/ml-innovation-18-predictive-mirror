"""CPU probe: python experiments/camera_return_memory.py --seed 11."""
import argparse
import json
import numpy as np

WIDTH = 32
HORIZON = 12


def episode(position, direction, stride):
    """Two pose branches share the first image. A screen hides the midpoint."""
    steps = np.arange(HORIZON + 1)
    poses = direction * stride * np.minimum(steps, HORIZON - steps)
    locations = position - poses
    frames = np.eye(WIDTH)[locations]
    frames[4:9] = 0
    return poses, frames, locations


def features(state, pose, delta, projection):
    controls = [pose / 12, delta / 2, 1]
    return np.tanh(np.r_[state, controls] @ projection)


def run(seed):
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(WIDTH + 3, WIDTH))
    positions = rng.permutation(np.arange(12, 20))
    result = {}
    for recurrent in (False, True):
        inputs, targets = [], []
        for position in positions[:6]:
            for direction in (-1, 1):
                for stride in (1, 2):
                    poses, frames, _ = episode(position, direction, stride)
                    state = frames[0]
                    for t in range(1, len(poses)):
                        hidden = features(state, poses[t], poses[t]-poses[t-1], projection)
                        inputs.append(hidden)
                        targets.append(frames[t])
                        state = hidden if recurrent else frames[t]
        x = np.asarray(inputs)
        readout = np.linalg.solve(x.T @ x + .01 * np.eye(WIDTH), x.T @ targets)
        scores = []
        for position in positions[6:]:
            for direction in (-1, 1):
                for stride in (1, 2):
                    poses, frames, locations = episode(position, direction, stride)
                    state = frames[0]
                    predictions = []
                    for t in range(1, len(poses)):
                        hidden = features(state, poses[t], poses[t]-poses[t-1], projection)
                        prediction = hidden @ readout
                        predictions.append(prediction)
                        state = hidden if recurrent else prediction
                    predictions = np.asarray(predictions)
                    visible = predictions.max(axis=1) >= .5
                    hits = np.flatnonzero(visible[3:]) + 4
                    scores.append([abs(predictions[8].argmax()-locations[9]),
                                   abs(predictions[-1].argmax()-position),
                                   abs(hits[0]-9) if hits.size else HORIZON,
                                   visible[3:8].mean()])
        result['recurrent' if recurrent else 'frame_feedback'] = dict(zip(
            ['reappearance_px', 'return_px', 'onset_error_steps_miss_penalty_12',
             'occluded_false_visible_rate'], np.mean(scores, axis=0)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()
    print(json.dumps(run(args.seed), indent=2))
