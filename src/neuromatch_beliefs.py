"""Small belief, object-memory, and information-gain references.

Inference receives only observations. Truth and identity labels remain in
experiment evaluators. No general-purpose object detector is claimed.
"""
import itertools
import math
import numpy as np


def kalman_track(observations, process_variance=0.01, observation_variance=0.04):
    """Constant-velocity filter [position, velocity] with missing observations."""
    if process_variance < 0 or observation_variance <= 0:
        raise ValueError("Invalid noise variances")
    if not observations:
        raise ValueError("Empty observations")
    f = np.array([[1., 1.], [0., 1.]])
    h = np.array([1., 0.])
    state = np.zeros(2)
    covariance = np.diag([100., 10.])
    estimate, uncertainty = [], []
    for t, observed in enumerate(observations):
        if t:
            state = f @ state
            covariance = f @ covariance @ f.T + process_variance * np.eye(2)
        if observed is not None:
            y = float(observed)
            if not math.isfinite(y):
                raise ValueError("Non-finite observation")
            innovation = y - h @ state
            s = float(h @ covariance @ h + observation_variance)
            k = covariance @ h / s
            state = state + k * innovation
            # Joseph form preserves covariance symmetry under finite precision.
            a = np.eye(2) - np.outer(k, h)
            covariance = a @ covariance @ a.T + observation_variance * np.outer(k, k)
        estimate.append(state.copy())
        uncertainty.append(covariance.copy())
    return np.stack(estimate), np.stack(uncertainty)


def hold_last(observations):
    """Simple causal position-persistence reference."""
    last = 0.
    output = []
    for observed in observations:
        if observed is not None:
            last = float(observed)
        output.append(last)
    return np.asarray(output)


def _cost(track, detection, appearance_weight):
    predicted_position = track["position"] + track["velocity"]
    return abs(predicted_position - detection[0]) + appearance_weight * abs(
        track["appearance"] - detection[1])


def update_object_memory(tracks, detections, t, appearance_weight=5.):
    """Follow two appearance-tagged detections with occlusion and crossing.

    A detection is (1D position, measured appearance). No object identity,
    hidden truth, or future observation is provided.
    """
    if t < 0 or appearance_weight < 0 or len(detections) > 2:
        raise ValueError("Invalid tracker inputs")
    detections = [(float(pos), float(appearance)) for pos, appearance in detections]
    if not all(math.isfinite(x) and math.isfinite(a) for x, a in detections):
        raise ValueError("Non-finite detection")
    if tracks is None:
        if len(detections) != 2 or t != 0:
            raise ValueError("Initialize with two unlabeled detections at t=0")
        return [{"position": p, "velocity": 0., "appearance": a,
                 "last_seen": t, "last_position": p} for p, a in detections]
    if len(tracks) != 2:
        raise ValueError("Expected two memory slots")
    result = [dict(track) for track in tracks]
    pairs = list(itertools.permutations(range(2), len(detections)))
    assignment = min(
        pairs, key=lambda perm: sum(
            _cost(result[slot], detection, appearance_weight)
            for slot, detection in zip(perm, detections))) if pairs else ()
    for track in result:
        track["position"] += track["velocity"]
    for slot, detection in zip(assignment, detections):
        position, appearance = detection
        previous = result[slot]
        dt = t - previous["last_seen"]
        if dt <= 0:
            raise ValueError("Time must advance")
        observed_velocity = (position - previous["last_position"]) / dt
        previous["velocity"] = 0.5 * previous["velocity"] + 0.5 * observed_velocity
        previous["position"] = position
        previous["appearance"] = 0.8 * previous["appearance"] + 0.2 * appearance
        previous["last_seen"] = t
        previous["last_position"] = position
    return result


def entropy(probabilities):
    x = np.asarray(probabilities, float)
    if x.ndim != 1 or not np.isfinite(x).all() or (x < 0).any() or x.sum() <= 0:
        raise ValueError("Invalid probabilities")
    x = x / x.sum()
    return float(-np.sum(x[x > 0] * np.log(x[x > 0])))


def expected_information_gain(prior, likelihood):
    """Mutual information I(hidden; observation | action), no future samples."""
    p = np.asarray(prior, float)
    model = np.asarray(likelihood, float)
    if p.ndim != 1 or model.ndim != 2 or model.shape[0] != len(p):
        raise ValueError("Expected prior [hidden] and likelihood [hidden, observed]")
    if not np.isfinite(model).all() or (model < 0).any() or not np.allclose(
            model.sum(axis=1), 1):
        raise ValueError("Invalid likelihood")
    prior_entropy = entropy(p)
    p = p / p.sum()
    marginal = p @ model
    conditional_entropy = 0.
    for j, probability in enumerate(marginal):
        if probability > 0:
            conditional_entropy += probability * entropy(p * model[:, j])
    return max(0., prior_entropy - conditional_entropy)


def select_active_view(prior, likelihoods):
    """Choose action by information gain; stable tie-break by input order."""
    if not likelihoods:
        raise ValueError("No candidate actions")
    scores = {name: expected_information_gain(prior, model)
              for name, model in likelihoods.items()}
    return max(scores, key=scores.get), scores


def replay_retention(report):
    """Augment the existing RGB replay results without re-training the NCA."""
    rows = []
    for entry in report["per_seed"]:
        stages = entry["stage_metrics"]
        if len(stages) < 3:
            raise ValueError("Expected A->B->A stages")
        task_a = stages[0]["task"]
        a0 = stages[0]["heldout_test"][task_a]
        a1 = stages[1]["heldout_test"][task_a]
        a2 = stages[2]["heldout_test"][task_a]
        rows.append({
            "seed": entry["seed"], "mode": entry["mode"],
            "initial_A_mse": a0, "A_after_B_mse": a1,
            "A_after_return_mse": a2,
            "forgetting_delta_mse": a1 - a0,
            "recovery_delta_mse": a2 - a1,
            "replay_updates": entry["replay_updates"],
            "memory_capacity": entry["memory_capacity"],
            "gradient_updates": entry["gradient_updates"],
        })
    paired = {}
    for seed in sorted({row["seed"] for row in rows}):
        by_mode = {r["mode"]: r for r in rows if r["seed"] == seed}
        if "replay" in by_mode and "recent_only" in by_mode:
            paired[str(seed)] = {
                "replay_minus_recent_forgetting": by_mode["replay"]["forgetting_delta_mse"]
                - by_mode["recent_only"]["forgetting_delta_mse"],
                "replay_minus_recent_final_A_mse": by_mode["replay"]["A_after_return_mse"]
                - by_mode["recent_only"]["A_after_return_mse"],
            }
    return {"per_seed": rows, "paired_by_seed": paired,
            "limitation": "MSE difference, not a hypothesis-test; memory reads and bytes not matched."}
