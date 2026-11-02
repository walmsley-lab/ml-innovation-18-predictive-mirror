"""Small moving-object active-gaze world with action-aware and blind controls.

A camera sees a short one-dimensional window. The policy gets only its
posterior belief, not the hidden target location. State dynamics and
observations are intentionally known; this is a sensing CONTROL baseline,
not a neural reinforcement-learning or video model.
"""
import numpy as np

WORLD = 32
FIELD = 8
START = 12
VELOCITIES = (-1, 0, 1)
ACTIONS = (-4, 0, 4)


def bounce(x, velocity):
    """Reflect a moving point inside a finite 1D world."""
    moved = x + velocity
    if moved < 0 or moved >= WORLD:
        return (x - velocity, -velocity)
    return moved, velocity


def states():
    return [(x, v) for x in range(WORLD) for v in VELOCITIES]


STATES = states()
NEXT = tuple(STATES.index(bounce(x, v)) for x, v in STATES)


def step_belief(belief):
    """One model timestep of state propagation; no true-state input."""
    return np.bincount(NEXT, weights=belief, minlength=len(STATES))


def sensor(x, view):
    """An observed LOCAL pixel offset, or -1 when target is outside crop."""
    return int(x-view) if view <= x < view+FIELD else -1


def posterior(prior, view, measurement):
    likelihood = np.asarray([sensor(x, view) == measurement
                             for x, _ in STATES], dtype=bool)
    result = prior * likelihood
    mass = result.sum()
    if mass <= 0:
        # An impossible observation must be flagged rather than silently
        # interpreted as successful inference.
        raise ValueError("Observation impossible under current belief")
    return result/mass


def future_position(belief, horizon):
    if horizon < 0:
        raise ValueError("Horizon cannot be negative")
    predicted = belief
    for _ in range(horizon):
        predicted = step_belief(predicted)
    return np.bincount(
        [x for x, _ in STATES], weights=predicted, minlength=WORLD)


def risk_position(distribution):
    xs = np.arange(WORLD)
    center = float(xs @ distribution)
    return float(np.sum(distribution * (xs-center)**2))


def expected_risk(belief, future_view, horizon):
    """Before the next observation, expected future location variance."""
    next_prior = step_belief(belief)
    total = 0.
    for reading in (-1, *range(FIELD)):
        likelihood = np.asarray(
            [sensor(x, future_view) == reading for x, _ in STATES])
        probability = float(next_prior[likelihood].sum())
        if probability > 0:
            updated = posterior(next_prior, future_view, reading)
            total += probability*risk_position(future_position(updated, horizon))
    return total


def choose_view(belief, current_view, horizon):
    """Greedy information-gathering action; no ground-truth input."""
    options = [(expected_risk(belief, int(np.clip(current_view+a, 0,
                    WORLD-FIELD)), horizon), abs(a), a) for a in ACTIONS]
    return min(options)[2]


def trajectory(seed, context, horizon):
    """Generate hidden simulation truth; never provide it to choose_view."""
    rng = np.random.default_rng(int(seed))
    x = int(rng.integers(WORLD))
    v = int(rng.choice(VELOCITIES))
    truth = [(x, v)]
    for _ in range(context+horizon-1):
        x, v = bounce(x, v)
        truth.append((x, v))
    return truth


def episode(seed, policy, context=8, horizon=10):
    if policy not in ("fixed", "random", "active") or context < 2 or horizon < 1:
        raise ValueError("Invalid gaze policy, context or horizon")
    ground_truth = trajectory(seed, context, horizon)
    posterior_aware = np.ones(len(STATES))/len(STATES)
    posterior_blind = posterior_aware.copy()
    rng = np.random.default_rng(int(seed)+33000)
    view = START
    readings = []
    for t in range(context):
        if t:
            posterior_aware = step_belief(posterior_aware)
            posterior_blind = step_belief(posterior_blind)
        observed = sensor(ground_truth[t][0], view)
        posterior_aware = posterior(posterior_aware, view, observed)
        # Matched action-blind control: assumes camera stayed at START
        # although it receives the same local measurement.
        try:
            posterior_blind = posterior(posterior_blind, START, observed)
        except ValueError:
            # Wrong stationary-camera hypothesis was contradicted. Record
            # failure, recover to a uniform prior rather than use hidden truth.
            posterior_blind = posterior(
                np.ones(len(STATES))/len(STATES), START, observed)
        readings.append({"step":t,"view":int(view),"local_observation":observed})
        if t < context-1:
            if policy == "active":
                action = choose_view(posterior_aware, view, horizon)
            elif policy == "random":
                action = int(rng.choice(ACTIONS))
            else:
                action = 0
            view = int(np.clip(view+action, 0, WORLD-FIELD))
    target_x = ground_truth[context+horizon-1][0]
    predictions = {}
    for name, belief in (("action_aware", posterior_aware),
                         ("action_blind", posterior_blind)):
        distribution = future_position(belief, horizon)
        predicted_x = float(np.arange(WORLD) @ distribution)
        predictions[name] = {
            "predicted_x": predicted_x,
            "absolute_error": abs(predicted_x-target_x),
            "forecast_entropy_nats": float(
                -np.sum(distribution*np.log(distribution+1e-12)))}
    return {"policy":policy,"seed":int(seed),"target_future_x":int(target_x),
            "views":readings,"predictions":predictions}


def compare(seeds=(11, 23, 37), episodes=24, context=8, horizon=10):
    if not seeds or episodes < 1 or context < 2 or horizon < 1:
        raise ValueError("Invalid benchmark settings")
    policies=("fixed","random","active")
    results={}
    for policy in policies:
        rows=[episode(int(seed)*100000 + index,policy,context,horizon)
              for seed in seeds for index in range(episodes)]
        results[policy]={
            "runs":len(rows),
            "mean_position_error":{
                arm:float(np.mean([r["predictions"][arm]["absolute_error"]
                                   for r in rows]))
                for arm in ("action_aware","action_blind")},
            "mean_forecast_entropy":{
                arm:float(np.mean([r["predictions"][arm]["forecast_entropy_nats"]
                                   for r in rows]))
                for arm in ("action_aware","action_blind")},
            "sample_trace":rows[0]}
    return {"experiment":"active_gaze_moving_object_known_dynamics",
            "seeds":list(map(int,seeds)),"episodes_per_seed":episodes,
            "context":context,"horizon":horizon,
            "policies":results,
            "limitations":"Belief-state dynamics and 1D observation likelihoods are "
                          "known to the agent; this is NOT learned world modeling or a "
                          "neural policy. Action-blind control incorrectly assumes a "
                          "fixed camera and can become inconsistent, in which case it "
                          "resets to an uninformative prior. No future hidden truth is "
                          "available to policy or filters. Action-aware and blind "
                          "predictions see the same observations and actions under "
                          "each policy. Synthetic steps are not guaranteed wall-clock time."}
