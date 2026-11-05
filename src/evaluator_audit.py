"""Deterministic observable-input and scorer controls for synthetic occlusions.

This audits test harness invariants; it is not a trained-model evaluation.
"""
import numpy as np
from src import guaranteed_reappearance as guaranteed
from src import paired_occlusion_histories as paired
from src import reappearance_scoring as scoring


def identifiable(prefix_a, prefix_b, future_a, future_b):
    """Whether the *observed* histories can distinguish conflicting futures.

    When inputs are identical but futures differ, deterministic future choice is
    unidentifiable from these observations; do not score an impossible winner.
    """
    a, b, fa, fb = map(np.asarray, (prefix_a, prefix_b, future_a, future_b))
    if a.shape != b.shape or fa.shape != fb.shape:
        raise ValueError("Mismatched paired shapes")
    if np.array_equal(fa, fb):
        return "same_future"
    return "identifiable_history" if not np.array_equal(a, b) else "ambiguous"


def validate_observation(record, context):
    """The only predictor-facing data is the observed RGB prefix."""
    actual = guaranteed.model_inputs(record)
    if set(actual) != {"frames"} or actual["frames"].shape != (
            context, *record["frames"].shape[1:]):
        raise ValueError("Predictor interface must expose past RGB only")
    np.testing.assert_array_equal(actual["frames"], record["frames"][:context])
    return True


def inspect_pair(seed=11, context=8, horizon=10):
    a, b = paired.pair(seed, context, horizon)
    if not np.array_equal(a["frames"][context-1], b["frames"][context-1]):
        raise ValueError("Pair final observed frames differ")
    if not np.any(a["frames"][:context-1] != b["frames"][:context-1]):
        raise ValueError("Pair histories do not differ")
    label = identifiable(a["frames"][:context], b["frames"][:context],
                         a["frames"][context:], b["frames"][context:])
    if label != "identifiable_history":
        raise ValueError("Paired history must identify opposing futures")
    return {"seed": seed, "case": label, "final_frame_equal": True,
            "future_equal": False}


def audit_controls(seed=11, context=4, horizon=10, mode="barrier"):
    record = guaranteed.episode(seed, context, horizon, mode=mode)
    validate_observation(record, context)
    truth = record["frames"][context:context+horizon][None]
    black = np.zeros_like(truth)
    # Roll zeroed images as a separate bad forecast control; ensures the
    # score does not mistake a missing object for a valid reappearance.
    perfect = scoring.score(truth, [record], context)["results"][0]["events"][0]
    absent = scoring.score(black, [record], context)["results"][0]["events"][0]
    if (perfect["missed_reappearance"] or perfect["timing_error_frames"] != 0
            or perfect["reappearance_centroid_error_px"] is None):
        raise ValueError("Positive oracle control failed")
    if (not absent["missed_reappearance"]
            or absent["reappearance_centroid_error_px"] is not None):
        raise ValueError("Missing-object negative control failed")
    return {"mode": mode, "oracle_correct": True, "blank_rejected": True,
            "oracle_only": True}


def suite(seeds=(11, 23, 41)):
    return {
        "purpose": "Synthetic evaluator and information-boundary controls only",
        "paired": [inspect_pair(int(s)) for s in seeds],
        "barrier": [audit_controls(int(s), mode="barrier") for s in seeds],
        "camera_exit": [audit_controls(int(s), mode="camera_exit") for s in seeds],
        "learned_object_permanence": "not_evaluated"
    }
