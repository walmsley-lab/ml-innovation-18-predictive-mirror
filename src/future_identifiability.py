"""Three distinct information regimes for paired-history visual prediction.

The unobserved-turn counterfactual allows a hidden stochastic reversal at the
last observed occluded position; it is NOT the deterministic paired benchmark.
"""
import numpy as np
from src import paired_occlusion_histories as paired


REGIMES = ("history_identifiable", "unobserved_turn_ambiguous", "same_future_null")


def pair(seed=11, regime="history_identifiable", context=8, horizon=10):
    if regime not in REGIMES:
        raise ValueError("Unknown identifiability regime")
    a, b = paired.pair(seed, context, horizon)
    pa = a["frames"][:context].copy()
    fa = a["frames"][context:context+horizon].copy()
    pb = b["frames"][:context].copy()
    fb = b["frames"][context:context+horizon].copy()
    if regime == "history_identifiable":
        pass  # Opposing observed approach history, fixed continuation rule.
    elif regime == "unobserved_turn_ambiguous":
        # Both branches begin with exactly the same history; the unobserved
        # random turn at the occluded last position selects +/- future motion.
        pb = pa.copy()
    else:
        pb, fb = pa.copy(), fa.copy()
    if not np.array_equal(pa[-1], pb[-1]):
        raise AssertionError("Paired cases must have identical last image")
    identical_prefix = bool(np.array_equal(pa, pb))
    identical_future = bool(np.array_equal(fa, fb))
    expected = {"history_identifiable": (False, False),
                "unobserved_turn_ambiguous": (True, False),
                "same_future_null": (True, True)}[regime]
    if (identical_prefix, identical_future) != expected:
        raise AssertionError("Regime is not identifiable as specified")
    return {"regime": regime, "seed": int(seed),
            "contexts": np.stack([pa,pb]), "futures": np.stack([fa,fb]),
            "identical_full_context": identical_prefix,
            "identical_future": identical_future,
            "information_rule": ("different full histories" if not identical_prefix
                else ("unobserved binary direction decision at last hidden point"
                      if not identical_future else "same deterministic future"))}


def ambiguity_bounds(futures):
    """Equal-probability conditional point-MSE Bayes risk for two branches.

    This is a mathematical **lower bound** for a deterministic point forecast
    given identical full context; it is not a trained model's result.
    """
    x=np.asarray(futures,np.float64)
    if x.ndim != 5 or x.shape[0] != 2 or x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError("Need two finite RGB futures [2,time,height,width,3]")
    midpoint=x.mean(axis=0)
    expected=float(np.square(x-midpoint).mean())
    guessing_first=float(np.square(x-x[0]).mean())
    return {"conditional_mean_mse":expected,
            "single_branch_guess_mse":guessing_first,
            "oracle_privileged_mse":0.0,
            "probabilities":[0.5,0.5],
            "scope":"equally likely hidden branch, identical complete observations"}


def audit(seeds=(11,23,41)):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Need unique scene seeds")
    records=[]
    for seed in seeds:
        for regime in REGIMES:
            test=pair(int(seed),regime)
            results={"seed":int(seed),"regime":regime,
                     "identical_full_context":test["identical_full_context"],
                     "identical_future":test["identical_future"]}
            if regime=="unobserved_turn_ambiguous":
                results["optimal_deterministic_point_mse"] = ambiguity_bounds(
                    test["futures"])["conditional_mean_mse"]
            records.append(results)
    return {"purpose":"information-theoretic evaluator audit, no learned model",
            "regimes":list(REGIMES),"records":records}
