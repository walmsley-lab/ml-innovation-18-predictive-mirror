"""Roadmap 32: prediction-first adaptation through recurrent domain shifts.

A deliberately small observed-feature regression workbench. The model commits
its forecast before the target arrives; SGD may use only *past* feedback.
The evaluator knows regime boundaries, the predictor does not.
"""
import argparse
import json
from pathlib import Path
import numpy as np

WEIGHTS = {
    "A": np.array([.35, 1.3, -.55]),
    "B": np.array([-.55, -.4, 1.15]),
}


def samples(seed, periods=(60,60,60)):
    if not periods or min(periods) < 3:
        raise ValueError("Need nonempty observable regimes")
    rng = np.random.default_rng(int(seed))
    x, y, labels = [], [], []
    for regime, length in zip(("A","B","A"), periods):
        features = np.column_stack((np.ones(length), rng.normal(size=(length,2))))
        targets = features @ WEIGHTS[regime] + rng.normal(0,.12,length)
        x.append(features)
        y.append(targets)
        labels.extend([regime] * length)
    return np.concatenate(x), np.concatenate(y), np.asarray(labels)


def initialize(seed=31000, episodes=3):
    if episodes < 1:
        raise ValueError("Need independent source episodes")
    x, y = [], []
    for i in range(episodes):
        features, targets, _ = source(seed+i)
        x.append(features)
        y.append(targets)
    xx, yy = np.concatenate(x), np.concatenate(y)
    return np.linalg.solve(xx.T@xx + .01*np.eye(xx.shape[1]), xx.T@yy)


def source(seed, count=60):
    rng = np.random.default_rng(int(seed))
    features = np.column_stack((np.ones(count),rng.normal(size=(count,2))))
    targets = features @ WEIGHTS["A"] + rng.normal(0,.12,count)
    return features, targets, np.repeat("A",count)


def prequential(initial, x, y, mode="adaptive", lr=.05, clip=2.0):
    """Every entry records a score before the same target can be learned."""
    if mode not in ("adaptive","frozen","stale_feedback") or lr <= 0 or clip <= 0:
        raise ValueError("Bad adaptation mode")
    x, y = np.asarray(x,float), np.asarray(y,float)
    if x.ndim != 2 or x.shape[0] != len(y) or x.shape[1] != len(initial):
        raise ValueError("Input/parameter mismatch")
    weights=np.asarray(initial,float).copy()
    prediction=[]
    previous = None
    for features, target in zip(x, y):
        prediction.append(float(features @ weights))
        # Loss and score are committed above, before reading current label
        # for a future update. Wrong-feedback arm uses previously observed
        # y with current x, never oracle or future values.
        if mode!="frozen":
            received = target if mode=="adaptive" else (
                previous if previous is not None else prediction[-1])
            error = float(np.clip(features @ weights - received,-clip,clip))
            weights -= lr * error * features / max(1., float(features @ features))
        previous = target
    return np.asarray(prediction), weights


def segment_scores(prediction, truth, period=60):
    error = np.square(np.asarray(prediction)-np.asarray(truth))
    if len(error) != 3*period:
        raise ValueError("Expected A/B/A evaluation stream")
    return {
        "source_before_shift":float(error[:period].mean()),
        "new_regime":float(error[period:2*period].mean()),
        "source_return":float(error[2*period:].mean()),
        "shift_first_ten":float(error[period:period+10].mean()),
        "shift_last_ten":float(error[2*period-10:2*period].mean()),
        "return_first_ten":float(error[2*period:2*period+10].mean()),
        "return_last_ten":float(error[-10:].mean()),
        "overall":float(error.mean()),
    }


def select_rate(initial, candidates=(.02,.06,.12),period=60):
    """Tune on validation only; no test seed participates."""
    valid=[samples(61000+i,(period,period,period)) for i in range(3)]
    rows={}
    for rate in candidates:
        errors=[]
        for x,y,_ in valid:
            pred,_=prequential(initial,x,y,"adaptive",rate)
            errors.append(float(np.mean(np.square(pred-y))))
        rows[float(rate)]=float(np.mean(errors))
    return min(rows,key=lambda rate:(rows[rate],rate)),rows


def run(out="runs/roadmap-32/metrics.json", test_seeds=(91000,91001,91002), period=60):
    if period < 20 or not test_seeds:
        raise ValueError("Need adequate independent streams")
    initial=initialize()
    rate,validation=select_rate(initial,period=period)
    results={name:[] for name in ("adaptive","frozen","stale_feedback")}
    for seed in test_seeds:
        x,y,_=samples(int(seed),(period,period,period))
        for mode in results:
            predicted,_=prequential(initial,x,y,mode,rate)
            results[mode].append(segment_scores(predicted,y,period))
    aggregate={
        mode:{metric:float(np.mean([row[metric] for row in entries]))
              for metric in entries[0]} for mode,entries in results.items()}
    report={
        "roadmap_id":32,"seed_list":list(map(int,test_seeds)),
        "source_seed_start":31000,"validation_seed_start":61000,
        "selected_learning_rate":rate,"validation_rate_mse":validation,
        "period":period,"test":aggregate,"per_seed":results,
        "limits":"Observed feature proxy with scored-before-update supervised feedback, "
                 "not RGB model adaptation. Regime boundaries are evaluator-only. "
                 "Shifts are engineered and recurrent; on-device optimization cost "
                 "and longer-term memory are assessed separately."}
    if out is not None:
        path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-32/metrics.json")
    a=p.parse_args()
    print(json.dumps(run(a.out)["test"],indent=2))


if __name__=="__main__":
    main()
