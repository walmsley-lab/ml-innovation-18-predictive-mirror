"""Roadmap 25: learned partial-object completion from previously seen images.

Self-supervised masking: training targets are fully observed RGB frames, with
synthetic masks added after observation. No simulator shape parameters/IDs are
passed to the model; test truth is evaluator-only. Fully hidden objects are a
distinct task requiring temporal memory, not single-frame completion.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def silhouette(seed, size=16):
    if size < 12:
        raise ValueError("size >= 12 required")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    cx, cy = rng.integers(5, size-4, size=2)
    kind = seed % 3
    if kind == 0:
        shape = (np.abs(x-cx) <= 3) & (np.abs(y-cy) <= 2)
    elif kind == 1:
        shape = ((np.abs(x-cx) <= 1) & (np.abs(y-cy) <= 4)
                 | (np.abs(y-cy) <= 1) & (np.abs(x-cx) <= 4))
    else:
        shape = ((x-cx)**2 + (y-cy)**2 <= 13)
    rgb = np.zeros((size,size,3), np.float32)
    rgb[shape] = [.90,.90,.90]
    return rgb, shape


def obscure(clean, seed=0, full=False):
    size = len(clean)
    rgb = clean.copy()
    stripe = np.zeros((size,size), bool)
    if full:
        stripe[:] = True
    else:
        x = int(size//2 + seed % 3-1)
        stripe[:, max(0,x-2):min(size,x+2)] = True
    rgb[stripe] = [0.15, 0.15, 0.85]
    return rgb


def features(rgb):
    """Color of visible occluder is part of the observed image."""
    rgb = np.asarray(rgb, float)
    object_pixels = (rgb[...,0] > .6).astype(float)
    occluder = (rgb[...,2] > .6).astype(float)
    return np.concatenate((object_pixels.ravel(), occluder.ravel(), [1.]))


def fit(seeds=range(31000,31100), regularization=2.0):
    data, truth = [], []
    for seed in seeds:
        clean, _ = silhouette(seed)
        data.append(features(obscure(clean, seed)))
        # Target is the previously unobscured observed RGB frame.
        truth.append((clean[...,0] > .6).astype(float).ravel())
    x = np.asarray(data)
    y = np.asarray(truth)
    # Dual ridge for small CPU runs; regularization keeps the inverse stable.
    return x.T @ np.linalg.solve(x@x.T + regularization*np.eye(len(x)), y)


def predict(weights, current, clip=True):
    scores = features(current) @ weights
    side = current.shape[0]
    return np.clip(scores,0,1).reshape(side,side) if clip else scores.reshape(side,side)


def measure(pred, truth, occluded):
    err = (pred-truth.astype(float))**2
    return {"hidden_mse": float(err[occluded].mean()) if occluded.any() else None,
            "whole_mse": float(err.mean()),
            "object_iou": float(np.logical_and(pred>.5, truth).sum() /
                max(1, np.logical_or(pred>.5, truth).sum()))}


def run(out="runs/development-25/metrics.json", train_count=80, test_count=20):
    if train_count < 4 or test_count < 1:
        raise ValueError("Need training and held-out samples")
    weights = fit(range(31000,31000+train_count))
    results = {"partial": [], "fully_hidden": []}
    for seed in range(91000,91000+test_count):
        clean, truth = silhouette(seed)
        for mode in results:
            image = obscure(clean, seed, full=(mode=="fully_hidden"))
            occluded = image[...,2] > .6
            learned = predict(weights, image)
            observed_only = (image[...,0] > .6).astype(float)
            results[mode].append({
                "learned": measure(learned, truth, occluded),
                "visible_only": measure(observed_only, truth, occluded)})
    means = {mode:{arm:{key:float(np.mean([r[arm][key] for r in rows]))
                        for key in ("hidden_mse","whole_mse","object_iou")}
                  for arm in ("learned","visible_only")}
             for mode, rows in results.items()}
    report = {"roadmap_id":25, "train_count":train_count, "test_count":test_count,
              "scores":means, "full_hidden_is_separate":True,
              "limits":"Linear self-supervised masked reconstruction from formerly visible "
                       "images; not learned object permanence. Fully hidden single-frame "
                       "completion is unidentifiable without history or stronger assumptions."}
    if out is not None:
        p=Path(out); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/development-25/metrics.json")
    p.add_argument("--train-count",type=int,default=80)
    p.add_argument("--test-count",type=int,default=20)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.train_count,a.test_count),indent=2))


if __name__=="__main__":
    main()
