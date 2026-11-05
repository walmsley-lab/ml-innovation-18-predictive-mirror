"""Roadmap 26: depth from two observed views, with known camera geometry.

Two calibrated camera views reveal disparity; depth and plane identity remain
evaluation truth. The vertical bands are fixed observation ROIs, not
simulator-provided per-object masks. A one-view constant-depth control is kept.
"""
import argparse
import json
from pathlib import Path
import numpy as np

FOCAL_TIMES_BASELINE = 24.0
BANDS = ((3, 12), (17, 26))
DEPTHS = (4., 6., 8., 12.)


def shift(image, dx):
    """Zero-padded horizontal translation, never wrap-around."""
    image = np.asarray(image)
    out = np.zeros_like(image)
    if abs(dx) >= image.shape[1]:
        return out
    if dx >= 0:
        out[:,dx:] = image[:,:image.shape[1]-dx]
    else:
        out[:,:dx] = image[:,-dx:]
    return out


def pair(seed=11, size=32):
    if size < 28:
        raise ValueError("Need at least 28 pixels")
    rng = np.random.default_rng(seed)
    left = np.zeros((size,size), np.float32)
    depths = rng.choice(DEPTHS, size=2, replace=False)
    disparities = np.rint(FOCAL_TIMES_BASELINE / depths).astype(int)
    for j,(start,end) in enumerate(BANDS):
        texture = rng.uniform(.35, .9, (end-start,10))
        left[start:end, 10:20] = texture
    right = np.zeros_like(left)
    for j, (start,end) in enumerate(BANDS):
        right[start:end] = shift(left[start:end], -int(disparities[j]))
    return left, right, depths, disparities


def disparity(left, right, band, max_shift=8):
    start,end = band
    a,b = np.asarray(left)[start:end], np.asarray(right)[start:end]
    if a.shape != b.shape or a.ndim != 2 or max_shift < 1:
        raise ValueError("Stereo frame mismatch")
    costs = [np.mean(np.square(shift(a,-d)-b)) for d in range(1,max_shift+1)]
    return int(np.argmin(costs)+1)


def estimate(left, right, focal_times_baseline=FOCAL_TIMES_BASELINE):
    if focal_times_baseline <= 0:
        raise ValueError("Camera calibration must be positive")
    d=np.asarray([disparity(left,right,band) for band in BANDS])
    return focal_times_baseline/d, d


def run(out="runs/development-26/metrics.json", seeds=range(91000,91024)):
    errors = {"stereo": [], "monocular_constant": [], "shuffled_right": []}
    disparity_errors=[]
    for seed in seeds:
        left,right,depths,true_d = pair(int(seed))
        z,d = estimate(left,right)
        false_right = pair(int(seed)+10000)[1]
        z_shuffled,_ = estimate(left,false_right)
        errors["stereo"].extend(np.abs(z-depths).tolist())
        errors["monocular_constant"].extend(np.abs(np.full(2,8.)-depths).tolist())
        errors["shuffled_right"].extend(np.abs(z_shuffled-depths).tolist())
        disparity_errors.extend(np.abs(d-true_d).tolist())
    report={"roadmap_id":26,"episodes":len(errors["stereo"])//2,
            "depth_mae":{k:float(np.mean(v)) for k,v in errors.items()},
            "disparity_mae_px":float(np.mean(disparity_errors)),
            "geometry":"two fixed-band textured patches, calibrated binocular shift",
            "limits":"Synthetic stereo with known baseline, fixed regions and integer disparity. "
                     "This is not learned monocular depth, general 3D reconstruction, "
                     "or real-camera calibration. Truth is evaluator-only."}
    if out is not None:
        p=Path(out); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="runs/development-26/metrics.json")
    a=parser.parse_args()
    print(json.dumps(run(a.out),indent=2))


if __name__=="__main__":
    main()
