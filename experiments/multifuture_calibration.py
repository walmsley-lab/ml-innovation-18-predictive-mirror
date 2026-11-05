"""Compare coherent alternatives with a deterministic single-future floor."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import multifuture_motion as motion


def run(out="runs/multifuture/metrics.json",
        train_pairs=40,test_pairs=24,context=5,horizon=6,size=32):
    if train_pairs<2 or test_pairs<2:
        raise ValueError("Need independent paired trajectories")
    train=[motion.episode(31000+i,b,context,horizon,size)
           for i in range(train_pairs) for b in (0,1)]
    weights=motion.train_probabilities(train)
    records=[motion.episode(91000+i,b,context,horizon,size)
             for i in range(test_pairs) for b in (0,1)]
    rows=[]
    for r in records:
        prediction=motion.predict(r["frames"][:context],horizon,weights)
        truth=r["positions"][context:]
        metrics=motion.measures(prediction,truth,r["branch"])
        deterministic=motion.coast(r["frames"][:context],horizon)
        metrics["coast_ade_px"]=float(np.linalg.norm(
                deterministic-truth,axis=-1).mean())
        metrics["coast_energy_score"]=float(np.linalg.norm(
                deterministic-truth))
        rows.append({"seed":r["pair_seed"],"branch":r["branch"],**metrics})
    means={key:float(np.mean([row[key] for row in rows]))
           for key in ("energy_score","top1_ade_px","expected_ade_px",
                "oracle_best_of_many_ade_px","coast_ade_px",
                "coast_energy_score","branch_brier","trajectory_diversity_px")}
    empirical=np.mean([r["branch"] for r in records])
    report={"protocol":"ambiguous-coherent-multifuture-v1",
            "train_pairs":train_pairs,"test_pairs":test_pairs,
            "train_seed_start":31000,"test_seed_start":91000,
            "context":context,"horizon":horizon,"branch_probabilities":weights.tolist(),
            "empirical_test_branch_1_frequency":float(empirical),
            "calibration_error_branch_1":float(abs(empirical-weights[1])),
            "aggregate":means,"rows":rows,
            "limits":"Known two-branch hypothesis family with supervised "
              "training-only frequency calibration. This is a strong "
              "coherent simple baseline, NOT a learned multimodal NCA. "
              "Predictor sees RGB past only, not hidden branch or future. "
              "Best-of-many ADE is diagnostic, not a proper score. "
              "Paired branches sharing one observed history must be "
              "resampled as one independent evaluation unit."}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/multifuture/metrics.json")
    p.add_argument("--train-pairs",type=int,default=40)
    p.add_argument("--test-pairs",type=int,default=24)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.train_pairs,a.test_pairs)["aggregate"],indent=2))


if __name__=="__main__":
    main()
