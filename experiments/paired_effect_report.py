"""Summarize frozen paired-history success rates without pixel pseudoreplication.

Input JSON: candidate_scores [model_seed][scene_pair]; baseline_scores may
be 1D [scene_pair] for a deterministic fixed baseline, or matching 2D.
"""
import argparse
import json
from pathlib import Path
from src.paired_statistics import paired_effect, pilot_precision


def run(input_path, out="runs/paired-effects/report.json", draws=2000,
        margin=0.10):
    data=json.loads(Path(input_path).read_text())
    required=("candidate_scores","baseline_scores","candidate","baseline",
              "scene_pair_ids","model_seeds")
    if any(k not in data for k in required):
        raise ValueError("Missing required paired-effect input fields")
    model_scores=data["candidate_scores"]
    if len(model_scores)!=len(data["model_seeds"]) or any(
            len(row)!=len(data["scene_pair_ids"]) for row in model_scores):
        raise ValueError("Model scores must align with declared seeds and scene pairs")
    if len(set(data["scene_pair_ids"]))!=len(data["scene_pair_ids"]):
        raise ValueError("Scene pair IDs must be unique")
    if len(set(data["model_seeds"]))!=len(data["model_seeds"]):
        raise ValueError("Model seed IDs must be unique")
    effect=paired_effect(model_scores,data["baseline_scores"],
                         draws=draws,practical_margin=margin)
    precision=pilot_precision(model_scores,data["baseline_scores"])
    report={"candidate":data["candidate"],"baseline":data["baseline"],
            "metric":data.get("metric","on_time_reappearance_success_rate"),
            "effect":effect,"pilot_precision":precision,
            "provenance":"Inputs supplied by caller; result validity and "
                         "checkpoint provenance must be verified separately"}
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--out",default="runs/paired-effects/report.json")
    p.add_argument("--draws",type=int,default=2000)
    p.add_argument("--margin",type=float,default=.10)
    a=p.parse_args()
    report=run(a.input,a.out,a.draws,a.margin)
    print(json.dumps(report["effect"],indent=2))


if __name__=="__main__":
    main()
