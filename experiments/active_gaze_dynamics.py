"""Compare fixed, random and information-seeking gaze on a moving 1D target."""
import argparse
import json
from pathlib import Path

from src.active_gaze_dynamics import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/active-gaze-dynamics/metrics.json")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,37])
    p.add_argument("--episodes",type=int,default=24)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    a=p.parse_args()
    report=compare(a.seeds,a.episodes,a.context,a.horizon)
    target=Path(a.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(target),
                      "policies":{k:v["mean_position_error"]
                                  for k,v in report["policies"].items()}},indent=2))


if __name__=="__main__":
    main()
