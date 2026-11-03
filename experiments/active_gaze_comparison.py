"""Evidence-first paired comparison of fixed/random/active 1D gaze."""
import argparse
import json
from pathlib import Path
from src.active_gaze_comparison import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/active-gaze-comparison/metrics.json")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,37])
    p.add_argument("--episodes",type=int,default=24)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    a=p.parse_args()
    report=compare(a.seeds,a.episodes,a.context,a.horizon)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["paired_comparisons"],indent=2))


if __name__=="__main__":
    main()
