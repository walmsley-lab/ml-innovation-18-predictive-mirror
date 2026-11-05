"""Audit observable history vs stochastic future ambiguity with no training."""
import argparse
import json
from pathlib import Path
from src.future_identifiability import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    p.add_argument("--out",default="runs/future-identifiability-audit.json")
    a=p.parse_args()
    report=audit(a.seeds)
    dest=Path(a.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"regimes":report["regimes"],
                      "count":len(report["records"]),"model_scores":"none"},indent=2))


if __name__=="__main__":
    main()
