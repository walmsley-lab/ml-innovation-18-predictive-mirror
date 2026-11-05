"""61: select a frozen visual model on validation performance and hardware.

The selector never sees test metrics. Every candidate must carry a trained
checkpoint SHA, comparable validation protocol and measured latency on the
target machine; missing metrics fail closed, not imputed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def validate(candidates):
    if not candidates:
        raise ValueError("Need one or more frozen trained candidates")
    fields=("name","checkpoint_sha256","validation_mse","latency_ms",
            "device","validation_session_sha256","protocol")
    protocol={(r.get("protocol"),r.get("validation_session_sha256"),
               r.get("device")) for r in candidates}
    if len(protocol)!=1:
        raise ValueError("Candidates must share validation split, protocol and hardware")
    for r in candidates:
        if any(not r.get(k) for k in fields if k not in ("validation_mse","latency_ms")):
            raise ValueError("Candidate missing required provenance")
        if r.get("evidence") not in (
                "trained_checkpoint","checkpoint_loaded_training_not_independently_verified"):
            raise ValueError("Require explicit trained-checkpoint evidence status")
        for k in ("validation_mse","latency_ms"):
            value=r.get(k)
            if value is None or not np.isfinite(float(value)) or (
                    float(value)<0 or (k=="latency_ms" and float(value)<=0)):
                raise ValueError("Candidate lacks finite measured "+k)
    if len({r["name"] for r in candidates})!=len(candidates):
        raise ValueError("Duplicate model entries")


def select(candidates,max_latency_ms):
    validate(candidates)
    if max_latency_ms<=0:
        raise ValueError("Hardware latency budget must be positive")
    eligible=[r for r in candidates if r["latency_ms"]<=max_latency_ms]
    if not eligible:
        return None
    return min(eligible,key=lambda r:(r["validation_mse"],r["latency_ms"],
                                      r["name"]))


def run(validations,test=None,out="runs/roadmap-61/selection.json",
        max_latency_ms=100.):
    candidates=json.loads(Path(validations).read_text())["candidates"]
    selected=select(candidates,max_latency_ms)
    selection_input=json.dumps({"candidates":candidates,"budget":max_latency_ms},
                               sort_keys=True).encode()
    selection_digest=hashlib.sha256(selection_input).hexdigest()
    result={"roadmap_id":61,"selected":selected,
            "selection_sha256":selection_digest,
            "latency_budget_ms":max_latency_ms,
            "status":"no_eligible_model" if selected is None else "selected_from_validation",
            "test":None,
            "limitations":"Selection uses only validation and per-device measured "
            "latency; benchmark comparability relies on truthful provenance. "
            "Do not use test accuracy to repair selection rules. "
            "A stale device/timing invalidates the gate."}
    if test:
        if selected is None:
            raise ValueError("Cannot score held-out test without selected model")
        unseen=json.loads(Path(test).read_text())
        if unseen.get("checkpoint_sha256")!=selected["checkpoint_sha256"]:
            raise ValueError("Test checkpoint differs from frozen selection")
        if unseen.get("test_session_sha256")==selected["validation_session_sha256"]:
            raise ValueError("Validation session reused as test")
        result["test"]={"source_sha256":unseen["test_session_sha256"],
                        "selected_model_mse":unseen["mse"],
                        "source":"separately produced independent test report"}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-manifest",required=True)
    p.add_argument("--test-report")
    p.add_argument("--max-latency-ms",type=float,default=100.)
    p.add_argument("--out",default="runs/roadmap-61/selection.json")
    a=p.parse_args()
    print(json.dumps(run(a.validation_manifest,a.test_report,a.out,
                         a.max_latency_ms),indent=2))


if __name__=="__main__":
    main()
