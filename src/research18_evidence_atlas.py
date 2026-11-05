"""No cross-protocol scoreboards: group evidence before aggregate inference."""
import json
from collections import defaultdict
from pathlib import Path

EVIDENCE=("hypothesis","simulation","observation","inference","reflection")
REQUIRED=("claim","experiment","protocol","seed","evidence",
          "metric","unit","value","scope","source_revision")

def validate(row):
    if any(key not in row for key in REQUIRED):
        raise ValueError("Missing evidence/provenance field")
    if row["evidence"] not in EVIDENCE:
        raise ValueError("Unsupported evidence class")
    if not isinstance(row["value"],(int,float)):
        raise ValueError("Metric value must be numeric")
    if not row["scope"] or not row["protocol"] or not row["source_revision"]:
        raise ValueError("Unscoped evidence")
    return row

def grouping_key(row):
    validate(row)
    return (row["protocol"],row["experiment"],row["metric"],
            row["unit"],row["scope"],row["evidence"])

def atlas(rows):
    groups=defaultdict(list)
    for row in rows:
        key=grouping_key(row)
        groups[key].append({"claim":row["claim"],"seed":row["seed"],
                            "value":row["value"],
                            "source_revision":row["source_revision"]})
    return [{"comparability":list(key),"evidence_rows":value}
            for key,value in sorted(groups.items())]

def read_records(paths):
    records=[]
    for path in paths:
        data=json.loads(Path(path).read_text())
        if isinstance(data,list):records.extend(data)
        else:records.append(data)
    return atlas(records)
