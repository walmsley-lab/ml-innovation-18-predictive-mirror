"""Compare capability and measured costs on compatible tasks and devices.

No universal scalar metric or fabricated energy estimate.
"""
import numpy as np


def nondominated(rows, success_key="success_rate", cost_keys=("latency_ms",)):
    if not rows:
        raise ValueError("Need model measurement rows")
    required=("model","protocol","device",success_key,*cost_keys)
    for row in rows:
        if any(k not in row or row[k] is None for k in required):
            raise ValueError("Missing comparable model measurement")
        nums=[row[success_key]]+[row[k] for k in cost_keys]
        if not all(isinstance(x,(int,float)) and np.isfinite(x) for x in nums):
            raise ValueError("Nonfinite or nonnumeric capability/cost")
    protocols={(r["protocol"],r["device"]) for r in rows}
    if len(protocols)!=1:
        raise ValueError("Cannot rank different tasks, normalization or devices")
    result=[]
    for i,a in enumerate(rows):
        dominated_by=[]
        for j,b in enumerate(rows):
            if i==j:
                continue
            no_worse=(b[success_key]>=a[success_key] and
                      all(b[k]<=a[k] for k in cost_keys))
            strictly=(b[success_key]>a[success_key] or
                      any(b[k]<a[k] for k in cost_keys))
            if no_worse and strictly:
                dominated_by.append(b["model"])
        result.append({**a,"pareto_efficient":not bool(dominated_by),
                       "dominated_by":dominated_by})
    return result
