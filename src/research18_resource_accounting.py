"""Analytic estimates only; not measured energy, latency or memory traffic."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Hierarchy:
    local_cost:float=1.
    remote_cost:float=5.
    offchip_cost:float=50.

def cellular(nodes,channels,ticks,edges,remote_edges=0,offchip_bytes=0,
             word_bytes=4,hierarchy=Hierarchy()):
    if min(nodes,channels,ticks,word_bytes)<1 or edges<0:
        raise ValueError("Invalid dimensions")
    if not 0<=remote_edges<=edges or offchip_bytes<0:
        raise ValueError("Invalid communication accounting")
    local=2*nodes*channels*ticks*word_bytes+(edges-remote_edges)*channels*ticks*word_bytes
    remote=remote_edges*channels*ticks*word_bytes
    return {"state_bytes":nodes*channels*word_bytes,
            "local_bytes":local,"remote_bytes":remote,
            "offchip_bytes":offchip_bytes,
            "abstract_cost":local*hierarchy.local_cost+
                remote*hierarchy.remote_cost+offchip_bytes*hierarchy.offchip_cost,
            "estimated":True}

def dense(inputs,outputs,batch=1,word_bytes=4):
    if min(inputs,outputs,batch,word_bytes)<1: raise ValueError("Invalid shape")
    return {"weights_bytes":inputs*outputs*word_bytes,
            "multiply_accumulates":inputs*outputs*batch,"estimated":True}

def iso_accuracy(rows,gap=.01):
    if not rows or gap<0: raise ValueError("Invalid conditions")
    best=max(x["accuracy"] for x in rows)
    return [x for x in rows if x["accuracy"]>=best-gap]
