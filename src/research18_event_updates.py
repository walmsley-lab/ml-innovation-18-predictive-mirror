"""Selective updates that skip inactive cell computation, not post-hoc masks."""
import numpy as np

def choose_active(errors,threshold=0.,max_cells=None):
    x=np.abs(np.asarray(errors,float).reshape(-1))
    if not np.isfinite(x).all() or threshold<0:raise ValueError("Invalid errors")
    selected=np.flatnonzero(x>threshold)
    if max_cells is not None:
        if max_cells<0:raise ValueError("Invalid budget")
        selected=selected[np.argsort(-x[selected],kind="stable")[:max_cells]]
    out=np.zeros(len(x),bool);out[selected]=True
    return out

def update(state,adjacency,active=None,retention=.5):
    x=np.asarray(state,float)
    a=np.asarray(adjacency,float)
    if x.ndim!=1 or a.shape!=(len(x),len(x)):
        raise ValueError("Expected vector + square adjacency")
    if not 0<=retention<=1:raise ValueError("Invalid retention")
    chosen=np.ones(len(x),bool) if active is None else np.asarray(active,bool)
    if chosen.shape!=(len(x),):raise ValueError("Invalid active mask")
    out=x.copy();reads=0;multiplies=0
    for i in np.flatnonzero(chosen):
        # Truly skip inactive cell updates in Python, including edge reads.
        neighbors=np.flatnonzero(a[i])
        degree=float(np.sum(a[i,neighbors]))
        incoming=float(np.dot(a[i,neighbors],x[neighbors]))/degree if degree else 0.
        out[i]=retention*x[i]+(1-retention)*incoming
        reads+=1+len(neighbors);multiplies+=2*len(neighbors)+2
    return out,{"updated_cells":int(chosen.sum()),
                "state_reads":int(reads),"scalar_multiplies_estimate":int(multiplies),
                "measured_energy":None}

def compare_accuracy(dense,selective,truth):
    d=np.asarray(dense);s=np.asarray(selective);t=np.asarray(truth)
    if d.shape!=s.shape or d.shape!=t.shape:raise ValueError("Shape mismatch")
    return {"dense_mse":float(np.mean((d-t)**2)),
            "selective_mse":float(np.mean((s-t)**2))}
