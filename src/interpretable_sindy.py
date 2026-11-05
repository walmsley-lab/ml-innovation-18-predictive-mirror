"""Sparse polynomial dynamics on *known* (x,y,vx,vy) variables.

This is a supplied-state system-identification reference, not an
RGB-inferred or causal explanation of a neural model.
"""
import numpy as np

NAMES=("1","x","y","vx","vy","x²","xy","xvx","xvy","y²",
       "yvx","yvy","vx²","vxvy","vy²")


def library(states,degree=2):
    states=np.asarray(states,np.float64)
    if states.shape[-1]!=4 or degree not in (1,2):
        raise ValueError("Four state variables and degree 1 or 2 required")
    columns=[np.ones(states.shape[:-1],np.float64)]
    columns.extend(states[...,j] for j in range(4))
    if degree==2:
        columns.extend(states[...,j]*states[...,k]
                       for j in range(4) for k in range(j,4))
    return np.stack(columns,axis=-1),NAMES[:len(columns)]


def fit(inputs,next_states,method="sindy",threshold=.05,ridge=1e-6,
        degree=2,iterations=6):
    inputs=np.asarray(inputs,np.float64)
    next_states=np.asarray(next_states,np.float64)
    if inputs.shape!=next_states.shape or inputs.ndim!=2 or inputs.shape[1]!=4:
        raise ValueError("Expected aligned [samples,4] state transitions")
    if method not in ("sindy","affine","dmd") or ridge<0 or threshold<0:
        raise ValueError("Invalid identification method")
    degree=degree if method=="sindy" else 1
    source,names=library(inputs,degree)
    if method=="dmd":
        source=source[:,1:]
        names=names[1:]
        output=next_states
    else:
        output=next_states-inputs
    # Column normalization prevents polynomial magnitude dominating selection.
    norm=np.maximum(np.linalg.norm(source,axis=0),1e-10)
    a=source/norm
    coef=np.linalg.solve(a.T@a+ridge*np.eye(a.shape[1]),a.T@output)
    if method=="sindy":
        for _ in range(iterations):
            mask=np.abs(coef)>=threshold
            mask[0,:]=True   # retain intercept rather than punish drift bias
            for j in range(4):
                keep=np.flatnonzero(mask[:,j])
                if len(keep):
                    coef[:,j]=0
                    coef[keep,j]=np.linalg.solve(
                        a[:,keep].T@a[:,keep]+ridge*np.eye(len(keep)),
                        a[:,keep].T@output[:,j])
    return {"method":method,"degree":degree,"names":names,
            "weights":coef/norm[:,None],"active_terms":int(
                np.count_nonzero(abs(coef)>1e-10))}


def next_step(model,state):
    state=np.asarray(state,np.float64)
    terms,_=library(state,model["degree"])
    if model["method"]=="dmd":
        return terms[...,1:]@model["weights"]
    return state+terms@model["weights"]


def rollout(model,initial,horizon):
    if horizon<1 or np.asarray(initial).shape[-1]!=4:
        raise ValueError("Invalid rollout")
    state=np.asarray(initial,np.float64)
    history=[]
    for _ in range(horizon):
        state=next_step(model,state)
        history.append(state.copy())
    return np.stack(history,axis=-2)


def coast(initial,horizon):
    state=np.asarray(initial,np.float64).copy()
    frames=[]
    for _ in range(horizon):
        state=state.copy()
        state[...,:2]+=state[...,2:]
        frames.append(state)
    return np.stack(frames,axis=-2)


def equations(model):
    labels=("Δx","Δy","Δvx","Δvy") if model["method"]!="dmd" else (
        "next_x","next_y","next_vx","next_vy")
    return {labels[j]:" + ".join(
        f"{v:+.5g}*{name}" for name,v in zip(model["names"],
        model["weights"][:,j]) if abs(v)>1e-7) or "0"
        for j in range(4)}
