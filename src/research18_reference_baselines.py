"""Tiny classical reference estimators (no oracle future observation)."""
import numpy as np

def persistence(observations):
    x=np.asarray(observations,float)
    return np.r_[x[0],x[:-1]]

def delayed(observations,lag):
    x=np.asarray(observations,float)
    if not 1<=lag<len(x): raise ValueError("Invalid lag")
    return np.r_[np.full(lag,np.nan),x[:-lag]]

def binary_filter(observations,transition=.1,error=.1,prior=.5):
    if not 0<=transition<=1 or not 0<error<.5 or not 0<=prior<=1:
        raise ValueError("Invalid probabilities")
    p=float(prior); out=[]
    for obs in observations:
        p=p*(1-transition)+(1-p)*transition
        if obs is not None:
            if obs not in (0,1): raise ValueError("Binary observation expected")
            l1=(1-error) if obs==1 else error
            l0=error if obs==1 else (1-error)
            p=p*l1/(p*l1+(1-p)*l0)
        out.append(p)
    return np.asarray(out)

def fit_affine(train,regularization=1e-3):
    x=np.asarray(train,float)
    if x.ndim!=2 or len(x)<3: raise ValueError("Expected T x D")
    a=np.c_[x[:-1],np.ones(len(x)-1)]
    reg=np.eye(a.shape[1]);reg[-1,-1]=0
    return np.linalg.solve(a.T@a+regularization*reg,a.T@x[1:])

def forecast_affine(state,weights):
    return np.r_[np.asarray(state),1.]@weights
