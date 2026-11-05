"""Timestamped camera evidence and causal replay perturbations.

True camera settings may be unavailable on a given backend. Preserve
null/unavailable metadata instead of fabricating the capture exposure.
"""
import numpy as np

FIELDS=("exposure","gain","white_balance")


def metadata(stamps,values=None,settings_locked=None):
    stamps=np.asarray(stamps,np.float64)
    if stamps.ndim!=1 or len(stamps)<2 or np.any(np.diff(stamps)<=0):
        raise ValueError("Need strictly increasing capture timestamps")
    values=values or {}
    output=[]
    for t,stamp in enumerate(stamps):
        row={"timestamp_s":float(stamp)}
        for key in FIELDS:
            source=values.get(key)
            value=None if source is None else source[t]
            row[key]=None if value is None else float(value)
        row["settings_locked"]=settings_locked if settings_locked is not None else None
        output.append(row)
    return output


def _drop_indices(length,rate,burst,seed):
    if not 0<=rate<=.1:
        raise ValueError("Drop-rate study limited to 0..10%")
    rng=np.random.default_rng(int(seed))
    count=int(round(rate*length))
    if count<1:return np.zeros(length,bool)
    missing=np.zeros(length,bool)
    if burst:
        first=int(rng.integers(1,max(2,length-count)))
        missing[first:first+count]=True
    else:
        choices=rng.choice(np.arange(1,length),size=min(count,length-1),
                            replace=False)
        missing[choices]=True
    return missing


def corrupt(frames,mode="exposure",rate=.05,seed=11):
    """Return sensor RGB and observed-frame status, not future labels."""
    frames=np.asarray(frames,np.float32)
    if frames.ndim!=4 or frames.shape[-1]!=3 or len(frames)<3:
        raise ValueError("Expected RGB sequence [T,H,W,3]")
    n,h,w,_=frames.shape
    observed=frames.copy()
    missing=np.zeros(n,bool)
    values={"exposure":[None]*n,"gain":[None]*n,"white_balance":[None]*n}
    if mode in ("isolated_drop","burst_drop"):
        missing=_drop_indices(n,rate,mode=="burst_drop",seed)
        # Last-seen-value hold is causal, but should be explicitly
        # distinguished from a truly acquired observation.
        for t in range(1,n):
            if missing[t]:
                observed[t]=observed[t-1]
    elif mode=="exposure":
        gains=1+.25*np.sin(.7*np.arange(n))
        observed=np.clip(frames*gains[:,None,None,None],0,1)
        values["exposure"]=gains.tolist()
    elif mode=="white_balance":
        r=np.arange(n)
        factors=np.stack([1+.2*np.sin(r*.57),
                          np.ones(n),1+.2*np.cos(r*.53)],axis=-1)
        observed=np.clip(frames*factors[:,None,None,:],0,1)
        values["white_balance"]=(factors[:,0]/factors[:,2]).tolist()
    elif mode=="rolling_shutter":
        # Synthetic row-time skew under constant horizontal sensor motion.
        # Not a calibrated sensor model; explicitly a stress test.
        yy=np.arange(h)
        shifts=np.rint((yy-(h-1)/2)*.10).astype(int)
        for t in range(n):
            for row in range(h):
                dst=np.arange(w);src=dst-shifts[row]
                valid=(src>=0)&(src<w)
                observed[t,row]=.035
                observed[t,row,valid]=frames[t,row,src[valid]]
    elif mode=="none":
        pass
    else:
        raise ValueError("Unknown camera corruption")
    stamps=np.arange(n,dtype=np.float64)*.1
    return {"frames":observed.astype(np.float32),
            "missing":missing,"capture":metadata(stamps,values),
            "mode":mode,"target_contract":"sensor frames include corruption; "
                "dropout uses causal last-observed imputation; dropout "
                "visibility is observable to the predictor"}


def strata(error,capture,missing=None,change_threshold=.08):
    """Report independent frame strata: exposure transition and dropout."""
    error=np.asarray(error)
    if len(error)!=len(capture):
        raise ValueError("Error/time metadata mismatch")
    values=np.asarray([row["exposure"] if row["exposure"] is not None
                       else np.nan for row in capture],np.float64)
    change=np.zeros(len(values),bool)
    change[1:]=np.abs(np.diff(values))>change_threshold
    missing=np.zeros(len(values),bool) if missing is None else np.asarray(missing,bool)
    if missing.shape!=(len(values),):
        raise ValueError("Dropout mask shape mismatch")
    output={}
    for name,mask in (("exposure_changed",change),("exposure_stable_known",
        np.isfinite(values)&~change),("frame_missing",missing),
        ("frame_received",~missing)):
        output[name]={"frames":int(mask.sum()),
                      "mean_error":float(error[mask].mean()) if mask.any() else None}
    return output


def effective_lead_ms(commit_time_s,target_capture_time_s,
                      display_delay_ms=0):
    value=(target_capture_time_s-commit_time_s)*1000-display_delay_ms
    return float(value)
