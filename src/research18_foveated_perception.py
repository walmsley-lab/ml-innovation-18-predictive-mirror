"""Real high-resolution crop plus low-resolution overview; no upsample cheat."""
import numpy as np

def peripheral(frame,factor=4):
    x=np.asarray(frame)
    if x.ndim not in (2,3) or factor<1 or x.shape[0]%factor or x.shape[1]%factor:
        raise ValueError("Expected HxW or HxWxC divisible by factor")
    h,w=x.shape[:2]
    shape=(h//factor,factor,w//factor,factor)+x.shape[2:]
    return x.reshape(shape).mean(axis=(1,3))

def choose_focus(coarse_history,mode="surprise"):
    """Only low-acuity history may select gaze. Return coarse pixel y,x."""
    x=np.asarray(coarse_history,float)
    if x.ndim<3:raise ValueError("Expected T,H,W[,C]")
    frame=x[-1]
    if mode=="center":return tuple(np.array(frame.shape[:2])//2)
    if mode=="salience":score=frame
    elif mode=="surprise":
        if len(x)<2:raise ValueError("Need previous coarse frame")
        score=np.abs(x[-1]-x[-2])
    else:raise ValueError("Unknown policy")
    if score.ndim==3:score=np.mean(score,axis=-1)
    return tuple(int(v) for v in np.unravel_index(np.argmax(score),score.shape))

def observe(raw,coarse_focus,crop_size=8,downsample=4):
    x=np.asarray(raw)
    low=peripheral(x,downsample)
    if crop_size<1 or crop_size>min(x.shape[:2]):
        raise ValueError("Invalid fovea")
    cy,cx=(int(v) for v in coarse_focus)
    if not (0<=cy<low.shape[0] and 0<=cx<low.shape[1]):
        raise ValueError("Gaze outside peripheral field")
    y=min(max(0,cy*downsample+downsample//2-crop_size//2),x.shape[0]-crop_size)
    xx=min(max(0,cx*downsample+downsample//2-crop_size//2),x.shape[1]-crop_size)
    crop=x[y:y+crop_size,xx:xx+crop_size].copy()
    return {"peripheral":low,"fovea":crop,"origin":(y,xx),
            "accounting":{"coarse_processed":int(low.shape[0]*low.shape[1]),
                          "fovea_processed":int(crop_size*crop_size),
                          "source_capture_pixels":int(x.shape[0]*x.shape[1])}}

def peripheral_event_miss(gaze_origins,events,crop_size):
    misses=0
    for (y,x),spots in zip(gaze_origins,events):
        for sy,sx in spots:
            misses+=not(y<=sy<y+crop_size and x<=sx<x+crop_size)
    total=sum(len(x) for x in events)
    return misses/max(1,total)
