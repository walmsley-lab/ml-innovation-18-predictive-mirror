"""Causal classical filtering and image transport (synthetic RGB baseline).

No future-frame access. Fixed color templates are a disclosed synthetic
prior, not a trained object detector; track-state oracle is evaluator-only.
"""
import numpy as np
from src import baselines
from src.rgb_motion_world import COLORS

BACKGROUND = .035


def backward_gather(image, flow_yx, background=BACKGROUND):
    """Bilinear source gather: output[y,x] samples source[(y,x)-flow].

    flow_yx is a destination-frame displacement [H,W,2], ordered (dy,dx).
    Does NOT wrap at the border, and the validity mask prevents dark edges
    being misread as model predictions.
    """
    image=np.asarray(image, np.float32)
    flow=np.asarray(flow_yx, np.float32)
    if image.ndim!=3 or flow.shape!=(*image.shape[:2],2):
        raise ValueError("Expected RGB image and [H,W,2] flow")
    h,w=image.shape[:2]
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    source_y=yy-flow[...,0];source_x=xx-flow[...,1]
    inside=(source_y>=0)&(source_y<=h-1)&(source_x>=0)&(source_x<=w-1)
    y0=np.floor(np.clip(source_y,0,h-1)).astype(int)
    x0=np.floor(np.clip(source_x,0,w-1)).astype(int)
    y1=np.minimum(y0+1,h-1);x1=np.minimum(x0+1,w-1)
    fy=np.clip(source_y-y0,0,1)[...,None]
    fx=np.clip(source_x-x0,0,1)[...,None]
    sample=((1-fy)*(1-fx)*image[y0,x0]+fy*(1-fx)*image[y1,x0]+
            (1-fy)*fx*image[y0,x1]+fy*fx*image[y1,x1])
    return np.where(inside[...,None],sample,background).astype(np.float32),inside


def forward_splat(image, flow_yx, background=BACKGROUND):
    """Scatter RGB from a source to destinations, returning occupancy weights.

    This is **forward splatting**, distinct from inverse/backward gathering;
    holes and collisions remain visible in the occupancy map. Expects a
    source-to-target flow, not the destination flow used by backward_gather.
    """
    image=np.asarray(image,np.float32);flow=np.asarray(flow_yx,np.float32)
    if image.ndim!=3 or flow.shape!=(*image.shape[:2],2):
        raise ValueError("Source/flow mismatch")
    h,w=image.shape[:2]
    yy,xx=np.mgrid[:h,:w]
    py=yy+flow[...,0];px=xx+flow[...,1]
    weighted=np.zeros_like(image,dtype=np.float64)
    counts=np.zeros((h,w),np.float64)
    for sy,sx,weight_y,weight_x in ((np.floor(py),np.floor(px),1-(py%1),1-(px%1)),
                                    (np.ceil(py),np.floor(px),py%1,1-(px%1)),
                                    (np.floor(py),np.ceil(px),1-(py%1),px%1),
                                    (np.ceil(py),np.ceil(px),py%1,px%1)):
        iy=sy.astype(int);ix=sx.astype(int)
        valid=(iy>=0)&(iy<h)&(ix>=0)&(ix<w)
        value=(weight_y*weight_x)[valid]
        np.add.at(counts,(iy[valid],ix[valid]),value)
        np.add.at(weighted,(iy[valid],ix[valid]),value[...,None]*image[valid])
    output=np.where(counts[...,None]>0,
                    weighted/np.maximum(counts[...,None],1e-12),background)
    return output.astype(np.float32),counts.astype(np.float32)


def detect(frame,colors=COLORS):
    """Observed colors only. Uses known palette, not evaluator object IDs."""
    frame=np.asarray(frame,np.float32)
    if frame.ndim!=3 or frame.shape[-1]!=3:
        raise ValueError("Expected RGB")
    rows=[]
    for color in colors:
        mask=np.max(abs(frame-np.asarray(color)),axis=-1)<.13
        if mask.sum()<3:
            rows.append(None);continue
        yy,xx=np.nonzero(mask)
        rows.append({"center":np.array([yy.mean(),xx.mean()],np.float32),
                     "radius":float(np.sqrt(len(yy)/np.pi)),
                     "color":np.asarray(color,np.float32)})
    return rows


def track(history,mode="coast",colors=COLORS):
    """Estimate motion from observed frames, coasting through missing views.

    'coast' estimates using the last two valid consecutive detections,
    retaining that velocity during blank frames. 'reid' uses the two most
    recent valid sightings, including after a gap, for appearance reacquisition.
    """
    if mode not in ("coast","reid") or len(history)<2:
        raise ValueError("Track needs >=2 observed frames")
    detections=[detect(frame,colors) for frame in history]
    states=[]
    for slot in range(len(colors)):
        sightings=[(t,r[slot]) for t,r in enumerate(detections)
                   if r[slot] is not None]
        if not sightings:
            states.append(None);continue
        t,now=sightings[-1]
        if len(sightings)>=2:
            p,prev=sightings[-2]
            # Coast uses last local velocity estimate if the last two
            # adjacent detections exist; otherwise conservatively stop.
            dt=t-p
            velocity=(now["center"]-prev["center"])/dt if (
                mode=="reid" or dt==1) else np.zeros(2,np.float32)
        else:
            velocity=np.zeros(2,np.float32)
        states.append({**now,"center":now["center"]+
                        (len(history)-1-t)*velocity,
                       "velocity":velocity})
    return states


def render_tracks(states,size,lead=1):
    frame=np.full((size,size,3),BACKGROUND,np.float32)
    yy,xx=np.mgrid[:size,:size]
    for item in states:
        if item is None:continue
        y,x=item["center"]+lead*item["velocity"]
        mask=(yy-y)**2+(xx-x)**2<=item["radius"]**2
        frame[mask]=item["color"]
    return frame


def forecast(context,horizon=5,mode="coast",max_shift=6):
    """Only observed RGB context enters any arm; no future truth or IDs."""
    context=np.asarray(context,np.float32)
    if (context.ndim!=4 or context.shape[-1]!=3 or
            context.shape[0]<2 or horizon<1):
        raise ValueError("Need RGB history and positive horizon")
    if mode=="persistence":
        return baselines.persistence_forecast(context,horizon)
    if mode=="pixel_velocity":
        return baselines.constant_velocity_forecast(context,horizon,max_shift)
    if mode=="flow_gather":
        dy,dx=baselines._estimate_translation(context[-2],context[-1],max_shift)
        h,w=context.shape[1:3]
        return np.stack([backward_gather(context[-1],
                            np.broadcast_to((step*dy,step*dx),(h,w,2)))[0]
                         for step in range(1,horizon+1)])
    if mode not in ("coast","reid"):
        raise ValueError("Unknown control")
    states=track(context,mode)
    return np.stack([render_tracks(states,len(context[0]),lead)
                     for lead in range(1,horizon+1)])
