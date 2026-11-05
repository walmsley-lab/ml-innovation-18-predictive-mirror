"""Deterministic, evaluator-traceable visible -> hidden -> returned worlds.

The old generic occlusion domain did not guarantee scored reappearance.
These explicit cases do, and distinguish partial barrier visibility from
full barrier occlusion and camera exit. RGB-only predictors see frames.
"""
import numpy as np
from src.rgb_motion_world import COLORS


def episode(seed=1, context=4, horizon=10, size=32, mode="barrier"):
    if (context < 4 or horizon < 9 or size < 32
            or mode not in ("barrier", "camera_exit")):
        raise ValueError("Require context>=4, horizon>=9, size>=32 and valid mode")
    rng=np.random.default_rng(int(seed))
    n=context+horizon
    yy,xx=np.mgrid[:size,:size]
    radius=2.5
    cy=size/2 + float(rng.choice([-1,0,1]))
    left,right=size//2-5,size//2+5
    stripe=(xx>=left)&(xx<=right)
    frames,visibility,reason,positions,offsets=[],[],[],[],[]
    for t in range(n):
        if mode=="barrier":
            # Last observed frame is fully visible; future crosses the bar,
            # vanishes for several frames and reappears by the +9 horizon.
            x=left-radius-3.0+2.0*(t-(context-1))
            camera_x=0.
        else:
            # Identical object with a prescribed camera excursion.
            # The return event is a field-of-view entry, not occlusion.
            x=float(size/2)
            camera_x=float(size) if context+1<=t<=context+3 else 0.
        mask=(xx-(x-camera_x))**2+(yy-cy)**2<=radius**2
        frame=np.full((size,size,3),.035,np.float32)
        visible_pixels=mask & (~stripe if mode=="barrier" else True)
        frame[visible_pixels]=COLORS[0]
        if mode=="barrier":
            frame[stripe]=[.13,.13,.15]
        visible_count=int(np.count_nonzero(visible_pixels))
        total=int(np.count_nonzero(mask))
        if visible_count==0:
            label=("barrier" if mode=="barrier" and total>0
                   else "out_of_view")
        elif mode=="barrier" and visible_count<total:
            label="partial"
        else:
            label="visible"
        frames.append(frame)
        visibility.append(visible_count>0)
        reason.append(label)
        positions.append((x,cy))
        offsets.append(camera_x)
    events=[]
    for t in range(1,n):
        if reason[t] in ("visible","partial") and reason[t-1] in (
                "barrier","out_of_view"):
            kind="barrier_return" if reason[t-1]=="barrier" else "camera_return"
            first_hidden=t-1
            while (first_hidden>0 and
                   reason[first_hidden-1] in ("barrier","out_of_view")):
                first_hidden-=1
            events.append({"frame":t,"kind":kind,
                           "hidden_start":first_hidden})
    relevant=[e for e in events if e["frame"]>=context and
              e["frame"]<n]
    if not relevant or (mode=="barrier" and not any(
            x=="partial" for x in reason[context:])):
        raise AssertionError("Generated episode lacks guaranteed event coverage")
    return {"frames":np.stack(frames),"visible":np.asarray(visibility)[:,None],
            "visibility_reason":np.asarray(reason),
            "positions":np.asarray(positions,np.float32),
            "camera_x":np.asarray(offsets,np.float32),
            "events":relevant,"seed":int(seed),"mode":mode,
            "color":COLORS[0].copy(),"context":context,"horizon":horizon}


def records(seeds=(11,23,41),context=4,horizon=10,size=32,mode="barrier"):
    return [episode(seed,context,horizon,size,mode) for seed in seeds]


def model_inputs(record):
    """No positions, visibility, events, future frames, or true actions."""
    return {"frames":record["frames"][:record["context"]]}
