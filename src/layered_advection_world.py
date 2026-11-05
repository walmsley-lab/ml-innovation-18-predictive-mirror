"""Analytic layered RGB advection with evaluator-only optical flow and masks.

Layer IDs and target/source visibility cannot be model inputs. Velocities
are fractional pixels. Flow conventions are explicit: source-to-target,
(dy,dx), with no implicit wrap or interpolation at boundaries.
"""
import numpy as np

MODES=("occlusion","subpixel","appearance_only","high_texture")


def texture(x,y,seed,high=False):
    """Continuous coordinate texture: moving a layer moves its appearance."""
    scale=2.9 if high else 1.1
    a=.5+.22*np.sin(x*scale+seed*.13)*np.cos(y*scale*.87+seed*.29)
    b=.5+.23*np.cos(x*.83*scale-seed*.31)
    c=.5+.21*np.sin((x+y)*scale*.47+seed*.51)
    return np.stack(np.broadcast_arrays(a,b,c),axis=-1).astype(np.float32)


def episode(seed=11,steps=12,size=32,mode="occlusion"):
    if mode not in MODES or steps<3 or size<20:
        raise ValueError("Invalid layered world")
    rng=np.random.default_rng(int(seed))
    y,x=np.mgrid[:size,:size]
    half=size*.17
    # Back object moves right; front object moves left. Their overlap and
    # disocclusion are visible during the sequence.
    initial=np.asarray([[size*.35,size*.5],[size*.65,size*.5]],np.float64)
    if mode=="appearance_only":
        velocity=np.zeros((2,2),np.float64)
    else:
        velocity=np.asarray([[.07,.59],[-.05,-.61]],np.float64)
    if mode=="subpixel":
        velocity*=.6
    frames=[];owners=[];positions=[]
    for t in range(steps):
        frame=texture(x/4.,y/4.,int(seed),mode=="high_texture")
        frame=.15+.65*frame
        owner=np.full((size,size),-1,np.int16)
        loc=initial+velocity*t
        for index in (0,1):  # Front layer 1 overwrites back layer 0.
            cy,cx=loc[index]
            mask=(abs(x-cx)<=half)&(abs(y-cy)<=half)
            palette=texture((x-cx)/2.3,(y-cy)/2.3,
                            int(seed)+20*(index+1),mode=="high_texture")
            palette=np.clip(palette*.68+(.18 if index else .05),0,1)
            frame[mask]=palette[mask]
            owner[mask]=index
        if mode=="appearance_only":
            # No movement: only sensor/lighting gain changes.
            frame=np.clip(frame*(.65+.25*np.sin(.55*t)+.15),0,1)
        frames.append(frame.astype(np.float32))
        owners.append(owner)
        positions.append(loc.copy())
    frames=np.stack(frames)
    ids=np.stack(owners)
    positions=np.asarray(positions,np.float32)
    src_flows=[];src_survive=[];target_from_source=[];about_to=[];newly_revealed=[]
    for t in range(steps-1):
        source=ids[t];target=ids[t+1]
        flow=np.zeros((size,size,2),np.float32)
        for index in (0,1):
            flow[source==index]=velocity[index]
        # Source owner at its advected target coordinate must still be frontmost.
        yy,xx=np.mgrid[:size,:size]
        ny=np.rint(yy+flow[...,0]).astype(int)
        nx=np.rint(xx+flow[...,1]).astype(int)
        in_view=(ny>=0)&(ny<size)&(nx>=0)&(nx<size)
        same=np.zeros((size,size),bool)
        same[in_view]=target[ny[in_view],nx[in_view]]==source[in_view]
        src_survive.append(same)
        about_to.append(~same)
        # For each destination pixel, back-project the target object's known
        # displacement; this is evaluator truth, not a model estimate.
        backward=np.zeros((size,size,2),np.float32)
        for index in (0,1):
            backward[target==index]=velocity[index]
        sy=np.rint(yy-backward[...,0]).astype(int)
        sx=np.rint(xx-backward[...,1]).astype(int)
        valid=(sy>=0)&(sy<size)&(sx>=0)&(sx<size)
        from_source=np.zeros((size,size),bool)
        from_source[valid]=source[sy[valid],sx[valid]]==target[valid]
        target_from_source.append(from_source)
        newly_revealed.append(~from_source)
        src_flows.append(flow)
    return {"frames":frames,"instance_ids":ids,
            "positions":positions,"velocities":velocity.astype(np.float32),
            "source_to_target_flow_yx":np.stack(src_flows),
            "source_survives":np.stack(src_survive),
            "about_to_be_occluded":np.stack(about_to),
            "target_source_visible":np.stack(target_from_source),
            "target_disoccluded":np.stack(newly_revealed),
            "depth_order":np.asarray([0,1],np.int16),
            "seed":int(seed),"mode":mode,
            "flow_unit":"pixels per frame; dy,dx; source to destination"}


def model_inputs(record,context):
    """Observations only, never future flow, identity or visibility."""
    if context<2 or context>len(record["frames"]):
        raise ValueError("Invalid history")
    return record["frames"][:context].copy()


def validate(record):
    frames=record["frames"];ids=record["instance_ids"]
    flow=record["source_to_target_flow_yx"]
    if frames.shape[:3]!=ids.shape or flow.shape!=(len(frames)-1,*ids.shape[1:],2):
        raise ValueError("Bad field dimensions")
    expected={"about_to_be_occluded","source_survives","target_source_visible",
              "target_disoccluded"}
    if not expected.issubset(record):
        raise ValueError("Visibility ground truth absent")
    if not np.array_equal(record["source_survives"],
                          ~record["about_to_be_occluded"]):
        raise ValueError("Source masks not complementary")
    if not np.array_equal(record["target_source_visible"],
                          ~record["target_disoccluded"]):
        raise ValueError("Target masks not complementary")
    if not np.all(np.isfinite(frames)):
        raise ValueError("Nonfinite imagery")
    if not set(np.unique(ids)).issubset({-1,0,1}):
        raise ValueError("Invalid instance IDs")
    return {"frames":len(frames),"source_occluded_pixels":int(
            record["about_to_be_occluded"].sum()),
            "target_disoccluded_pixels":int(
            record["target_disoccluded"].sum()),
            "subpixel_max_motion":float(np.max(abs(flow)))}
