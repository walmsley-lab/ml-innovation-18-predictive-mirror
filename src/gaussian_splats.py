"""Tiny anisotropic-ready (currently isotropic) metric-3D Gaussian renderer.

Reference implementation in NumPy, not a GPU 3DGS engine. Scene state
from the simulator is privileged until an RGB-to-state encoder is trained.
"""
import numpy as np

BACKGROUND = .035


def state_from_record(record, context):
    """Oracle geometry from frames < context; explicitly *not* RGB inference."""
    if context<2 or context>=len(record["rgb"]):
        raise ValueError("Invalid observed history")
    centers=record["centers_world"]
    return {
        "center":np.asarray(centers[context-1],np.float32).copy(),
        "velocity":np.asarray(centers[context-1]-centers[context-2],np.float32),
        "scale":np.asarray(record["radii"],np.float32).copy(),
        "color":np.asarray(record["colors"],np.float32).copy(),
        "opacity":np.full(len(record["radii"]),.85,np.float32),
    }


def render(centers,scale,color,opacity,K,T_world_camera,size,background=BACKGROUND):
    """Approximate alpha compositing of projected 3D Gaussians.

    Metric 3D centers, metric isotropic scales and calibrated pinhole
    projection. Front-to-back splats are ordered by camera-space depth.
    This does not model sphere surfaces/physically correct volume rendering.
    """
    center=np.asarray(centers,np.float32)
    radius=np.asarray(scale,np.float32)
    colors=np.asarray(color,np.float32)
    alpha=np.asarray(opacity,np.float32)
    if (center.ndim!=2 or center.shape[1]!=3 or radius.shape!=(len(center),)
            or colors.shape!=(len(center),3) or alpha.shape!=(len(center),)
            or np.any(radius<=0) or np.any((alpha<0)|(alpha>1)) or size<8):
        raise ValueError("Invalid Gaussian state")
    transform=np.asarray(T_world_camera,np.float32)
    K=np.asarray(K,np.float32)
    local=(center-transform[:3,3])@transform[:3,:3]
    yy,xx=np.mgrid[0:size,0:size]
    image=np.full((size,size,3),background,np.float32)
    transmittance=np.ones((size,size),np.float32)
    # Near -> far; alpha over background.
    for index in np.argsort(local[:,2]):
        x,y,z=local[index]
        if z<=1e-4:
            continue
        u=K[0,0]*x/z+K[0,2]
        v=K[1,1]*(-y)/z+K[1,2]
        sigma=max(float(K[0,0]*radius[index]/z),.20)
        strength=alpha[index]*np.exp(-.5*((xx-u)**2+(yy-v)**2)/sigma**2)
        weight=transmittance*strength
        image+=weight[...,None]*(colors[index]-background)
        transmittance*=1-strength
    return np.clip(image,0,1).astype(np.float32)


def predict_centers(state,horizon,mode="velocity"):
    if horizon<1 or mode not in ("static","velocity"):
        raise ValueError("Invalid state forecast")
    steps=np.arange(1,horizon+1,dtype=np.float32)[:,None,None]
    velocity=state["velocity"] if mode=="velocity" else np.zeros_like(state["velocity"])
    return state["center"][None]+steps*velocity[None]


def render_forecast(record,positions,context,state):
    """Known future camera actions are allowed to move the renderer.

    Future world positions are inputs ONLY to evaluator-only oracle mode.
    """
    positions=np.asarray(positions)
    if positions.ndim!=3 or positions.shape[1:]!=state["center"].shape:
        raise ValueError("Expected [horizon,gaussian,xyz]")
    if context+len(positions)>len(record["rgb"]):
        raise ValueError("World record too short")
    return np.stack([render(positions[t],state["scale"],state["color"],
                  state["opacity"],record["K"],
                  record["T_world_camera"][context+t],
                  record["rgb"].shape[1])
                  for t in range(len(positions))])


def evaluate(record,context,horizon):
    """Oracle geometry sanity check with static and velocity controls."""
    state=state_from_record(record,context)
    true=record["centers_world"][context:context+horizon]
    target=record["rgb"][context:context+horizon]
    if len(target)!=horizon:
        raise ValueError("Insufficient future for evaluation")
    records={}
    # All arms share the Gaussian renderer; oracle is an evaluator ceiling
    # for scene representation, not a learnable RGB-only competitor.
    for mode in ("static","velocity","oracle_future_geometry"):
        positions=(np.asarray(true) if mode=="oracle_future_geometry" else
                   predict_centers(state,horizon,mode))
        forecast=render_forecast(record,positions,context,state)
        records[mode]={
            "mse_vs_analytic_rgb":float(np.square(forecast-target).mean()),
            "mse_vs_gaussian_oracle":None,
            "centroid_3d_mae_m":float(
                np.linalg.norm(positions-true,axis=-1).mean()),
            "frames":forecast,
        }
    ideal=records["oracle_future_geometry"]["frames"]
    for mode in records:
        records[mode]["mse_vs_gaussian_oracle"]=float(
            np.square(records[mode]["frames"]-ideal).mean())
    return records
