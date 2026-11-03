"""Frozen 3D perspective metrics, with explicitly privileged geometry diagnostic."""
import numpy as np
from src import world3d


def reproject(rgb, depth, K, old_pose, new_pose, background=.035):
    """Backward-known-depth **forward** projection of last seen pixels.

    This diagnostic has access to TRUE depth and both camera poses. It is
    an oracle geometry control, not a fair RGB-only prediction baseline.
    """
    h,w=depth.shape
    yy,xx=np.indices((h,w))
    valid=np.isfinite(depth)&(depth>0)
    z=depth[valid]
    if not len(z):
        return np.full_like(rgb,background)
    camera=np.stack(((xx[valid]-K[0,2])*z/K[0,0],
                     -(yy[valid]-K[1,2])*z/K[1,1],z),axis=-1)
    xyz=camera@old_pose[:3,:3].T+old_pose[:3,3]
    in_new=(xyz-new_pose[:3,3])@new_pose[:3,:3]
    positive=in_new[:,2]>1e-5
    u=np.rint(K[0,0]*in_new[:,0]/np.maximum(in_new[:,2],1e-5)+K[0,2]).astype(int)
    v=np.rint(-K[1,1]*in_new[:,1]/np.maximum(in_new[:,2],1e-5)+K[1,2]).astype(int)
    inside=positive&(u>=0)&(u<w)&(v>=0)&(v<h)
    # Paint far-to-near, so nearest object survives collisions.
    order=np.argsort(-in_new[:,2])
    result=np.full_like(rgb,background)
    colors=rgb[valid]
    for index in order:
        if inside[index]:
            result[v[index],u[index]]=colors[index]
    return result


def score(records,predictions,context,leads=(1,3,5,10)):
    """Model predictions use only RGB and (if allowed) planned actions."""
    predictions=np.asarray(predictions)
    if (not records or predictions.ndim!=5 or predictions.shape[-1]!=3
            or len(records)!=len(predictions) or context<2):
        raise ValueError("Invalid held-out 3D forecast")
    if not leads or min(leads)<1 or max(leads)>predictions.shape[1]:
        raise ValueError("Outside available horizon")
    output={}
    for lead in leads:
        full,foreground,counts=[],[],0
        for i,record in enumerate(records):
            idx=context+lead-1
            actual=record["rgb"][idx]
            prediction=predictions[i,lead-1]
            if actual.shape!=prediction.shape:
                raise ValueError("Prediction shape mismatch")
            error=np.square(actual-prediction).mean(-1)
            full.append(float(error.mean()))
            mask=record["instance_id"][idx]>=0
            counts+=int(mask.sum())
            foreground.extend(error[mask].tolist())
        output[str(lead)]={
            "mse":float(np.mean(full)),
            "visible_geometry_mse":float(np.mean(foreground)) if foreground else None,
            "visible_pixels":counts,
            "episodes":len(records)}
    return output


def depth_errors(predicted_depth,true_depth,instance_id):
    mask=(instance_id>=0)&np.isfinite(true_depth)
    predicted_depth=np.asarray(predicted_depth)
    if predicted_depth.shape!=true_depth.shape:
        raise ValueError("Depth shape mismatch")
    if not np.any(mask):
        return {"rmse_m":None,"valid_pixels":0}
    errors=predicted_depth[mask]-true_depth[mask]
    return {"rmse_m":float(np.sqrt(np.mean(errors**2))),
            "valid_pixels":int(mask.sum())}


def diagnostic_baselines(records,context,horizon):
    hold=np.stack([np.repeat(x["rgb"][context-1:context],horizon,axis=0)
                   for x in records])
    privileged=[]
    for record in records:
        frames=[]
        for idx in range(context,context+horizon):
            frames.append(reproject(
                record["rgb"][context-1],record["depth"][context-1],
                record["K"],record["T_world_camera"][context-1],
                record["T_world_camera"][idx]))
        privileged.append(frames)
    return {"rgb_persistence":hold,"oracle_depth_reprojection":np.asarray(privileged)}
