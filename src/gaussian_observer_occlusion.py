"""Controlled object-only occlusion of past observations.

Uses evaluator-only object positions/poses to DRAW an opaque obstruction,
never to initialize an observer or to supply future state.
"""
import numpy as np
from src import world3d


def occlude(record,context,hidden_length=3,object_id=0):
    if context<3 or hidden_length<1 or hidden_length>=context:
        raise ValueError("Need a visible context before a hidden interval")
    rgb=np.array(record["rgb"],copy=True)
    affected=[]
    K=record["K"]
    for t in range(context-hidden_length,context):
        pose=record["T_world_camera"][t]
        yaw=float(np.arctan2(pose[0,2],pose[0,0]))
        uv,z=world3d.project(record["centers_world"][t,object_id],
                             pose[:3,3],yaw,K)
        radius=float(record["radii"][object_id])
        projected=K[0,0]*radius/max(float(z),.1)
        span=projected+2.
        xx,yy=np.meshgrid(np.arange(rgb.shape[2]),np.arange(rgb.shape[1]))
        screen=(np.abs(xx-uv[0])<=span)&(np.abs(yy-uv[1])<=span)
        rgb[t][screen]=[.13,.13,.15]
        affected.append({"frame":t,"occluder_pixels":int(screen.sum()),
                         "target_pixels_hidden":int(np.sum(
                             screen & (record["instance_id"][t]==object_id)))})
    return rgb,affected
