"""Small, fully calibrated 3D ray-sphere world, CPU-only NumPy.

A real perspective ray renderer, not a 2D image translation. Fixed
primitive spheres and a prescribed camera path keep physics separate
from the perception question. Depth/identity/pose are evaluator-only.
"""
import numpy as np

DOMAINS=("train","ood_depth","ood_camera","ood_appearance")


def intrinsics(size, fov_degrees=60):
    if size < 12 or not 20 <= fov_degrees <= 120:
        raise ValueError("Invalid camera calibration")
    f = .5*size/np.tan(np.deg2rad(fov_degrees)/2)
    return np.array([[f,0,(size-1)/2],
                     [0,f,(size-1)/2],[0,0,1]],np.float32)


def rotation(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]],np.float32)


def project(points,camera_xyz,yaw,K):
    local=(np.asarray(points,dtype=np.float32)-camera_xyz)@rotation(yaw)
    z=local[...,2]
    uv=np.stack((K[0,0]*local[...,0]/z+K[0,2],
                 K[1,1]*(-local[...,1])/z+K[1,2]),axis=-1)
    return uv,z


def render(centers,radii,colors,camera_xyz,yaw,K,size):
    """Vectorized nearest positive ray intersection with actual 3D spheres."""
    yy,xx=np.mgrid[0:size,0:size]
    directions=np.stack(((xx-K[0,2])/K[0,0],
                         -(yy-K[1,2])/K[1,1],
                         np.ones_like(xx)),axis=-1).astype(np.float32)
    directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
    rays=directions@rotation(yaw).T
    depth=np.full((size,size),np.inf,np.float32)
    ids=np.full((size,size),-1,np.int32)
    rgb=np.full((size,size,3),.035,np.float32)
    for index,(center,radius,color) in enumerate(zip(centers,radii,colors)):
        offset=np.asarray(camera_xyz)-center
        b=np.sum(rays*offset,axis=-1)
        discriminant=b*b-(np.sum(offset**2)-radius**2)
        t=-b-np.sqrt(np.maximum(discriminant,0))
        z=t*directions[...,2]
        visible=(discriminant>=0)&(t>0)&(z<depth)
        depth=np.where(visible,z,depth)
        ids=np.where(visible,index,ids)
        rgb=np.where(visible[...,None],color,rgb)
    return rgb,depth,ids


def episode(seed=1,steps=12,size=32,domain="train",
            actions=None,fps=10):
    if (steps<3 or size<12 or fps<=0 or domain not in DOMAINS):
        raise ValueError("Invalid calibrated 3D world")
    rng=np.random.default_rng(seed)
    K=intrinsics(size)
    count=3
    centers=np.stack((rng.uniform(-.9,.9,count),
                      rng.uniform(-.4,.4,count),
                      rng.uniform(2.8,5.5,count)),axis=1).astype(np.float32)
    if domain=="ood_depth":
        centers[:,2]+=2.8
    radii=rng.uniform(.28,.48,count).astype(np.float32)
    colors=np.array([[.95,.2,.12],[.10,.75,.95],[.93,.75,.13]],np.float32)
    if domain=="ood_appearance":
        colors=rng.uniform(.15,1.,(count,3)).astype(np.float32)
    if actions is None:
        scale=.11 if domain=="ood_camera" else .04
        actions=np.stack((rng.choice([-1.,0.,1.],steps-1)*scale,
                          rng.choice([-1.,0.,1.],steps-1)*.012),axis=-1).astype(np.float32)
    else:
        actions=np.asarray(actions,np.float32)
    if actions.shape!=(steps-1,2):
        raise ValueError("Camera actions must be [steps-1, x_translation_and_yaw]")
    camera=np.zeros(3,np.float32)
    yaw=0.
    frames,depths,ids,poses,positions=[],[],[],[],[]
    for tick in range(steps):
        rgb,depth,instance=render(centers,radii,colors,camera,yaw,K,size)
        pose=np.eye(4,dtype=np.float32)
        pose[:3,:3]=rotation(yaw)
        pose[:3,3]=camera
        frames.append(rgb)
        depths.append(depth)
        ids.append(instance)
        poses.append(pose)
        positions.append(centers.copy())
        if tick<steps-1:
            camera[0]+=actions[tick,0]
            yaw+=float(actions[tick,1])
            # A small deterministic 3D velocity is part of world dynamics.
            centers[:,1]+=.009*np.sin(.4*tick+np.arange(count))
    return {
        "rgb":np.stack(frames),"depth":np.stack(depths),
        "instance_id":np.stack(ids),
        "K":K,"T_world_camera":np.stack(poses),
        "camera_actions":actions,"centers_world":np.stack(positions),
        "radii":radii,"colors":colors,"times_s":np.arange(steps)/fps,
        "backend":"ray-sphere-analytic-v1","domain":domain,
        "units":"meters_xyz_seconds",
    }


def model_inputs(record,known_actions=False):
    result={"rgb":record["rgb"]}
    if known_actions:
        result["camera_actions"]=record["camera_actions"]
    return result
