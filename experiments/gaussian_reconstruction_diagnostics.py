"""Compare current RGB reconstruction with supplied vs inferred Gaussian scene.

The privileged oracle gets world positions/radii/colors from the simulator
ONLY for evaluation. The inferred model receives RGB and known camera actions.
No future frames or oracle geometry are given to its encoder.
"""
import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import world3d, rgb_gaussian_nca as model, gaussian_slot_matching
from experiments.gaussian_active_view import load


def _camera(record,frame):
    x,yaw=model.camera_traces(jnp.asarray(record["camera_actions"]))
    return x[frame],yaw[frame]


def _projection(centers,K,x,yaw):
    camera=(jnp.asarray(centers)-jnp.array([x,0,0]))@model.rotation(yaw)
    z=jnp.maximum(camera[:,2],.01)
    return np.asarray(jnp.stack((K[0,0]*camera[:,0]/z+K[0,2],
                     -K[1,1]*camera[:,1]/z+K[1,2]),axis=-1))


def _png(path,picture):
    """Small dependency-free PNG gallery writer."""
    rgb=np.asarray(np.clip(picture*255,0,255),np.uint8)
    h,w,_=rgb.shape
    raw=b"".join(b"\x00"+rgb[y].tobytes() for y in range(h))
    def chunk(tag,payload):
        return (struct.pack(">I",len(payload))+tag+payload+
                struct.pack(">I",zlib.crc32(tag+payload)&0xffffffff))
    png=(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(
        ">IIBBBBB",w,h,8,2,0,0,0))+
        chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))
    Path(path).write_bytes(png)


def evaluate(record,params,context=5):
    """Score only the last observed frame, isolating renderer vs encoder."""
    t=context-1
    if context<2 or context>len(record["rgb"]):
        raise ValueError("Invalid observed context")
    K=jnp.asarray(record["K"])
    camera_x,yaw=_camera(record,t)
    actual=np.asarray(record["rgb"][t])
    oracle=np.asarray(model.render(
        jnp.asarray(record["centers_world"][t]),jnp.asarray(record["radii"]),
        jnp.asarray(record["colors"]),
        jnp.full((len(record["radii"]),),.85),K,camera_x,yaw,actual.shape[0]))
    inferred,scale,colors,opacity=model.infer(
        params,jnp.asarray(actual),K,camera_x,yaw)
    predicted=np.asarray(model.render(
        inferred,scale,colors,opacity,K,camera_x,yaw,actual.shape[0]))
    instance=record["instance_id"][t]
    visible=instance>=0
    def errors(image):
        diff=(image-actual)**2
        return {"rgb_mse":float(diff.mean()),
                "visible_pixel_mse":float(diff[visible].mean()) if visible.any() else None,
                "background_pixel_mse":float(diff[~visible].mean()) if (~visible).any() else None}
    true_xy=_projection(record["centers_world"][t],K,camera_x,yaw)
    pred_xy=_projection(inferred,K,camera_x,yaw)
    costs=np.linalg.norm(true_xy[:,None]-pred_xy[None],axis=-1)
    matches,missing=gaussian_slot_matching.assignment(costs)
    # No identities assigned to slots; permutation-invariant per-frame matching.
    match_errors=[distance for gt,slot,distance in matches
                  if (instance==gt).any()]
    return {"analytic_rgb":actual,"oracle_rgb":oracle,"inferred_rgb":predicted,
            "oracle":errors(oracle),"inferred":errors(predicted),
            "extra_encoder_error_mse":float(errors(predicted)["rgb_mse"]-errors(oracle)["rgb_mse"]),
            "projected_center_errors_px":match_errors,
            "unmatched_true_objects":missing,
            "visible_true_objects":int(len(np.unique(instance[visible]))) if visible.any() else 0,
            "slot_count":int(len(inferred)),
            "oracle_information":"privileged simulator centers/radii/colors",
            "inferred_information":"current observed RGB plus known camera calibration/actions"}


def run(out="runs/gaussian-reconstruction-diagnostics",checkpoint=None,
        untrained_smoke=False,seeds=(11,23,41),context=5,size=24,
        domain="ood_camera",slots=3,width=8):
    if not seeds or context<2 or size<12 or (checkpoint is None and not untrained_smoke):
        raise ValueError("Provide a trained checkpoint or explicit --untrained-smoke")
    params=(load(checkpoint) if checkpoint else
            model.init(jax.random.PRNGKey(7),slots,width))
    folder=Path(out)
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        record=world3d.episode(int(seed),context+1,size,domain)
        value=evaluate(record,params,context)
        gallery=np.concatenate([value[k] for k in (
            "analytic_rgb","oracle_rgb","inferred_rgb")],axis=1)
        path=folder/f"seed-{seed}-actual-oracle-inferred.png"
        _png(path,gallery)
        rows.append({"seed":int(seed),"gallery":path.name,
                     **{k:v for k,v in value.items() if not k.endswith("_rgb")}})
    summary={"protocol":"gaussian-current-reconstruction-v1",
             "source":"observed frame t=context-1, no future frames or geometry given to inferred arm",
             "checkpoint_sha256":(hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                                  if checkpoint else None),
             "untrained_smoke":bool(untrained_smoke),"domain":domain,
             "context":context,"size":size,"seeds":list(map(int,seeds)),
             "rows":rows,
             "limits":"The oracle and inferred scenes use the SAME soft Gaussian "
             "renderer; oracle RGB mismatch is a renderer/primitive gap, "
             "not a perception failure. The signed excess MSE is NOT "
             "a guaranteed nonnegative decomposition. Slot centers are "
             "matched per frame by assignment, not learned identities."}
    (folder/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/gaussian-reconstruction-diagnostics")
    p.add_argument("--checkpoint")
    p.add_argument("--untrained-smoke",action="store_true")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    p.add_argument("--context",type=int,default=5)
    p.add_argument("--size",type=int,default=24)
    p.add_argument("--domain",choices=world3d.DOMAINS,default="ood_camera")
    a=p.parse_args()
    print(json.dumps({"rows":len(run(a.out,a.checkpoint,a.untrained_smoke,
                 a.seeds,a.context,a.size,a.domain)["rows"])},indent=2))


if __name__=="__main__":
    main()
