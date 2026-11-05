"""Permutable slots vs actual object loss across 3D observations.

Ground-truth simulator identity is evaluator-only. The RGB encoder is
evaluated frame by frame, so this diagnoses representation consistency,
not persistent learned memory.
"""
import numpy as np
import jax.numpy as jnp
from src import world3d, rgb_gaussian_nca as gaussian, gaussian_slot_matching


def partial_episode(seed=11, steps=10, size=32):
    """Controlled 3D overlapping spheres with camera movement and changing occlusion."""
    if steps<5 or size<24:
        raise ValueError("Partial-occlusion test needs >=5 frames, >=24 pixels")
    base=world3d.episode(seed,steps,size,"train")
    radii=np.array([.52,.57,.33],np.float32)
    colors=np.array([[.94,.17,.12],[.12,.80,.94],[.93,.80,.12]],np.float32)
    positions=[]
    for t in range(steps):
        # Rear object 0 is partially or entirely covered by moving closer object 1.
        positions.append(np.array([[0.,0.,4.],[-.8+.20*t,0.,3.2],[.85,.3,4.8]],
                                  np.float32))
    positions=np.stack(positions)
    poses=base["T_world_camera"]
    K=base["K"]
    rgb,depth,instance=[],[],[]
    for t in range(steps):
        # Reuse the exact camera from the known-action world3d episode.
        pose=poses[t]
        yaw=float(np.arctan2(pose[0,2],pose[0,0]))
        image,d,ids=world3d.render(
            positions[t],radii,colors,pose[:3,3],yaw,K,size)
        rgb.append(image);depth.append(d);instance.append(ids)
    return {**base,"rgb":np.stack(rgb),"depth":np.stack(depth),
            "instance_id":np.stack(instance),
            "centers_world":positions,"radii":radii,"colors":colors,
            "domain":"controlled_partial_occlusion"}


def visible_fractions(record,t):
    """Evaluate partial occlusion from object-only projected support."""
    K=record["K"]
    pose=record["T_world_camera"][t]
    yaw=float(np.arctan2(pose[0,2],pose[0,0]))
    visible=record["instance_id"][t]
    values=[]
    for i,(center,radius,color) in enumerate(zip(
            record["centers_world"][t],record["radii"],record["colors"])):
        _,_,alone=world3d.render(
            center[None],np.asarray([radius]),color[None],
            pose[:3,3],yaw,K,visible.shape[0])
        count=int(np.count_nonzero(alone>=0))
        count_seen=int(np.count_nonzero(visible==i))
        values.append(count_seen/count if count else 0.)
    return np.asarray(values)


def _project(xyz,K,x,yaw):
    camera=(jnp.asarray(xyz)-jnp.array([x,0,0]))@gaussian.rotation(yaw)
    z=jnp.maximum(camera[:,2],1e-3)
    return np.asarray(jnp.stack(
        (K[0,0]*camera[:,0]/z+K[0,2],
         -K[1,1]*camera[:,1]/z+K[1,2]),axis=-1))


def trace(params,record,tolerance_px=4.):
    """Oracle-truth identity used ONLY after image-based slot inference."""
    from src import gaussian_slot_matching
    frames=record["rgb"]
    K=jnp.asarray(record["K"])
    x,yaw=gaussian.camera_traces(jnp.asarray(record["camera_actions"]))
    results=[]
    for t,frame in enumerate(frames):
        state,_,_,_=gaussian.infer(params,jnp.asarray(frame),K,x[t],yaw[t])
        pred=_project(state,K,x[t],yaw[t])
        truth=_project(record["centers_world"][t],K,x[t],yaw[t])
        fractions=visible_fractions(record,t)
        visible_ids=np.flatnonzero(fractions>0)
        if not len(visible_ids):
            results.append({"frame":t,"objects":{},"partial_objects":0,
                            "hidden_objects":len(fractions)})
            continue
        cost=np.linalg.norm(truth[visible_ids,None]-pred[None],axis=-1)
        matches,missing=gaussian_slot_matching.assignment(
            cost,miss_penalty=np.hypot(*frame.shape[:2]))
        objects={int(visible_ids[i]):{"slot":int(slot),"error_px":float(distance),
                                  "visible_fraction":float(fractions[visible_ids[i]]),
                                  "localized":bool(distance<=tolerance_px)}
                 for i,slot,distance in matches}
        for index in missing:
            objects[int(visible_ids[index])]={"slot":None,"error_px":None,
                "visible_fraction":float(fractions[visible_ids[index]]),
                "localized":False}
        results.append({"frame":t,"objects":objects,
            "partial_objects":int(np.count_nonzero((fractions>0)&(fractions<.98))),
            "hidden_objects":int(np.count_nonzero(fractions==0))})
    return summarize(results,tolerance_px)


def summarize(frames,tolerance_px=4.):
    """A changed slot index is not identity loss when both matches stay good."""
    switches,harmless,losses,pairs=0,0,0,0
    errors=[]
    last={}
    for frame in frames:
        for identity,item in frame["objects"].items():
            if item["error_px"] is not None:
                errors.append(item["error_px"])
            if not item["localized"]:
                losses+=1
            prior=last.get(identity)
            if prior and item["slot"] is not None and prior["slot"] is not None:
                pairs+=1
                if item["slot"]!=prior["slot"]:
                    switches+=1
                    if item["localized"] and prior["localized"]:
                        harmless+=1
            last[identity]=item
    return {"per_frame":frames,
            "mean_aligned_center_error_px":float(np.mean(errors)) if errors else None,
            "visible_identity_failures":losses,
            "raw_slot_switches":switches,
            "localized_permutation_only_switches":harmless,
            "unresolved_switches":switches-harmless,
            "adjacent_visible_pairs":pairs,
            "frames_with_partial_occlusion":sum(
                r["partial_objects"]>0 for r in frames),
            "frames_with_full_invisibility":sum(
                r["hidden_objects"]>0 for r in frames),
            "tolerance_px":tolerance_px,
            "limits":"Assignment uses privileged truth only for scoring, "
             "not for inference. Harmless reordering means both adjacent "
             "positions were accurate after permutation; it is not learned "
             "identity persistence. Hidden objects cannot be evaluated from "
             "visual correspondence at those times."}
