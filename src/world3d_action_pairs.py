"""Paired interventions in actual perspective-rendered 3D observations.

Both futures share exactly the same RGB past and initial world; only
the *planned* camera actions beyond the last observation differ.
"""
import numpy as np
from src import world3d


def pair(seed, context=5, horizon=5, size=24, translation=.15, yaw=.02,
         domain="train"):
    if context<2 or horizon<1 or size<12 or translation<=0 or yaw<0:
        raise ValueError("Invalid paired-action protocol")
    steps=context+horizon
    actions_plus=np.zeros((steps-1,2),np.float32)
    actions_minus=np.zeros_like(actions_plus)
    # Past actions common: future action on last context frame starts at C-1.
    actions_plus[context-1:]=[translation,yaw]
    actions_minus[context-1:]=[-translation,-yaw]
    plus=world3d.episode(seed,steps,size,domain,actions=actions_plus)
    minus=world3d.episode(seed,steps,size,domain,actions=actions_minus)
    if not np.array_equal(plus["rgb"][:context],minus["rgb"][:context]):
        raise AssertionError("Paired input frames diverged before intervention")
    if not np.array_equal(plus["centers_world"],minus["centers_world"]):
        raise AssertionError("World trajectories changed across camera policies")
    return plus,minus


def scores(positive,negative,pred_pos=None,pred_neg=None,context=5):
    """Contrast-focused action sensitivity; no world metadata enters model."""
    target_pos=np.asarray(positive["rgb"][context:],np.float32)
    target_neg=np.asarray(negative["rgb"][context:],np.float32)
    if target_pos.shape!=target_neg.shape or not len(target_pos):
        raise ValueError("Invalid true counterfactuals")
    contrast=target_pos-target_neg
    world_l1=np.mean(np.abs(contrast),axis=(1,2,3))
    result={"world_action_l1_by_lead":world_l1.tolist(),
            "world_action_l1":float(np.mean(world_l1)),
            "comparison":"identical observed RGB; only future x/yaw camera actions differ"}
    if (pred_pos is None)!=(pred_neg is None):
        raise ValueError("Predictions must come in pairs")
    if pred_pos is None:
        return result
    p=np.asarray(pred_pos,np.float32)
    n=np.asarray(pred_neg,np.float32)
    if p.shape!=contrast.shape or n.shape!=contrast.shape:
        raise ValueError("Model prediction has wrong horizon/image shape")
    estimated=p-n
    predicted_l1=np.mean(np.abs(estimated),axis=(1,2,3))
    denominator=np.sum(contrast*contrast,axis=(1,2,3))+1e-12
    alignment=np.sum(contrast*estimated,axis=(1,2,3))/denominator
    residual=np.mean((estimated-contrast)**2,axis=(1,2,3))
    result.update({
        "model_action_l1_by_lead":predicted_l1.tolist(),
        "model_action_l1":float(np.mean(predicted_l1)),
        "model_to_world_action_ratio":float(np.mean(predicted_l1)/
                                            max(float(np.mean(world_l1)),1e-12)),
        "signed_action_alignment_by_lead":alignment.tolist(),
        "counterfactual_difference_mse_by_lead":residual.tolist(),
        "positive_prediction_mse":float(np.mean((p-target_pos)**2)),
        "negative_prediction_mse":float(np.mean((n-target_neg)**2)),
    })
    return result
