"""Prequential visual adaptation and bounded rehearsal on frozen forecasters.

Frozen RGB NCA/CNN/transport checkpoints stay unchanged; a small per-channel
bias adapter is updated only AFTER the previously committed prediction has
been scored. Memory stores previously observed residual labels. Both adaptive
arms expend two SGD steps per observation; replay carries additional bytes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import frozen_visual_bridge as bridge
from src import rgb_world_ladder as worlds, camera_robustness as camera

STAGES=(("A","linear","none"),("B","appearance","exposure"),
        ("A_return","linear","none"))
MODES=("frozen","recent_only","replay")


def episodes(seed, context=4, length=10, size=20):
    """Repeated A/B/A conditions with disjoint trajectory seeds per stage."""
    records=[]
    for stage,(_,domain,corruption) in enumerate(STAGES):
        r=worlds.episode(seed+1000*stage,context+length,size,domain=domain)
        observed=camera.corrupt(r["frames"],corruption,seed+1000*stage)["frames"]
        records.append(observed)
    return records


def update(bias, forecast, observed, rate=.15):
    """Small bounded online gradient on per-RGB-channel additive correction."""
    pred=np.clip(forecast+bias,0,1)
    # Straight-through boundary gradient: a dark (0) or saturated (1)
    # forecast must still be able to adapt toward the observed target.
    # Otherwise the clipped readout creates an absorbing fixed point.
    grad=np.mean(2*(pred-observed),axis=(0,1))
    return np.clip(bias-rate*grad,-.25,.25)


def event_stream(name,params,frames,context,length,ticks):
    """Compute frozen image forecasts causally; labels stored separately."""
    if len(frames)!=context+length:
        raise ValueError("Bad streaming episode length")
    events=[]
    for t in range(context,len(frames)):
        # This forecast must not see image[t] (the target).
        future=bridge.predict(name,params,frames[None,t-context:t],1,ticks)[0,0]
        events.append((future,frames[t]))
    return events


def train_events(mode,events,rate=.15,capacity=12,seed=11):
    if mode not in MODES or capacity<1 or rate<=0:
        raise ValueError("Bad adaptation configuration")
    rng=np.random.default_rng(seed)
    bias=np.zeros(3)
    memory=[]
    reports=[]
    replayed=0
    updates=0
    step=0
    for stage,stream in enumerate(events):
        errors=[]
        for pred,truth in stream:
            committed=np.clip(pred+bias,0,1)
            errors.append(float(np.square(committed-truth).mean()))
            if mode!="frozen":
                bias=update(bias,pred,truth,rate)
                updates+=1
                if mode=="replay" and memory:
                    index=int(rng.integers(len(memory)))
                    prior=memory[index]
                    if prior[0]>=step:
                        raise AssertionError("Future-target replay leakage")
                    bias=update(bias,prior[1],prior[2],rate)
                    replayed+=1
                else:
                    # Equal gradient count; recent-only also spends two steps.
                    bias=update(bias,pred,truth,rate)
                updates+=1
                entry=(step,pred.copy(),truth.copy())
                if len(memory)<capacity:
                    memory.append(entry)
                else:
                    replacement=int(rng.integers(0,step+1))
                    if replacement<capacity:
                        memory[replacement]=entry
            step+=1
        reports.append({"stage":STAGES[stage][0],
                        "prequential_mse":float(np.mean(errors)),
                        "start_mse":float(np.mean(errors[:min(3,len(errors))])),
                        "end_mse":float(np.mean(errors[-min(3,len(errors)):])),
                        "bias_at_end":bias.tolist()})
    return {"stages":reports,"updates":updates,"replayed":replayed,
            "memory_entries":len(memory),"memory_bytes":sum(
                a.nbytes+b.nbytes for _,a,b in memory),"final_bias":bias.tolist()}


def action_control(name,params,record,context,ticks=1):
    """Separately compare known *committed* camera action vs no action.

    Image shift uses zero-filled edges, not wrapping. This is a classical
    action-aware postprocessing control, NOT a trained multimodal NCA.
    """
    from experiments.development_27_egomotion import shift
    rgb=record["frames"]
    if len(rgb)<context+1 or context<2:
        raise ValueError("Need visible context and target")
    predicted=bridge.predict(name,params,rgb[None,:context],1,ticks)[0,0]
    action=record["camera_actions"][context-1]
    # RGB roll in y and x with a zero border (one-axis helper twice).
    def shift2(image,dy,dx):
        row_first=np.swapaxes(shift(np.swapaxes(image,0,1),int(dx)),0,1)
        return shift(row_first,int(dy))
    adjusted=shift2(predicted,int(-np.rint(action[1])),int(-np.rint(action[0])))
    target=rgb[context]
    return {"no_action_mse":float(np.square(predicted-target).mean()),
            "known_action_mse":float(np.square(adjusted-target).mean()),
            "planned_action":action.tolist()}


def visual_gate(validation_by_stage,thresholds):
    """Only calibration/validation performance may gate stage advancement."""
    if len(validation_by_stage)!=len(thresholds):
        raise ValueError("Gate data mismatch")
    reached=0
    for i,(errors,limit) in enumerate(zip(validation_by_stage,thresholds)):
        if i!=reached:
            break
        if len(errors)<2 or not np.isfinite(errors).all() or limit<=0:
            raise ValueError("Need independent validation and prespecified threshold")
        if np.sum(np.asarray(errors)<=limit)>=2:
            reached+=1
        else:
            break
    return reached


def run(out="runs/visual-bridge/memory.json",checkpoints=None,smoke=False,
        seeds=(11,23),context=4,length=6,size=20,width=4,ticks=1,
        capacity=12,rate=.15):
    if not seeds or len(set(seeds))!=len(seeds) or context<2 or length<2:
        raise ValueError("Need distinct episode seeds and context")
    checkpoints=checkpoints or {}
    result={"protocol":"frozen-visual-prequential-memory-v1","rows":[],
            "budgets":{"context":context,"length_per_stage":length,"capacity":capacity,
                       "rate":rate,"seeds":list(seeds)},
            "limits":"Model weights frozen; only small RGB output bias adapts. "
             "True frame is read only after forecast scoring. Reservoir contains "
             "past labeled frames, not future targets. 2 gradient steps per "
             "observation for both trained modes, but memory I/O differs. "
             "Cross-modal action arm is a postprocessing control only."}
    for name in bridge.MODELS:
        params,meta=bridge.checkpoint(name,checkpoints.get(name),smoke,width)
        for seed in seeds:
            scenes=episodes(seed,context,length,size)
            streams=[event_stream(name,params,frames,context,length,ticks)
                     for frames in scenes]
            arms={mode:train_events(mode,streams,rate,capacity,int(seed))
                  for mode in MODES}
            # Known scripted actions are not silently supplied to these RGB models.
            action_record=worlds.episode(99000+seed,context+1,size,domain="camera_pan")
            result["rows"].append({"model":name,"seed":int(seed),"checkpoint":meta,
                                   "modes":arms,
                                   "action_control":action_control(
                                       name,params,action_record,context,ticks)})
    if out:
        path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/visual-bridge/memory.json")
    p.add_argument("--nca-checkpoint")
    p.add_argument("--cnn-checkpoint")
    p.add_argument("--transport-checkpoint")
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    manifest=run(a.out,{"nca":a.nca_checkpoint,"cnn":a.cnn_checkpoint,
                        "transport":a.transport_checkpoint},a.smoke)
    print(json.dumps({"models":len(manifest["rows"]),"protocol":manifest["protocol"]}))


if __name__=="__main__":
    main()
