"""55: an executable, provenance-linked attention/scene behavior campaign.

Independent experiments preserve their individual units and input privileges.
A trained RGB comparison requires named checkpoints; absent models are skipped
and NEVER replaced with random-init results masquerading as research.
"""
import argparse
import hashlib
import json
from pathlib import Path
from experiments import (
    roadmap_46_attention as e46,
    roadmap_47_budgeted_attention as e47,
    roadmap_48_working_memory as e48,
    roadmap_49_lifecycle as e49,
    roadmap_50_correspondence as e50,
    roadmap_51_irregular_memory as e51,
    roadmap_52_surprise as e52,
    roadmap_53_interventions as e53,
    roadmap_54_active_gaze as e54,
    frozen_visual_bridge as visual)

MODULES=((46,e46),(47,e47),(48,e48),(49,e49),(50,e50),(51,e51),
         (52,e52),(53,e53),(54,e54))


def run(out="runs/roadmap-55",seeds=(91000,91001,91002),
        checkpoints=None,recordings=None):
    if len(set(seeds))!=len(seeds) or len(seeds)<2:
        raise ValueError("Need at least two distinct evaluation episodes")
    folder=Path(out) if out else None
    if folder:folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n,module in MODULES:
        result=module.run(None,seeds=seeds)
        rows.append({"roadmap_id":n,"module":module.__name__,
                     "source_sha256":hashlib.sha256(
                         Path(module.__file__).read_bytes()).hexdigest(),
                     "episode_seeds":list(seeds),"report":result})
    checkpoint_status="skipped_missing_checkpoints"
    rgb=None
    if checkpoints:
        if set(checkpoints)!=set(visual.MODELS):
            raise ValueError("Supply all three named trained checkpoints")
        recorded=recordings or {}
        cal=recorded.get("calibration")
        test=recorded.get("test")
        if bool(cal)!=bool(test):
            raise ValueError("Recording calibration/test must be paired")
        rgb=visual.run(None,checkpoints,smoke=False,
                       calibration_count=12,test_count=3,
                       context=4,horizon=2,size=20,width=8,ticks=1,
                       calibration_recording=cal,test_recording=test)
        checkpoint_status="measured_frozen_models"
    result={"protocol":"attention-scene-evidence-v1",
            "roadmap_ids":[x[0] for x in MODULES],
            "parts":rows,"trained_rgb_status":checkpoint_status,
            "trained_rgb_metrics":rgb,
            "constraints":"No comparison between scalar surprise, structured "
              "color-marker tracking and RGB pixel-MSE experiments. "
              "A 'winner' must be evaluated within matched task, observed "
              "information and measured compute budgets.",
            "limits":"A small multi-experiment launch, NOT evidence that learned "
              "attention or working memory outperforms simple models. "
              "No real camera evidence unless independent recordings and "
              "trained checkpoints supplied."}
    if folder:
        path=folder/"manifest.json"
        path.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-55")
    p.add_argument("--seeds",nargs="+",type=int,default=[91000,91001,91002])
    p.add_argument("--nca-checkpoint")
    p.add_argument("--cnn-checkpoint")
    p.add_argument("--transport-checkpoint")
    p.add_argument("--calibration-recording")
    p.add_argument("--test-recording")
    a=p.parse_args()
    paths=(a.nca_checkpoint,a.cnn_checkpoint,a.transport_checkpoint)
    checkpoints=(dict(zip(visual.MODELS,paths)) if any(paths) else None)
    print(json.dumps({"parts":len(run(a.out,a.seeds,checkpoints,
        {"calibration":a.calibration_recording,"test":a.test_recording}
        )["parts"])},indent=2))


if __name__=="__main__":
    main()
