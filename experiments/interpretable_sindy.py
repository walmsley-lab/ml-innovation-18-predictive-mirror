"""SINDy vs affine/DMD/coast on disjoint synthetic state trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import interpretable_sindy as sid, rgb_world_ladder as worlds

METHODS=("sindy","affine","dmd")


def trajectories(start,count,domain,length=12):
    records=[worlds.episode(start+i,length,32,balls=2,domain=domain)
             for i in range(count)]
    return np.stack([np.concatenate((r["positions"][:,0],
                    r["velocities"][:,0]),axis=-1) for r in records])


def run(out="runs/interpretable-sindy/metrics.json",seeds=(11,23,41),
        train_domain="linear",test_domains=("linear","accelerating","collision"),
        noise_levels=(0,.02,.1),horizon=5):
    if not seeds or horizon<1:
        raise ValueError("Invalid SINDy protocol")
    test={domain:trajectories(91000,24,domain,4+horizon)
          for domain in test_domains}
    reports=[]
    for seed in seeds:
        train=trajectories(31000+int(seed)*200,40,train_domain,4+horizon)
        train_x=train[:,:-1].reshape(-1,4)
        train_y=train[:,1:].reshape(-1,4)
        for noise in noise_levels:
            rng=np.random.default_rng(int(seed)+int(noise*10000))
            noisy=train_x+rng.normal(0,noise,train_x.shape)
            for method in METHODS:
                model=sid.fit(noisy,train_y,method)
                scores={}
                for domain,records in test.items():
                    guess=sid.rollout(model,records[:,3],horizon)
                    target=records[:,4:]
                    control=sid.coast(records[:,3],horizon)
                    scores[domain]={"position_mae_px":float(np.mean(np.linalg.norm(
                        guess[...,:2]-target[...,:2],axis=-1))),
                        "state_rmse":float(np.sqrt(np.mean((guess-target)**2))),
                        "coast_position_mae_px":float(np.mean(np.linalg.norm(
                        control[...,:2]-target[...,:2],axis=-1)))}
                reports.append({"seed":int(seed),"noise_sd":noise,
                    "method":method,"active_terms":model["active_terms"],
                    "equations":sid.equations(model),"heldout":scores})
    output={"protocol":"supplied-state-sindy-v1","results":reports,
            "input":"Privileged ground-truth x,y,vx,vy from synthetic worlds",
            "time_unit":"one generated frame",
            "limits":"Tests interpretable supplied-state dynamics, not visual perception "
              "or learned hidden NCA states. Noise is added to training "
              "inputs, not target state. Full state availability is more "
              "informative than RGB-only controls; compare only inside "
              "this privileged-state task. Confidence needs independent "
              "episode resampling and training seeds."}
    target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,indent=2)+"\n")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/interpretable-sindy/metrics.json")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    a=p.parse_args()
    report=run(a.out,a.seeds)
    print(json.dumps({"fit_variants":len(report["results"])}))


if __name__=="__main__":
    main()
