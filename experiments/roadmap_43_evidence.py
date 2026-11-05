"""Roadmap 43: provenance-first architecture evidence comparisons.

Ingest independently generated experiment metrics without erasing their input
privilege, task, units, horizon, seeds or hardware. Only within-protocol,
same-input, same-metric comparisons are eligible for paired uncertainty.
No global architecture leaderboard; mixed measurements stay separate.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMAS=(36,37,38,39,40,41,42)


def _rows(report):
    """Yield canonical scalar records; keep privileged inputs visibly separate."""
    roadmap=report.get("roadmap_id",report.get("protocol"))
    if roadmap not in SCHEMAS and roadmap!="frozen-visual-bridge-v1":
        raise ValueError("Unsupported roadmap report: "+str(roadmap))
    emitted=[]
    def add(seed,model,domain,lead,metric,unit,value,representation,scope):
        if value is None or not np.isfinite(float(value)):
            return
        emitted.append({"roadmap_id":roadmap,"seed":seed,
                        "model":str(model),"domain":str(domain),
                        "lead":int(lead),"metric":str(metric),
                        "unit":str(unit),"value":float(value),
                        "input_representation":representation,
                        "protocol_scope":scope})

    if roadmap=="frozen-visual-bridge-v1":
        # Each model block repeats identical persistence/velocity controls;
        # include those controls once per frozen scene, not as extra evidence.
        for domain,models in report["cases"].items():
            first=sorted(models)[0]
            for checkpointed,comparison in models.items():
                for name,scores in comparison["metrics"].items():
                    if name!=checkpointed and checkpointed!=first:
                        continue
                    for lead,values in scores.items():
                        for metric in ("mse","motion_mse"):
                            add(None,name,domain,lead,metric,"rgb_mse",
                                values.get(metric),"observed_rgb",
                                "frozen-visual-bridge-v1")
    elif roadmap==36:
        # The top level is domain -> model -> forecast lead -> metrics.
        # Roadmap 36 reports an aggregate only; it has no per-seed rows.
        for domain,models in report["cases"].items():
            for name,scores in models.items():
                for lead,values in scores.items():
                    for metric in ("mse","motion_mse"):
                        add(None,name,domain,lead,metric,"rgb_mse",
                            values.get(metric),"observed_rgb","rgb-latent")
    elif roadmap in (37,41):
        for item in report["results"]:
            name=item.get("mode",item.get("arm"))
            for domain,scores in item["heldout"].items():
                for lead,values in scores.items():
                    for metric in ("mse","motion_mse"):
                        add(item.get("seed"),name,domain,lead,metric,"rgb_mse",
                            values.get(metric),"observed_rgb",
                            "phase-video" if roadmap==37 else "signal-transfer")
    elif roadmap in (38,39):
        measure=("visible_joint_mae_px" if roadmap==38 else "position_mae_px")
        for row in report["rows"]:
            representation=("given_skeleton_joints" if roadmap==38
                            else row["input"])
            scope="skeleton-messages" if roadmap==38 else "particle-positions"
            name=row.get("arm","messages" if row.get("messages") else "isolated")
            for model,metrics in row["scores"].items():
                for lead,values in metrics.items():
                    add(row["seed"],name+"/"+model,"heldout",lead,measure,
                        "pixels",values.get(measure),representation,scope)
    elif roadmap==40:
        for domain,arms in report["test"].items():
            for name,scores in arms.items():
                for metric in ("displacement_mae_px","displacement_mse_px2"):
                    add(None,name,domain,1,metric,
                        "pixels" if metric.endswith("mae_px") else "pixels_squared",
                        scores.get(metric),"observed_rgb_center_plus_planned_dt",
                        "irregular-temporal")
    else:  # 42
        # Each circuit arm repeats the same pixel baselines. A baseline
        # computed on one test set must count only once per resolution.
        seen_controls=set()
        for row in report["rows"]:
            for name,metrics in [(row["mode"],row["scores"]),*row["baselines"].items()]:
                if name!=row["mode"]:
                    marker=(row["size"],name)
                    if marker in seen_controls:
                        continue
                    seen_controls.add(marker)
                for lead,values in metrics.items():
                    add(None,name,"resolution-"+str(row["size"]),lead,
                        "mse","rgb_mse",values.get("mse"),
                        "observed_rgb","circuit-hardware")
    return emitted


def summarize(files):
    if not files:
        raise ValueError("Supply generated metrics JSON paths")
    sources=[]
    rows=[]
    for filename in files:
        path=Path(filename)
        report=json.loads(path.read_text())
        added=_rows(report)
        rows.extend(added)
        sources.append({"path":str(path),
                        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "roadmap_id":report.get("roadmap_id",report.get("protocol")),
                        "normalized_rows":len(added)})
    grouped=defaultdict(list)
    for r in rows:
        key=(r["roadmap_id"],r["protocol_scope"],r["input_representation"],
             r["domain"],r["lead"],r["metric"],r["unit"])
        grouped[key].append(r)
    output=[]
    for key,group in sorted(grouped.items(),key=lambda x:str(x[0])):
        per_model=defaultdict(dict)
        for r in group:
            # One independent seed per model/group. Repeated results under
            # different files are not independent replicates.
            if r["seed"] in per_model[r["model"]]:
                raise ValueError("Duplicate model/seed/case: "+str((key,r["model"],r["seed"])))
            per_model[r["model"]][r["seed"]]=r["value"]
        arms={name:{"n_independent_seeds":sum(s is not None for s in seed_map),
                    "descriptive_mean":float(np.mean(list(seed_map.values())))}
              for name,seed_map in per_model.items()}
        paired={}
        names=sorted(per_model)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                shared=sorted(s for s in per_model[a]
                              if s is not None and s in per_model[b])
                if len(shared)<2:
                    continue
                difference=np.asarray([per_model[a][s]-per_model[b][s]
                                       for s in shared])
                rng=np.random.default_rng(20261105)
                sampled=rng.choice(difference,(1000,len(shared)),replace=True)
                ci=np.quantile(np.mean(sampled,axis=1),[.025,.975])
                paired[a+" minus "+b]={
                    "n_paired_seeds":len(shared),"mean_difference":float(difference.mean()),
                    "bootstrap_ci_95":ci.tolist(),
                    "interpretation":"negative favors first arm (lower is better)",
                    "unit":key[-1]}
        output.append({
            "roadmap_id":key[0],"scope":key[1],"representation":key[2],
            "domain":key[3],"lead":key[4],"metric":key[5],"unit":key[6],
            "arms":arms,"paired":paired,
            "status":"descriptive_only" if not paired else "paired_exploratory"})
    return {"protocol":"architecture-evidence-v1","sources":sources,
            "groups":output,"limits":"No cross-representation/global ranking. "
            "Only within-roadmap/protocol/input/domain/horizon/metric/units "
            "groups are comparable; descriptive means are not superiority claims. "
            "Paired seed bootstraps are exploratory and require distinct "
            "independent training/world seeds. Hardware cost and calibration "
            "are separate axes; absent values are not guessed."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics",nargs="+",help="Generated roadmap JSON manifest paths")
    parser.add_argument("--out",default="runs/roadmap-43/evidence.json")
    args=parser.parse_args()
    result=summarize(args.metrics)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"groups":len(result["groups"]),"sources":len(result["sources"])},
                     indent=2))


if __name__=="__main__":
    main()
