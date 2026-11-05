"""63: register neural graph coordinates to a visual sheet explicitly.

A connectome's node order is NOT retinotopy. Synthetic layout is labeled.
Measured inputs require provided coordinates: no anatomical projection can
be inferred from a connectivity matrix alone.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import roadmap_62_connectome_contract as contract


def toy_positions(count):
    theta=2*np.pi*np.arange(count)/count
    return np.stack((.5+.38*np.cos(theta),.5+.38*np.sin(theta)),axis=-1)


def validate(positions,n):
    x=np.asarray(positions,np.float64)
    if x.shape!=(n,2) or not np.isfinite(x).all() or (
            np.any(x<0) or np.any(x>1)):
        raise ValueError("Need explicit normalized 2D node-to-field positions")
    return x


def rasterize(signals,positions,size=24,sigma=.12):
    values=np.asarray(signals,float)
    if values.ndim!=2 or size<8 or sigma<=0:
        raise ValueError("Signals [time,nodes], valid image size and sigma")
    coords=validate(positions,values.shape[1])
    y,x=np.mgrid[:size,:size]/(size-1)
    distance=(x[:,:,None]-coords[None,None,:,0])**2+(
               y[:,:,None]-coords[None,None,:,1])**2
    weights=np.exp(-distance/(2*sigma*sigma))
    weights/=np.maximum(weights.sum(axis=2,keepdims=True),1e-12)
    return np.einsum("hwn,tn->thw",weights,values)


def signals(graph,steps=10,seed=11):
    if steps<2:
        raise ValueError("Need more than one graph step")
    rng=np.random.default_rng(seed)
    x=rng.normal(size=len(graph))
    p=np.asarray(graph,float)
    p=p/np.maximum(p.sum(axis=1,keepdims=True),1)
    out=[x.copy()]
    for _ in range(steps-1):
        x=.7*x+.3*p@x
        out.append(x.copy())
    return np.stack(out)


def run(matrix=None,metadata=None,coordinates=None,out="runs/roadmap-63/metrics.json",
        size=24,seeds=(11,23,41)):
    if bool(matrix)!=bool(metadata):
        raise ValueError("Matrix and source metadata must be paired")
    if matrix:
        graph,ids,meta=contract.load(matrix,metadata)
        if not coordinates:
            raise ValueError("External graph requires explicit coordinate mapping")
        with np.load(coordinates,allow_pickle=False) as data:
            layout=validate(data["positions"],len(graph))
        layout_source="explicit_external_registration_unverified"
    else:
        graph,ids=contract.toy()
        meta={"source_id":"toy-12","source_type":"synthetic"}
        layout=toy_positions(len(graph))
        layout_source="synthetic_ring_layout"
    rows=[]
    for seed in seeds:
        original=signals(graph,seed=int(seed))
        raster=rasterize(original,layout,size)
        shuffled=layout[np.random.default_rng(int(seed)+712).permutation(len(layout))]
        control=rasterize(original,shuffled,size)
        rows.append({"seed":int(seed),
                     "temporal_change_energy":float(np.square(np.diff(raster,axis=0)).mean()),
                     "permuted_layout_change_energy":float(
                         np.square(np.diff(control,axis=0)).mean()),
                     "layout_effect_mse":float(np.square(raster-control).mean())})
    report={"roadmap_id":63,"connectome_source":meta["source_id"],
            "source_type":meta["source_type"],"layout_source":layout_source,
            "nodes":len(ids),"size":size,"rows":rows,
            "limitations":"Graph node coordinates must be externally registered "
             "to genuine retinotopy; synthetic ring is not biological. "
             "Permutation null alters registration, not synaptic topology. "
             "Raster statistics do not establish neural agreement."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix");p.add_argument("--metadata");p.add_argument("--coordinates")
    p.add_argument("--out",default="runs/roadmap-63/metrics.json")
    a=p.parse_args()
    print(json.dumps(run(a.matrix,a.metadata,a.coordinates,a.out),indent=2))
