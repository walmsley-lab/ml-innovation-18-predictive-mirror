"""62: strict, provenance-linked connectome data for causal graph studies.

The reference is SYNTHETIC unless independently acquired external data with
source metadata is supplied. Directed and undirected adjacency are distinct;
never silently symmetrize synaptic graphs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

REQUIRED=("source_id","source_type","species","citation","license",
          "preprocessing","connectome_version","matrix_sha256")


def validate(adjacency,node_ids):
    a=np.asarray(adjacency,dtype=np.float64)
    ids=np.asarray(node_ids)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)<3 or (
            ids.shape!=(len(a),)) or len(np.unique(ids))!=len(ids) or
            not np.isfinite(a).all() or np.any(a<0) or
            not np.allclose(np.diag(a),0)):
        raise ValueError("Need finite nonnegative zero-diagonal adjacency with unique IDs")
    return a


def toy(seed=11,n=12):
    if n<4:
        raise ValueError("At least four nodes")
    rng=np.random.default_rng(seed)
    a=np.zeros((n,n))
    for i in range(n):
        a[i,(i+1)%n]=1.+.4*rng.random()
        a[(i+1)%n,i]=1.+.4*rng.random()
    for i in range(n):
        if rng.random()<.3:
            j=(i+3)%n
            a[i,j]=.3+.3*rng.random()
    return a,np.array([f"cell-{i:03d}" for i in range(n)])


def synthetic_artifact(out,seed=11,n=12):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    a,ids=toy(seed,n)
    path=root/"adjacency.npz"
    np.savez_compressed(path,adjacency=a,node_ids=ids)
    info={"source_id":f"toy-{seed}-{n}","source_type":"synthetic",
          "species":"not_applicable","citation":"self-generated controlled reference",
          "license":"generated-in-repository","preprocessing":"directed ring with shortcuts",
          "connectome_version":"toy-v1",
          "matrix_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (root/"metadata.json").write_text(json.dumps(info,indent=2)+"\n")
    return path,root/"metadata.json"


def load(matrix,metadata):
    with open(metadata) as f:
        meta=json.load(f)
    if any(not meta.get(key) for key in REQUIRED) or meta["source_type"] not in (
            "synthetic","measured"):
        raise ValueError("Missing source provenance or illegal evidence type")
    digest=hashlib.sha256(Path(matrix).read_bytes()).hexdigest()
    if digest!=meta["matrix_sha256"]:
        raise ValueError("Connectome file SHA does not match source record")
    with np.load(matrix,allow_pickle=False) as d:
        a=validate(d["adjacency"],d["node_ids"])
        ids=d["node_ids"].copy()
    return a,ids,meta


def summarize(a,meta):
    return {"source_id":meta["source_id"],"source_type":meta["source_type"],
            "nodes":len(a),"directed":not bool(np.allclose(a,a.T)),
            "weighted_edges":int(np.count_nonzero(a)),
            "out_degree_mean":float((a>0).sum(1).mean()),
            "total_connection_weight":float(a.sum()),
            "evidence_level":("controlled_synthetic" if meta["source_type"]=="synthetic"
                else "external_record_with_self_reported_metadata_not_independently_verified")}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-62")
    p.add_argument("--matrix")
    p.add_argument("--metadata")
    a=p.parse_args()
    matrix,info=(a.matrix,a.metadata) if a.matrix and a.metadata else synthetic_artifact(a.out)
    graph,_,record=load(matrix,info)
    print(json.dumps(summarize(graph,record),indent=2))


if __name__=="__main__":
    main()
