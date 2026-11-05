"""45: deterministic release evidence with verified file hashes.

Default: raw train/validation/test footage is NOT embedded (privacy); only
SHA256 and metadata are emitted. Checkpoints/metrics are included when listed.
Never automatically publish recordings or claim to reproduce training without
source data and an exact environment.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

REQUIRED=("train","validation","test","checkpoint","metrics")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def evidence(spec,include_recordings=False):
    if not re.fullmatch("[0-9a-f]{40}",spec.get("git_sha","")):
        raise ValueError("Require exact 40-character commit SHA")
    sources=spec.get("sources",[])
    roles={x.get("role") for x in sources}
    if not set(REQUIRED).issubset(roles):
        raise ValueError("Need train, validation, test, checkpoint and metrics")
    records=[];embedded={}
    for i,entry in enumerate(sources):
        path=Path(entry["path"])
        if not path.is_file():
            raise ValueError("Missing evidence artifact "+str(path))
        if path.stat().st_size>100_000_000:
            raise ValueError("Refuse extremely large archive input")
        data=path.read_bytes()
        digest=sha(data)
        role=entry["role"]
        include=(include_recordings or role not in ("train","validation","test"))
        archive_name=f"evidence/{i:03d}-{digest[:12]}-{path.name}"
        if include:
            embedded[archive_name]=data
        records.append({"role":role,"name":path.name,"sha256":digest,
                        "size_bytes":len(data),"included":include,
                        "archive_path":archive_name if include else None,
                        "capture_session":entry.get("capture_session"),
                        "representation":entry.get("representation")})
    splits={role:{r["sha256"] for r in records if r["role"]==role}
            for role in ("train","validation","test")}
    if any(splits[a]&splits[b] for a,b in
           (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Identical bytes reused across train/validation/test")
    sessions={role:{r["capture_session"] for r in records
                   if r["role"]==role and r["capture_session"]}
              for role in splits}
    if any(sessions[a]&sessions[b] for a,b in
           (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Capture session reused across splits")
    manifest={"protocol":"predictive-mirror-release-v1",
              "git_sha":spec["git_sha"],"hardware":spec.get("hardware"),
              "software":spec.get("software"),"run_config":spec.get("run_config"),
              "sources":records,"recordings_embedded":include_recordings,
              "constraints":"File hashes and session labels cannot prove independent "
               "capture or full reproducibility. Raw footage omitted by default; "
               "operator must approve sharing private video and model weights."}
    return manifest,embedded


def pack(spec_path,output,include_recordings=False):
    spec=json.loads(Path(spec_path).read_text())
    manifest,files=evidence(spec,include_recordings)
    content=json.dumps(manifest,indent=2,sort_keys=True).encode()+b"\n"
    payload={"manifest.json":content,**files}
    destination=Path(output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(payload):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,payload[name])
    return verify(destination)


def verify(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or "manifest.json" not in names:
            raise ValueError("Malformed release archive")
        manifest=json.loads(z.read("manifest.json"))
        for item in manifest["sources"]:
            if item["included"]:
                if sha(z.read(item["archive_path"]))!=item["sha256"]:
                    raise ValueError("Corrupted archived source "+item["name"])
            elif item["archive_path"] is not None:
                raise ValueError("External evidence must have no archive path")
    return {"archive":str(path),"sha256":sha(Path(path).read_bytes()),
            "source_count":len(manifest["sources"]),
            "raw_recordings_embedded":manifest["recordings_embedded"],
            "verification":"included file SHA256 verified; withheld files metadata only"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec",required=True)
    p.add_argument("--out",default="runs/release/predictive-mirror-evidence.zip")
    p.add_argument("--include-recordings",action="store_true")
    a=p.parse_args()
    print(json.dumps(pack(a.spec,a.out,a.include_recordings),indent=2))


if __name__=="__main__":
    main()
