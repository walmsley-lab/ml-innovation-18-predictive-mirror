"""Versioned reports and tiny deterministic environments; model inputs exclude truth."""
import hashlib
import json
import random
PROTOCOL="emergence-v1"
TASKS=("memory","routing","switch","damage")

def validate(record):
    needed={"protocol","experiment","seed","condition","metrics","costs","provenance"}
    if not isinstance(record,dict) or not needed <= record.keys():
        raise ValueError("Incomplete run record")
    if record["protocol"] != PROTOCOL or type(record["seed"]) is not int:
        raise ValueError("Wrong protocol or seed")
    if any(not isinstance(record[x],dict) for x in ("metrics","costs","provenance")):
        raise ValueError("Metrics/costs/provenance must be mappings")
    return record

def digest(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()

def split_seed(split,index):
    roots={"train":100000,"validation":200000,"test":300000}
    if split not in roots or not 0<=index<100000:
        raise ValueError("Invalid split")
    return roots[split]+index

def episode(task,seed,length=32,demand=4):
    if task not in TASKS or length<8 or not 1<=demand<length:
        raise ValueError("Invalid task or dimensions")
    rng=random.Random(seed)
    bits=[rng.randrange(2) for _ in range(length)]
    if task=="memory":
        observed=bits
        targets=[None]*demand+bits[:-demand]
        truth={"lag":demand}
    elif task=="routing":
        observed=[(rng.randrange(demand),bit) for bit in bits]
        targets=[None]*demand+bits[:-demand]
        truth={"distance":demand}
    elif task=="switch":
        observed=bits
        targets=[bit if t<length//2 else 1-bit for t,bit in enumerate(bits)]
        truth={"switch_at":length//2}
    else:
        observed=bits
        targets=bits[:]
        truth={"damage_from":length//2,"damage_fraction":min(1.,demand/10.)}
    return {"task":task,"observations":observed,"targets":targets,
            "evaluator_only":truth,"seed":int(seed)}
