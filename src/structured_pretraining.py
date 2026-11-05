"""Matched-budget source curricula for CNN and recurrent NCA transfer.

Training-time future RGB is a supervision target; inference sees only
context. Synthetic wave/diffusion patterns are *engineering* stimuli,
not a claim about homologous fly or mammalian visual development.
"""
import jax
import jax.numpy as jnp
import numpy as np
from src import physical_fields as physical, rgb_world_ladder as worlds
from src import tiny_cnn_video as cnn, predictive_rgb_nca as nca

SOURCES=("wave","diffusion","sprites","motion","shuffled","scratch")
ARCHITECTURES=("cnn","nca")


def samples(start,count,source,context=5,horizon=3,size=24):
    if source not in SOURCES or count<1:
        raise ValueError("Unknown source")
    total=context+horizon
    records=[]
    for i in range(count):
        seed=start+i
        if source in ("wave","diffusion"):
            episode=physical.episode(seed,kind=source,size=size,steps=total)
            u=episode["observed"].astype(np.float32)
            # Fixed channel projection, shared across every episode.
            frames=np.stack((.5+.40*u,.5+.20*u,.5-.30*u),axis=-1)
            frames=np.clip(frames,0,1)
        else:
            domain="occlusion_short" if source=="sprites" else "linear"
            frames=worlds.episode(seed,total,size,balls=2,
                                  domain=domain)["frames"]
            if source=="shuffled":
                rng=np.random.default_rng(int(seed)+918)
                frames=frames[rng.permutation(total)]
        records.append(frames)
    return np.stack(records).astype(np.float32)


def architecture(name,key,width=8,ticks=1,horizon=3):
    if name=="cnn":
        return cnn.init(key,width),lambda p,c:cnn.rollout(p,c,horizon)
    if name=="nca":
        return nca.init(key,width),lambda p,c:nca.rollout(p,c,horizon,ticks)
    raise ValueError("Unknown model family")


def objective(params,examples,context,predict):
    actual=examples[:,context:]
    output=predict(params,examples[:,:context])
    return jnp.mean(jnp.square(output-actual))
