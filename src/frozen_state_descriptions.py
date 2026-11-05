"""Fit sparse descriptors to a FROZEN NCA hidden trajectory.

PCA coordinates are learned only from training states. The fitted
SINDy/linear dynamics are descriptive approximations, NOT claims
that the network literally implements these equations.
"""
import numpy as np
import jax.numpy as jnp
from src import predictive_rgb_nca as nca, interpretable_sindy as sindy


def hidden_trace(params,frame,ticks=10,perturb=False):
    frame=jnp.asarray(frame,jnp.float32)
    if frame.ndim!=3 or frame.shape[-1]!=3 or ticks<3:
        raise ValueError("Expected one RGB frame and >=3 cell updates")
    hidden=jnp.zeros((1,*frame.shape[:2],params["rgb_readout"].shape[0]),
                     jnp.float32)
    image=frame[None]
    records=[np.asarray(hidden[0].mean(axis=(0,1)))]
    for t in range(ticks):
        if perturb and t==2:
            hidden=hidden.at[:,frame.shape[0]//2-1:frame.shape[0]//2+2,
                              frame.shape[1]//2-1:frame.shape[1]//2+2,0].add(.8)
        hidden=nca.cell_update(params,hidden,image,jnp.zeros_like(image))
        records.append(np.asarray(hidden[0].mean(axis=(0,1))))
    return np.stack(records)


def fit_pca(training,components=4):
    training=np.asarray(training,np.float64)
    if training.ndim!=3 or training.shape[-1]<components:
        raise ValueError("Need [sequences,steps,channels] and >=4 channels")
    observations=training.reshape(-1,training.shape[-1])
    mean=observations.mean(axis=0)
    _,_,v=np.linalg.svd(observations-mean,full_matrices=False)
    return {"mean":mean,"basis":v[:components].T}


def project(traces,encoding):
    traces=np.asarray(traces,np.float64)
    return (traces-encoding["mean"])@encoding["basis"]


def fit_descriptors(training,methods=("sindy","affine","dmd")):
    encoding=fit_pca(training)
    z=project(training,encoding)
    source=z[:,:-1].reshape(-1,4)
    target=z[:,1:].reshape(-1,4)
    return encoding,{m:sindy.fit(source,target,method=m,threshold=.01)
                     for m in methods}


def evaluate(models,encoding,traces,context=3,horizon=5):
    latent=project(traces,encoding)
    if context<2 or context+horizon>latent.shape[1]:
        raise ValueError("Missing held-out rollout truth")
    initial=latent[:,context-1]
    truth=latent[:,context:context+horizon]
    outputs={}
    for method,model in models.items():
        guess=sindy.rollout(model,initial,horizon)
        outputs[method]={"latent_rmse":float(np.sqrt(np.mean((guess-truth)**2))),
                         "per_horizon_rmse":np.sqrt(np.mean((guess-truth)**2,
                                                  axis=(0,2))).tolist()}
    outputs["persistence"]={"latent_rmse":float(np.sqrt(np.mean(
        (initial[:,None]-truth)**2)))}
    return outputs
