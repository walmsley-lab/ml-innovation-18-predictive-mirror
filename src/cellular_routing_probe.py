"""Causal routing diagnostics on a frozen cellular-reference checkpoint.

The probe reads source/target positions as evaluator-only labels; they are
not additional model inputs. A checkpoint is never tuned on the test set.
"""
import numpy as np
import jax.numpy as jnp
from src import cellular_references as cells


def analyze(params, examples, ticks=4):
    pulses, destinations, signs = examples
    if len(signs) < 1 or len(pulses) != len(destinations):
        raise ValueError("Need aligned episodes")
    size = pulses.shape[1]
    if pulses.shape[1:3] != destinations.shape[1:3]:
        raise ValueError("Inconsistent spatial grid")
    x, y = jnp.asarray(pulses), jnp.asarray(destinations)
    predictions = {}
    for mode in cells.MODES:
        guess, routing = cells.rollout(params,x,y,ticks,mode,True)
        guess, routing = np.asarray(guess), np.asarray(routing)
        dst = np.argmax(destinations[...,0].reshape(len(signs),-1),axis=1)
        src = np.argmax(np.abs(pulses[...,0]).reshape(len(signs),-1),axis=1)
        src_mass = routing[np.arange(len(signs)),dst,src]
        predictions[mode] = {
            "mse":float(np.mean((guess-signs)**2)),
            "sign_accuracy":float(np.mean(np.sign(guess)==signs)),
            "source_routing_mass":float(np.mean(src_mass)),
            "source_mass_uniform_null":1.0/(size*size),
            "routing_entropy_nats":float(np.mean(-np.sum(
                routing*np.log(routing+1e-9),axis=-1))),
            "per_episode_signed_output":guess.tolist(),
        }
    # Exact reference knockout: make all query-key logits zero, keeping the
    # checkpoint's learned message production and receiver weights fixed.
    muted={**params,"keys":jnp.zeros_like(params["keys"])}
    knocked=np.asarray(cells.rollout(muted,x,y,ticks,"learned"))
    predictions["key_lesion"]={
        "mse":float(np.mean((knocked-signs)**2)),
        "sign_accuracy":float(np.mean(np.sign(knocked)==signs)),
        "per_episode_signed_output":knocked.tolist(),
    }
    return predictions
