"""Frozen action/hidden-state interventions on an action-conditioned RGB NCA.

Interventions occur after all observed context frames. With no lesion
the rollout must be exactly equal to the baseline model's rollout.
"""
import jax
import jax.numpy as jnp
from src import action_rgb_nca as model

MODES=("intact","zero_hidden","shuffle_hidden",
       "blind_actions","shifted_actions")


def rollout(params,context,actions,horizon=5,mode="intact"):
    if mode not in MODES:
        raise ValueError("Unknown intervention")
    if (context.ndim!=5 or context.shape[-1]!=3 or context.shape[1]<2
            or horizon<1 or actions.shape!=(
                context.shape[0],context.shape[1]+horizon-1,2)):
        raise ValueError("Context/actions mismatch")
    if mode=="blind_actions":
        actions=jnp.zeros_like(actions)
    elif mode=="shifted_actions":
        actions=jnp.roll(actions,1,axis=0)
    batch,count,height,width,_=context.shape
    hidden=jnp.zeros((batch,height,width,params["readout"].shape[0]))
    state=model.observe(params,(hidden,jnp.zeros_like(context[:,0])),
                        context[:,0])
    def absorb(state,pair):
        frame,action=pair
        state,_=model.dream(params,state,action)
        return model.observe(params,state,frame),None
    state,_=jax.lax.scan(absorb,state,(
        jnp.swapaxes(context[:,1:],0,1),
        jnp.swapaxes(actions[:,:count-1],0,1)))
    if mode=="zero_hidden":
        state=(jnp.zeros_like(state[0]),state[1])
    if mode=="shuffle_hidden":
        state=(jnp.roll(state[0],3,axis=2),state[1])
    _,output=jax.lax.scan(
        lambda s,a:model.dream(params,s,a),state,
        jnp.swapaxes(actions[:,count-1:],0,1))
    return jnp.swapaxes(output,0,1)
