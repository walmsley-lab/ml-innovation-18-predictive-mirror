"""Explicit, partial initialization bridge from a signal NCA to an RGB NCA.

The two architectures are NOT identical: only hidden-channel 3x3
communication and a proposal-mixing block are mapped. No RGB weights,
readout, or gates are inherited.
"""
import jax.numpy as jnp


def transfer(rgb_params,signal_params,shift=0):
    channels=rgb_params["rgb_readout"].shape[0]
    if (signal_params["readout"].shape[0]!=channels or
            signal_params["perception"].shape!=(3,3,channels+2,channels) or
            rgb_params["perception"].shape!=(3,3,channels+6,2*channels)):
        raise ValueError("Communication and RGB hidden channel widths must match")
    source=signal_params["perception"][:,:,:channels,:]
    if shift:
        source=jnp.roll(source,int(shift),axis=0)
    perception=rgb_params["perception"].at[:,:,:channels,channels:].set(source)
    mix=rgb_params["mix"].at[channels:,channels:].set(signal_params["mix"])
    return {**rgb_params,"perception":perception,"mix":mix}
