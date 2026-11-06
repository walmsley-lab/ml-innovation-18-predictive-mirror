"""Explicit inference dispatch for frozen RGB-only checkpoints.

No silent fallback to NCA: wrong model/checkpoint combinations are errors.
All candidates predict from observed history only.
"""
from src import predictive_rgb_nca as nca
from src import rgb_grounded_nca as grounded
from src import tiny_cnn_video as cnn
from src import rgb_motion_transport as transport

MODELS = ("nca", "nca_blend", "cnn", "transport")
REQUIRED = {
    "nca": {"perception", "mix", "rgb_readout", "rgb_bias"},
    "nca_blend": {"perception", "mix", "rgb_readout", "rgb_bias",
                  "gate_readout", "gate_bias"},
    "cnn": {"conv1", "conv2", "readout", "bias"},
    "transport": {"perception", "router", "router_bias", "residual", "residual_bias"},
}


def validate(params, model):
    if model not in MODELS:
        raise ValueError("Unsupported RGB model: " + str(model))
    present = set(params)
    missing = REQUIRED[model] - present
    if missing:
        raise ValueError(f"{model} checkpoint missing: {sorted(missing)}")
    if model == "nca" and {"gate_readout", "gate_bias"} & present:
        raise ValueError("Blend-head checkpoint requires --model nca_blend")


def rollout(params, context, horizon=10, ticks=2, model="nca"):
    validate(params, model)
    if model == "nca":
        return nca.rollout(params, context, horizon, ticks)
    if model == "nca_blend":
        return grounded.rollout(params, context, horizon, ticks, "blend")
    if model == "cnn":
        return cnn.rollout(params, context, horizon)
    return transport.rollout(params, context, horizon, "combined")
