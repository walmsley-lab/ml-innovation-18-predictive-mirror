"""Optional PyTorch companion to src.predictive_rgb_nca (same RGB NCA rule).

Public inputs/outputs are NHWC [batch, time, height, width, 3]. Internal
PyTorch convolutions use NCHW. This is a comparison, not a second trainer.
"""

import numpy as np
import torch
from torch import nn


class RGBNCA(nn.Module):
    def __init__(self, channels=12):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.perception = nn.Conv2d(channels + 6, channels * 2, 3, padding=1, bias=False)
        self.mix = nn.Conv2d(channels * 2, channels * 2, 1)
        self.readout = nn.Conv2d(channels, 3, 1)
        with torch.no_grad():
            nn.init.normal_(self.perception.weight, std=0.05)
            nn.init.normal_(self.mix.weight, std=0.05)
            nn.init.zeros_(self.mix.bias)
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

    def load_jax_params(self, params):
        """Copy corresponding parameters; avoids random-initialization confounds."""
        with torch.no_grad():
            self.perception.weight.copy_(torch.as_tensor(
                np.array(params["perception"]).transpose(3, 2, 0, 1).copy()))
            self.mix.weight.copy_(torch.as_tensor(
                np.array(params["mix"]).T[:, :, None, None].copy()))
            self.mix.bias.copy_(torch.as_tensor(np.array(params["cell_bias"])))
            self.readout.weight.copy_(torch.as_tensor(
                np.array(params["rgb_readout"]).T[:, :, None, None].copy()))
            self.readout.bias.copy_(torch.as_tensor(np.array(params["rgb_bias"])))
        return self

    def cell_update(self, hidden, image, surprise):
        perceived = torch.tanh(self.perception(torch.cat((hidden, image, surprise), dim=1)))
        gate, proposal = self.mix(perceived).chunk(2, dim=1)
        return hidden + 0.25 * torch.sigmoid(gate) * torch.tanh(proposal)

    def observe(self, state, observed, ticks):
        hidden, previous_forecast = state
        surprise = observed - previous_forecast
        for _ in range(ticks):
            hidden = self.cell_update(hidden, observed, surprise)
        return hidden, observed

    def dream(self, state, ticks):
        hidden, image = state
        quiet = torch.zeros_like(image)
        for _ in range(ticks):
            hidden = self.cell_update(hidden, image, quiet)
        future = torch.clamp(image + 0.20 * torch.tanh(self.readout(hidden)), 0.0, 1.0)
        return (hidden, future), future

    def forward(self, observed_context, horizon, ticks=2, return_activity=False):
        if (observed_context.ndim != 5 or observed_context.shape[-1] != 3
                or observed_context.shape[1] < 2 or horizon < 1 or ticks < 1):
            raise ValueError("Expected [B,C,H,W,3], C>=2, positive horizon/ticks")
        observed = observed_context.permute(0, 1, 4, 2, 3)
        batch, _, _, height, width = observed.shape
        hidden = observed.new_zeros((batch, self.channels, height, width))
        state = (hidden, torch.zeros_like(observed[:, 0]))
        for index in range(observed.shape[1] - 1):
            state = self.observe(state, observed[:, index], ticks)
            state, _ = self.dream(state, ticks)
        state = self.observe(state, observed[:, -1], ticks)
        predictions, activities = [], []
        for _ in range(horizon):
            state, frame = self.dream(state, ticks)
            predictions.append(frame)
            if return_activity:
                activities.append(state[0].abs().mean(dim=1, keepdim=True))
        rgb = torch.stack(predictions, 1).permute(0, 1, 3, 4, 2)
        if return_activity:
            activity = torch.stack(activities, 1).permute(0, 1, 3, 4, 2)
            return rgb, activity
        return rgb

    def loss(self, examples, context=6, horizon=5, ticks=2):
        forecasts = self(examples[:, :context], horizon, ticks)
        target = examples[:, context:context + horizon]
        motion = (target - examples[:, context - 1:context]).abs().mean(dim=-1, keepdim=True)
        return ((1.0 + 8.0 * motion) * (forecasts - target).square()).mean()
