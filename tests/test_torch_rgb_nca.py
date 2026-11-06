"""Optional cross-framework equivalence checks; no PyTorch needed for default suite."""
import unittest
from importlib.util import find_spec

import jax
import jax.numpy as jnp
import numpy as np

if find_spec("torch") is not None:
    import torch
    from experiments.torch_rgb_nca import RGBNCA
else:
    torch = None
    RGBNCA = None

from src import predictive_rgb_nca as jax_nca


@unittest.skipIf(torch is None, "optional PyTorch not installed")
class TorchRGBNCATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        params = jax_nca.init(jax.random.PRNGKey(9), channels=4)
        # Nonzero readout exercises dream dynamics beyond the identity baseline.
        self.params = dict(params, rgb_readout=jnp.ones((4, 3), dtype=jnp.float32) * 0.013)
        self.torch_model = RGBNCA(4).load_jax_params(self.params)
        self.examples = np.random.default_rng(12).random((1, 7, 8, 8, 3), dtype=np.float32)

    def test_forward_parity_and_activity(self):
        expected, activity = jax_nca.rollout_and_activity(
            self.params, jnp.asarray(self.examples[:, :4]), 3, ticks=1)
        actual, torch_activity = self.torch_model(
            torch.from_numpy(self.examples[:, :4]), 3, ticks=1, return_activity=True)
        np.testing.assert_allclose(actual.detach().numpy(), np.asarray(expected), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(torch_activity.detach().numpy(), np.asarray(activity), rtol=1e-4, atol=1e-5)

    def test_loss_and_gradient_parity(self):
        jax_value, jax_grad = jax.value_and_grad(jax_nca.loss)(
            self.params, jnp.asarray(self.examples), 4, 3, 1)
        torch_value = self.torch_model.loss(torch.from_numpy(self.examples), 4, 3, 1)
        torch_value.backward()
        self.assertAlmostEqual(float(jax_value), torch_value.item(), delta=1e-5)
        gradients = {
            "perception": self.torch_model.perception.weight.grad.permute(2, 3, 1, 0),
            "mix": self.torch_model.mix.weight.grad.squeeze().T,
            "cell_bias": self.torch_model.mix.bias.grad,
            "rgb_readout": self.torch_model.readout.weight.grad.squeeze().T,
            "rgb_bias": self.torch_model.readout.bias.grad,
        }
        for name, gradient in gradients.items():
            np.testing.assert_allclose(gradient.detach().numpy(), np.asarray(jax_grad[name]),
                                       rtol=1e-3, atol=1e-5, err_msg=name)

    def test_initialization_is_persistence_and_future_cannot_leak(self):
        model = RGBNCA(4)
        x = torch.from_numpy(self.examples[:, :4])
        forecast = model(x, 3, ticks=1)
        np.testing.assert_allclose(forecast.detach().numpy(),
                                   np.repeat(self.examples[:, 3:4], 3, axis=1))
        changed = self.examples.copy()
        changed[:, 4:] = 0
        np.testing.assert_allclose(
            model(torch.from_numpy(self.examples[:, :4]), 3, 1).detach().numpy(),
            model(torch.from_numpy(changed[:, :4]), 3, 1).detach().numpy())
        self.assertNotAlmostEqual(
            model.loss(torch.from_numpy(self.examples), 4, 3, 1).item(),
            model.loss(torch.from_numpy(changed), 4, 3, 1).item(), places=5)


if __name__ == "__main__":
    unittest.main()
