import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import cellular_signal as signal,predictive_rgb_nca as rgb
from src.transmission_to_rgb import transfer
from experiments.transmission_rgb_transfer import run


class TransferTests(unittest.TestCase):
    def test_explicit_partial_mapping_and_shifted_null(self):
        comm=signal.init(jax.random.PRNGKey(2),channels=4)
        base=rgb.init(jax.random.PRNGKey(3),channels=4)
        adapted=transfer(base,comm)
        np.testing.assert_array_equal(
            np.asarray(adapted["perception"][:,:,:4,4:]),
            np.asarray(comm["perception"][:,:,:4,:]))
        np.testing.assert_array_equal(np.asarray(adapted["rgb_readout"]),
                                      np.asarray(base["rgb_readout"]))
        shuffled=transfer(base,comm,shift=1)
        self.assertFalse(np.array_equal(np.asarray(shuffled["perception"]),
                                        np.asarray(adapted["perception"])))

    def test_mismatched_width_is_rejected(self):
        comm=signal.init(jax.random.PRNGKey(2),channels=4)
        base=rgb.init(jax.random.PRNGKey(3),channels=8)
        with self.assertRaises(ValueError):
            transfer(base,comm)

    def test_short_train_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"signal.npz"
            comm=signal.init(jax.random.PRNGKey(2),channels=4)
            np.savez_compressed(source,**jax.tree.map(np.asarray,comm))
            result=run(source,out=Path(directory)/"rgb",seeds=(3,),
                       steps=1,context=3,horizon=2,size=16,batch_size=2)
            self.assertEqual(len(result["results"]),1)
            self.assertEqual(set(result["results"][0]["arms"]),
                             {"scratch","transmission","spatially_shifted_transfer"})
            measurement=result["results"][0]["measurement"]
            self.assertIn("clip_rail_saturation",measurement)
            self.assertIn("scratch_readout_x3",
                          measurement["positive_control_response"])
            # The init effect under test is reported alongside the positive
            # control so a reader can compare the two magnitudes directly.
            self.assertEqual(set(measurement["init_effect_on_forecast"]),
                             {"transmission","spatially_shifted_transfer"})


if __name__=="__main__":
    unittest.main()
