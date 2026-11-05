import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from src import field_predictors as field
from experiments.physical_field_prediction import run


class PhysicalFieldPredictionTests(unittest.TestCase):
    def test_wave_full_state_and_oracle(self):
        contexts,truth,records=field.dataset((11,12),"wave",3,3,20)
        self.assertEqual(contexts.shape,(2,3,20,20,2))
        np.testing.assert_allclose(field.numerical(records,3,3),truth,atol=2e-7)

    def test_stencil_and_neural_shapes(self):
        contexts,truth,_=field.dataset(range(11,16),"diffusion",3,2,20)
        weights=field.fit_stencil(contexts[:,-2],contexts[:,-1])
        self.assertEqual(weights.shape,(5,2))
        prediction=field.stencil_rollout(weights,contexts[:,-1],2,"diffusion")
        self.assertEqual(prediction.shape,truth.shape)
        for arm in field.ARMS:
            params=field.init(jax.random.PRNGKey(1),arm,4)
            pred=field.rollout(params,jnp.asarray(contexts[:,-1]),2,arm,"diffusion")
            self.assertEqual(pred.shape,truth.shape)
            self.assertTrue(np.isfinite(np.asarray(pred)).all())

    def test_short_training(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run(folder,"wave",seeds=(1,),steps=1,context=3,
                       horizon=2,size=16,channels=4,batch_size=2)
            self.assertEqual(len(result["results"]["learned"]),2)


if __name__=="__main__":
    unittest.main()
