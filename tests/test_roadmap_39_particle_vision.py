import tempfile
import unittest
import numpy as np
from experiments import roadmap_39_particle_vision as m


class ParticleVisionTests(unittest.TestCase):
    def test_observation_and_occlusion(self):
        r=np.stack([[[.3,.3],[.7,.7]],[[.34,.3],[.67,.7]],[[.38,.3],[.64,.7]]])
        missing=np.zeros((3,2),bool);missing[1,0]=True
        f=m.render(r,24,missing)
        positions,visible=m.observe(f,2)
        self.assertFalse(visible[1,0])
        np.testing.assert_allclose(positions[1,0],positions[0,0])
        self.assertTrue(visible[2,0])

    def test_future_suffix_cannot_change_observed_prefix(self):
        from src import particle_nca as nca
        import jax
        import jax.numpy as jnp
        a,_,_=m.pools([91000],4,2,24,2,True)
        params=nca.init(jax.random.PRNGKey(1),2)
        x=np.asarray(nca.rollout(params,jnp.asarray(a[:,:4]),2))
        altered=a.copy();altered[:,4:]=0.
        y=np.asarray(nca.rollout(params,jnp.asarray(altered[:,:4]),2))
        np.testing.assert_array_equal(x,y)

    def test_one_step_four_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            r=m.run(folder,seeds=(3,),steps=1,batch=1,heldout=2,
                    context=3,horizon=2,size=20,channels=2)
            self.assertEqual(len(r["rows"]),4)
            self.assertEqual(set(x["input"] for x in r["rows"]),
                             {"rgb_centroids","privileged_positions"})


if __name__=="__main__":
    unittest.main()
