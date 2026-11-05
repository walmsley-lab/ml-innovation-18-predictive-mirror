import tempfile
import unittest
import numpy as np
from experiments import roadmap_37_oscillatory as m


class OscillatoryVideoTests(unittest.TestCase):
    def test_world_motion_and_fixed_splits(self):
        a=m.episode(11,7,20,"periodic")
        b=m.episode(11,7,20,"periodic")
        np.testing.assert_array_equal(a,b)
        self.assertGreater(np.abs(a[1]-a[0]).sum(),0)
        self.assertFalse(np.array_equal(a,m.episode(11,7,20,"chirp")))

    def test_forecast_does_not_see_future(self):
        import jax
        import jax.numpy as jnp
        from src import predictive_rgb_nca as nca, rgb_phase_nca as phase
        x=m.pool(91000,1,4,2,20,"periodic")
        p=nca.init(jax.random.PRNGKey(1),2)
        for mode in m.MODES:
            a=np.asarray(phase.rollout(p,jnp.asarray(x[:,:4]),2,1,mode))
            altered=x.copy();altered[:,4:]=1.
            np.testing.assert_array_equal(a,np.asarray(
                phase.rollout(p,jnp.asarray(altered[:,:4]),2,1,mode)))

    def test_paired_one_step_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            r=m.run(d,seeds=(3,),steps=1,context=3,horizon=2,
                    size=16,width=2,ticks=1,batch=1)
            self.assertEqual(len(r["results"]),3)
            self.assertEqual(set(x["mode"] for x in r["results"]),set(m.MODES))
            self.assertTrue(all(x["best_step"]<=1 for x in r["results"]))


if __name__=="__main__":
    unittest.main()
