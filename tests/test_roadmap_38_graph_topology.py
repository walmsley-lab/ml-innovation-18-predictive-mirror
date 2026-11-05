import tempfile
import unittest
import numpy as np
from experiments import roadmap_38_graph_topology as m
from src import graph_skeleton_nca as nca, communication_graph_harmonics as h


class GraphTopologyTests(unittest.TestCase):
    def test_equal_degree_and_network_shape(self):
        actual=h.adjacency_from_edges(len(nca.JOINTS),nca.EDGES)
        options=m.graphs(13)
        self.assertEqual(set(options),{"skeleton","rewired","no_messages"})
        for name in ("skeleton","rewired"):
            adjacency,_=options[name]
            self.assertEqual(adjacency.shape,actual.shape)
            np.testing.assert_allclose(np.asarray(adjacency).sum(1),
                                       np.minimum(actual.sum(1),1))

    def test_training_consumes_visible_joints_only(self):
        from experiments.graph_skeleton_nca import examples
        from experiments.graph_skeleton_horizons import rollout
        import jax
        import jax.numpy as jnp
        obs,mask,truth=examples([91000],steps=12)
        params=nca.init(jax.random.PRNGKey(1),2)
        graph=m.graphs()[ "skeleton"][0]
        predicted=rollout(params,jnp.asarray(obs[:,:8]),jnp.asarray(mask[:,:8]),
                          5,graph)
        altered=obs.copy();altered[:,8:]=1.
        compare=rollout(params,jnp.asarray(altered[:,:8]),jnp.asarray(mask[:,:8]),
                        5,graph)
        np.testing.assert_array_equal(np.asarray(predicted),np.asarray(compare))

    def test_one_update_three_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            r=m.run(folder,seeds=(3,),steps=1,batch=1,heldout=2,channels=2)
            self.assertEqual(len(r["rows"]),3)
            self.assertIn("velocity",r["rows"][0]["scores"])


if __name__=="__main__":
    unittest.main()
