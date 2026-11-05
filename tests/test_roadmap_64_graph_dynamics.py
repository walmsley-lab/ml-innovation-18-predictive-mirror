import unittest
import numpy as np
from experiments import roadmap_64_graph_dynamics as m
from experiments import roadmap_62_connectome_contract as c


class ConnectomeDynamicsTests(unittest.TestCase):
    def test_outgoing_budget_and_future_free(self):
        a,_=c.toy(11)
        b=m.null_graph(a,7)
        np.testing.assert_allclose(a.sum(1),b.sum(1))
        train=m.episodes(a,31000,4,8)
        w=m.fit(a,train)
        pred=m.predict(a,w,train[:,3],3)
        altered=train.copy();altered[:,4:]=1e4
        np.testing.assert_allclose(pred,m.predict(a,w,altered[:,3],3))
        self.assertEqual(pred.shape,(4,3,len(a)))

    def test_seeded_protocol(self):
        r=m.run(out=None,seeds=(11,),steps=9,horizon=3)
        self.assertEqual(len(r["rows"]),3)
        self.assertEqual(r["source_type"],"synthetic")
        self.assertTrue(all(np.isfinite(x["test_mse"]) for x in r["rows"]))


if __name__=="__main__":
    unittest.main()
