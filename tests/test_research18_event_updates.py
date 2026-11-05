import unittest
import numpy as np
from src import research18_event_updates as m
class Tests(unittest.TestCase):
    def test_all_active_equal_dense(self):
        a=np.array([[0,1,0],[1,0,1],[0,1,0]],float)
        x=np.array([1.,0.,0.])
        full,c=m.update(x,a)
        all_active,c2=m.update(x,a,np.ones(3,bool))
        np.testing.assert_allclose(full,all_active)
        self.assertEqual(c["state_reads"],c2["state_reads"])
    def test_skip_work_and_state(self):
        a=np.array([[0,1],[1,0]],float)
        x=np.array([1.,0.])
        frozen,c=m.update(x,a,[False,False])
        np.testing.assert_array_equal(frozen,x)
        self.assertEqual(c["state_reads"],0)
        self.assertEqual(m.choose_active([.2,.1],max_cells=1).sum(),1)
