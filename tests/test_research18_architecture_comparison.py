import unittest
import numpy as np
from src import research18_architecture_comparison as e
class Tests(unittest.TestCase):
    def test_static_persistence(self):
        f=np.full((8,4,4),.2)
        a=e.score(e.persistence,f,5,3)
        self.assertEqual(a["mse"],0)
    def test_no_leak(self):
        frames=np.arange(8*2*2).reshape(8,2,2)/100.
        captured=[]
        def model(past,horizon):
            captured.append(len(past))
            return e.persistence(past,horizon)
        e.score(model,frames)
        self.assertEqual(captured,[5])
    def test_competitors(self):
        x=np.zeros((8,3,3))
        out=e.paired_compare({"hold":e.persistence,"move":e.constant_velocity},[x])
        self.assertEqual(set(out),{"hold","move"})
