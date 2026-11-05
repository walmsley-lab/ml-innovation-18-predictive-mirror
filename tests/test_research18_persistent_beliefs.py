import unittest
from src import research18_persistent_beliefs as b
class Tests(unittest.TestCase):
    def test_bayes_cue(self):
        p=b.posterior([.5,.5],[.1,.9])
        self.assertAlmostEqual(p[1],.9)
        self.assertLess(b.neg_log_likelihood(p,1),b.neg_log_likelihood([.5,.5],1))
        self.assertEqual(b.time_to_collapse([[.5,.5],p]),1)
    def test_coverage(self):
        hit,count=b.support_coverage([.5,.5],1,.95)
        self.assertEqual((hit,count),(1,2))
        with self.assertRaises(ValueError):b.normalize([0,0])
