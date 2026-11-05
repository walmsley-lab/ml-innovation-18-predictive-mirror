import unittest
from src import research18_evidence_atlas as e
class Tests(unittest.TestCase):
    def item(self,**kwargs):
        x=dict(claim="C1",experiment="routing",protocol="v1",seed=1,
               evidence="simulation",metric="accuracy",unit="fraction",
               value=.8,scope="toy-scale",source_revision="abc123")
        x.update(kwargs)
        return x
    def test_unmatched_metrics_not_collapsed(self):
        out=e.atlas([self.item(),self.item(evidence="observation")])
        self.assertEqual(len(out),2)
    def test_provenance_gate(self):
        with self.assertRaises(ValueError):
            e.validate(self.item(source_revision=""))
        with self.assertRaises(ValueError):
            e.validate(self.item(evidence="unlabeled"))
