import tempfile
import unittest
from experiments import roadmap_70_biology as m


class BiologicalLedgerTests(unittest.TestCase):
    def test_synthetic_campaign_cannot_promote_biology(self):
        with tempfile.TemporaryDirectory() as directory:
            report=m.run(directory,seeds=(11,23))
            self.assertEqual(len(report["parts"]),8)
            self.assertEqual(report["evidence_status"],"synthetic_controls_only")
            self.assertTrue(all(c["status"]=="not_demonstrated"
                                for c in report["claims"].values()))
            self.assertTrue(all(row["source_type"]=="synthetic"
                                for row in report["parts"]))

    def test_measured_claim_without_provenance_fails(self):
        with self.assertRaises(ValueError):
            m.assert_evidence([{"source_type":"measured","provenance":None}])
        with self.assertRaises(ValueError):
            m.run(None,seeds=(11,11))


if __name__=="__main__":
    unittest.main()
