"""Model seeds and opposing scene histories are not independent pixels."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from src.paired_statistics import paired_effect,pilot_precision
from experiments.paired_effect_report import run


class PairedStatisticsTests(unittest.TestCase):
    def test_deterministic_cluster_effect_with_fixed_baseline(self):
        a=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]],float)
        b=np.zeros(4)
        one=paired_effect(a,b,draws=300,practical_margin=.10,seed=11)
        two=paired_effect(a,b,draws=300,practical_margin=.10,seed=11)
        self.assertEqual(one,two)
        self.assertEqual(one["absolute_effect"],1.)
        self.assertEqual(one["ci_95"],[1.,1.])
        self.assertEqual(one["decision"],"supports_practical_advantage")
        self.assertEqual(one["n_independent_scene_pairs"],4)

    def test_undecided_effect_has_no_fake_significance(self):
        a=np.array([[0,1,0,1],[1,0,1,0],[0,1,1,0]],float)
        b=np.array([0,1,0,1],float)
        r=paired_effect(a,b,draws=300,practical_margin=.1)
        self.assertIn(r["decision"],("inconclusive","no_supported_practical_advantage"))
        self.assertLessEqual(r["ci_95"][0],r["absolute_effect"])
        self.assertGreaterEqual(r["ci_95"][1],r["absolute_effect"])

    def test_reject_pixel_pseudoreplication_and_missing(self):
        with self.assertRaisesRegex(ValueError, ">=2"):
            paired_effect([1,0,1],[0,0,0])
        with self.assertRaisesRegex(ValueError, "Missing"):
            paired_effect([[1,1],[0,float("nan")]],[0,0])
        with self.assertRaisesRegex(ValueError, "align"):
            paired_effect([[1,1],[1,0]],[0,0,0])

    def test_pilot_widths_have_honest_label(self):
        result=pilot_precision([[1,0,1],[0,1,1]],[0,0,0],draws=120)
        self.assertEqual(len(result["scenarios"]),9)
        self.assertIn("not power",result["warning"])

    def test_runner_checks_pair_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source=root/"scores.json"
            values={"candidate":"nca","baseline":"kalman",
                    "model_seeds":[11,23],"scene_pair_ids":[101,102,103],
                    "candidate_scores":[[1,0,1],[1,1,0]],
                    "baseline_scores":[0,0,1]}
            source.write_text(json.dumps(values))
            r=run(source,root/"report.json",draws=150)
            self.assertEqual(r["effect"]["n_model_seeds"],2)
            values["scene_pair_ids"]=[101,101,103]
            source.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError,"unique"):
                run(source,root/"report.json",draws=150)


if __name__=="__main__":
    unittest.main()
