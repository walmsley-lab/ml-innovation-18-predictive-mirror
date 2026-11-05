import unittest
import numpy as np
from experiments import roadmap_51_irregular_memory as m
from experiments import roadmap_49_lifecycle as lifecycle


class IrregularMemoryTests(unittest.TestCase):
    def test_timestamp_causality_and_validation(self):
        frames,_,_=lifecycle.episode(11)
        stamps=m.timestamps(11,len(frames),"burst")
        self.assertTrue(np.all(np.diff(stamps)>0))
        np.testing.assert_array_equal(
            list(m.infer(frames,stamps)[:8]),
            list(m.infer(frames[:8],stamps[:8])))
        with self.assertRaises(ValueError):
            m.infer(frames,stamps[::-1])

    def test_report_and_ablation(self):
        r=m.run(None,seeds=(11,))
        self.assertEqual(len(r["rows"]),6)
        self.assertEqual(set(x["timing"] for x in r["rows"]),
                         {"regular","jitter","burst"})


if __name__=="__main__":
    unittest.main()
