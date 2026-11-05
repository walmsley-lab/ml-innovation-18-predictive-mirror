import unittest
from experiments.rgb_staged_learning import schedule


class StageTests(unittest.TestCase):
    def test_increasing_horizons(self):
        self.assertEqual([x["horizon"] for x in schedule((1,3,10),2)],[1,3,10])
        with self.assertRaises(ValueError):
            schedule((1,3,3))
        with self.assertRaises(ValueError):
            schedule((3,1))
        with self.assertRaises(ValueError):
            schedule((1,3),0)


if __name__=="__main__":
    unittest.main()
