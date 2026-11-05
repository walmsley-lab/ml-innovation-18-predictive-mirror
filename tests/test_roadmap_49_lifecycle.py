import unittest
from experiments import roadmap_49_lifecycle as m


class LifecycleTests(unittest.TestCase):
    def test_birth_missing_reappearance_and_retirement(self):
        x=m.update({}, {0:None},0,2)
        self.assertEqual(x[0]["state"],"visible")
        x=m.update(x,{},1,2)
        self.assertEqual(x[0]["state"],"uncertain")
        x=m.update(x,{},3,2)
        self.assertEqual(x[0]["state"],"retired")
        x=m.update(x,{0:None},4,2)
        self.assertEqual(x[0]["state"],"reappeared")

    def test_future_observations_cannot_change_prefix(self):
        frames,_,_=m.episode(11)
        self.assertEqual(m.trajectory(frames)[:8],m.trajectory(frames[:8]))

    def test_retirement_not_confused_with_truth(self):
        r=m.run(None,seeds=(11,))
        self.assertEqual(len(r["rows"]),3)
        self.assertTrue(all(x["false_retire_during_occlusion"]<=
                            x["occluded_alive_events"] for x in r["rows"]))


if __name__=="__main__":
    unittest.main()
