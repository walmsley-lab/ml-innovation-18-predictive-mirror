import tempfile
import unittest
import numpy as np
from src import guaranteed_reappearance as world
from src import reappearance_scoring as events
from experiments.guaranteed_reappearance import run


class ReappearanceTests(unittest.TestCase):
    def test_guarantees_barrier_partial_and_return(self):
        rec=world.episode(11)
        labels=rec["visibility_reason"]
        self.assertIn("partial",labels)
        self.assertIn("barrier",labels)
        self.assertEqual(rec["events"][0]["kind"],"barrier_return")
        self.assertTrue(4<=rec["events"][0]["frame"]<14)
        self.assertEqual(set(world.model_inputs(rec)),{"frames"})

    def test_perfect_oracle_scores_time_location_and_mass(self):
        rec=world.episode(11)
        pred=rec["frames"][4:][None]
        result=events.score(pred,[rec],4)
        self.assertEqual(result["barrier_events"],1)
        event=result["results"][0]["events"][0]
        self.assertEqual(event["timing_error_frames"],0)
        self.assertEqual(event["missed_reappearance"],False)
        self.assertAlmostEqual(event["reappearance_centroid_error_px"],0)
        self.assertGreater(event["reappearance_mass_ratio"],0)
        self.assertLessEqual(event["reappearance_mass_ratio"],1.05)
        self.assertEqual(event["premature_visible_frames"],0)

    def test_broken_forecasts_cannot_fake_object_permanence(self):
        rec=world.episode(11)
        dark=np.zeros_like(rec["frames"][4:])[None]
        scored=events.score(dark,[rec],4)["results"][0]["events"][0]
        self.assertIsNone(scored["timing_error_frames"])
        self.assertIsNone(scored["reappearance_centroid_error_px"])
        self.assertTrue(scored["missed_reappearance"])
        shifted=np.roll(rec["frames"][4:][None],5,axis=3)
        event=events.score(shifted,[rec],4)["results"][0]["events"][0]
        self.assertGreater(event["reappearance_centroid_error_px"],1)

    def test_camera_exit_not_barrier(self):
        rec=world.episode(13,mode="camera_exit")
        result=events.score(rec["frames"][4:][None],[rec],4)
        self.assertEqual(result["barrier_events"],0)
        self.assertEqual(result["camera_exit_events"],1)

    def test_cli_oracle_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            result=run(directory+"/metrics.json",seeds=(11,),mode="barrier")
            self.assertEqual(result["barrier_events"],1)


if __name__=="__main__":
    unittest.main()
