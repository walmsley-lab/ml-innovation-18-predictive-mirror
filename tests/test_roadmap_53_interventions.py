import unittest
from experiments import roadmap_53_interventions as m
from src import rgb_world_ladder as world


class InterventionTests(unittest.TestCase):
    def test_matched_window_count_and_output_control(self):
        rgb=world.episode(91000,8,24,domain="linear")["frames"]
        s=m.evaluate(rgb)
        self.assertEqual(s["high_priority_cut"]["window_count"],
                         s["matched_random_window_cut"]["window_count"])
        self.assertEqual(s["intact"]["causal_output_delta"],0.)
        self.assertTrue(all(x["selected_pixels"]>=0 for x in s.values()))

    def test_seeded_heldout(self):
        r=m.run(None,seeds=(91000,),domains=("linear",))
        self.assertEqual(len(r["rows"]),1)
        self.assertIn("no_attention",r["rows"][0]["interventions"])


if __name__=="__main__":
    unittest.main()
