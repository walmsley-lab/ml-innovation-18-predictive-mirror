import unittest
from experiments import roadmap_47_budgeted_attention as m
from src import rgb_world_ladder as worlds


class BudgetedAttentionTests(unittest.TestCase):
    def test_budgeted_arms_and_unbudgeted_control(self):
        x=worlds.episode(91000,8,24,domain="linear")["frames"]
        r=m.one(x)
        self.assertEqual(set(r),set(m.MODES))
        self.assertLess(r["surprise"]["unique_updated_pixels"],
                        r["full_velocity"]["unique_updated_pixels"])
        self.assertEqual(r["persistence"]["unique_updated_pixels"],0)
        self.assertTrue(all(v["mse"]>=0 for v in r.values()))

    def test_forecast_uses_only_prior_selection(self):
        from experiments import roadmap_46_attention as attention
        x=worlds.episode(91000,8,24)["frames"]
        origins=attention.choose(x[:5],"surprise")
        y=x.copy();y[5:]=0
        self.assertEqual(origins,attention.choose(y[:5],"surprise"))

    def test_domains(self):
        self.assertEqual(len(m.run(None,seeds=(91000,),domains=("linear",))["rows"]),1)


if __name__=="__main__":
    unittest.main()
