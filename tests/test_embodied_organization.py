"""Deterministic checks for local constraints, partial observation and controls."""
import tempfile
import unittest
from pathlib import Path

from experiments import embodied_organization as lab


def item(index, width, depth, shape="round", seen=None):
    visible = seen if seen is not None else (width, depth)
    return {"id": index, "shape": shape, "true_width": width,
            "true_depth": depth, "seen_width": visible[0],
            "seen_depth": visible[1]}


class EmbodiedOrganizationTests(unittest.TestCase):
    def test_fit_is_directed_and_shape_sensitive(self):
        large = item(0, 1.0, 1.0)
        small = item(1, .5, .5)
        square = item(2, .35, .35, "square")
        self.assertTrue(lab.fits(large, small))
        self.assertFalse(lab.fits(small, large))
        self.assertFalse(lab.fits(large, square))

    def test_compositional_towers_and_oracle(self):
        items = [item(0, 1.0, 1.0), item(1, .75, .75), item(2, .45, .45)]
        true = lab.relations(items, observed=False)
        self.assertEqual(lab.minimum_towers(true), ((0, 1, 2),))
        outcome = lab.strategy(items, "oracle", budget=2, seed=11)
        self.assertEqual(outcome["tower_count"], 1)
        self.assertTrue(outcome["optimal_valid"])
        self.assertEqual(outcome["failed_joins"], 0)

    def test_noncomparable_items_require_separate_towers(self):
        items = [item(0, 1.0, .5), item(1, .5, 1.0)]
        self.assertEqual(len(lab.minimum_towers(lab.relations(items))), 2)

    def test_bad_visual_fit_fails_and_recovers_with_split(self):
        items = [item(0, .4, .4, seen=(.95, .95)),
                 item(1, .9, .9, seen=(.2, .2))]
        plan = lab.minimum_towers(lab.relations(items))
        self.assertEqual(plan, ((0, 1),))
        result = lab.execute(plan, lab.relations(items, observed=False))
        self.assertEqual(result["failed_joins"], 1)
        self.assertEqual(result["towers"], [(0,), (1,)])
        self.assertEqual(sorted(sum((list(t) for t in result["towers"]), [])), [0, 1])

    def test_zero_noise_is_perfect_observation(self):
        for seed in (1, 2, 3):
            items = lab.make_items(seed, count=6, noise=0)
            outcome = lab.strategy(items, "vision", budget=0, seed=seed)
            self.assertTrue(outcome["optimal_valid"])
            self.assertEqual(outcome["probe_count"], 0)

    def test_active_and_random_use_same_possible_probe_budget(self):
        items = [item(i,1.0-0.13*i,1.0-0.12*i) for i in range(5)]
        for method in ("active", "random"):
            result = lab.strategy(items, method, budget=4, seed=21)
            pairs = [(p["outer"],p["inner"]) for p in result["probes"]]
            self.assertEqual(len(pairs), 4)
            self.assertEqual(len(set(pairs)), 4)
            for probe in result["probes"]:
                i,j = probe["outer"],probe["inner"]
                self.assertEqual(probe["fits"],lab.fits(items[i],items[j]))

    def test_one_episode_comparison_is_paired_and_repeatable(self):
        a = lab.score(seed=11, episodes=5, count=6, budget=2)
        self.assertEqual(a, lab.score(seed=11, episodes=5, count=6, budget=2))
        for row in a["episodes"]:
            self.assertEqual(set(row["arms"]), {"vision","random","active","oracle"})
            self.assertTrue(row["arms"]["oracle"]["optimal_valid"])
            self.assertEqual(row["arms"]["active"]["oracle_towers"],
                             row["arms"]["vision"]["oracle_towers"])
            self.assertEqual(row["arms"]["active"]["probe_count"],
                             row["arms"]["random"]["probe_count"])
            for result in row["arms"].values():
                self.assertGreaterEqual(result["excess_towers"],0)

    def test_interaction_changes_planning_not_visual_inputs(self):
        items = [item(0,.45,.45,seen=(.95,.95)),
                 item(1,.8,.8,seen=(.25,.25)),
                 item(2,.9,.9,seen=(.5,.5))]
        learned = {(0,1):False}
        self.assertTrue(lab.relations(items)[0][1])
        self.assertFalse(lab.relations(items,learned)[0][1])
        self.assertTrue(lab.relations(items,observed=False)[1][0])

    def test_output_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"metrics.json"
            result = lab.run(str(target),episodes=3,count=4)
            self.assertEqual(len(result["sha256"]), 64)
            self.assertTrue(target.is_file())
            with self.assertRaises(FileExistsError):
                lab.run(str(target),episodes=3,count=4)

    def test_validation(self):
        with self.assertRaises(ValueError):
            lab.make_items(1,count=8)
        with self.assertRaises(ValueError):
            lab.score(episodes=0)
        with self.assertRaises(ValueError):
            lab.strategy([item(0,1,1),item(1,.5,.5)],"invalid",1,1)
        with self.assertRaises(ValueError):
            lab.minimum_towers(((False,True),))


if __name__ == "__main__":
    unittest.main()
