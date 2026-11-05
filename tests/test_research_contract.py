"""Contract checks prevent silent protocol drift and invalid claims."""
import json
import tempfile
import unittest
from pathlib import Path

from src.research_contract import load, validate


class ContractTests(unittest.TestCase):
    def test_current_contract_is_valid_and_hashed(self):
        contract, digest = load("experiments/research_contract_v1.json")
        self.assertEqual(contract["status"], "proposed")
        self.assertEqual(len(digest), 64)
        self.assertEqual(len(contract["hypotheses"]), 4)

    def test_deny_truth_as_input_and_ambiguous_units(self):
        data, _ = load("experiments/research_contract_v1.json")
        data["task"]["input"] = "frames plus simulator positions"
        with self.assertRaisesRegex(ValueError, "past RGB"):
            validate(data)
        data["task"]["input"] = "past RGB only"
        data["task"]["statistical_unit"] = "each pixel"
        with self.assertRaisesRegex(ValueError, "statistical unit"):
            validate(data)

    def test_reject_duplicate_split_and_post_hoc_comparator(self):
        data, _ = load("experiments/research_contract_v1.json")
        data["split_seed_roots"]["test"] = data["split_seed_roots"]["train"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate(data)
        data["split_seed_roots"]["test"] = 71000
        data["primary_comparator_rule"] = "choose on final test"
        with self.assertRaisesRegex(ValueError, "Comparator"):
            validate(data)

    def test_content_hash_changes_with_amendment(self):
        data, digest = load("experiments/research_contract_v1.json")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "contract.json"
            data["status"] = "amended"
            p.write_text(json.dumps(data))
            _, changed = load(p)
            self.assertNotEqual(digest, changed)


if __name__ == "__main__":
    unittest.main()
