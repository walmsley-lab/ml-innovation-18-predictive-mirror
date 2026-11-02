"""Keep the three newcomer documents navigable as experiments accumulate."""
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDES = ("README.md", "docs/EXPERIMENTS.md", "docs/GETTING_STARTED.md")


class DocumentationNavigationTests(unittest.TestCase):
    def test_newcomer_markdown_links_resolve(self):
        pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        for name in GUIDES:
            source = ROOT / name
            with self.subTest(guide=name):
                self.assertTrue(source.is_file())
            for link in pattern.findall(source.read_text()):
                if "://" in link or link.startswith("#"):
                    continue
                target = (source.parent / link.split("#")[0]).resolve()
                with self.subTest(guide=name, link=link):
                    self.assertTrue(target.is_file(), f"{name} links to {link}")

    def test_demo_commands_point_to_existing_modules(self):
        for name in ("experiments/rgb_curriculum.py",
                     "experiments/rgb_prediction_gallery.py",
                     "experiments/rgb_benchmark.py",
                     "experiments/rgb_nca.py",
                     "experiments/rgb_verified_live.py"):
            with self.subTest(path=name):
                self.assertTrue((ROOT / name).is_file())

    def test_readme_is_an_entry_point_not_old_merge_instructions(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("docs/EXPERIMENTS.md", readme)
        self.assertIn("docs/GETTING_STARTED.md", readme)
        self.assertNotIn("Stacked implementation PRs", readme)


if __name__ == "__main__":
    unittest.main()
