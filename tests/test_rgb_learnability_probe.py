"""A learnability gate has one training scene and no false test claim."""
import unittest

from experiments.rgb_learnability_probe import run


class LearnabilityTests(unittest.TestCase):
    def test_invalid_settings_rejected_before_training(self):
        with self.assertRaises(ValueError):
            run(steps=0)
        with self.assertRaises(ValueError):
            run(output="unknown")
        with self.assertRaises(ValueError):
            run(foreground_weight=-1)


if __name__ == "__main__":
    unittest.main()
