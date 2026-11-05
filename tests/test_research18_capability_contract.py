import unittest
from src import research18_capability_contract as c
class Tests(unittest.TestCase):
    def test_known_answer_and_splits(self):
        for task in c.TASKS:
            e=c.episode(task,5,demand=3)
            self.assertEqual(len(e["targets"]),32)
        e=c.episode("memory",5,demand=3)
        self.assertEqual(e["observations"][:-3],e["targets"][3:])
        self.assertNotEqual(c.split_seed("train",3),c.split_seed("test",3))
    def test_validator(self):
        r=dict(protocol=c.PROTOCOL,experiment="memory",seed=5,
               condition="grid",metrics={},costs={},provenance={})
        self.assertIs(c.validate(r),r)
        self.assertEqual(c.digest(r),c.digest(dict(r)))
        with self.assertRaises(ValueError): c.validate({"protocol":c.PROTOCOL})
