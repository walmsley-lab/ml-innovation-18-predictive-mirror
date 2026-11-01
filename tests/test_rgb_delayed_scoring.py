"""Prove target-frame matching and distinguish forecast from live copying."""
import unittest
import numpy as np
from src import rgb_delayed_scoring as d


class DelayedScoringTests(unittest.TestCase):
    def test_only_scores_after_target_arrives(self):
        frame=np.zeros((4,4,3),np.float32)
        future=np.ones_like(frame)
        predictions=np.stack([frame,frame,future])
        pending=d.issue([],4,1000,frame,predictions)
        pending,pairs=d.resolve(pending,6,3000,frame,lead=3)
        self.assertFalse(pairs)
        self.assertEqual(len(pending),1)
        pending,pairs=d.resolve(pending,7,4000,future,lead=3)
        self.assertFalse(pending)
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0]["issued_at"],4)
        self.assertEqual(pairs[0]["target_index"],7)
        score=d.metrics(pairs)
        self.assertEqual(score["model_mse"],0.)
        self.assertEqual(score["persistence_mse"],1.)

    def test_missing_pairs_do_not_claim_success(self):
        self.assertEqual(d.metrics([])["scored"],0)
