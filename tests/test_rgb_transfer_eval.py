"""Verify replay boundary, chronological holdout and training-leakage guard."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.rgb_transfer_eval import replay_windows,sha,evaluate


class RGBTransferTests(unittest.TestCase):
    def test_final_fifth_has_valid_future(self):
        frames=np.arange(600*4,dtype=np.float32).reshape(600,2,2,1)
        clips,anchors=replay_windows(frames,context=8,horizon=10,
                                     stride=2,cap=120)
        self.assertEqual(anchors[0],487)
        self.assertEqual(anchors[-1],589)
        self.assertEqual(clips.shape,(52,18,2,2,1))
        np.testing.assert_array_equal(clips[-1,-1],frames[599])

    def test_reject_short_recording(self):
        with self.assertRaisesRegex(ValueError,"too short"):
            replay_windows(np.zeros((20,4,4,3)),context=8,horizon=10)

    def test_known_training_hash_blocks_eval_before_loading_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/"same.npz"
            np.savez(file,frames=np.zeros((200,16,16,3),np.float32))
            checkpoint=Path(folder)/"checkpoint.npz"
            np.savez(checkpoint,irrelevant=np.array([0]))
            with self.assertRaisesRegex(ValueError,"Training recording"):
                evaluate(checkpoint,[file],excluded_training_hashes=(sha(file),))
