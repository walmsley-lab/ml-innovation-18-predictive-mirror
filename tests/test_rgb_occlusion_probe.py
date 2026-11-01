"""Foreground occlusion and reappearance must be scored honestly."""
import unittest
import numpy as np
from src import rgb_motion_world as world
from experiments.rgb_occlusion_probe import reappearance_count


class OcclusionProbeTests(unittest.TestCase):
    def test_visibility_after_foreground_mask(self):
        size=20
        obj={"position":np.asarray((10.,10.),np.float32),
             "radius":1,"color":np.asarray((1.,0.,0.),np.float32)}
        frame,visible=world._render([obj],np.zeros(2,np.float32),
                                    size,size,occlusion=True)
        self.assertFalse(bool(visible[0]))
        self.assertFalse(np.any(frame[:,:,0]>.5))
        _,clear=world._render([obj],np.zeros(2,np.float32),
                              size,size,occlusion=False)
        self.assertTrue(bool(clear[0]))

    def test_reappearance_event_is_only_posthoc(self):
        records=[{"visible":np.asarray([[1],[0],[1],[1]],bool)},
                 {"visible":np.asarray([[1],[1],[1],[1]],bool)}]
        self.assertEqual(reappearance_count(records,2,2),1)
