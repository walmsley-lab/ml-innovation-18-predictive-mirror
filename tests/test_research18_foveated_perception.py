import unittest
import numpy as np
from src import research18_foveated_perception as f
class Tests(unittest.TestCase):
    def test_true_high_frequency_crop(self):
        x=np.zeros((16,16))
        x[4:8,4:8]=np.indices((4,4)).sum(0)%2
        low=f.peripheral(x,4)
        self.assertEqual(low[1,1],.5)
        view=f.observe(x,(1,1),crop_size=4,downsample=4)
        self.assertGreater(np.var(view["fovea"]),0)
        self.assertEqual(view["accounting"]["source_capture_pixels"],256)
        self.assertEqual(view["accounting"]["coarse_processed"],16)
    def test_gaze_uses_only_coarse(self):
        x=np.zeros((2,4,4));x[1,2,2]=1
        self.assertEqual(f.choose_focus(x,"surprise"),(2,2))
    def test_peripheral_misses(self):
        self.assertEqual(f.peripheral_event_miss([(0,0)],[[(7,7)]],3),1)
