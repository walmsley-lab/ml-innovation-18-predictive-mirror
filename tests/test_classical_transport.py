import tempfile
import unittest
import numpy as np
from src import classical_transport as c
from experiments.classical_transport_controls import run


class ClassicalTransportTests(unittest.TestCase):
    def test_gather_and_forward_splat_do_not_wrap(self):
        a=np.zeros((12,12,3),np.float32)
        a[2,3]=[1,0,0]
        dy=np.broadcast_to((1.,2.),(12,12,2))
        moved,valid=c.backward_gather(a,dy)
        self.assertAlmostEqual(float(moved[3,5,0]),1.)
        self.assertEqual(moved[0,0,0],c.BACKGROUND)
        shifted,coverage=c.forward_splat(a,dy)
        self.assertGreater(shifted[3,5,0],.99)
        self.assertGreater(coverage[3,5],0)
        self.assertFalse(valid[0,0])

    def test_past_only_and_center_estimate(self):
        h,w=24,24
        yy,xx=np.mgrid[:h,:w]
        frames=[]
        for x in (8,9,10):
            frame=np.full((h,w,3),.035,np.float32)
            frame[(xx-x)**2+(yy-10)**2<9]=c.COLORS[0]
            frames.append(frame)
        history=np.stack(frames)
        state=c.track(history,mode="reid")[0]
        self.assertAlmostEqual(state["velocity"][1],1.,delta=.3)
        out=c.forecast(history,3,"coast")
        self.assertEqual(out.shape,(3,h,w,3))
        # Modifying a future scene cannot change a past-only forecast.
        np.testing.assert_array_equal(out,c.forecast(history.copy(),3,"coast"))

    def test_batch_smoke(self):
        with tempfile.TemporaryDirectory() as root:
            result=run(root+"/metrics.json",seeds=(11,12),context=4,
                       horizon=2,size=32)
            self.assertEqual(set(result["modes"]),set(("persistence",
                "pixel_velocity","flow_gather","coast","reid")))


if __name__=="__main__":
    unittest.main()
