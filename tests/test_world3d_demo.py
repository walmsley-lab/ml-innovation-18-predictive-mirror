import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.world3d_storyboard import png_data_url,html
from experiments.world3d_demo import run


class World3DDemoTests(unittest.TestCase):
    def test_png_and_html_without_external_libraries(self):
        frame=np.zeros((4,4,3),np.float32)
        url=png_data_url(frame)
        self.assertTrue(url.startswith("data:image/png;base64,"))
        report=html(frame,frame[None],frame[None],frame[None])
        self.assertIn("Prediction",report.replace("Predicted","Prediction"))

    def test_smoke_requires_explicit_untrained_mode(self):
        with self.assertRaises(ValueError):
            run(out="runs/missing",active=False)
        with tempfile.TemporaryDirectory() as folder:
            result=run(out=folder,seed=11,context=3,horizon=2,size=16,
                       untrained_smoke=True,active=False)
            self.assertTrue(result["untrained_smoke"])
            self.assertTrue((Path(folder)/"storyboard.html").exists())
            self.assertTrue((Path(folder)/"metrics.json").exists())


if __name__=="__main__":
    unittest.main()
