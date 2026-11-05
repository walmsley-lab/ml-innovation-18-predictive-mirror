import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from experiments import roadmap_45_release as m


class ReleaseTests(unittest.TestCase):
    def test_deterministic_and_private_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            entries=[]
            for i,role in enumerate(m.REQUIRED):
                path=root/f"{role}.txt"
                path.write_text(f"{role} artifact")
                entries.append({"role":role,"path":str(path),"capture_session":role+"-id"})
            config=root/"input.json"
            config.write_text(json.dumps({"git_sha":"a"*40,"sources":entries}))
            first=m.pack(config,root/"one.zip")
            second=m.pack(config,root/"two.zip")
            self.assertEqual(first["sha256"],second["sha256"])
            with zipfile.ZipFile(root/"one.zip") as z:
                names=z.namelist()
                self.assertEqual(len(names),3)
                self.assertTrue(any("metrics" in name for name in names))
                self.assertFalse(any("train.txt" in name for name in names))

    def test_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            file=root/"same";file.write_text("one")
            entries=[{"role":role,"path":str(file)} for role in m.REQUIRED]
            with self.assertRaises(ValueError):
                m.evidence({"git_sha":"a"*40,"sources":entries})


if __name__=="__main__":
    unittest.main()
