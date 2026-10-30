import tempfile
import unittest
from pathlib import Path

import jax

from deploy.artifacts import export, restore, safe_relative
from src.checkpoint import save, load
from src.model import init


class ArtifactsTests(unittest.TestCase):
    def test_export_verify_and_restore_from_local_url(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            remote = root / "provider" / "pilot"
            remote.mkdir(parents=True)
            params = init(jax.random.PRNGKey(7))
            zeros = jax.tree.map(lambda v: v * 0, params)
            save(remote / "checkpoint.npz", params, zeros, zeros, 5, 7, 2)
            records = export((root / "provider").as_uri(), root / "archive", ["pilot/checkpoint.npz"])
            self.assertEqual(len(records), 2)
            self.assertEqual(restore(root / "archive", root / "fresh"), 2)
            restored = load(root / "fresh" / "pilot" / "checkpoint.npz", 7, 2)
            self.assertEqual(restored[-1], 5)
            self.assertEqual(restore(root / "archive", root / "fresh"), 2)
            (root / "fresh" / "pilot" / "checkpoint.npz").write_bytes(b"tampered")
            with self.assertRaises(FileExistsError):
                restore(root / "archive", root / "fresh")
            (root / "archive" / "pilot" / "checkpoint.npz").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                restore(root / "archive", root / "another")

    def test_checkpoint_detects_tampered_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.npz"
            p = init(jax.random.PRNGKey(8))
            save(path, p, p, p, 1, 4, 2)
            (Path(folder) / "checkpoint.npz.sha256").write_text("0" * 64 + "  checkpoint.npz\n")
            with self.assertRaises(ValueError):
                load(path, 4, 2)

    def test_rejects_paths_outside_synthetic_runs(self):
        for bad in ("../secret", "/etc/passwd", "pilot/../../etc", "hidden/checkpoint.npz"):
            with self.assertRaises(ValueError):
                safe_relative(bad)


if __name__ == "__main__":
    unittest.main()
