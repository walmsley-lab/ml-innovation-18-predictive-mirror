import subprocess
import unittest
from pathlib import Path


class CampaignPackagingTests(unittest.TestCase):
    def test_container_has_camapaign_code_and_shell_is_valid(self):
        docker = Path("Dockerfile").read_text()
        shell = Path("deploy/entrypoint.sh").read_text()
        manifest = Path("deploy/akash-gpu.yaml").read_text()
        self.assertIn("campaign.py ./", docker)
        self.assertIn('CAMPAIGN_APPROVED:-0', shell)
        self.assertIn('CAMPAIGN_MAX_RUNS:-1', shell)
        self.assertIn("python -m campaign", shell)
        self.assertIn("CAMPAIGN_APPROVED=0", manifest)
        self.assertIn("RUN_MODE=train", manifest)
        subprocess.run(["sh", "-n", "deploy/entrypoint.sh"], check=True)


if __name__ == "__main__":
    unittest.main()
