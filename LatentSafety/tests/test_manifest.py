from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from latent_safety.manifest import base_manifest


ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_environment_snapshot_is_path_free_and_hashed(self) -> None:
        manifest = base_manifest(
            repo_root=ROOT,
            config_path=ROOT / "configs" / "pilot" / "smoke.toml",
        )
        environment = manifest["environment"]
        installed = tuple(environment["installed_distributions"])

        self.assertTrue(installed)
        self.assertEqual(installed, tuple(sorted(set(installed), key=str.casefold)))
        self.assertTrue(all("==" in distribution for distribution in installed))
        expected_hash = hashlib.sha256("\n".join(installed).encode("utf-8")).hexdigest()
        self.assertEqual(environment["installed_distributions_sha256"], expected_hash)


if __name__ == "__main__":
    unittest.main()
