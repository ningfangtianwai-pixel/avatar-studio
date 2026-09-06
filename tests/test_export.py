"""Verify release boundary using isolated synthetic files."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("export_source", Path(__file__).resolve().parents[1] / "scripts/export-source.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class SourceExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="avatar-source-test-")
        self.root = Path(self.temp.name)
        for name in ("LICENSE", "README.md", "package.json"):
            (self.root / name).write_text("{}" if name.endswith(".json") else "Synthetic", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_private_folders_and_config_excluded(self):
        for name in ("runtime/studio.db", "media/person.wav", "studio.local.json", ".env",
                     "backend/.venv/private.py", "backend/__pycache__/private.py", ".git/config"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Private synthetic data", encoding="utf-8")
        self.assertEqual(set(exporter.collect_sources(self.root)), {"LICENSE", "README.md", "package.json"})

    def test_personal_home_path_fails_without_echoing_value(self):
        (self.root / "README.md").write_text("/home/" + "private-person/example", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "personal Linux home path") as error:
            exporter.collect_sources(self.root)
        self.assertNotIn("private-person", str(error.exception))

    def test_credentials_rejected(self):
        (self.root / "README.md").write_text("ghp_" + "a" * 30, encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "credential"):
            exporter.collect_sources(self.root)

    def test_symlink_rejected(self):
        (self.root / "scripts").mkdir()
        (self.root / "scripts/linked.py").symlink_to(self.root / "README.md")
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            exporter.collect_sources(self.root)


if __name__ == "__main__":
    unittest.main()
