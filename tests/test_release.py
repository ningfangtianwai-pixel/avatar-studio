"""Check public product naming, attribution and contact information."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_frontend_identity(self):
        layout = (ROOT / "app/layout.tsx").read_text(encoding="utf-8")
        ui = (ROOT / "components/studio-app.tsx").read_text(encoding="utf-8")
        self.assertIn("title: '数字人口播工作台'", layout)
        self.assertIn('>Avatar Studio</p>', ui)
        self.assertIn('mailto:ningfangtianwai@gmail.com', ui)
        self.assertIn('本系统由 Manny 与 Codex 协作完成', ui)

    def test_package_and_service_identity(self):
        package = json.loads((ROOT / "package.json").read_text())
        lock = json.loads((ROOT / "package-lock.json").read_text())
        self.assertEqual(package["name"], "avatar-studio")
        self.assertEqual(package["author"], "Manny")
        self.assertEqual(package["license"], "MIT")
        self.assertEqual(package["version"], lock["version"])
        self.assertEqual(package["name"], lock["packages"][""]["name"])
        keepalive = (ROOT / "scripts/keepalive.sh").read_text()
        self.assertIn('SERVICE_NAME="avatar-studio.service"', keepalive)

    def test_public_documentation_and_license(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        product = (ROOT / "docs/PRODUCT.md").read_text(encoding="utf-8")
        license_text = (ROOT / "LICENSE").read_text()
        self.assertIn("Manny 与 Codex 协作完成", readme)
        self.assertIn("mailto:ningfangtianwai@gmail.com", readme)
        self.assertIn("不是多样本受控对照实验", product)
        self.assertIn("尚未实现逐字强制对齐", product)
        self.assertIn("Copyright (c) 2026 Manny", license_text)


if __name__ == "__main__":
    unittest.main()
