"""A stale build tree must not enter the wheel and must not be deleted."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackageSmokeTests(unittest.TestCase):
    def test_stale_namespace_is_not_packaged_or_deleted(self):
        with tempfile.TemporaryDirectory(prefix="wsm-package-regression-") as temporary:
            root = Path(temporary)
            package = root / "packages/python"
            shutil.copytree(ROOT / "packages/python", package,
                            ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__", "*.pyc"))
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/package_smoke.py", root / "scripts/package_smoke.py")
            (root / "fixtures").mkdir()
            shutil.copyfile(ROOT / "fixtures/addition.json", root / "fixtures/addition.json")
            stale = package / "build/lib/weightless/__init__.py"
            stale.parent.mkdir(parents=True)
            stale.write_text("legacy = True\n")
            result = subprocess.run([sys.executable, str(root / "scripts/package_smoke.py")],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(stale.read_text(), "legacy = True\n")


if __name__ == "__main__":
    unittest.main()
