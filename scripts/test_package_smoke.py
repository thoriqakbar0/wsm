"""A stale build tree must not enter the wheel and must not be deleted."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackageSmokeTests(unittest.TestCase):
    def test_rejects_broken_wheel(self):
        with tempfile.TemporaryDirectory(prefix="wsm-broken-wheel-") as temporary:
            root = Path(temporary)
            package = root / "package"
            shutil.copytree(
                ROOT / "packages/python",
                package,
                ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__", "*.pyc"),
            )
            config = package / "pyproject.toml"
            config.write_text(
                config.read_text().replace("wsm.cli:main", "wsm.missing_cli:main")
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", str(package), "-w", str(root)],
                check=True,
                capture_output=True,
                text=True,
            )
            wheel, = root.glob("*.whl")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/package_smoke.py"), wheel.name],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("No module named 'wsm.missing_cli'", result.stderr)

    def test_stale_namespace_is_not_packaged_or_deleted(self):
        with tempfile.TemporaryDirectory(prefix="wsm-package-regression-") as temporary:
            root = Path(temporary)
            package = root / "packages/python"
            shutil.copytree(ROOT / "packages/python", package,
                            ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__", "*.pyc"))
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/package_smoke.py", root / "scripts/package_smoke.py")
            stale = package / "build/lib/weightless/__init__.py"
            stale.parent.mkdir(parents=True)
            stale.write_text("legacy = True\n")
            result = subprocess.run([sys.executable, str(root / "scripts/package_smoke.py")],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(stale.read_text(), "legacy = True\n")


if __name__ == "__main__":
    unittest.main()
