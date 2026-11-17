"""Inspect an installed wheel outside the source checkout."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

API_CHECK = '''
import importlib.util
import sys
import wsm
from wsm import WeightModel, execute_program
assert set(wsm.__all__) == {'WeightModel', 'execute_program'}
assert callable(execute_program)
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
for name in ('core', 'session', 'client', 'harness', 'tinker_backend', 'binary64', 'broker'):
    assert importlib.util.find_spec('wsm.' + name) is None, name
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", type=Path, help="wheel to test (default: build from source)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="wsm-wheel-") as temporary:
        root = Path(temporary)
        if args.wheel is None:
            source = root / "source"
            shutil.copytree(
                ROOT / "packages/python",
                source,
                ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__", "*.pyc"),
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", str(source), "-w", str(root)],
                check=True,
            )
            wheel, = root.glob("*.whl")
        else:
            wheel = args.wheel.resolve()

        target = root / "python"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
            check=True,
        )
        env = {**os.environ, "PYTHONPATH": str(target)}
        subprocess.run([sys.executable, "-S", "-c", API_CHECK], cwd=root, env=env, check=True)

        command = [sys.executable, "-S", str(target / "bin/wsm")]
        help_result = subprocess.run(
            [*command, "--help"], cwd=root, env=env, stdout=subprocess.PIPE, text=True, check=True,
        )
        assert "execute" in help_result.stdout
        legacy_result = subprocess.run(
            [*command, "init"], cwd=root, env=env, capture_output=True, text=True,
        )
        assert legacy_result.returncode != 0
    print(f"Installed wheel {wheel.name} passed public API and CLI checks")


if __name__ == "__main__":
    main()
