"""Build and inspect an installed wheel outside the source checkout."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='wsm-wheel-') as temporary:
    root = Path(temporary)
    source = root / 'source'
    shutil.copytree(ROOT / 'packages/python', source,
                    ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__', '*.pyc'))
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', str(source), '-w', str(root)], check=True)
    wheel, = root.glob('*.whl')
    target = root / 'python'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(target), str(wheel)], check=True)
    env = {**os.environ, 'PYTHONPATH': str(target)}
    code = '''
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
    subprocess.run([sys.executable, '-S', '-c', code], cwd=root, env=env, check=True)
    command = [sys.executable, str(target / 'bin/wsm')]
    result = subprocess.run([*command, '--help'], cwd=root, env=env, capture_output=True, text=True, check=True)
    assert 'execute' in result.stdout
    result = subprocess.run([*command, 'init'], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode != 0
print('Installed wheel exposes only Python editing and rejects legacy commands')
