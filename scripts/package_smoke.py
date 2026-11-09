import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CODE = '''
import importlib.util
import json
import os
from pathlib import Path
import sys
import wsm
from importlib.metadata import distribution
from wsm import (MetricPolicy, NumericalEvaluation, SearchRun, replay, Session,
                        Verifier, DockerRunner, BubblewrapRunner, ResponsesTeacher, replay_session,
                        verify_session, stage_smollm)
from wsm.binary64 import create_checkpoint, predict
from wsm.harness import run as run_harness
from importlib.resources import files
assert Path(wsm.__file__).resolve().is_relative_to(Path('python').resolve())
installed = distribution('wsm')
assert installed.metadata['Name'] == 'wsm'
assert installed.version == wsm.__version__
assert Path(installed.locate_file('')).resolve() == Path('python').resolve()
assert any(entry.name == 'wsm' and entry.value == 'wsm.cli:main'
           for entry in installed.entry_points if entry.group == 'console_scripts')
assert callable(run_harness)
assert json.loads(files('wsm').joinpath('tasks/addition.json').read_text())['format'] == 'wsm.task.v1'
assert installed.metadata['Summary'].startswith('WSM (Weight Space Modifiers):')
assert importlib.util.find_spec('weightless') is None
assert '"format": "wsm.task.v1"' in files('wsm').joinpath('tasks/addition.json').read_text()
assert all(name not in sys.modules for name in ('verifiers', 'harbor', 'torch', 'transformers'))
fixture = json.loads(Path(sys.argv[1]).read_text())
run = SearchRun(create_checkpoint(fixture['graph'], fixture['tensors']),
                evaluator=NumericalEvaluation(fixture['task']),
                policy=MetricPolicy(('development.loss', 'minimize'), ('train.loss', 'minimize')),
                budget=3)
for attempt in fixture['attempts']:
    decision = run.submit({'base': run.inspect()['checkpoint']['id'],
                           'author': 'wheel-smoke', 'edits': attempt['edits']})
    assert decision['accepted'] == attempt['accepted']
assert predict(replay(run.finish()), [-8, 3]) == [-5]
assert all(name not in sys.modules for name in ('verifiers', 'harbor', 'torch', 'transformers'))
if os.environ.get('WEIGHTLESS_LOCAL_SANDBOX') == '1':
    runner = BubblewrapRunner(timeout=10)
    target = {'kind': 'binary64', 'checkpoint': create_checkpoint(fixture['graph'], fixture['tensors'])}
    public = [{'id': str(index), **case} for index, case in enumerate(fixture['task']['development'])]
    declaration = {'hypothesis': 'Installed package scalar edit', 'arms': ['teacher'],
                   'editable': ['output.weight'], 'max_edits': 2, 'max_inspect': 2, 'wall_seconds': 60,
                   'budgets': {'attempts': 1, 'evaluations': 2, 'inspections': 0,
                               'teacher_requests': 0, 'final_evaluations': 0}}
    session = Session.create('native-session.db', runner=runner, target=target,
                             declaration=declaration, development=public)
    try:
        view = session.observe('teacher')
        proposal = {key: view[key] for key in ('session', 'arm', 'declaration', 'observation', 'base')}
        proposal['edits'] = [{'tensor': 'output.weight', 'index': 0, 'value': 1},
                             {'tensor': 'output.weight', 'index': 1, 'value': -1}]
        result = session.submit(proposal, key='installed-edit')
        assert result['selection']['accepted']
        frozen = session.freeze(key='installed-freeze')
        assert frozen['complete']
        assert replay_session(session.export()) == frozen
        assert verify_session(session.export(), runner=runner, target=target)['frozen'] == frozen['id']
    finally:
        session.close()
if len(sys.argv) > 2:
    external = json.loads(Path(sys.argv[2]).read_text())
    assert predict(replay(external), [-8, 3]) == [-5]
'''
with tempfile.TemporaryDirectory(prefix="wsm-packages-") as temporary:
    root = Path(temporary)
    clean = root / "source"
    shutil.copytree(ROOT / "packages/python", clean,
                    ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__", "*.pyc"))
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", str(clean), "--wheel-dir", str(root)], check=True)
    wheel, = root.glob("*.whl")
    assert wheel.name.startswith('wsm-'), wheel.name
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert any(name.startswith('wsm/') for name in names)
        assert not any(name.startswith('weightless/') for name in names)
        metadata, = [name for name in names if name.endswith('.dist-info/METADATA')]
        assert metadata.startswith('wsm-'), metadata
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(root / "python"), str(wheel)], check=True)
    help_result = subprocess.run([sys.executable, str(root / "python/bin/wsm"), "--help"],
                                cwd=root, env={**os.environ, "PYTHONPATH": str(root / "python")},
                                check=True, capture_output=True, text=True)
    assert 'weight-editing experiments' in help_result.stdout
    subprocess.run([sys.executable, "-S", "-c", CODE, str(ROOT / "fixtures/addition.json"), *sys.argv[1:]],
                   cwd=root, env={**os.environ, "PYTHONPATH": str(root / "python")}, check=True)
print("Built wheel runs the WSM CLI, search, and pure replay outside the source tree without optional imports")
