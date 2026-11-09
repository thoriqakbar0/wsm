import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CODE = '''
import json
from pathlib import Path
import sys
from weightless import (MetricPolicy, NumericalEvaluation, SearchRun, replay, Session,
                        Verifier, DockerRunner, ResponsesTeacher, replay_session,
                        verify_session, stage_smollm)
from weightless.binary64 import create_checkpoint, predict
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
if len(sys.argv) > 2:
    external = json.loads(Path(sys.argv[2]).read_text())
    assert predict(replay(external), [-8, 3]) == [-5]
'''
with tempfile.TemporaryDirectory(prefix="weightless-packages-") as temporary:
    root = Path(temporary)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", str(ROOT / "packages/python"), "--wheel-dir", str(root)], check=True)
    wheel, = root.glob("*.whl")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(root / "python"), str(wheel)], check=True)
    subprocess.run([sys.executable, "-c", CODE, str(ROOT / "fixtures/addition.json"), *sys.argv[1:]],
                   cwd=root, env={**os.environ, "PYTHONPATH": str(root / "python")}, check=True)
print("Built wheel runs search and pure replay outside the source tree without optional imports")
