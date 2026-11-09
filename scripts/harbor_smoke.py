import copy
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from harbor.models.task.task import Task
from jsonschema import Draft202012Validator
from verifiers.v1.tasksets.harbor.taskset import HarborConfig, HarborTask, parse_task
from weightless import MetricPolicy, NumericalEvaluation, SearchRun
from weightless.binary64 import create_checkpoint
from weightless.core import _binding, _sealed, evaluator_config

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "examples/harbor/signed-addition"
POLICY = MetricPolicy(("development.loss", "minimize"), ("train.loss", "minimize"))
for package, pinned in (("harbor", "0.25.0"), ("verifiers", "0.4.1"), ("jsonschema", "4.26.0")):
    assert version(package) == pinned, f"Expected {package}=={pinned}"
schema = json.loads((ROOT / "schemas/v1.json").read_text())
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema)


def run_for(public, initial=None):
    return SearchRun(initial or create_checkpoint(public["graph"], public["tensors"]),
                     evaluator=NumericalEvaluation(public["task"]), policy=POLICY, budget=8)


with tempfile.TemporaryDirectory(prefix="weightless-harbor-") as temporary:
    root = Path(temporary)
    development = root / "development"
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_harbor.py"), "--development", str(development)], check=True)
    for directory in (TASK, development):
        actual = Task(directory)
        assert actual.config.verifier.environment_mode.value == "separate"
        assert actual.config.artifacts == ["/app/submission.json"]
        config = HarborConfig(ignore_dockerfile=True, ignore_timeouts=False)
        upstream = HarborTask(parse_task(directory, 0, config), config.task)
        assert upstream.scoring.mode == "isolated" and upstream.data.verifier.fresh_copy
        assert upstream.data.network.allow == []
        assert [a.source for a in upstream.data.artifacts] == ["/app/submission.json"]
        context = directory / "environment/generated"
        inventory = {path.relative_to(context).as_posix() for path in context.rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts}
        expected_modules = {"__init__.py", "binary64.py", "core.py", "harbor.py", "session.py",
                            "verifier.py", "sandbox.py", "local_sandbox.py", "target_worker.py", "teacher.py"}
        assert inventory == {"public.json", *["weightless/" + name for name in expected_modules]}, inventory
        assert not any(name in inventory for name in ("heldout.json", "grade.py", "solve.sh"))
    assert not (development / "tests/heldout.json").exists()
    public = json.loads((TASK / "environment/generated/public.json").read_text())
    checkpoint = create_checkpoint(public["graph"], public["tensors"])
    for record in (checkpoint, public["task"], POLICY):
        validator.validate(record)
    agent = root / "agent"
    agent.mkdir()
    shutil.copy(TASK / "environment/generated/public.json", agent / "public.json")
    env = {**os.environ, "PYTHON": sys.executable, "PYTHONPATH": str(TASK / "environment/generated"), "WEIGHTLESS_APP": str(agent)}
    subprocess.run(["bash", str(TASK / "solution/solve.sh")], env=env, check=True)
    oracle = json.loads((agent / "submission.json").read_text())
    validator.validate(oracle)
    for event in oracle["events"]:
        validator.validate(event["proposal"])
        validator.validate(event["evaluation"])
    validator.validate(oracle["baseline"])
    validator.validate(oracle["config"])
    unchanged = run_for(public).finish()
    tampered = copy.deepcopy(oracle)
    tampered["checkpoint"]["tensors"]["output.weight"]["values"][0] = 0
    forged = copy.deepcopy(oracle)
    forged["config"]["settings"]["task"]["name"] = "different-problem"
    forged["config"] = _sealed({key: value for key, value in forged["config"].items() if key != "id"})
    forged = _sealed({key: value for key, value in forged.items() if key != "id"})
    other_tensors = copy.deepcopy(public["tensors"])
    other_tensors["output.bias"]["values"][0] = 1
    wrong_initial = run_for(public, create_checkpoint(public["graph"], other_tensors)).finish()
    wrong_policy = copy.deepcopy(oracle)
    wrong_policy["policy"]["metrics"].reverse()
    wrong_split = copy.deepcopy(unchanged)
    source = json.loads((development / "tests/manifest.json").read_text())
    source["split"] = "sha256:" + "1" * 64
    wrong_split["config"] = evaluator_config("harbor", "development", {"manifest": source})
    wrong_split = _sealed({key: value for key, value in wrong_split.items() if key != "id"})
    cases = [("oracle", oracle, 1.0, "graded"), ("unchanged", unchanged, 0.0, "graded"),
             ("malformed", {"format": "weightless.run.v2"}, 0.0, "rejected"),
             ("tampered", tampered, 0.0, "rejected"), ("wrong-task", forged, 0.0, "rejected"),
             ("wrong-initial", wrong_initial, 0.0, "rejected"), ("wrong-policy", wrong_policy, 0.0, "rejected"),
             ("wrong-development-split", wrong_split, 0.0, "rejected"),
             ("oversize", "x" * 8_000_001, 0.0, "rejected"),
             ("duplicate-json", '{"format":0,"format":1}', 0.0, "rejected")]
    for name, submission, reward, status in cases:
        app, logs = root / name, root / "logs" / name
        app.mkdir()
        (app / "submission.json").write_text(submission if isinstance(submission, str) else json.dumps(submission))
        process = subprocess.run(["bash", str(TASK / "tests/test.sh")], env={**env,
            "WEIGHTLESS_APP": str(app), "WEIGHTLESS_TESTS": str(TASK / "tests"), "WEIGHTLESS_LOGS": str(logs)}, capture_output=True, text=True)
        result = json.loads((logs / "diagnostics.json").read_text())
        assert result["status"] == status and result["reward"] == reward, (name, result, process.stderr)
        assert process.returncode == (0 if status == "graded" else 1), (name, process)
        if name == "oracle":
            assert result["metrics"]["heldout"]["loss"] == 0
        if name in ("wrong-task", "wrong-development-split"):
            assert "Public configuration differs from evaluator problem" in result["error"]
        print(f"Local final {name}: reward={reward}, status={status}")
    manifest = json.loads((development / "tests/manifest.json").read_text())
    config = evaluator_config("harbor", "development", {"manifest": manifest})
    request = _binding(oracle["run"], 1, oracle["checkpoint"]["id"], config)
    dev_submission = {"format": "weightless.development.v1", "request": request, "checkpoint": oracle["checkpoint"]}
    validator.validate(dev_submission)
    dev_app, dev_logs = root / "dev-app", root / "dev-logs"
    dev_app.mkdir()
    (dev_app / "submission.json").write_text(json.dumps(dev_submission))
    dev_env = {**env, "WEIGHTLESS_APP": str(dev_app), "WEIGHTLESS_TESTS": str(development / "tests"),
               "WEIGHTLESS_LOGS": str(dev_logs), "WEIGHTLESS_CONFIG": json.dumps(config), "WEIGHTLESS_REQUEST": json.dumps(request)}
    subprocess.run(["bash", str(development / "tests/test.sh")], env=dev_env, check=True, capture_output=True)
    result = json.loads((dev_logs / "diagnostics.json").read_text())
    assert set(result["metrics"]) == {"train", "development"}
    assert result["reward"] == 1 and result["request"] == request
    (dev_app / "submission.json").write_text(json.dumps(oracle))
    process = subprocess.run(["bash", str(development / "tests/test.sh")], env=dev_env, capture_output=True)
    assert process.returncode == 1
    assert not validator.is_valid(dict(oracle, extra=0))
print("Schema, upstream Harbor/Verifiers loading, split isolation, and local verifier passed")
print("This local smoke does not execute the native Docker adapter. Use scripts/harbor_native.py.")
