import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from weightless import MetricPolicy, SearchRun, replay
from weightless.binary64 import apply_patch, create_checkpoint
from weightless.core import _binding
from weightless.harbor import HarborEvaluation, resolve_image

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Execute the actual optional adapter in Docker and retain native evidence.")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--image", default="weightless-signed-addition:local")
parser.add_argument("--failures", action="store_true", help="Also run native task, output, and infrastructure failure cases")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
development = args.output / "development"
subprocess.run([sys.executable, str(ROOT / "scripts/prepare_harbor.py"), "--development", str(development)], check=True)
fixture = json.loads((ROOT / "fixtures/addition.json").read_text())
image = resolve_image(args.image)
evaluator = HarborEvaluation(task_dir=development, image=image)
run = SearchRun(create_checkpoint(fixture["graph"], fixture["tensors"]), evaluator=evaluator,
                policy=MetricPolicy(("development.loss", "minimize"), ("train.loss", "minimize")), budget=3)
for number, attempt in enumerate(fixture["attempts"], 1):
    decision = run.submit({"base": run.inspect()["checkpoint"]["id"], "author": "scripted-teacher", "edits": attempt["edits"]})
    evidence = run.inspect()["events"][-1]["evaluation"]
    (args.output / f"development-{number}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(number, evidence["status"], evidence["native"]["reward"], decision["reason"], flush=True)
    assert evidence["status"] == "completed", evidence
    assert (decision["accepted"], decision["reason"]) == (attempt["accepted"], attempt["reason"]), decision
frozen = run.finish()
(args.output / "harbor-frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
(args.output / "baseline.json").write_text(json.dumps(frozen["baseline"], indent=2) + "\n")
assert frozen["baseline"]["status"] == "completed"
assert frozen["baseline"]["native"]["reward"] == {"reward": 0.0}
assert replay(frozen) == frozen["checkpoint"]
final = HarborEvaluation(task_dir=ROOT / "examples/harbor/signed-addition", image=image)
result = final.grade(frozen)
(args.output / "final.json").write_text(json.dumps(result, indent=2) + "\n")
print("final", result["status"], result["native"]["reward"], flush=True)
assert result["status"] == "completed" and result["native"]["reward"] == {"reward": 1.0}, result
print("Native adapter development reject/accept/reject and independent final passed", flush=True)

if args.failures:
    def record(name, configured, checkpoint):
        request = _binding(frozen["run"], 1, checkpoint["id"], configured.config)
        result = configured(request, checkpoint)
        (args.output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        print(name, result["status"], result["native"]["reward"], flush=True)
        return result

    overflow = apply_patch(frozen["checkpoint"], {"base": frozen["checkpoint"]["id"],
        "author": "native-failure-check", "edits": [{"tensor": "output.weight", "index": 0, "value": 1e308}]})
    failed = record("numerical-failure", evaluator, overflow)
    assert failed["status"] == "task_failure" and failed["native"]["reward"] == {"reward": 0.0}
    assert failed["native"]["diagnostics"]["execution"]["exit_code"] == 1
    wrong = create_checkpoint({"inputSize": 1, "nodes": [{"id": "o", "op": "affine", "input": "input", "weight": "w", "bias": "b"}], "output": "o"},
                              {"w": {"shape": [1, 1], "values": [0]}, "b": {"shape": [1], "values": [0]}})
    rejected = record("wrong-structure", evaluator, wrong)
    assert rejected["status"] == "task_failure" and rejected["native"]["status"] == "rejected"
    for name, script in (
        ("malformed-diagnostics", '#!/bin/sh\npython3 /tests/grade.py\nprintf malformed > /logs/verifier/diagnostics.json\n'),
        ("diagnostics-list", '#!/bin/sh\npython3 /tests/grade.py\nprintf "[]" > /logs/verifier/diagnostics.json\n'),
        ("diagnostics-null", '#!/bin/sh\npython3 /tests/grade.py\nprintf "null" > /logs/verifier/diagnostics.json\n'),
        ("invalid-metrics", '#!/bin/sh\npython3 /tests/grade.py\npython3 - <<\'PY\'\nimport json\nfrom pathlib import Path\np = Path("/logs/verifier/diagnostics.json")\ndata = json.loads(p.read_text())\ndata["metrics"]["development"]["loss"] = "not a number"\np.write_text(json.dumps(data))\nPY\n'),
        ("deep-diagnostics", '#!/bin/sh\npython3 /tests/grade.py\npython3 - <<\'PY\'\nimport json\nfrom pathlib import Path\np = Path("/logs/verifier/diagnostics.json")\ndata = json.loads(p.read_text())\nnested = None\nfor _ in range(30):\n    nested = {"nested": nested}\ndata["debug"] = nested\np.write_text(json.dumps(data))\nPY\n'),
        ("missing-reward", '#!/bin/sh\necho "intentional native verifier failure" >&2\nexit 7\n')):
        directory = args.output / name
        if directory.exists():
            raise ValueError("Use a fresh output directory for failure proof")
        shutil.copytree(development, directory)
        (directory / "tests/test.sh").write_text(script)
        configured = HarborEvaluation(task_dir=directory, image=image)
        failed = record(name, configured, frozen["checkpoint"])
        assert failed["status"] == "invalid_response", failed
        if name != "missing-reward":
            assert failed["native"]["reward"] == {"reward": 1.0}, failed
        else:
            assert failed["native"]["diagnostics"]["execution"]["exit_code"] == 7
            assert "intentional native verifier failure" in failed["native"]["diagnostics"]["execution"]["stderr"]
    previous = os.environ.get("DOCKER_HOST")
    try:
        os.environ["DOCKER_HOST"] = "unix:///tmp/weightless-intentionally-missing-docker.sock"
        failed = record("infrastructure-failure", evaluator, frozen["checkpoint"])
        assert failed["status"] == "infrastructure_error" and failed["native"]["reward"] is None
    finally:
        if previous is None:
            os.environ.pop("DOCKER_HOST", None)
        else:
            os.environ["DOCKER_HOST"] = previous
    print("Native task failures, malformed output, missing rewards, and infrastructure failure passed", flush=True)
