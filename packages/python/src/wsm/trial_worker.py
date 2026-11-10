"""Run one coding-API command in a disposable process.

Worker results are untrusted observations when the process executes agent code.
Checkpoint fingerprints describe file-boundary changes, not individual writes.
"""

import copy
import hashlib
import json
from pathlib import Path
import sys

from .editing import WeightModel
from .program import execute_program


def fingerprints(model):
    return {name: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, tensor in model.model.state_dict().items()}


def run(request):
    checkpoint = request["checkpoint"]
    device = request.get("device", "cpu")
    cases = request["cases"]
    if request["action"] == "evaluate":
        return WeightModel.load(checkpoint, device=device).evaluate(cases)
    if request["action"] != "execute":
        raise ValueError("Unknown trial-worker action")
    before_model = WeightModel.load(checkpoint, device=device)
    before = fingerprints(before_model)
    del before_model
    output = Path(request["output"])
    program = output.with_suffix(".py")
    with program.open("x") as stream:
        stream.write(request["source"])
    evaluations = []
    evaluate = WeightModel.evaluate

    def record_evaluation(model, examples):
        result = evaluate(model, examples)
        evaluations.append(copy.deepcopy({"cases": examples, "result": result}))
        return result

    WeightModel.evaluate = record_evaluation
    try:
        receipt = execute_program(program, model_path=checkpoint, output=output, device=device, cases=cases)
    finally:
        WeightModel.evaluate = evaluate
    after = fingerprints(WeightModel.load(output / "checkpoint", device=device))
    changes = {name: {"before": before.get(name), "after": after.get(name)}
               for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)}
    return {"checkpoint": str(output / "checkpoint"), "receipt": receipt, "changes": changes,
            "evaluations": evaluations, "python_stdout": (output / "stdout.txt").read_text(),
            "python_stderr": (output / "stderr.txt").read_text(),
            "trust": "sandbox observation; final grading is independent"}


def main():
    request_path, response_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    exit_code = 0
    try:
        result = run(request)
    except BaseException as error:
        exit_code = 1
        result = {"error": {"type": type(error).__name__, "message": str(error)}}
        if "output" in request:
            output = Path(request["output"])
            for name in ("stdout.txt", "stderr.txt", "run.json"):
                path = output / name
                if path.is_file():
                    result[name] = path.read_text()
    with response_path.open("x") as stream:
        json.dump(result, stream, allow_nan=False)
        stream.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
