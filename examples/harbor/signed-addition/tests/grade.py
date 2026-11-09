import json
import os
from pathlib import Path
import sys

from wsm import WeightlessError, replay
from wsm.binary64 import create_checkpoint, evaluate, parse_checkpoint
from wsm.core import _config, _digest, _object, _read, _request, _same


def grade(submission, expected, manifest, tests, config=None, request=None):
    initial = create_checkpoint(expected["graph"], expected["tensors"])
    if manifest["initial"] != initial["id"] or manifest["problem"] != _digest("weightless.problem.v1", expected):
        raise WeightlessError("invalid_config", "Trusted problem manifest differs")
    purpose = manifest["purpose"]
    if config is not None:
        config = _config(config)
        request = _request(request)
        if (config["kind"] != "harbor" or config["purpose"] != purpose
                or config["settings"]["manifest"] != manifest
                or request["config"] != config["id"] or request["purpose"] != purpose):
            raise WeightlessError("invalid_result", "Trusted evaluation binding differs")
    if purpose == "development":
        _object(submission, ("format", "request", "checkpoint"), "invalid_artifact")
        if submission["format"] != "weightless.development.v1" or config is None:
            raise WeightlessError("invalid_artifact", "Development requires a bound candidate checkpoint")
        if not _same(submission["request"], request):
            raise WeightlessError("invalid_result", "Submission request differs from trusted request")
        checkpoint = parse_checkpoint(submission["checkpoint"])
        if (checkpoint["graph"] != initial["graph"]
                or {k: t["shape"] for k, t in checkpoint["tensors"].items()}
                != {k: t["shape"] for k, t in initial["tensors"].items()}):
            raise WeightlessError("invalid_checkpoint", "Candidate structure differs from evaluator problem")
        splits = {split: expected["task"][split] for split in ("train", "development")}
    elif purpose == "final":
        if not _same(submission.get("initial"), initial) or not _same(submission.get("policy"), manifest["policy"]):
            raise WeightlessError("invalid_artifact", "Initial checkpoint or policy differs from evaluator problem")
        if type(submission.get("budget")) is not int or not 1 <= submission["budget"] <= expected["task"]["budget"]:
            raise WeightlessError("invalid_artifact", "Proposal budget exceeds evaluator problem")
        run_config = _config(submission.get("config"))
        settings = run_config["settings"]
        if run_config["kind"] == "numerical":
            valid = _same(settings, {"arithmetic": "binary64-row-major-v1", "task": expected["task"]})
        elif run_config["kind"] == "harbor":
            source = settings.get("manifest", {})
            development = {key: expected["task"][key] for key in ("train", "development")}
            expected_source = dict(manifest, purpose="development",
                                   split=_digest("weightless.split.v1", development))
            valid = _same(source, expected_source)
        else:
            valid = False
        if not valid:
            raise WeightlessError("invalid_artifact", "Public configuration differs from evaluator problem")
        checkpoint = replay(submission)
        splits = {"heldout": _read((tests / "heldout.json").read_text(), "invalid_task")}
        if request is not None and (request["run"] != submission["run"] or request["attempt"] != submission["used"]):
            raise WeightlessError("invalid_result", "Frozen run differs from final request")
    else:
        raise WeightlessError("invalid_config", "Unsupported trusted purpose")
    if manifest["split"] != _digest("weightless.split.v1", splits if purpose == "development" else splits["heldout"]):
        raise WeightlessError("invalid_config", "Trusted split identity differs")
    if request is not None and request["checkpoint"] != checkpoint["id"]:
        raise WeightlessError("invalid_result", "Actual checkpoint differs from trusted request")
    metrics = {split: evaluate(checkpoint, examples) for split, examples in splits.items()}
    selected = metrics["development" if purpose == "development" else "heldout"]
    return {"reward": float(selected["exact"] == selected["count"]), "metrics": metrics,
            "checkpoint": checkpoint["id"], "status": "graded", "request": request, "manifest": manifest}


def main():
    tests = Path(os.environ.get("WEIGHTLESS_TESTS", "/tests"))
    app = Path(os.environ.get("WEIGHTLESS_APP", "/app"))
    logs = Path(os.environ.get("WEIGHTLESS_LOGS", "/logs/verifier"))
    logs.mkdir(parents=True, exist_ok=True)
    request = config = manifest = None
    try:
        manifest = _read((tests / "manifest.json").read_text(), "invalid_config")
        config = _read(os.environ["WEIGHTLESS_CONFIG"], "invalid_config") if "WEIGHTLESS_CONFIG" in os.environ else None
        request = _read(os.environ["WEIGHTLESS_REQUEST"], "invalid_result") if "WEIGHTLESS_REQUEST" in os.environ else None
        with (app / "submission.json").open("rb") as file:
            raw = file.read(8_000_001)
        if len(raw) > 8_000_000:
            raise WeightlessError("invalid_artifact", "Artifact exceeds byte limit")
        submission = _read(raw.decode(), "invalid_artifact")
        if type(submission) is not dict:
            raise WeightlessError("invalid_artifact", "Artifact must be an object")
        result = grade(submission, _read((tests / "expected.json").read_text(), "invalid_config"),
                       manifest, tests, config, request)
    except (WeightlessError, ValueError, OSError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as error:
        result = {"reward": 0.0, "metrics": {}, "status": "task_failure" if getattr(error, "code", None) == "numerical_failure" else "rejected",
                  "error": str(error), "code": getattr(error, "code", "invalid_artifact"),
                  "request": request, "checkpoint": request.get("checkpoint") if type(request) is dict else None,
                  "manifest": manifest}
    (logs / "reward.json").write_text(json.dumps({"reward": result["reward"]}) + "\n")
    (logs / "reward.txt").write_text(str(result["reward"]) + "\n")
    (logs / "diagnostics.json").write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "graded" else 1


if __name__ == "__main__":
    sys.exit(main())
