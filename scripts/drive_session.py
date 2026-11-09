"""Run one explicit weight-editing session and write evidence.json.

Usage: python scripts/drive_session.py CONFIG.json

The config names the target, runner, declaration, development cases, output
directory, arm, and max_new_tokens. max_new_tokens must be 16 because Session
reserves generation at that fixed value. budgets.forward_calls is required.
final_forward_calls is required when a private verifier is configured.
Choose proposals or teacher, not both. Paths are relative to the working directory.

proposals runs are recorded as scripted_plumbing. A teacher run is recorded as
fresh_agent. Neither mode claims that an edit improved the model. Supported
targets are binary64 and HuggingFaceTB/SmolLM2-135M-Instruct revision
12fd25f77366fa6b3b4b768ec3050bf629380bac. The in-process runner executes the
real binary64 worker in this process and does not isolate it. SmolLM uses
docker or bubblewrap.

Teacher credentials come from an environment variable or from a JSON or YAML
file with teacher.base_url, teacher.model, and one access.api-keys entry.
Secret values are not written to the output. Private verifier cases are read
only to construct Verifier and are not placed in teacher prompts.

This command writes evidence.json, target.json, and driver.json. It does not
export weight files. Run scripts/export_checkpoint.py on a machine that has
the numerical runtime.
"""
import copy
import json
import os
from pathlib import Path
import sys

from wsm.binary64 import WeightlessError
from wsm.local_sandbox import BubblewrapRunner
from wsm.sandbox import DockerRunner
from wsm.session import Session
from wsm.teacher import ResponsesTeacher
from wsm.verifier import Verifier

_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
_ACTIONS = ("observe", "inspect", "submit", "diagnose", "status", "freeze", "finalize", "export", "cancel")
_RESERVED_KEYS = {"driver-freeze", "driver-finalize"}


def _fail(code, message):
    raise WeightlessError(code, message)


def _load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("invalid_request", "Duplicate JSON field")
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    except WeightlessError:
        raise
    except (OSError, ValueError, RecursionError):
        _fail("invalid_request", "Configuration is not JSON")


def _resolve(path):
    if type(path) is not str or not path:
        _fail("invalid_request", "Expected a path")
    return Path(path).expanduser().resolve()


def _nonnegative(value, message):
    if type(value) is not int or isinstance(value, bool) or value < 0:
        _fail("invalid_declaration", message)
    return value


def _check_target(target):
    if type(target) is not dict:
        _fail("invalid_target", "Target configuration is invalid")
    if target.get("kind") == "binary64" and set(target) == {"kind", "checkpoint"} and type(target["checkpoint"]) is dict:
        return
    pinned = (target.get("kind") == "smollm" and set(target) == {"kind", "source", "revision", "model", "dtype"}
              and target.get("revision") == _REVISION and target.get("model") == _MODEL and target.get("dtype") == "float32"
              and type(target.get("source")) is str and os.path.isabs(target["source"]))
    if pinned:
        return
    _fail("unsupported_target", "Supported targets are binary64 and the pinned SmolLM2 checkpoint only")


class InProcessBinaryRunner:
    """The real binary64 worker in this process, without sandbox isolation."""

    def __init__(self):
        self._config = {"mode": "inprocess-binary64", "isolation": "none", "worker": "wsm.target_worker"}

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def execute(self, action, target, *, patches=None, edits=None, cases=None, query=None,
                generate=False, purpose="development", timeout=None, on_dispatch=None, forward_limit=None):
        if type(target) is not dict or target.get("kind") != "binary64":
            _fail("unsupported_target", "In-process runner accepts binary64 only")
        request = {"action": action, "target": target, "patches": [] if patches is None else patches,
                   "edits": [] if edits is None else edits, "cases": cases, "query": query,
                   "generate": generate, "purpose": purpose}
        if forward_limit is not None:
            request["forward_limit"] = forward_limit
        if on_dispatch is not None:
            on_dispatch()
        from wsm.target_worker import run
        return run(request)


def _runner(spec, target):
    if type(spec) is not dict or type(spec.get("kind")) is not str:
        _fail("invalid_request", "Runner configuration is invalid")
    kind = spec["kind"]
    if kind == "inprocess-binary64":
        if set(spec) != {"kind"}:
            _fail("invalid_request", "In-process runner accepts no hidden options")
        if target.get("kind") != "binary64":
            _fail("unsupported_target", "In-process runner accepts binary64 only")
        return InProcessBinaryRunner()
    if kind == "docker":
        required = {"kind", "image", "docker", "timeout", "cpus", "memory"}
        if not required <= set(spec) <= required | {"host"}:
            _fail("invalid_request", "Docker runner fields are incomplete")
        return DockerRunner(spec["image"], docker=spec["docker"], host=spec.get("host"), timeout=spec["timeout"],
                            cpus=spec["cpus"], memory=spec["memory"])
    if kind == "bubblewrap":
        required = {"kind", "python", "bwrap", "systemd_run", "systemctl", "timeout", "cpus", "memory_bytes", "pids", "tmp_bytes"}
        if set(spec) != required:
            _fail("invalid_request", "Bubblewrap runner fields are incomplete")
        return BubblewrapRunner(python=spec["python"], bwrap=spec["bwrap"], systemd_run=spec["systemd_run"],
                                systemctl=spec["systemctl"], timeout=spec["timeout"], cpus=spec["cpus"],
                                memory_bytes=spec["memory_bytes"], pids=spec["pids"], tmp_bytes=spec["tmp_bytes"])
    _fail("invalid_request", "Runner must be inprocess-binary64, docker, or bubblewrap")


def _parse(config):
    allowed = {"target", "runner", "declaration", "development", "output", "max_new_tokens", "arm",
               "proposals", "teacher", "development_retention", "verifier"}
    if type(config) is not dict or set(config) - allowed or not {"target", "runner", "declaration", "development", "output", "max_new_tokens", "arm"} <= set(config):
        _fail("invalid_request", "Configuration is missing a required field or contains an unknown field")
    if ("proposals" in config) == ("teacher" in config):
        _fail("invalid_request", "Choose scripted proposals or a teacher")
    if type(config["max_new_tokens"]) is not int or config["max_new_tokens"] != 16:
        _fail("unsupported", "Session reserves max_new_tokens 16 only")
    _check_target(config["target"])
    declaration = config["declaration"]
    if type(declaration) is not dict:
        _fail("invalid_declaration", "Declaration is invalid")
    if declaration.get("claim") not in ("exploratory", "confirmatory"):
        _fail("invalid_declaration", "claim must be exploratory or confirmatory")
    if declaration.get("selection") != "loss":
        _fail("unsupported", "Only strict loss minimization is supported")
    budgets = declaration.get("budgets")
    if type(budgets) is not dict:
        _fail("invalid_declaration", "forward_calls budget is required")
    _nonnegative(budgets.get("forward_calls"), "forward_calls budget is required")
    if "verifier" in config:
        _nonnegative(declaration.get("final_forward_calls"), "final_forward_calls is required when a private verifier is configured")
    if type(config["development"]) is not list or not config["development"]:
        _fail("invalid_declaration", "Public development cases are required")
    if "development_retention" in config and type(config["development_retention"]) is not list:
        _fail("invalid_declaration", "Public retention cases must be a list")
    if type(config["arm"]) is not str or not config["arm"]:
        _fail("invalid_request", "Arm is required")
    arms = declaration.get("arms")
    if type(arms) is list and config["arm"] not in arms:
        _fail("invalid_request", "Arm is not in the declaration")
    spec = {"target": config["target"], "runner": config["runner"], "declaration": declaration,
            "development": config["development"], "output": _resolve(config["output"]), "arm": config["arm"],
            "retention": config.get("development_retention"), "mode": "scripted_plumbing" if "proposals" in config else "fresh_agent"}
    if spec["output"].exists() or spec["output"].is_symlink():
        _fail("invalid_request", "Output directory already exists")
    if spec["mode"] == "scripted_plumbing":
        proposals = config["proposals"]
        if type(proposals) is not list:
            _fail("invalid_request", "Proposals must be a list")
        keys = []
        for item in proposals:
            if type(item) is not dict or set(item) != {"key", "edits"} or type(item["key"]) is not str or type(item["edits"]) is not list:
                _fail("invalid_request", "Each proposal needs a key and edits")
            if item["key"] in _RESERVED_KEYS:
                _fail("invalid_request", "Proposal key is reserved by the driver")
            keys.append(item["key"])
        if len(keys) != len(set(keys)):
            _fail("invalid_request", "Proposal keys must be unique")
        spec["proposals"] = proposals
    else:
        spec["teacher"] = _parse_teacher(config["teacher"])
    if "verifier" in config:
        verifier = config["verifier"]
        if type(verifier) is not dict or set(verifier) != {"database", "dataset", "cases"}:
            _fail("invalid_request", "Verifier configuration is invalid")
        if type(verifier["dataset"]) is not str or not verifier["dataset"]:
            _fail("invalid_request", "Verifier dataset name is required")
        spec["verifier"] = {"database": _resolve(verifier["database"]), "dataset": verifier["dataset"], "cases": _resolve(verifier["cases"])}
    return spec


def _parse_teacher(teacher):
    if type(teacher) is not dict or type(teacher.get("instructions")) is not str or not teacher["instructions"]:
        _fail("invalid_teacher", "Teacher instructions are required")
    if len(teacher["instructions"].encode()) > 100000:
        _fail("invalid_teacher", "Teacher instructions exceed the prompt bound")
    required = {"instructions", "timeout", "max_output_tokens", "directory"}
    endpoint = {"endpoint", "model", "api_key_env"}
    if required | endpoint == set(teacher):
        if type(teacher["api_key_env"]) is not str or not teacher["api_key_env"].isidentifier():
            _fail("invalid_teacher", "Teacher credential environment variable name is invalid")
        credential = ("env", teacher["api_key_env"], teacher["endpoint"], teacher["model"])
    elif set(teacher) == required | {"credentials"}:
        credential = ("file", _resolve(teacher["credentials"]))
    else:
        _fail("invalid_teacher", "Teacher configuration needs an endpoint and environment credential, or a credentials file")
    if type(teacher["directory"]) is not str:
        _fail("invalid_teacher", "Teacher artifact directory is required")
    return {"instructions": teacher["instructions"], "timeout": teacher["timeout"],
            "max_output_tokens": teacher["max_output_tokens"], "directory": _resolve(teacher["directory"]),
            "credential": credential}


def _credentials(credential):
    if credential[0] == "env":
        try:
            key = os.environ[credential[1]]
        except KeyError:
            _fail("invalid_teacher", "Teacher credential environment variable is missing")
        if type(key) is not str or not key:
            _fail("invalid_teacher", "Teacher credential environment variable is empty")
        return credential[2], credential[3], key
    path = credential[1]
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            data = yaml.safe_load(path.read_text())
        else:
            data = json.loads(path.read_text())
    except ImportError:
        _fail("invalid_teacher", "YAML credentials require PyYAML, which is not installed")
    except (OSError, ValueError):
        _fail("invalid_teacher", "Teacher credentials file is invalid")
    if (type(data) is not dict or set(data) != {"teacher", "access"} or type(data["teacher"]) is not dict
            or set(data["teacher"]) != {"base_url", "model"} or type(data["access"]) is not dict):
        _fail("invalid_teacher", "Teacher credentials file fields are invalid")
    keys = data["access"].get("api-keys")
    if type(keys) is not list or len(keys) != 1 or type(keys[0]) is not str or not keys[0]:
        _fail("invalid_teacher", "Teacher credentials file must contain one api key")
    return data["teacher"]["base_url"], data["teacher"]["model"], keys[0]


def _load_cases(path):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        _fail("invalid_request", "Verifier cases file is invalid")
    if type(data) is list or (type(data) is dict and type(data.get("task")) is list):
        return data
    _fail("invalid_request", "Verifier cases file is invalid")


def _prompt(instructions, view, previous):
    return instructions + "\n" + json.dumps({"view": view, "previous": previous}, sort_keys=True, allow_nan=False)


def _dispatch(session, command, arm, dataset):
    try:
        result = session.dispatch(command, arm=arm, allowed_actions=_ACTIONS, final_dataset=dataset)
    except WeightlessError as error:
        return {"action": command.get("action"), "error": error.code, "message": str(error)}
    summary = {"action": command.get("action"), "status": result.get("status", "completed")}
    selection = result.get("selection")
    if type(selection) is dict and "accepted" in selection:
        summary["accepted"] = selection["accepted"]
    return summary


def _scripted(session, spec, dataset):
    feedback = []
    for item in spec["proposals"]:
        view = session.observe(spec["arm"])
        if view["phase"] != "developing":
            break
        payload = {key: view[key] for key in ("session", "arm", "declaration", "observation", "base")}
        payload["edits"] = item["edits"]
        feedback.append({"key": item["key"], **_dispatch(session, {"action": "submit", "key": item["key"], "payload": payload}, spec["arm"], dataset)})
    return feedback


def _teacher(session, spec, dataset, credential):
    teacher_spec = spec["teacher"]
    endpoint, model, key = credential
    teacher = ResponsesTeacher(base_url=endpoint, model=model, api_key=key, directory=teacher_spec["directory"],
                               timeout=teacher_spec["timeout"], max_output_tokens=teacher_spec["max_output_tokens"])
    feedback = []
    previous = None
    turns = spec["declaration"]["budgets"]["teacher_requests"]
    for turn in range(1, turns + 1):
        view = session.observe(spec["arm"])
        purpose = "report" if view["phase"] in ("frozen", "closed") else "development"
        try:
            response = teacher.ask(session, _prompt(teacher_spec["instructions"], view, previous),
                                   key=f"teacher-{turn}", arm=spec["arm"], purpose=purpose)
            command = json.loads(response["text"])
        except WeightlessError as error:
            feedback.append({"turn": turn, "error": error.code, "message": str(error)})
            break
        except (ValueError, TypeError, KeyError):
            previous = {"turn": turn, "error": "invalid_teacher_command", "message": "Teacher response was not a JSON command"}
            feedback.append(previous)
            continue
        if type(command) is not dict or command.get("action") == "report":
            feedback.append({"turn": turn, "action": "report"})
            break
        summary = {"turn": turn, **_dispatch(session, command, spec["arm"], dataset)}
        feedback.append(summary)
        previous = summary
    return feedback


def _finish(session, arm, dataset):
    feedback = []
    view = session.observe(arm)
    if view["phase"] == "developing":
        feedback.append(_dispatch(session, {"action": "freeze", "key": "driver-freeze", "payload": {}}, arm, dataset))
    if dataset and "finalize" in session.observe(arm)["actions"]:
        feedback.append(_dispatch(session, {"action": "finalize", "key": "driver-finalize", "payload": {}}, arm, dataset))
    return feedback


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def drive(config):
    spec = _parse(config)
    credential = _credentials(spec["teacher"]["credential"]) if spec["mode"] == "fresh_agent" else None
    private_cases = _load_cases(spec["verifier"]["cases"]) if "verifier" in spec else None
    runner = _runner(spec["runner"], spec["target"])
    dataset = spec["verifier"]["dataset"] if "verifier" in spec else None
    verifier = None
    session = None
    spec["output"].mkdir(mode=0o700)
    try:
        if spec["mode"] == "fresh_agent":
            spec["teacher"]["directory"].mkdir(parents=True, mode=0o700)
            spec["teacher"]["directory"].chmod(0o700)
        if private_cases is not None:
            verifier = Verifier(spec["verifier"]["database"], runner=runner, datasets={dataset: private_cases})
        session = Session.create(spec["output"] / "session.db", runner=runner, target=spec["target"],
                                 declaration=spec["declaration"], development=spec["development"],
                                 development_retention=spec["retention"], verifier=verifier, final_dataset=dataset)
        feedback = _scripted(session, spec, dataset) if spec["mode"] == "scripted_plumbing" else _teacher(session, spec, dataset, credential)
        feedback.extend(_finish(session, spec["arm"], dataset))
        bundle = session.export()
        _write(spec["output"] / "evidence.json", bundle)
        _write(spec["output"] / "target.json", spec["target"])
        frozen = bundle.get("frozen") or {}
        report = {"format": "weightless.driver.v1", "mode": spec["mode"], "efficacy_claim": False,
                  "runner": spec["runner"]["kind"], "max_new_tokens": 16, "arm": spec["arm"],
                  "evidence": "evidence.json", "target": "target.json",
                  "frozen_complete": frozen.get("complete") is True,
                  "final_status": None if bundle.get("final") is None else bundle["final"].get("status"),
                  "supported_targets": ["binary64", _MODEL + "@" + _REVISION],
                  "forward_calls": spec["declaration"]["budgets"]["forward_calls"],
                  "final_forward_calls": spec["declaration"].get("final_forward_calls"),
                  "feedback": feedback,
                  "limitations": [
                      "scripted_plumbing records supplied edits and does not claim an improvement." if spec["mode"] == "scripted_plumbing"
                      else "fresh_agent records one bounded teacher session and does not claim an improvement.",
                      "Generation reservation uses the session max_new_tokens value of 16.",
                      "Supported targets are binary64 and the pinned SmolLM2-135M-Instruct revision only.",
                      "This command writes evidence. Checkpoint files are a separate operator export.",
                      "Private verifier cases are not placed in teacher prompts."]}
        _write(spec["output"] / "driver.json", report)
        return report
    finally:
        if session is not None:
            session.close()
        if verifier is not None:
            verifier.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in (["-h"], ["--help"]):
        sys.stdout.write(__doc__)
        return 0
    if len(argv) != 1:
        sys.stderr.write(__doc__ or "")
        return 2
    try:
        report = drive(_load_json(argv[0]))
    except WeightlessError as error:
        sys.stderr.write(f"{error.code}: {error}\n")
        return 1
    summary = {key: report[key] for key in ("mode", "efficacy_claim", "frozen_complete", "evidence")}
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
