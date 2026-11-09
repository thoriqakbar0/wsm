"""Operator workspaces around the durable experiment runtime."""
import copy
import json
import os
from pathlib import Path
import tempfile

from .binary64 import WeightlessError, create_checkpoint
from .local_sandbox import BubblewrapRunner
from .sandbox import DockerRunner
from .session import Session
from .verifier import Verifier


def read_json(path):
    """Read finite JSON, rejecting duplicate fields at the input boundary."""
    def fields(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("Duplicate JSON field")
            result[name] = value
        return result

    def nonfinite(value):
        raise ValueError("Nonfinite JSON number")

    try:
        value = json.loads(Path(path).read_text(), object_pairs_hook=fields,
                           parse_constant=nonfinite)
        json.dumps(value, allow_nan=False)
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise WeightlessError("invalid_json", "Expected finite JSON with unique fields") from None


def write_json(path, value):
    """Durably publish a complete artifact without overwriting an existing file."""
    path = Path(path)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".wsm-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def addition_config():
    """A synthetic signed-addition example with explicit experiment limits."""
    graph = {"inputSize": 2, "nodes": [
        {"id": "hidden", "op": "affine", "input": "input", "weight": "hidden.weight", "bias": "hidden.bias"},
        {"id": "activation", "op": "relu", "input": "hidden"},
        {"id": "output", "op": "affine", "input": "activation", "weight": "output.weight", "bias": "output.bias"},
    ], "output": "output"}
    tensors = {"hidden.weight": {"shape": [2, 2], "values": [1, 1, -1, -1]},
               "hidden.bias": {"shape": [2], "values": [0, 0]},
               "output.weight": {"shape": [1, 2], "values": [0, 0]},
               "output.bias": {"shape": [1], "values": [0]}}
    return {"format": "wsm.workspace.v1", "runner": {"backend": "bubblewrap"},
            "target": {"kind": "binary64", "checkpoint": create_checkpoint(graph, tensors)},
            "declaration": {
                "hypothesis": "Two output-weight replacements implement signed addition on the fixed feature basis.",
                "arms": ["researcher"], "editable": ["output.weight", "output.bias"],
                "budgets": {"attempts": 8, "evaluations": 12, "inspections": 8,
                            "diagnostics": 2, "teacher_requests": 8, "final_evaluations": 2},
                "max_edits": 4, "max_inspect": 8, "wall_seconds": 86400,
                "claim": "exploratory"},
            "development": [{"id": str(i), "input": pair, "target": [sum(pair)]}
                            for i, pair in enumerate(([3, -1], [-3, 1], [4, 5]))]}


def create_example_verifier(directory):
    """Create separate synthetic demo data, never evidence of a hidden benchmark."""
    directory = Path(directory).resolve()
    directory.mkdir(mode=0o700)
    path = directory / "verifier.json"
    write_json(path, {"registry": str(directory / "claims.db"), "dataset": "addition-demo",
                     "cases": {"task": [{"id": f"demo-final-{i}", "input": pair, "target": [sum(pair)]}
                                        for i, pair in enumerate(([8, 9], [-8, -9], [-7, 11], [0, 0]))]}})
    path.chmod(0o600)
    return path


def make_runner(config):
    if type(config) is not dict:
        raise WeightlessError("invalid_config", "Runner configuration must be an object")
    options = dict(config)
    backend = options.pop("backend", None)
    constructors = {"bubblewrap": BubblewrapRunner, "docker": DockerRunner}
    if backend not in constructors:
        raise WeightlessError("invalid_config", "Choose an explicit bubblewrap or docker runner")
    try:
        return constructors[backend](**options)
    except TypeError:
        raise WeightlessError("invalid_config", "Unsupported runner options") from None


class Experiment:
    """A trusted operator's workspace. Expose Session.dispatch to agents instead."""

    @classmethod
    def create(cls, directory, config, *, verifier_config=None):
        config = copy.deepcopy(config)
        required = {"format", "runner", "target", "declaration", "development"}
        if (type(config) is not dict or not required <= config.keys()
                or set(config) - required - {"development_retention"}
                or config["format"] != "wsm.workspace.v1"):
            raise WeightlessError("invalid_config", "Expected a wsm.workspace.v1 configuration")
        runner = make_runner(config["runner"])
        directory = Path(directory).resolve()
        directory.mkdir(mode=0o700)
        write_json(directory / "experiment.json", config)
        obj = cls(directory, config, runner, verifier_config)
        try:
            obj.session = Session.create(directory / "session.db", runner=runner, target=config["target"],
                                         declaration=config["declaration"], development=config["development"],
                                         development_retention=config.get("development_retention"),
                                         verifier=obj.verifier, final_dataset=obj.final_dataset)
            obj._bind_verifier(create=True)
            return obj
        except BaseException:
            obj.close()
            raise

    @classmethod
    def open(cls, directory, *, verifier_config=None):
        directory = Path(directory).resolve()
        if not (directory / "session.db").is_file():
            raise WeightlessError("workspace_missing", "No session.db found; create an experiment first")
        config = read_json(directory / "experiment.json")
        if type(config) is not dict or config.get("format") != "wsm.workspace.v1":
            raise WeightlessError("invalid_config", "Expected a wsm.workspace.v1 workspace")
        obj = None
        try:
            runner = make_runner(config["runner"])
            obj = cls(directory, config, runner, verifier_config)
            obj.session = Session.open(directory / "session.db", runner=runner, target=config["target"],
                                       verifier=obj.verifier, final_dataset=obj.final_dataset)
            obj._bind_verifier()
            return obj
        except KeyError:
            if obj is not None:
                obj.close()
            raise WeightlessError("invalid_config", "Workspace configuration is missing required fields") from None
        except BaseException:
            if obj is not None:
                obj.close()
            raise

    def __init__(self, directory, config, runner, verifier_config):
        self.directory, self.config, self.runner = directory, config, runner
        self.session, self.verifier, self.final_dataset = None, None, None
        if verifier_config is not None:
            private = Path(verifier_config).resolve()
            if private.is_relative_to(directory):
                raise WeightlessError("invalid_config", "Keep verifier configuration outside the public workspace")
            record = read_json(private)
            if (type(record) is not dict or set(record) != {"registry", "dataset", "cases"}
                    or type(record["registry"]) is not str or not record["registry"]
                    or type(record["dataset"]) is not str or not record["dataset"]):
                raise WeightlessError("invalid_config", "Verifier configuration needs registry, dataset, and cases")
            registry = (private.parent / record["registry"]).resolve()
            if registry.is_relative_to(directory):
                raise WeightlessError("invalid_config", "Keep the verifier registry outside the public workspace")
            source = config.get("target", {}).get("source")
            if source and (private.is_relative_to(Path(source).resolve())
                           or registry.is_relative_to(Path(source).resolve())):
                raise WeightlessError("invalid_config", "Keep verifier files outside the mounted model directory")
            self.final_dataset = record["dataset"]
            self.verifier = Verifier(registry, runner=runner, datasets={self.final_dataset: record["cases"]})

    def _bind_verifier(self, *, create=False):
        if self.verifier is not None:
            self.verifier.bind_session(self.session.export()["session"], self.final_dataset, create=create)

    def proposal(self, edits, *, arm=None):
        """Bind edits to the current observation; save this before submitting."""
        arm = self.arm(arm)
        view = self.session.observe(arm)
        return {key: view[key] for key in ("session", "arm", "declaration", "observation", "base")} | {"edits": edits}

    def arm(self, value=None):
        arms = self.config["declaration"]["arms"]
        if value is None:
            if len(arms) != 1:
                raise WeightlessError("invalid_request", "Select an arm explicitly for a multi-arm experiment")
            return arms[0]
        return value

    def execute(self, command, *, arm=None):
        """Dispatch an operator command through the existing session contract."""
        if type(command) is dict and command.get("action") == "finalize":
            self._bind_verifier()
        return self.session.dispatch(command, arm=self.arm(arm), allowed_actions=(
            "observe", "inspect", "submit", "diagnose", "status", "freeze", "cancel", "export", "finalize"))

    def close(self):
        if self.session is not None:
            self.session.close()
        if self.verifier is not None:
            self.verifier.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
