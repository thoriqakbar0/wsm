"""Controller-owned history for sequential Python weight-editing trials.

The controller directory must remain outside the code-execution sandbox. Runner
results are observations, not authenticated scores. Private grading is recorded
separately and never appears in the agent's history.
"""

import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .errors import WeightlessError
from .editing import _cases


def _encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


class TrialExecutionError(RuntimeError):
    """A failed command with retained process output and worker observations."""

    def __init__(self, message, observation):
        super().__init__(message)
        self.observation = observation


class Trial:
    """A single writer advances checkpoints without comparing their scores."""

    def __init__(self, directory, runner):
        self.directory = Path(directory).resolve()
        self.runner = runner
        self.config = json.loads((self.directory / "trial.json").read_text())

    @classmethod
    def create(cls, directory, *, runner, initial_checkpoint, cases, max_actions=16):
        if type(initial_checkpoint) is not str or not initial_checkpoint:
            raise ValueError("initial_checkpoint must be a nonempty string")
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        public = _cases(cases)
        if not public:
            raise ValueError("A trial requires public evaluation cases")
        root = Path(directory).resolve()
        root.mkdir(mode=0o700)
        config = {"format": "wsm.coding-trial.v1", "initial_checkpoint": initial_checkpoint,
                  "cases": public, "max_actions": max_actions}
        (root / "trial.json").write_text(_encode(config) + "\n")
        (root / "events.jsonl").touch(mode=0o600)
        (root / "actions").mkdir()
        trial = cls(root, runner)
        with trial._lock():
            trial._append("created", {"checkpoint": initial_checkpoint})
        return trial

    @contextlib.contextmanager
    def _lock(self):
        with (self.directory / ".writer.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise WeightlessError("trial_busy", "Another controller is writing this trial") from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def history(self):
        """Return public events only. Held-out results never enter this file."""
        events = []
        with (self.directory / "events.jsonl").open() as stream:
            for line in stream:
                events.append(json.loads(line))
        return events

    def _append(self, kind, data):
        events = self.history()
        event = {"sequence": len(events), "time": time.time(), "kind": kind, "data": data}
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(_encode(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event

    def observation(self):
        events = self.history()
        checkpoint = self.config["initial_checkpoint"]
        state, pending, actions = "running", None, 0
        for event in events:
            kind, data = event["kind"], event["data"]
            if kind == "action":
                actions += 1
                pending = data["action"]
            elif kind == "result":
                pending = None
                if data["operation"] == "execute":
                    checkpoint = data["result"]["checkpoint"]
            elif kind == "failure":
                state = "failed"
                pending = None
            elif kind == "frozen":
                state = "frozen"
        if pending is not None:
            state = "outcome_unknown"
        return {"state": state, "checkpoint": checkpoint, "actions": actions,
                "remaining_actions": self.config["max_actions"] - actions,
                "history": events}

    def _running(self):
        view = self.observation()
        if view["state"] != "running":
            raise WeightlessError("trial_closed", "Trial is " + view["state"])
        if view["remaining_actions"] <= 0:
            raise WeightlessError("trial_budget", "Trial action limit exhausted")
        return view

    def record_message(self, role, content, **metadata):
        if role not in ("system", "user", "assistant", "tool") or type(content) is not str:
            raise ValueError("Expected a conversation role and string content")
        data = {"role": role, "content": content, "metadata": metadata}
        _encode(data)
        with self._lock():
            if self.observation()["state"] != "running":
                raise WeightlessError("trial_closed", "Messages require a running trial")
            return self._append("message", data)

    def execute(self, source, *, message=""):
        if type(source) is not str or not source.strip() or type(message) is not str:
            raise ValueError("Expected Python source and an agent message")
        with self._lock():
            view = self._running()
            action = view["actions"] + 1
            directory = self.directory / "actions" / f"{action:04d}"
            directory.mkdir()
            (directory / "edit.py").write_text(source)
            intent = {"action": action, "operation": "execute", "message": message,
                      "source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "command": ["python", "edit.py"], "checkpoint": view["checkpoint"]}
            self._append("action", intent)
            try:
                result = self.runner.execute(source, view["checkpoint"], copy.deepcopy(self.config["cases"]))
                if (type(result) is not dict or type(result.get("checkpoint")) is not str
                        or not result["checkpoint"]):
                    raise ValueError("Runner returned no published checkpoint")
                _encode(result)
                self._append("result", {"action": action, "operation": "execute", "result": result})
                return copy.deepcopy(result)
            except BaseException as error:
                self._append("failure", {"action": action, "type": type(error).__name__,
                                         "message": str(error), "outcome": "unknown",
                                         "observation": getattr(error, "observation", None)})
                raise

    def evaluate(self):
        with self._lock():
            view = self._running()
            action = view["actions"] + 1
            self._append("action", {"action": action, "operation": "evaluate", "checkpoint": view["checkpoint"]})
            try:
                result = self.runner.evaluate(view["checkpoint"], copy.deepcopy(self.config["cases"]))
                _encode(result)
                self._append("result", {"action": action, "operation": "evaluate", "result": result})
                return copy.deepcopy(result)
            except BaseException as error:
                self._append("failure", {"action": action, "type": type(error).__name__,
                                         "message": str(error), "outcome": "unknown",
                                         "observation": getattr(error, "observation", None)})
                raise

    def finish(self):
        with self._lock():
            view = self.observation()
            if view["state"] == "frozen":
                return {"checkpoint": view["checkpoint"], "state": "frozen"}
            if view["state"] != "running":
                raise WeightlessError("trial_closed", "Cannot finish a " + view["state"] + " trial")
            self._append("frozen", {"checkpoint": view["checkpoint"]})
            return {"checkpoint": view["checkpoint"], "state": "frozen"}

    def record_grade(self, result):
        """Write a private controller receipt after freezing; do not disclose it."""
        _encode(result)
        with self._lock():
            if self.observation()["state"] != "frozen":
                raise WeightlessError("trial_not_frozen", "Freeze the trial before grading")
            with (self.directory / "grade.json").open("x") as stream:
                stream.write(_encode({"checkpoint": self.observation()["checkpoint"], "result": result}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return copy.deepcopy(result)


class LocalTrialRunner:
    """Trusted local subprocess execution for SDK use and offline verification.

    This runner is not an isolation boundary. Use Modal for agent-authored code.
    """

    def __init__(self, directory, *, python=sys.executable, device="cpu", timeout=120):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.python, self.device, self.timeout = str(python), device, timeout

    def set_history(self, events):
        """Publish the public history projection for the next Python command."""
        temporary = self.directory / (".history-" + os.urandom(8).hex())
        temporary.write_text(_encode(events) + "\n")
        os.replace(temporary, self.directory / "history.json")

    def _run(self, request):
        with tempfile.TemporaryDirectory(dir=self.directory, prefix="command-") as temporary:
            root = Path(temporary)
            request_path, response_path = root / "request.json", root / "response.json"
            request_path.write_text(_encode({**request, "device": self.device}))
            try:
                process = subprocess.run([self.python, "-m", "wsm.trial_worker", str(request_path), str(response_path)],
                                         capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired as error:
                def decoded(value):
                    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
                raise TrialExecutionError("Trial worker timed out", {"stdout": decoded(error.stdout),
                                          "stderr": decoded(error.stderr), "outcome": "unknown"}) from error
            result = json.loads(response_path.read_text()) if response_path.is_file() else {}
            if process.returncode or "error" in result:
                raise TrialExecutionError("Trial worker failed", {"worker": result,
                                          "returncode": process.returncode, "stdout": process.stdout,
                                          "stderr": process.stderr})
            if request["action"] == "execute":
                result["stdout"] = process.stdout
                result["stderr"] = process.stderr
            return result

    def execute(self, source, checkpoint, cases):
        output = self.directory / ("edit-" + os.urandom(8).hex())
        return self._run({"action": "execute", "source": source, "checkpoint": checkpoint,
                          "cases": cases, "output": str(output)})

    def evaluate(self, checkpoint, cases):
        return self._run({"action": "evaluate", "checkpoint": checkpoint, "cases": cases})
