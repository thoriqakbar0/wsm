from contextlib import contextmanager
import copy
import hashlib
import json
import math
import re
import secrets
import threading

from .binary64 import (WeightlessError, __version__, _array, _canonical, _decode,
                       _fail, _identity, _integer, _object, apply_patch,
                       parse_checkpoint, validate_proposal)

MAX_BYTES = 8_000_000
MAX_RESULT_BYTES = 262144
STATUSES = ("completed", "task_failure", "infrastructure_error", "invalid_response")


def _json(value, code):
    def visit(item, depth=0):
        if depth > 32:
            _fail(code, "JSON exceeds depth limit")
        if item is None or type(item) in (str, bool):
            return
        if type(item) in (int, float):
            try:
                finite = math.isfinite(item)
            except OverflowError:
                finite = False
            if not finite:
                _fail(code, "JSON contains nonfinite number")
            return
        if type(item) is list and len(item) <= 100000:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict and len(item) <= 100000 and all(type(k) is str for k in item):
            for child in item.values():
                visit(child, depth + 1)
            return
        _fail(code, "Expected bounded JSON data")
    visit(value)
    if len(_canonical(value)) > MAX_BYTES:
        _fail(code, "JSON exceeds byte limit")
    return copy.deepcopy(value)


def _read(value, code):
    if type(value) is str and len(value.encode()) > MAX_BYTES:
        _fail(code, "JSON exceeds byte limit")
    return _json(_decode(value, code), code)


def _digest(domain, value):
    return "sha256:" + hashlib.sha256(domain.encode() + b"\0" + _canonical(value)).hexdigest()


def _same(left, right):
    return _canonical(left) == _canonical(right)


def _sealed(value):
    return dict(value, id=_digest(value["format"], value))


def _check_id(value, code):
    _identity(value["id"], code)
    body = {k: v for k, v in value.items() if k != "id"}
    if value["id"] != _digest(value["format"], body):
        _fail(code, "Content identity mismatch")
    return value


def evaluator_config(kind, purpose, settings):
    """Create the copied configuration record shared by evaluators."""
    return _config(_sealed({"format": "weightless.evaluator.v1", "kind": kind,
                            "purpose": purpose, "settings": settings}))


def _config(value):
    code = "invalid_config"
    value = _read(value, code)
    _object(value, ("format", "kind", "purpose", "settings", "id"), code)
    if (value["format"] != "weightless.evaluator.v1"
            or type(value["kind"]) is not str
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value["kind"]) is None
            or value["purpose"] not in ("development", "final")
            or type(value["settings"]) is not dict):
        _fail(code, "Unsupported evaluator configuration")
    return _check_id(value, code)


def MetricPolicy(*terms):
    """Declare ordered metric paths. Equal tuples retain the incumbent."""
    if any(type(term) not in (tuple, list) or len(term) != 2 for term in terms):
        _fail("invalid_policy", "Each metric term must be a path and direction")
    return _policy({"format": "weightless.policy.v1", "metrics": [
        {"path": path, "direction": direction} for path, direction in terms], "ties": "retain"})


def _policy(value):
    code = "invalid_policy"
    value = _read(value, code)
    _object(value, ("format", "metrics", "ties"), code)
    if value["format"] != "weightless.policy.v1" or value["ties"] != "retain":
        _fail(code, "Unsupported policy")
    seen = set()
    for term in _array(value["metrics"], code, 8, 1):
        _object(term, ("path", "direction"), code)
        path = term["path"]
        if (type(path) is not str or len(path) > 256
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*(\.[A-Za-z][A-Za-z0-9_-]*)*", path) is None
                or path in seen or term["direction"] not in ("minimize", "maximize")):
            _fail(code, "Invalid metric path or direction")
        seen.add(path)
    return value


def _request(value):
    code = "invalid_result"
    _object(value, ("format", "run", "attempt", "checkpoint", "config", "purpose"), code)
    if value["format"] != "weightless.request.v1" or value["purpose"] not in ("development", "final"):
        _fail(code, "Invalid evaluation request")
    for key in ("run", "checkpoint", "config"):
        _identity(value[key], code)
    value["attempt"] = _integer(value["attempt"], 0, 1000, code)
    return value


def evaluation_result(request, *, status, metrics, reward=None, native_status=None,
                      diagnostics=None, error=None):
    """Build a bound result while preserving native scores and failure evidence."""
    return _result(_sealed({"format": "weightless.evaluation.v1", "request": request,
                            "status": status, "native": {"reward": reward, "metrics": metrics,
                            "status": native_status, "diagnostics": diagnostics or {}}, "error": error}))


def _result(value, expected=None):
    code = "invalid_result"
    value = _read(value, code)
    _object(value, ("format", "request", "status", "native", "error", "id"), code)
    if len(_canonical(value)) > MAX_RESULT_BYTES:
        _fail(code, "Evaluation record exceeds byte limit")
    if value["format"] != "weightless.evaluation.v1" or value["status"] not in STATUSES:
        _fail(code, "Invalid result format or status")
    _request(value["request"])
    if expected is not None and not _same(value["request"], expected):
        _fail(code, "Evaluation request binding mismatch")
    native = _object(value["native"], ("reward", "metrics", "status", "diagnostics"), code)
    if (type(native["metrics"]) is not dict or type(native["diagnostics"]) is not dict
            or (native["status"] is not None and type(native["status"]) is not str)
            or (native["reward"] is not None and type(native["reward"]) is not dict)):
        _fail(code, "Invalid native evidence")
    if len(_canonical(native["diagnostics"])) > 262144:
        _fail(code, "Diagnostics exceed byte limit")
    def metrics(items):
        for item in items.values():
            if type(item) is dict:
                metrics(item)
            elif type(item) not in (int, float):
                _fail(code, "Native metric leaves must be finite numbers")
    metrics(native["metrics"])
    if native["reward"] is not None:
        for reward in native["reward"].values():
            if type(reward) not in (int, float):
                _fail(code, "Native rewards must be numbers")
    if value["status"] == "completed":
        if value["error"] is not None:
            _fail(code, "Completed result has an error")
    else:
        error = _object(value["error"], ("type", "message"), code)
        if any(type(error[k]) is not str or len(error[k]) > 8192 for k in error):
            _fail(code, "Invalid failure detail")
    return _check_id(value, code)


def _binding(run, attempt, checkpoint, config):
    return {"format": "weightless.request.v1", "run": run, "attempt": attempt,
            "checkpoint": checkpoint, "config": config["id"], "purpose": config["purpose"]}


def _metric_key(result, policy):
    values = []
    for term in policy["metrics"]:
        value = result["native"]["metrics"]
        for part in term["path"].split("."):
            if type(value) is not dict or part not in value:
                _fail("invalid_result", "Missing policy metric " + term["path"])
            value = value[part]
        if type(value) not in (int, float) or not math.isfinite(value):
            _fail("invalid_result", "Policy metric must be finite numeric data")
        values.append(value if term["direction"] == "minimize" else -value)
    return tuple(values)


def _decision(attempt, before, result, current, baseline, policy):
    accepted = False
    if baseline is None or baseline["status"] != "completed":
        reason = "baseline_failed"
    elif result["status"] != "completed":
        reason = "evaluation_failed"
    else:
        accepted = _metric_key(result, policy) < _metric_key(current, policy)
        reason = "improved" if accepted else "not_improved"
    return {"attempt": attempt, "accepted": accepted, "reason": reason, "before": before,
            "after": attempt if accepted else before, "evaluation": result["id"],
            "incumbentEvaluation": current["id"] if current else None,
            "policy": _digest(policy["format"], policy)}


class SearchRun:
    """Single owner of candidates, evidence, budget, and incumbent selection."""

    def __init__(self, initial, *, evaluator, policy, budget):
        self._initial = parse_checkpoint(initial)
        self._config = _config(evaluator.config)
        if self._config["purpose"] != "development" or not callable(evaluator):
            _fail("invalid_config", "SearchRun requires a development evaluator")
        self._policy = _policy(policy)
        self._budget = _integer(budget, 1, 1000, "invalid_config")
        self._evaluator = evaluator
        self._run = _digest("weightless.run.identity.v1", secrets.token_hex(32))
        self._incumbent = {"run": self._run, "attempt": 0, "base": None,
                           "checkpoint": self._initial}
        self._baseline = None
        self._events = []
        self._pending = None
        self._finished = None
        self._lock = threading.Lock()
        body = _json(self._body(), "invalid_config")
        if len(_canonical(body)) + 2 * MAX_RESULT_BYTES + 2048 > MAX_BYTES:
            _fail("invalid_config", "Initial experiment leaves no space for bounded evaluation evidence")

    def _body(self):
        return {"format": "weightless.run.v2", "run": self._run,
                "initial": self._initial, "config": self._config, "policy": self._policy,
                "budget": self._budget, "used": len(self._events), "baseline": self._baseline,
                "events": self._events, "checkpoint": self._incumbent["checkpoint"],
                "provenance": {"sdk": "python", "version": __version__}}

    @contextmanager
    def _mutation(self):
        if not self._lock.acquire(blocking=False):
            _fail("invalid_state", "Run already has an operation in progress")
        try:
            if self._finished is not None:
                _fail("closed", "Run is closed")
            yield
        finally:
            self._lock.release()

    def inspect(self):
        return copy.deepcopy({"run": self._run, "checkpoint": self._incumbent["checkpoint"],
            "incumbent": self._incumbent, "config": self._config, "policy": self._policy,
            "budget": self._budget, "used": len(self._events), "baseline": self._baseline,
            "events": self._events, "closed": self._finished is not None})

    def propose(self, proposal):
        with self._mutation():
            if self._pending is not None:
                _fail("pending", "Resolve the pending candidate before proposing another")
            if len(self._events) >= self._budget:
                _fail("budget_exhausted", "Proposal budget exhausted")
            checkpoint = self._incumbent["checkpoint"]
            proposal = validate_proposal(_read(proposal, "invalid_proposal"), checkpoint)
            attempt = len(self._events) + 1
            event = {"attempt": attempt, "baseAttempt": self._incumbent["attempt"],
                     "proposal": proposal, "checkpoint": checkpoint["id"], "evaluation": None, "decision": None}
            growth = sum(len(_canonical(edit["value"])) - len(_canonical(
                checkpoint["tensors"][edit["tensor"]]["values"][edit["index"]])) for edit in proposal["edits"])
            reserved = MAX_RESULT_BYTES * (2 if self._baseline is None else 1) + 2048
            if len(_canonical(self._body())) + len(_canonical(event)) + max(0, growth) + reserved > MAX_BYTES:
                _fail("invalid_proposal", "Proposal exceeds frozen history byte limit")
            self._events.append(event)
            candidate = apply_patch(checkpoint, proposal)
            event["checkpoint"] = candidate["id"]
            self._pending = {"run": self._run, "attempt": attempt,
                             "base": checkpoint["id"], "checkpoint": candidate}
            return copy.deepcopy(self._pending)

    def _dispatch(self, candidate):
        request = _binding(self._run, candidate["attempt"], candidate["checkpoint"]["id"], self._config)
        try:
            if not _same(_config(self._evaluator.config), self._config):
                _fail("invalid_config", "Evaluator configuration changed")
            response = self._evaluator(copy.deepcopy(request), copy.deepcopy(candidate["checkpoint"]))
        except Exception as error:
            return evaluation_result(request, status="infrastructure_error", metrics={},
                error={"type": type(error).__name__, "message": str(error)[:8192]})
        result = None
        try:
            result = _result(response, request)
            if result["status"] == "completed":
                _metric_key(result, self._policy)
            envelope = {"baseline": result} if candidate["attempt"] == 0 else {"events": [{"evaluation": result}]}
            _json(envelope, "invalid_result")
            return result
        except (WeightlessError, TypeError, ValueError, RecursionError, OverflowError) as error:
            raw = repr(response)
            return evaluation_result(request, status="invalid_response", metrics={},
                reward=result["native"]["reward"] if result is not None else None,
                native_status=result["native"]["status"] if result is not None else None,
                diagnostics={"response": raw[:8192], "responseTruncated": len(raw) > 8192},
                error={"type": type(error).__name__, "message": str(error)[:8192]})

    def evaluate(self, candidate):
        with self._mutation():
            candidate = _read(candidate, "foreign_candidate")
            if _same(candidate, {"run": self._run, "attempt": 0, "base": None, "checkpoint": self._initial}):
                if self._baseline is not None:
                    _fail("invalid_state", "Baseline already evaluated")
                self._baseline = self._dispatch(candidate)
                return copy.deepcopy(self._baseline)
            if self._pending is None or not _same(candidate, self._pending):
                _fail("foreign_candidate", "Candidate is not this run's pending candidate")
            event = self._events[-1]
            if event["evaluation"] is not None:
                _fail("invalid_state", "Candidate already evaluated")
            if self._baseline is None:
                baseline = {"run": self._run, "attempt": 0, "base": None, "checkpoint": self._initial}
                self._baseline = self._dispatch(baseline)
            event["evaluation"] = self._dispatch(self._pending)
            return copy.deepcopy(event["evaluation"])

    def select(self, candidate, result):
        with self._mutation():
            candidate = _read(candidate, "foreign_candidate")
            if self._pending is None or not _same(candidate, self._pending):
                _fail("foreign_candidate", "Candidate is not this run's pending candidate")
            event = self._events[-1]
            result = _read(result, "invalid_result")
            if event["evaluation"] is None or not _same(result, event["evaluation"]):
                _fail("invalid_result", "Selection requires the retained evaluation")
            before = self._incumbent["attempt"]
            current = self._baseline if before == 0 else self._events[before - 1]["evaluation"]
            decision = _decision(event["attempt"], before, event["evaluation"], current,
                                 self._baseline, self._policy)
            if decision["accepted"]:
                self._incumbent = self._pending
            event["decision"] = decision
            self._pending = None
            return copy.deepcopy(decision)

    def submit(self, proposal):
        candidate = self.propose(proposal)
        result = self.evaluate(candidate)
        return self.select(candidate, result)

    def finish(self):
        if self._finished is not None:
            return copy.deepcopy(self._finished)
        with self._mutation():
            self._finished = _sealed(self._body())
            return copy.deepcopy(self._finished)


def replay(artifact):
    """Reconstruct weights and verify recorded evidence. Never run an evaluator."""
    code = "invalid_artifact"
    try:
        artifact = _read(artifact, code)
        _object(artifact, ("format", "run", "initial", "config", "policy", "budget", "used",
                           "baseline", "events", "checkpoint", "provenance", "id"), code)
        if artifact["format"] != "weightless.run.v2":
            _fail(code, "Unsupported artifact format")
        _check_id(artifact, code)
        run = _identity(artifact["run"], code)
        initial = parse_checkpoint(artifact["initial"])
        config, policy = _config(artifact["config"]), _policy(artifact["policy"])
        if config["purpose"] != "development":
            _fail(code, "Run configuration must have development purpose")
        provenance = _object(artifact["provenance"], ("sdk", "version"), code)
        if provenance != {"sdk": "python", "version": __version__}:
            _fail(code, "Unsupported provenance")
        budget = _integer(artifact["budget"], 1, 1000, code)
        events = _array(artifact["events"], code, budget)
        if type(artifact["used"]) is not int or artifact["used"] != len(events):
            _fail(code, "Incorrect budget usage")
        baseline = artifact["baseline"]
        if baseline is not None:
            baseline = _result(baseline, _binding(run, 0, initial["id"], config))
            if baseline["status"] == "completed":
                _metric_key(baseline, policy)
        checkpoint, incumbent, current = initial, 0, baseline
        for attempt, event in enumerate(events, 1):
            _object(event, ("attempt", "baseAttempt", "proposal", "checkpoint", "evaluation", "decision"), code)
            if (type(event["attempt"]) is not int or event["attempt"] != attempt
                    or type(event["baseAttempt"]) is not int or event["baseAttempt"] != incumbent):
                _fail(code, "Invalid event order or incumbent")
            candidate = apply_patch(checkpoint, event["proposal"])
            if candidate["id"] != event["checkpoint"]:
                _fail(code, "Candidate checkpoint mismatch")
            result, decision = event["evaluation"], event["decision"]
            if result is not None:
                if baseline is None:
                    _fail(code, "Candidate evaluation has no baseline evidence")
                result = _result(result, _binding(run, attempt, candidate["id"], config))
                if result["status"] == "completed":
                    _metric_key(result, policy)
            if decision is None:
                if attempt != len(events):
                    _fail(code, "Unresolved candidate must be last")
                continue
            if result is None:
                _fail(code, "Decision has no evaluation")
            expected = _decision(attempt, incumbent, result, current, baseline, policy)
            if type(decision) is not dict or not _same(decision, expected):
                _fail(code, "Recorded decision differs from policy")
            if expected["accepted"]:
                checkpoint, incumbent, current = candidate, attempt, result
        selected = parse_checkpoint(artifact["checkpoint"])
        if selected != checkpoint:
            _fail(code, "Selected checkpoint differs from replay")
        return selected
    except (WeightlessError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as error:
        if isinstance(error, WeightlessError) and error.code == code:
            raise
        _fail(code, "Replay rejected: " + str(error))
