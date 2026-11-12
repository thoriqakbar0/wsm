"""Frozen benchmark protocols and sequential campaign ownership.

The controller owns private data and receipts. Editing receives public inputs only.
Local execution is for explicitly trusted synthetic fixtures, never isolation.
"""

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import unicodedata

from .errors import WeightlessError

FORMAT = "wsm.benchmark.v1"
ARMS = ("unchanged", "scripted", "agent")
STATUSES = ("completed", "invalid_artifact", "failed", "timed_out", "budget_blocked", "outcome_unknown")


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def read(path):
    def nonfinite(value):
        raise ValueError("Nonfinite JSON")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=nonfinite)


def write(path, value):
    """Durably publish one new controller fact without overwriting an earlier fact."""
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + os.urandom(8).hex())
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            stream.write(encoded(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        with _directory_fd(path.parent) as fd:
            os.fsync(fd)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _directory_fd(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield fd
    finally:
        os.close(fd)


@contextmanager
def campaign_lock(directory):
    with (Path(directory) / ".controller.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Campaign already has a controller") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _fields(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("Missing or unknown protocol fields")


def _number(value, name, *, positive=False, integer=False):
    if (type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(name + " must be finite and " + ("positive" if positive else "nonnegative"))
    return value


def _label(value):
    if type(value) is not str or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", value):
        raise ValueError("Identifiers must contain 1-64 letters, digits, dots, underscores, or hyphens")
    return value


def tree_digest(directory):
    root = Path(directory)
    files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError("Model snapshot is empty")
    return identity(files)


def code_digest():
    return identity({p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    return {"python": sys.version.split()[0], "machine": platform.machine(), "processor": platform.processor(),
            "numeric_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "packages": sorted(
        [d.metadata["Name"], d.version] for d in importlib.metadata.distributions())}


def grader_identity(python):
    from .trial_grading import grader_environment
    script = "import json; from wsm.benchmark import environment; print(json.dumps(environment()))"
    completed = subprocess.run([str(python), "-c", script], capture_output=True, text=True,
                               env=grader_environment(), cwd="/", timeout=30, check=True)
    return {"executable_sha256": digest(Path(python).resolve()), "environment": json.loads(completed.stdout)}


@dataclass(frozen=True)
class TrialSpec:
    id: str
    task: str
    arm: str
    repetition: int


@dataclass(frozen=True)
class Protocol:
    config: dict
    manifest: dict
    schedule: tuple[TrialSpec, ...]


def _file(reference, base):
    _fields(reference, ("path", "sha256"))
    path = (base / reference["path"]).resolve(strict=True)
    if not path.is_file() or digest(path) != reference["sha256"]:
        raise ValueError("Input digest mismatch")
    return {"path": str(path), "sha256": reference["sha256"]}


def _cases(reference):
    from .editing import _cases as validate_cases
    rows = read(reference["path"])
    cases = validate_cases(rows)
    if not cases or any(not row["prompt"].strip() or not row["target"].strip() for row in cases):
        raise ValueError("Every partition requires nonempty prompt and target strings")
    if rows != cases:
        raise ValueError("Cases require exactly prompt and target fields")
    return cases


def load_protocol(path):
    """Validate all inputs offline and derive the exact schedule and public manifest."""
    path = Path(path).resolve()
    config = read(path)
    _fields(config, ("format", "name", "model", "provenance", "runtime", "limits", "budget",
                     "statistics", "review", "regression", "tasks", "repetitions"), ("thresholds",))
    if config["format"] != FORMAT:
        raise ValueError("Unsupported benchmark format")
    _label(config["name"])
    _number(config["repetitions"], "repetitions", positive=True, integer=True)
    if config["repetitions"] > 10000:
        raise ValueError("Too many repetitions")
    model = config["model"]
    _fields(model, ("path", "id", "revision", "sha256"))
    if not all(type(model[k]) is str and model[k].strip() for k in ("id", "revision")):
        raise ValueError("Model identity and revision are required")
    model["path"] = str((path.parent / model["path"]).resolve(strict=True))
    if not (Path(model["path"]) / "model.safetensors").is_file() or tree_digest(model["path"]) != model["sha256"]:
        raise ValueError("Trusted model snapshot digest mismatch")
    provenance = config["provenance"]
    _fields(provenance, ("code_revision", "dependency_lock"))
    if type(provenance["code_revision"]) is not str or not re.fullmatch(r"[0-9a-f]{40}", provenance["code_revision"]):
        raise ValueError("A full code revision is required")
    provenance["dependency_lock"] = _file(provenance["dependency_lock"], path.parent)
    runtime = config["runtime"]
    _fields(runtime, ("backend", "grader_python"), ("policy", "actions"))
    if runtime["backend"] not in ("local_trusted", "modal"):
        raise ValueError("Select local_trusted fixtures or modal isolation")
    runtime["grader_python"] = str((path.parent / runtime["grader_python"]).absolute())
    if not Path(runtime["grader_python"]).is_file():
        raise ValueError("Grader interpreter does not exist")
    if runtime["backend"] == "local_trusted":
        if "policy" in runtime or "actions" not in runtime:
            raise ValueError("Local fixtures require frozen action JSON and cannot call an agent provider")
        runtime["actions"] = _file(runtime["actions"], path.parent)
        actions = read(runtime["actions"]["path"])
        if type(actions) is not list or not actions or any(type(a) is not dict for a in actions):
            raise ValueError("Synthetic actions must be a nonempty JSON action list")
    else:
        from .modal_trial import MODEL, REVISION
        if (model["id"], model["revision"]) != (MODEL, REVISION):
            raise ValueError("Modal v1 requires the existing pinned SmolLM2 target")
        if "actions" in runtime or "policy" not in runtime:
            raise ValueError("Modal requires a Responses proxy policy file")
        runtime["policy"] = _file(runtime["policy"], path.parent)
        from .trial_editor import validate_config
        policy = validate_config(read(runtime["policy"]["path"]))
        runtime["policy_identity"] = {k: v for k, v in policy.items() if k != "base_url"}
        route = os.environ.get(policy["model_env"])
        runtime["policy_identity"]["route_sha256"] = hashlib.sha256(route.encode()).hexdigest() if route else None
    limits = config["limits"]
    _fields(limits, ("turns", "evaluations", "output_tokens", "wall_seconds", "command_seconds", "grader_seconds"))
    for key, value in limits.items():
        _number(value, key, positive=True, integer=True)
    if limits["command_seconds"] > limits["wall_seconds"]:
        raise ValueError("Command timeout exceeds trial timeout")
    if runtime["backend"] == "modal":
        if (policy["max_turns"] != limits["turns"] or policy["max_tokens"] != limits["output_tokens"]
                or policy["request_seconds"] > limits["wall_seconds"]):
            raise ValueError("Policy limits must match campaign limits")
    budget = config["budget"]
    _fields(budget, ("max_usd", "prior_reserved_usd", "rates", "price_checked_on"))
    _number(budget["max_usd"], "max_usd")
    _number(budget["prior_reserved_usd"], "prior_reserved_usd")
    if budget["prior_reserved_usd"] > budget["max_usd"]:
        raise ValueError("Prior reservation exceeds budget")
    rates = budget["rates"]
    _fields(rates, ("build_usd", "sandbox_hourly_usd", "egress_per_gib", "teacher_request_usd"))
    for key, value in rates.items():
        _number(value, key, positive=runtime["backend"] == "modal")
    if runtime["backend"] == "modal" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", budget["price_checked_on"]):
        raise ValueError("Record when conservative prices were checked")
    statistics = config["statistics"]
    _fields(statistics, ("seed", "bootstrap_samples"))
    _number(statistics["seed"], "seed", integer=True)
    _number(statistics["bootstrap_samples"], "bootstrap_samples", positive=True, integer=True)
    if not 100 <= statistics["bootstrap_samples"] <= 100000:
        raise ValueError("bootstrap_samples must be between 100 and 100000")
    thresholds = config.get("thresholds")
    if thresholds is not None:
        _fields(thresholds, ("improvement", "regression", "numeric_tolerance"))
        for key, value in thresholds.items():
            _number(value, key)
    review = config["review"]
    _fields(review, ("data_rights", "semantic_overlap", "power", "owner", "private_artifact_store"))
    if any(type(review[k]) is not bool for k in ("data_rights", "semantic_overlap", "power")):
        raise ValueError("Review approvals must be booleans")
    if any(v is not None and (type(v) is not str or not v.strip()) for v in
           (review["owner"], review["private_artifact_store"])):
        raise ValueError("Supply owner and restricted artifact destination or null")
    regression = _file(config["regression"], path.parent)
    config["regression"] = regression
    partitions = [("regression", _cases(regression))]
    if type(config["tasks"]) is not list or not config["tasks"]:
        raise ValueError("At least one task is required")
    tasks, ids, scripts, schedule = [], set(), {}, []
    for task in config["tasks"]:
        _fields(task, ("id", "family", "instruction", "public", "private", "script"), ("scripted_stochastic",))
        stochastic = task.setdefault("scripted_stochastic", False)
        if type(stochastic) is not bool:
            raise ValueError("scripted_stochastic must be a boolean")
        _label(task["id"])
        _label(task["family"])
        if task["id"] in ids or type(task["instruction"]) is not str or not task["instruction"].strip():
            raise ValueError("Task identities must be unique and instructions nonempty")
        ids.add(task["id"])
        for kind in ("public", "private", "script"):
            task[kind] = _file(task[kind], path.parent)
        script_identity = (task["script"]["sha256"], stochastic)
        if scripts.setdefault(task["family"], script_identity) != script_identity:
            raise ValueError("Freeze one scripted heuristic per family")
        for kind in ("public", "private"):
            cases = _cases(task[kind])
            partitions.append((kind, cases))
            task[kind]["count"] = len(cases)
        tasks.append(task)
        for arm in ARMS:
            repeated = arm == "agent" or arm == "scripted" and stochastic
            for repetition in range(config["repetitions"] if repeated else 1):
                schedule.append(TrialSpec(f"{task['id']}.{arm}.{repetition:04d}", task["id"], arm, repetition))
    seen = {}
    for kind, cases in partitions:
        for case in cases:
            key = tuple(unicodedata.normalize("NFC", case[k]).strip() for k in ("prompt", "target"))
            if key in seen and (seen[key] != kind or kind == "regression"):
                raise ValueError("Normalized duplicate across benchmark partitions")
            seen[key] = kind
    package = Path(__file__).parent.resolve()
    image_files = {p.resolve() for root in (Path(model["path"]), package)
                   for p in root.rglob("*") if p.is_file()}
    for reference in [regression] + [t["private"] for t in tasks]:
        private = Path(reference["path"])
        if private in image_files or private.is_relative_to(Path(model["path"])) or private.is_relative_to(package):
            raise ValueError("Private inputs must stay outside model and package image inputs")
    family_counts = Counter(t["family"] for t in tasks)
    adequate = (len(family_counts) >= 2 and min(family_counts.values()) >= 3 and config["repetitions"] >= 5
                and all(t["public"]["count"] >= 20 and t["private"]["count"] >= 50 for t in tasks)
                and len(partitions[0][1]) >= 100)
    confirmatory = (adequate and min(family_counts.values()) > 3 and runtime["backend"] == "modal" and thresholds is not None
                    and all(review[k] for k in review))
    manifest = {"format": FORMAT, "name": config["name"], "model": {k: v for k, v in model.items() if k != "path"},
                "provenance": {"code_revision": provenance["code_revision"], "code_sha256": code_digest(),
                               "lock_sha256": provenance["dependency_lock"]["sha256"], "environment": environment(),
                               "grader": grader_identity(runtime["grader_python"])},
                "backend": runtime["backend"], "synthetic": runtime["backend"] == "local_trusted",
                "limits": limits, "budget": budget, "statistics": statistics, "thresholds": thresholds,
                "confirmatory": bool(confirmatory), "coverage_adequate": adequate,
                "scoring": "case-mean-target-nll/no-special-tokens/no-chat-template/v1",
                "regression": {"sha256": regression["sha256"], "count": len(partitions[0][1])},
                "policy": runtime.get("policy_identity", {"actions_sha256": runtime.get("actions", {}).get("sha256")}),
                "tasks": [{"id": t["id"], "family": t["family"], "scripted_stochastic": t["scripted_stochastic"],
                           "instruction_sha256": identity(t["instruction"]),
                           **{kind: {k: v for k, v in t[kind].items() if k != "path"}
                              for kind in ("public", "private", "script")}} for t in tasks],
                "schedule": [asdict(row) for row in schedule]}
    return Protocol(config, manifest, tuple(schedule))


def preflight(path):
    protocol = load_protocol(path)
    _require_editor_environment(protocol.config)
    return {"valid": True, "external_calls": 0, "scheduled_trials": len(protocol.schedule),
            "confirmatory": protocol.manifest["confirmatory"],
            "warning": "Exact matching does not establish semantic split independence or adequate power",
            "worst_case_new_reservation_usd": sum(reservation(protocol.config, row.arm) for row in protocol.schedule)}


def _require_editor_environment(config):
    if config["runtime"]["backend"] != "modal":
        return
    policy = read(config["runtime"]["policy"]["path"])
    if any(not os.environ.get(policy[key], "").strip() for key in ("model_env", "api_key_env")):
        raise WeightlessError("editor_credentials_missing",
                              "Set the configured editor model and credential environment variables before dispatch")


def reservation(config, arm):
    if config["runtime"]["backend"] == "local_trusted" or arm == "unchanged":
        return 0.0
    rates, limits = config["budget"]["rates"], config["limits"]
    from .modal_trial import MAX_CHECKPOINT_BYTES, MAX_COMMAND_OUTPUT_BYTES
    export_bytes = MAX_CHECKPOINT_BYTES + MAX_COMMAND_OUTPUT_BYTES + 1
    return (rates["build_usd"] + rates["sandbox_hourly_usd"] * limits["wall_seconds"] / 3600
            + rates["egress_per_gib"] * export_bytes / 1024**3 * (limits["turns"] if arm == "agent" else 1)
            + (rates["teacher_request_usd"] * limits["turns"] if arm == "agent" else 0))


def _freeze(protocol, root):
    root.mkdir(mode=0o700)
    restricted = root / "restricted"
    restricted.mkdir(mode=0o700)
    (root / "trials").mkdir(mode=0o700)
    config = json.loads(encoded(protocol.config))
    references = [config["regression"], config["provenance"]["dependency_lock"]]
    references += [t[k] for t in config["tasks"] for k in ("public", "private", "script")]
    references += [config["runtime"][k] for k in ("actions", "policy") if k in config["runtime"]]
    inputs = restricted / "inputs"
    inputs.mkdir(mode=0o700)
    for reference in references:
        target = inputs / reference["sha256"]
        if not target.exists():
            with target.open("xb") as stream:
                os.chmod(target, 0o600)
                stream.write(Path(reference["path"]).read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
        if digest(target) != reference["sha256"]:
            raise ValueError("Input changed while freezing")
        reference["path"] = target.relative_to(root).as_posix()
    manifest = {**protocol.manifest, "config_sha256": identity(config)}
    write(restricted / "config.json", config)
    write(root / "manifest.json", manifest)
    write(root / "seal.json", {"manifest_sha256": identity(manifest)})
    write(root / "budget-initial.json", {"reserved_usd": config["budget"]["prior_reserved_usd"]})
    return campaign(root)[2], manifest


def campaign(directory):
    root = Path(directory).resolve()
    manifest = read(root / "manifest.json")
    if read(root / "seal.json") != {"manifest_sha256": identity(manifest)}:
        raise ValueError("Campaign manifest changed")
    config = read(root / "restricted/config.json")
    if identity(config) != manifest["config_sha256"]:
        raise ValueError("Frozen config changed")
    references = [config["regression"], config["provenance"]["dependency_lock"]]
    references += [t[k] for t in config["tasks"] for k in ("public", "private", "script")]
    references += [config["runtime"][k] for k in ("actions", "policy") if k in config["runtime"]]
    for reference in references:
        reference["path"] = str(root / reference["path"])
    return root, manifest, config


def _verify_inputs(config, manifest):
    if tree_digest(config["model"]["path"]) != config["model"]["sha256"]:
        raise ValueError("Pristine snapshot changed")
    if code_digest() != manifest["provenance"]["code_sha256"] or environment() != manifest["provenance"]["environment"]:
        raise ValueError("Execution code or runtime differs from frozen protocol")
    if grader_identity(config["runtime"]["grader_python"]) != manifest["provenance"]["grader"]:
        raise ValueError("Grader interpreter or runtime differs from frozen protocol")
    references = [config["regression"], config["provenance"]["dependency_lock"]]
    references += [t[k] for t in config["tasks"] for k in ("public", "private", "script")]
    references += [config["runtime"][k] for k in ("actions", "policy") if k in config["runtime"]]
    if any(digest(ref["path"]) != ref["sha256"] for ref in references):
        raise ValueError("Frozen input changed")
    if config["runtime"]["backend"] == "modal":
        policy = read(config["runtime"]["policy"]["path"])
        route = os.environ.get(policy["model_env"])
        route_hash = hashlib.sha256(route.encode()).hexdigest() if route else None
        if route_hash != manifest["policy"]["route_sha256"]:
            raise ValueError("Editor route changed after protocol freeze")


def _grade(config, task, weights):
    from .trial_grading import grade_in_process
    return {kind: grade_in_process(config["model"]["path"], weights, reference["path"],
                                  scoring="loss_improvement", python=config["runtime"]["grader_python"],
                                  timeout=config["limits"]["grader_seconds"], detailed=True)
            for kind, reference in (("task", task["private"]), ("regression", config["regression"]))}


def _save_grade(root, trial_id, config, task, frozen, *, filename="grade.json"):
    weights = root / frozen["weights"]
    if digest(weights) != frozen["sha256"] or not frozen["editor_closed"]:
        raise ValueError("Frozen artifact is missing, changed, or editor termination unconfirmed")
    grades = _grade(config, task, weights)
    expected = "sha256:" + frozen["sha256"]
    if any(g["checkpoint"] != expected for g in grades.values()) or digest(weights) != frozen["sha256"]:
        raise ValueError("Grader artifact identity mismatch")
    receipt = {"manifest_sha256": identity(read(root / "manifest.json")), "trial_id": trial_id,
               "checkpoint": expected, "grades": grades}
    location = root / "restricted" / trial_id
    location.mkdir(mode=0o700, exist_ok=True)
    path = location / filename
    write(path, receipt)
    return digest(path)


def run(config_path, directory, *, local_trusted=False):
    protocol = load_protocol(config_path)
    if (protocol.config["runtime"]["backend"] == "local_trusted") != local_trusted:
        raise ValueError("Local fixture execution requires --local-trusted; never use it for untrusted edits")
    _require_editor_environment(protocol.config)
    root = Path(directory).resolve()
    if root.is_relative_to(Path(protocol.config["model"]["path"])) or root.is_relative_to(Path(__file__).parent):
        raise ValueError("Campaign output must stay outside model and package image inputs")
    config, manifest = _freeze(protocol, root)
    with campaign_lock(root):
        tasks = {t["id"]: t for t in config["tasks"]}
        reserved = config["budget"]["prior_reserved_usd"]
        interrupted = False
        for row in protocol.schedule:
            folder = root / "trials" / row.id
            folder.mkdir(mode=0o700)
            outcome = {"trial_id": row.id, "manifest_sha256": identity(manifest), "status": "failed",
                       "reserved_usd": 0.0, "actual_usd": None, "usage_status": "unreconciled"}
            if interrupted:
                outcome["reason"] = "campaign_interrupted"
                write(folder / "outcome.json", outcome)
                continue
            amount = reservation(config, row.arm)
            if reserved + amount > config["budget"]["max_usd"]:
                outcome.update(status="budget_blocked", reason="reservation_exceeds_allocation")
                write(folder / "outcome.json", outcome)
                continue
            started = time.monotonic()
            try:
                _verify_inputs(config, manifest)
                reserved += amount
                outcome["reserved_usd"] = amount
                write(folder / "reservation.json", {"usd": amount, "cumulative_usd": reserved})
                write(folder / "intent.json", {"trial_id": row.id, "manifest_sha256": identity(manifest)})
                if row.arm == "unchanged":
                    weights = folder / "final.safetensors"
                    shutil.copyfile(Path(config["model"]["path"]) / "model.safetensors", weights)
                    frozen = {"weights": str(weights), "sha256": digest(weights), "editor_closed": True}
                else:
                    from .benchmark_runtime import execute
                    task = tasks[row.task]
                    frozen = execute(source=config["model"]["path"], public=read(task["public"]["path"]),
                                     instruction=task["instruction"],
                                     script=Path(task["script"]["path"]).read_text() if row.arm == "scripted" else None,
                                     runtime=config["runtime"], limits=config["limits"], folder=folder)
                with Path(frozen["weights"]).open("rb") as stream:
                    os.fsync(stream.fileno())
                frozen["weights"] = Path(frozen["weights"]).relative_to(root).as_posix()
                write(folder / "frozen.json", frozen)
                _verify_inputs(config, manifest)
                grade_hash = _save_grade(root, row.id, config, tasks[row.task], frozen)
                outcome.update(status="completed", checkpoint="sha256:" + frozen["sha256"], grade_sha256=grade_hash)
            except BaseException as error:
                code = getattr(error, "code", "execution_error")
                outcome["reason"] = str(code) if isinstance(code, str) and re.fullmatch(r"[a-z_]+", code) else "execution_error"
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    interrupted = True
                local = config["runtime"]["backend"] == "local_trusted"
                observation = getattr(error, "observation", {})
                observed_failure = (type(observation) is dict and type(observation.get("worker")) is dict
                                    and "error" in observation["worker"] and observation.get("returncode") is not None)
                rejected_action = code in {"evaluation_limit", "trial_budget", "trial_closed"}
                if ((folder / "intent.json").exists() and not (folder / "frozen.json").exists()
                        and not (local or observed_failure or rejected_action)):
                    outcome["status"] = "outcome_unknown"
                elif (isinstance(error, (subprocess.TimeoutExpired, TimeoutError))
                      or local and isinstance(error.__cause__, subprocess.TimeoutExpired)):
                    outcome["status"] = "timed_out"
                    outcome["reason"] = "execution_timeout"
                elif code in ("invalid_checkpoint", "nonfinite"):
                    outcome["status"] = "invalid_artifact"
                private = root / "restricted" / row.id
                private.mkdir(mode=0o700, exist_ok=True)
                write(private / "error.json", {"type": type(error).__name__, "message": str(error),
                                              "diagnostics": getattr(error, "diagnostics", None)})
            outcome["seconds"] = time.monotonic() - started
            write(folder / "outcome.json", outcome)
        return {"directory": str(root), "scheduled_trials": len(protocol.schedule), "interrupted": interrupted}


def regrade(directory, trial_id):
    """Recompute a verified frozen artifact without replaying an edit or provider call."""
    root, manifest, config = campaign(directory)
    with campaign_lock(root):
        _verify_inputs(config, manifest)
        row = next((r for r in manifest["schedule"] if r["id"] == trial_id), None)
        if row is None:
            raise ValueError("Unknown scheduled trial")
        task = next(t for t in config["tasks"] if t["id"] == row["task"])
        frozen = read(root / "trials" / trial_id / "frozen.json")
        private = root / "restricted" / trial_id
        original = private / "grade.json"
        if not original.exists():
            sha = _save_grade(root, trial_id, config, task, frozen)
            # Recovery records grading evidence without erasing the earlier failed outcome.
            return {"regraded": True, "grade_sha256": sha, "outcome_unchanged": True}
        previous = read(original)
        outcome_path = root / "trials" / trial_id / "outcome.json"
        if outcome_path.exists():
            outcome = read(outcome_path)
            if outcome.get("status") == "completed" and digest(original) != outcome["grade_sha256"]:
                raise ValueError("Original grade receipt changed")
        if (previous["manifest_sha256"] != identity(manifest) or previous["trial_id"] != trial_id
                or previous["checkpoint"] != "sha256:" + digest(root / frozen["weights"])
                or not frozen["editor_closed"]):
            raise ValueError("Frozen artifact changed")
        grades = _grade(config, task, root / frozen["weights"])
        tolerance = (manifest["thresholds"] or {}).get("numeric_tolerance", 0.0)
        for kind in grades:
            for phase in ("baseline", "final"):
                old, new = previous["grades"][kind][phase], grades[kind][phase]
                if len(old["cases"]) != len(new["cases"]) or any(
                    abs(a["target_nll"] - b["target_nll"]) > tolerance
                    for a, b in zip(old["cases"], new["cases"], strict=True)):
                    raise ValueError("Independent regrade differs beyond frozen numeric tolerance")
        return {"regraded": True, "matches": True, "outcome_unchanged": True}
