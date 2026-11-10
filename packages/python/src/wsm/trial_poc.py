"""Controller for the fixed-model Modal and hosted-Tinker proof of concept."""

import json
import hashlib
import math
import os
from pathlib import Path
import sys
import time

from .binary64 import WeightlessError
from .editing import _cases
from .modal_trial import INITIAL_CHECKPOINT, MODEL, REVISION, ModalTrialRunner, trial_image
from .trial import Trial


INSTRUCTIONS = """You edit the weights of a target language model to solve the supplied task.
Return exactly one JSON object per turn. Available actions:
{"action":"python","message":"why","source":"def edit(model):\\n    ..."}
{"action":"evaluate"}
{"action":"history"}
{"action":"finish"}
Python files receive the existing WSM WeightModel as model. model.weights maps
names to live PyTorch tensors. model.evaluate(cases) returns public loss. Use
model.evaluate on the supplied public cases or request the evaluate action.
Each successful Python file publishes a new checkpoint, even if it scores worse.
The next file starts from that checkpoint. Python globals do not persist.
Undo by writing another Python edit, or loading a prior checkpoint with
WeightModel.load and copying its tensors. Public history lists checkpoint paths
and is also readable at /trial/history.json. All commands and outputs are recorded.
The fixed target is SmolLM2-135M-Instruct. Only final held-out performance supplies
the training reward. Held-out tests and training credentials are not in your sandbox.
Finish within the configured action limit. Return JSON without Markdown fences.
"""


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive number")
    return value


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    required = {"model_source", "public_cases", "heldout_cases", "instruction", "scoring",
                "policy", "budget", "sandbox_seconds", "command_seconds", "grader_python", "grader_seconds"}
    if type(config) is not dict or set(config) - {"student"} != required:
        raise ValueError("Trial config must contain exactly " + ", ".join(sorted(required)))
    for key in ("model_source", "public_cases", "heldout_cases", "grader_python"):
        candidate = path.parent / config[key]
        config[key] = os.path.abspath(candidate) if key == "grader_python" else str(candidate.resolve())
        if not Path(config[key]).exists():
            raise ValueError(key + " does not exist")
    private_path = Path(config["heldout_cases"])
    if (private_path.is_relative_to(config["model_source"])
            or private_path.is_relative_to(Path(__file__).resolve().parent)):
        raise ValueError("Keep held-out cases outside every model and package image input")
    snapshot = Path(config["model_source"]) / "snapshots" / REVISION
    if not (snapshot / "model.safetensors").is_file():
        raise ValueError("model_source must contain the pinned SmolLM2 snapshot")
    for file in snapshot.iterdir():
        if file.is_file() and (not file.resolve().is_relative_to(config["model_source"])
                               or file.samefile(private_path)):
            raise ValueError("Model image inputs must not link to external or held-out files")
    if type(config["instruction"]) is not str or not config["instruction"].strip():
        raise ValueError("instruction must describe the selected task")
    if config["scoring"] not in ("negative_loss", "loss_improvement"):
        raise ValueError("Select negative_loss or loss_improvement explicitly")
    for name in ("sandbox_seconds", "command_seconds", "grader_seconds"):
        if type(config[name]) is not int:
            raise ValueError(name + " must be an integer")
        _positive(config[name], name)
    if config["command_seconds"] > config["sandbox_seconds"]:
        raise ValueError("command_seconds cannot exceed sandbox_seconds")
    public = _cases(json.loads(Path(config["public_cases"]).read_text()))
    private = _cases(json.loads(Path(config["heldout_cases"]).read_text()))
    if not public or not private:
        raise ValueError("Public and held-out cases must both be nonempty")
    if {json.dumps(case, sort_keys=True) for case in public} & {json.dumps(case, sort_keys=True) for case in private}:
        raise ValueError("Public and held-out cases must not overlap")
    # Validate without constructing a network client.
    if config["policy"].get("provider") == "responses_proxy":
        from .trial_editor import validate_config
        from .trial_distillation import validate_student_config
        validate_config(config["policy"])
        validate_student_config(config.get("student"))
        training_provider = config["student"]["provider"]
    else:
        from .trial_training import validate_config
        validate_config(config["policy"])
        if "student" in config:
            raise ValueError("A separate student requires a Responses-proxy teacher")
        training_provider = config["policy"]["provider"]
    Budget.validate(config["budget"])
    expected_price = "spindle_hourly_usd" if training_provider == "spindle" else "tinker_train_per_million"
    if expected_price not in config["budget"]:
        raise ValueError("Training provider and budget pricing must agree")
    return config


class Budget:
    """Conservative pre-dispatch reservations, not a provider billing limit."""

    @staticmethod
    def validate(config):
        common = {"max_usd", "modal_hourly_usd", "modal_build_usd", "modal_egress_per_gib"}
        hosted = {"tinker_prefill_per_million", "tinker_sample_per_million", "tinker_train_per_million",
                  "tinker_storage_usd"}
        spindle = {"spindle_hourly_usd", "spindle_seconds"}
        if (type(config) is not dict
                or set(config) - {"prior_reserved_usd"} not in (common | hosted, common | spindle)):
            raise ValueError("Budget requires explicit current prices and build/storage allowances")
        for key, value in config.items():
            if key == "prior_reserved_usd" and value == 0:
                continue
            _positive(value, key)

    def __init__(self, config, path):
        self.validate(config)
        self.config, self.path = config, Path(path)
        if self.path.exists():
            raise FileExistsError("Budget journal exists; inspect it before another run")
        self.reserved = 0.0
        self.spindle_started = None
        with self.path.open("x"):
            pass
        if config.get("prior_reserved_usd", 0):
            self.reserve("prior_work", {})

    def reserve(self, operation, units):
        prices = self.config
        if operation == "prior_work":
            if self.reserved:
                raise ValueError("Prior work can only be reserved once, before new work")
            amount = prices["prior_reserved_usd"]
        elif operation == "modal_build":
            amount = prices["modal_build_usd"]
        elif operation == "modal_sandbox":
            amount = prices["modal_hourly_usd"] * units.get("seconds", 0) / 3600
        elif operation == "modal_egress":
            amount = prices["modal_egress_per_gib"] * units.get("bytes", 0) / 1024**3
        elif operation == "spindle_create":
            if self.spindle_started is not None:
                raise ValueError("Only one Spindle session is admitted per proof budget")
            amount = prices["spindle_hourly_usd"] * prices["spindle_seconds"] / 3600
        elif operation in ("spindle_snapshot", "spindle_sample", "spindle_train", "spindle_save"):
            if (self.spindle_started is None
                    or time.monotonic() - self.spindle_started > prices["spindle_seconds"]):
                raise WeightlessError("spending_limit", "No active prepaid Spindle session envelope")
            amount = 0.0
        elif operation == "tinker_create":
            if "tinker_storage_usd" not in prices:
                raise ValueError("Hosted Tinker is not priced by this budget")
            amount = 0.0
        elif operation == "teacher_request":
            # CLI-proxy usage is separately funded; this journal caps Modal compute.
            amount = 0.0
        elif operation in ("tinker_snapshot", "tinker_save"):
            amount = prices["tinker_storage_usd"]
        elif operation in ("tinker_sample", "tinker_train"):
            amount = sum(units.get(kind + "_tokens", 0) * prices["tinker_" + kind + "_per_million"]
                         for kind in ("prefill", "sample", "train")) / 1_000_000
        else:
            raise ValueError("Unpriced operation " + operation)
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in units.values()):
            raise ValueError("Reservation units must be finite and nonnegative")
        if not math.isfinite(amount) or amount < 0 or self.reserved + amount > prices["max_usd"]:
            raise WeightlessError("spending_limit", "The next operation exceeds the reserved dollar budget")
        self.reserved += amount
        if operation == "spindle_create":
            self.spindle_started = time.monotonic()
        with self.path.open("a") as stream:
            stream.write(json.dumps({"operation": operation, "units": units, "usd": amount,
                                     "reserved_usd": self.reserved}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def run_agent(trial, policy, instruction):
    """Only generated agent tokens enter the later training trajectory."""
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": instruction + "\nPublic cases:\n" + json.dumps(trial.config["cases"])}]
    for message in messages:
        trial.record_message(**message)
    trajectory = []
    for _ in range(policy.config["max_turns"]):
        trial.runner.set_history(trial.history())
        prompt = json.dumps(messages, ensure_ascii=True)
        # Record intent before any remote sampler call. Never silently retry it.
        trial.record_message("system", "Requesting next agent action")
        sample = policy.sample(prompt)
        trajectory.append(sample)
        trial.record_message("assistant", sample["text"], sample=sample)
        messages.append({"role": "assistant", "content": sample["text"]})
        try:
            action = json.loads(sample["text"])
            name = action["action"]
            if name == "python" and set(action) <= {"action", "source", "message"}:
                result = trial.execute(action["source"], message=action.get("message", ""))
            elif name == "evaluate" and set(action) == {"action"}:
                result = trial.evaluate()
            elif name == "history" and set(action) == {"action"}:
                # Exact records are on disk. Avoid recursively quoting prior history replies.
                result = [event for event in trial.history() if event["kind"] != "message"]
            elif name == "finish" and set(action) == {"action"}:
                trial.finish()
                break
            else:
                raise ValueError("Unknown action or fields")
        except (ValueError, KeyError, TypeError) as error:
            result = {"error": str(error), "instructions": "Return one valid action JSON object"}
        content = json.dumps(result, allow_nan=False)
        if len(content) > 6000:
            content = json.dumps({"truncated": True, "preview": content[:6000],
                                  "full_history": "/trial/history.json",
                                  "instruction": "Read the full result from history with a Python action if needed."})
        messages.append({"role": "tool", "content": content})
        trial.record_message("tool", content)
    else:
        trial.finish()
    trial.runner.set_history(trial.history())
    return trajectory


def run_poc(config, output, *, budget=None, app_name="wsm-coding-trial"):
    import modal
    from .trial_training import TinkerPolicy
    from .trial_grading import grade_in_process
    output = Path(output).resolve()
    output.mkdir(mode=0o700)
    if budget is None:
        budget = Budget(config["budget"], output / "budget.jsonl")
    manifest = {"format": "wsm.trial-poc.v1", "target": MODEL, "revision": REVISION,
                "state": "starting", "budget_usd": config["budget"]["max_usd"]}

    def record(**fields):
        manifest.update(fields)
        (output / "poc.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    policy, runner, student = None, None, None
    try:
        if config["policy"]["provider"] == "responses_proxy":
            from .trial_editor import ProxyEditor
            policy = ProxyEditor(config["policy"], directory=output / "editor", reserve=budget.reserve)
        else:
            policy = TinkerPolicy(config["policy"], reserve=budget.reserve)
        budget.reserve("modal_build", {})
        app = modal.App(app_name, include_source=False)
        with app.run():
            image = trial_image(config["model_source"], app)
            budget.reserve("modal_sandbox", {"seconds": config["sandbox_seconds"]})
            sandbox = modal.Sandbox.create("sleep", "infinity", app=app, image=image,
                                           gpu="T4", cpu=(2, 2), memory=(8192, 8192), workdir="/trial",
                                           timeout=config["sandbox_seconds"], block_network=True,
                                           include_oidc_identity_token=False)
            runner = ModalTrialRunner(sandbox, command_seconds=config["command_seconds"],
                                      artifacts=output / "checkpoints", reserve=budget.reserve)
            record(state="running", sandbox_id=sandbox.object_id)
            trial = Trial.create(output / "trial", runner=runner, initial_checkpoint=INITIAL_CHECKPOINT,
                                 cases=json.loads(Path(config["public_cases"]).read_text()),
                                 max_actions=config["policy"]["max_turns"])
            trajectory = run_agent(trial, policy, config["instruction"])
            with (output / "trajectory.json").open("x") as stream:
                json.dump(trajectory, stream, allow_nan=False)
            frozen = trial.finish()
            weights = output / "final.safetensors"
            runner.export_weights(frozen["checkpoint"], weights)
            runner.close()
            record(state="frozen", sandbox_terminated=True)
        # Neither agent code nor agent-supplied config enters this fresh process.
        source = Path(config["model_source"])
        if (source / "snapshots" / REVISION).is_dir():
            source = source / "snapshots" / REVISION
        grade = grade_in_process(source, weights, config["heldout_cases"], scoring=config["scoring"],
                                 python=config["grader_python"], timeout=config["grader_seconds"])
        trial.record_grade(grade)
        record(state="graded", checkpoint=grade["checkpoint"])
        if config["policy"]["provider"] == "responses_proxy":
            from .trial_distillation import StudentTrainer
            student = StudentTrainer(config["student"], reserve=budget.reserve)
            training = student.train(trajectory, grade["reward"], output=output / "training")
        else:
            training = policy.train(trajectory, grade["reward"], output=output / "training")
        record(state="complete", training=training, reserved_usd=budget.reserved)
        return manifest
    except BaseException as error:
        record(state="failed", error={"type": type(error).__name__, "message": str(error)[:2000]},
               reserved_usd=budget.reserved)
        raise
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            status = "success" if manifest["state"] == "complete" else "errored"
            try:
                if policy is not None:
                    policy.close(status)
            finally:
                if student is not None:
                    student.close(status)


def watch(directory, *, follow=False):
    """Print the public host journal. No control channel or mutation is exposed."""
    path = Path(directory) / "events.jsonl"
    with path.open() as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if line and line.endswith("\n"):
                event = json.loads(line)
                print(json.dumps(event, ensure_ascii=True), flush=True)
                if follow and event["kind"] in ("frozen", "failure"):
                    return
            elif follow:
                stream.seek(position)
                time.sleep(0.2)
            else:
                return


def resume_student(config, source_run, output, *, retry_pretraining=False):
    """Grade a frozen teacher artifact and train once without rerunning its edits."""
    from .trial_distillation import StudentTrainer
    from .trial_grading import grade_in_process
    from .program import _write_json

    source_run, output = Path(source_run).resolve(), Path(output).absolute()
    if config["policy"]["provider"] != "responses_proxy":
        raise ValueError("Recovery requires a recorded teacher/student trial")
    trial = Trial(source_run / "trial", None)
    view = trial.observation()
    intent = source_run / "student-training-intent.json"
    previous = None
    if retry_pretraining:
        previous = json.loads(intent.read_text())
        prior_output = Path(previous["output"])
        receipt = json.loads((prior_output / "training/receipt.json").read_text())
        ledger = [json.loads(line) for line in (prior_output / "budget.jsonl").read_text().splitlines()]
        cleanup = json.loads((prior_output.parent / "cleanup-confirmed.json").read_text())
        local_failure = (receipt.get("state") == "failed_before_training" or
                         (receipt.get("last_state") == "create_dispatched" and
                          receipt.get("error_type") == "ImportError"))
        if (previous.get("state") != "failed" or not local_failure
                or any(row["operation"] in {"spindle_train", "spindle_save"} for row in ledger)
                or not cleanup.get("confirmed") or cleanup.get("remaining_tasks") != 0):
            raise ValueError("Previous training outcome is uncertain; recovery refused")
    if (view["state"] != "frozen" or (source_run / "training").exists()
            or (intent.exists() and previous is None)):
        raise ValueError("Only a frozen trial without a previous training attempt can be recovered")
    trajectory = json.loads((source_run / "trajectory.json").read_text())
    recorded = [event["data"]["metadata"]["sample"] for event in trial.history()
                if event["kind"] == "message" and event["data"]["role"] == "assistant"]
    if trajectory != recorded:
        raise ValueError("Teacher trajectory differs from the canonical action journal")
    result = next(event["data"]["result"] for event in reversed(trial.history())
                  if event["kind"] == "result" and event["data"]["operation"] == "execute")
    weights = source_run / "final.safetensors"
    with weights.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if result["checkpoint"] != view["checkpoint"] or result["artifact"]["sha256"] != digest:
        raise ValueError("Frozen weights differ from the retained final checkpoint")
    output.mkdir(mode=0o700)
    budget = Budget(config["budget"], output / "budget.jsonl")
    intent_record = {"output": str(output), "checkpoint": "sha256:" + digest}
    if previous is None:
        with intent.open("x") as stream:
            json.dump(intent_record, stream)
    else:
        intent_record["previous_attempt"] = previous
        _write_json(intent, intent_record)
    manifest = {"state": "grading", "source_trial_directory": str(trial.directory),
                "checkpoint": "sha256:" + digest, "teacher_rerun": False}
    trainer = None
    try:
        source = Path(config["model_source"]) / "snapshots" / REVISION
        grade = grade_in_process(source, weights, config["heldout_cases"], scoring=config["scoring"],
                                 python=config["grader_python"], timeout=config["grader_seconds"])
        if not grade["changed_tensors"]:
            raise ValueError("The frozen proof has no actual tensor changes")
        if (trial.directory / "grade.json").exists():
            old_grade = json.loads((trial.directory / "grade.json").read_text())["result"]
            if previous is None or old_grade != grade:
                raise ValueError("Existing final grade differs; recovery refused")
        else:
            trial.record_grade(grade)
        _write_json(output / "grade.json", grade)
        manifest["state"] = "graded"
        _write_json(output / "poc.json", manifest)
        trainer = StudentTrainer(config["student"], reserve=budget.reserve)
        training = trainer.train(trajectory, grade["reward"], output=output / "training")
        manifest.update(state="complete", training=training)
        return manifest
    except BaseException as error:
        manifest.update(state="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["reserved_usd"] = budget.reserved
        _write_json(output / "poc.json", manifest)
        intent_record["state"] = manifest["state"]
        _write_json(intent, intent_record)
        if trainer is not None:
            trainer.close("success" if manifest["state"] == "complete" else "errored")


def inspect_weights(path, *, tensor=None, indices=None):
    """Read retained numeric tensors without importing or executing model code."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as weights:
        if tensor is None:
            if indices is not None:
                raise ValueError("Select a tensor before supplying indices")
            return {name: {"shape": weights.get_slice(name).get_shape(),
                           "dtype": weights.get_slice(name).get_dtype()} for name in weights.keys()}
        if indices is None or not 1 <= len(indices) <= 64 or any(index < 0 for index in indices):
            raise ValueError("Supply between 1 and 64 nonnegative flat tensor indices")
        values = weights.get_tensor(tensor).reshape(-1)
        if any(index >= values.numel() for index in indices):
            raise ValueError("Tensor index out of range")
        return {"tensor": tensor, "indices": indices, "values": values[indices].tolist()}
