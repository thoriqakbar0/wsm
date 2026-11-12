"""Public-only editing executors. This module never receives held-out paths."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import signal
import time

from .benchmark import digest, read, write
from .errors import WeightlessError
from .trial import LocalTrialRunner, Trial


@contextmanager
def _deadline(seconds):
    def expired(signum, frame):
        raise TimeoutError("Trial wall deadline reached")
    previous = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, previous)


class _ActionLimits:
    def __init__(self, runner, evaluations):
        self.runner = runner
        self.remaining = evaluations

    def execute(self, source, checkpoint, cases):
        return self.runner.execute(source, checkpoint, cases)

    def evaluate(self, checkpoint, cases):
        if self.remaining <= 0:
            raise WeightlessError("evaluation_limit", "Public evaluation action limit exhausted")
        self.remaining -= 1
        return self.runner.evaluate(checkpoint, cases)

    def set_history(self, events):
        self.runner.set_history(events)


class _RecordedPolicy:
    def __init__(self, actions, limits):
        self.actions = iter(actions)
        self.config = {"max_turns": limits["turns"]}
        self.tokens = limits["output_tokens"] * limits["turns"]

    def sample(self, prompt):
        action = next(self.actions, {"action": "finish"})
        text = json.dumps(action)
        # Synthetic action tapes have no tokenizer. Bound bytes and label them synthetic.
        if len(text.encode()) > self.tokens:
            raise ValueError("Synthetic action tape exceeds its byte allowance")
        self.tokens -= len(text.encode())
        return {"prompt": prompt, "text": text, "synthetic": True}


def _edit(runner, initial, public, instruction, script, policy, limits, folder):
    from .trial_poc import INSTRUCTIONS, run_agent
    trial = Trial.create(folder / "trial", runner=_ActionLimits(runner, limits["evaluations"]),
                         initial_checkpoint=initial, cases=public, max_actions=limits["turns"])
    if script is not None:
        trial.execute(script, message="Frozen scripted baseline")
        trial.finish()
    else:
        system = INSTRUCTIONS[:INSTRUCTIONS.index("The fixed target")]
        system += "Only public cases are available. Finish before independent private evaluation. No training occurs."
        run_agent(trial, policy, instruction, system_instruction=system)
    return trial.finish()["checkpoint"]


def execute(*, source, public, instruction, script, runtime, limits, folder):
    """Execute with public inputs only; private campaign data stays with the controller."""
    if runtime["backend"] == "local_trusted":
        return _local(source, public, instruction, script,
                      read(runtime["actions"]["path"]), limits, folder)
    return _modal(source, public, instruction, script,
                  read(runtime["policy"]["path"]), limits, folder)


def _local(source, public, instruction, script, actions, limits, folder):
    # These programs are trusted fixtures and retain the invoking user's permissions.
    runner = LocalTrialRunner(folder / "worker", timeout=limits["command_seconds"])
    with _deadline(limits["wall_seconds"]):
        checkpoint = _edit(runner, source, public, instruction, script,
                           _RecordedPolicy(actions, limits), limits, folder)
        weights = folder / "final.safetensors"
        with (Path(checkpoint) / "model.safetensors").open("rb") as incoming, weights.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    return {"weights": str(weights), "sha256": digest(weights), "editor_closed": True, "synthetic": True}


def _modal(source, public, instruction, script, policy_config, limits, folder):
    import modal
    from .modal_trial import INITIAL_CHECKPOINT, REVISION, ModalTrialRunner, trial_image
    from .trial_editor import ProxyEditor
    source = Path(source)
    if source.name != REVISION or source.parent.name != "snapshots":
        raise ValueError("Modal snapshot must use snapshots/<pinned-revision> layout")
    weights = folder / "final.safetensors"
    runner, policy = None, None
    started = time.monotonic()
    intent = {"phase": "provisioning", "sandbox_id": None}
    write(folder / "modal-intent.json", intent)
    try:
        app = modal.App("wsm-benchmark", include_source=False)
        with app.run():
            with _deadline(limits["wall_seconds"]):
                image = trial_image(source.parent.parent, app)
                sandbox = modal.Sandbox.create("sleep", "infinity", app=app, image=image,
                    gpu="T4", cpu=(2, 2), memory=(8192, 8192), workdir="/trial",
                    timeout=limits["wall_seconds"], block_network=True, include_oidc_identity_token=False)
                write(folder / "sandbox.json", {"id": sandbox.object_id})
                # The whole build, sandbox, request, and export envelope was reserved before dispatch.
                runner = ModalTrialRunner(sandbox, command_seconds=limits["command_seconds"],
                                          artifacts=folder / "checkpoints", reserve=lambda operation, units: None)
                if script is None:
                    policy = ProxyEditor(policy_config, directory=folder / "editor",
                                         reserve=lambda operation, units: None)
                checkpoint = _edit(runner, INITIAL_CHECKPOINT, public, instruction, script, policy, limits, folder)
                runner.export_weights(checkpoint, weights)
            runner.close()
    finally:
        if runner is not None and not runner.closed:
            runner.close()
        if runner is not None and runner.closed:
            write(folder / "closed.json", {"sandbox_id": runner.sandbox.object_id, "confirmed": True})
        if policy is not None:
            policy.close()
    if time.monotonic() - started > limits["wall_seconds"]:
        raise TimeoutError("Trial exceeded wall allowance")
    return {"weights": str(weights), "sha256": digest(weights), "editor_closed": True, "synthetic": False}
