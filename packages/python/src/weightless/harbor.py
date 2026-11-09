import asyncio
import copy
from importlib.metadata import version
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import threading

from .binary64 import _canonical, _fail, _object, parse_checkpoint
from .core import (_binding, _config, _digest, _json, _read, _request, _same,
                   evaluation_result, evaluator_config, replay)


def resolve_image(reference):
    """Explicitly resolve a local Docker image before creating an evaluator."""
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", reference],
                            capture_output=True, text=True, check=True, timeout=30)
    image = result.stdout.strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        _fail("invalid_config", "Docker did not return an immutable image ID")
    return image


def _task_digest(directory):
    files = [directory / "task.toml", directory / "instruction.md"]
    files += sorted(path for path in (directory / "tests").rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    return _digest("weightless.harbor.package.v1", {
        path.relative_to(directory).as_posix(): path.read_bytes().hex() for path in files})


class HarborEvaluation:
    """Trusted purpose-specific task package, isolated grading, and native evidence."""

    def __init__(self, *, task_dir, image):
        if type(image) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
            _fail("invalid_config", "Pass resolve_image(tag), an immutable Docker image ID")
        from verifiers.v1.tasksets.harbor.taskset import HarborConfig, HarborTask, parse_task
        self._directory = Path(task_dir).resolve()
        manifest = _read((self._directory / "tests/manifest.json").read_text(), "invalid_config")
        _object(manifest, ("format", "purpose", "problem", "initial", "policy", "split"), "invalid_config")
        if manifest["format"] != "weightless.harbor-task.v1" or manifest["purpose"] not in ("development", "final"):
            _fail("invalid_config", "Invalid trusted task manifest")
        config = HarborConfig(ignore_dockerfile=True, ignore_timeouts=False)
        task = HarborTask(parse_task(self._directory, 0, config), config.task)
        if (task.scoring.mode != "isolated" or task.data.verifier is None
                or not task.data.verifier.fresh_copy or task.data.verifier_image is not None
                or task.data.network.allow != []
                or [a.source for a in task.data.artifacts] != ["/app/submission.json"]):
            _fail("invalid_config", "Task must use the declared isolated, no-network JSON boundary")
        self._task = task.with_data(image=image, upload_environment=False)
        self._config = evaluator_config("harbor", manifest["purpose"], {
            "manifest": manifest, "package": _task_digest(self._directory), "image": image,
            "harbor": version("harbor"), "verifiers": version("verifiers"),
            "timeout": task.data.timeout.model_dump(mode="json"),
            "resources": task.data.resources.model_dump(mode="json"),
            "network": task.data.network.model_dump(mode="json"), "retries": 0,
            "rewardKey": task.config.reward_key})
        self._lock = threading.Lock()

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def __call__(self, request, checkpoint):
        if self._config["purpose"] != "development":
            _fail("invalid_config", "Final evaluator accepts only grade(frozen)")
        checkpoint = parse_checkpoint(checkpoint)
        request = _request(_json(request, "invalid_result"))
        if request["checkpoint"] != checkpoint["id"]:
            _fail("invalid_result", "Checkpoint differs from request")
        return self._execute(request, {"format": "weightless.development.v1",
                                       "request": request, "checkpoint": checkpoint})

    def grade(self, frozen):
        if self._config["purpose"] != "final":
            _fail("invalid_config", "Development evaluator accepts candidate checkpoints only")
        frozen = _read(frozen, "invalid_artifact")
        checkpoint = replay(frozen)
        request = _binding(frozen["run"], frozen["used"], checkpoint["id"], self._config)
        return self._execute(request, frozen)

    def _execute(self, request, submission):
        if request["purpose"] != self._config["purpose"] or request["config"] != self._config["id"]:
            _fail("invalid_result", "Evaluator purpose or configuration differs from request")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            _fail("invalid_state", "HarborEvaluation requires a synchronous caller outside an active event loop")
        if not self._lock.acquire(blocking=False):
            _fail("invalid_state", "Harbor evaluator already has an operation in progress")
        try:
            return asyncio.run(self._grade(request, submission))
        finally:
            self._lock.release()

    async def _grade(self, request, submission):
        from verifiers.v1.configs.agent import AgentConfig
        from verifiers.v1.configs.retries import RetryConfig
        from verifiers.v1.configs.runtime import DockerConfig
        from verifiers.v1.grading import grade_task
        from verifiers.v1.runtimes import backend_cls
        from verifiers.v1.runtimes.deployment import Deployment
        from verifiers.v1.tasksets.harbor.taskset import HarborTask, verifier_box_data
        from verifiers.v1.trace import AgentInfo, Trace, TraceTask
        from verifiers.v1.utils.compile import resolve_runtime_config

        diagnostics, native_rewards, trace, deployment = {}, None, None, None
        status, error = "infrastructure_error", None
        try:
            if _task_digest(self._directory) != self._config["settings"]["package"]:
                _fail("invalid_config", "Trusted task package changed after configuration")
            policy = self._task.scoring.model_copy(update={"retries": RetryConfig(max_retries=0),
                "env": {"WEIGHTLESS_CONFIG": _canonical(self._config).decode(),
                        "WEIGHTLESS_REQUEST": _canonical(request).decode()}})
            task = HarborTask(verifier_box_data(self._task.data), self._task.config).with_data(scoring=policy)
            config = resolve_runtime_config(DockerConfig(image=self._config["settings"]["image"]), task)
            trace = Trace(task=TraceTask(type="harbor", data=task.data),
                          agent=AgentInfo(config=AgentConfig(runtime=config)))
            buffer = io.BytesIO()
            payload = _canonical(submission)
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                member = tarfile.TarInfo("app/submission.json")
                member.size, member.mode = len(payload), 0o644
                archive.addfile(member, io.BytesIO(payload))
            trace.state.artifacts = {"/app/submission.json": buffer.getvalue()}
            deployment = Deployment(backend_cls(config)(config))
            await deployment.start()

            async def capture(result, runtime):
                nonlocal native_rewards
                diagnostics["execution"] = result.info.get("scoring_diagnostics", {}).get("harbor", {})
                diagnostics["upstreamMetrics"] = dict(result.metrics)
                diagnostics["upstreamRewards"] = {key: value.model_dump(mode="json") if value else None
                                                   for key, value in result.rewards.items()}
                native_rewards = await task.read_reward_json(runtime)
                if native_rewards is None:
                    # Preserve the native scalar fallback through the upstream parsed reward.
                    reward = result.rewards.get(task.config.reward_key)
                    native_rewards = {task.config.reward_key: reward.score} if reward else None
                try:
                    raw = await runtime.read("/logs/verifier/diagnostics.json", max_bytes=131072)
                    diagnostics["verifier"] = _read(raw.decode(), "invalid_result")
                except Exception as failure:
                    diagnostics["captureError"] = {"type": type(failure).__name__, "message": str(failure)[:8192]}
                    if "raw" in locals():
                        diagnostics["diagnosticsRaw"] = raw.decode(errors="replace")[:131072]
                    _fail("invalid_result", "Invalid native diagnostics: " + str(failure)[:8192])

            await grade_task(task, trace, config, task.data.timeout, policy, deployment, on_score=capture)
            verifier = diagnostics.get("verifier")
            if type(verifier) is not dict:
                _fail("invalid_result", "Missing native diagnostics")
            if (not _same(verifier.get("request"), request)
                    or verifier.get("checkpoint") != request["checkpoint"]
                    or verifier.get("manifest") != self._config["settings"]["manifest"]):
                _fail("invalid_result", "Native verifier binding mismatch")
            if verifier.get("status") == "graded":
                status, error = "completed", None
            elif verifier.get("status") in ("rejected", "task_failure"):
                status = "task_failure"
                error = {"type": verifier.get("code", "task_failure"),
                         "message": str(verifier.get("error", "Verifier rejected submission"))[:8192]}
            else:
                _fail("invalid_result", "Unknown native verifier status")
        except Exception as failure:
            executed = trace is not None and trace.info.get("scoring_diagnostics", {}).get("harbor")
            malformed = (getattr(failure, "code", None) == "invalid_result"
                         or "captureError" in diagnostics or (executed and native_rewards is None))
            status = "invalid_response" if malformed else "infrastructure_error"
            error = {"type": type(failure).__name__, "message": str(failure)[:8192]}
        finally:
            if trace is not None:
                diagnostics.setdefault("execution", trace.info.get("scoring_diagnostics", {}).get("harbor", {}))
            if deployment is not None:
                try:
                    await deployment.close()
                except Exception as failure:
                    diagnostics["cleanupError"] = {"type": type(failure).__name__, "message": str(failure)[:8192]}
                    status, error = "infrastructure_error", diagnostics["cleanupError"]
        verifier = diagnostics.get("verifier")
        verifier = verifier if type(verifier) is dict else {}
        try:
            return evaluation_result(request, status=status, metrics=verifier.get("metrics", {}),
                                     reward=native_rewards, native_status=verifier.get("status"),
                                     diagnostics=diagnostics, error=error)
        except (ValueError, TypeError, OverflowError, RecursionError) as failure:
            failed_status = "infrastructure_error" if "cleanupError" in diagnostics else "invalid_response"
            raw = _canonical(diagnostics).decode()
            diagnostics = {"evidenceRaw": raw[:32768], "evidenceTruncated": len(raw) > 32768}
            native_status = verifier.get("status")
            if type(native_status) is not str or len(native_status) > 256:
                native_status = None
            return evaluation_result(request, status=failed_status, metrics={}, reward=native_rewards,
                                     native_status=native_status, diagnostics=diagnostics,
                                     error={"type": type(failure).__name__, "message": str(failure)[:8192]})
