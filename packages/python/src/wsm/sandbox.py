"""Shared numerical-worker protocol and one-shot Docker execution."""
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import uuid

from .binary64 import WeightlessError

_MAX_INPUT = 4 * 1024 * 1024
_MAX_OUTPUT = 4 * 1024 * 1024
_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
_GENERATIVE = ("smollm", "huggingface")


def _error(code, message):
    raise WeightlessError(code, message)


def forward_reservation(target, case_count, *, generate=False, max_new_tokens=16):
    """Conservative top-level target invocations, independent of observed usage."""
    if (type(case_count) is not int or case_count < 0 or type(generate) is not bool
            or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128):
        _error("invalid_request", "Invalid forward reservation")
    return case_count * (1 + max_new_tokens if generate and target.get("kind") in _GENERATIVE else 1)


def stage_smollm(source, destination):
    source, destination = Path(source), Path(destination)
    snapshot = source / "snapshots" / _REVISION
    if destination.exists() or not (snapshot / "model.safetensors").is_file():
        _error("invalid_target", "A pinned source and unused staging directory are required")
    staged = destination / "snapshots" / _REVISION
    staged.mkdir(parents=True, mode=0o755)
    for path in snapshot.iterdir():
        if path.is_file() and path.suffix in (".json", ".txt", ".model", ".jinja", ".safetensors"):
            shutil.copyfile(path.resolve(strict=True), staged / path.name)
            (staged / path.name).chmod(0o444)
    return {"kind": "smollm", "source": str(destination.resolve()), "revision": _REVISION,
            "model": "HuggingFaceTB/SmolLM2-135M-Instruct", "dtype": "float32"}


def _nested(parent, child):
    parent = os.path.realpath(parent)
    child = os.path.realpath(child)
    try:
        return os.path.commonpath((parent, child)) == parent
    except ValueError:
        return False


def _copy_snapshot(snapshot, staged, source_root):
    from .target_worker import resolved_snapshot_file

    for dirpath, dirnames, filenames in os.walk(snapshot, followlinks=False):
        relative_dir = Path(dirpath).relative_to(snapshot)
        for dirname in list(dirnames):
            if os.path.islink(os.path.join(dirpath, dirname)):
                _error("invalid_target", "Snapshot directory links are unsupported")
        for name in filenames:
            source = Path(dirpath) / name
            resolved = Path(resolved_snapshot_file(str(source_root), str(source)))
            copied = staged / relative_dir / name
            copied.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            shutil.copyfile(resolved, copied)
            copied.chmod(0o444)


def stage_model(source, destination, *, model, revision):
    """Copy a local snapshot at ``source/snapshots/<revision>`` into an unused directory."""
    from .target_worker import huggingface_snapshot, model_id_ok, revision_ok

    if not model_id_ok(model) or not revision_ok(revision):
        _error("invalid_target", "A model id and immutable revision are required")
    try:
        root = Path(source).resolve(strict=True)
    except (OSError, RuntimeError):
        _error("invalid_target", "A local snapshot and unused staging directory are required")
    if not root.is_dir():
        _error("invalid_target", "A local snapshot and unused staging directory are required")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        _error("invalid_target", "A local snapshot and unused staging directory are required")
    if _nested(root, destination) or _nested(destination, root):
        _error("invalid_target", "Staging destination must stay outside the source")
    snapshot = Path(huggingface_snapshot(str(root), revision))
    created = False
    try:
        destination.mkdir(parents=True)
        created = True
        staged = destination / "snapshots" / revision
        staged.mkdir(parents=True, mode=0o755)
        _copy_snapshot(snapshot, staged, root)
    except BaseException:
        if created:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    return {"kind": "huggingface", "source": str(destination.resolve()), "revision": revision,
            "model": model, "dtype": "float32"}


def _model_mount(target):
    from .target_worker import huggingface_snapshot, model_id_ok, revision_ok

    kind = target.get("kind")
    if kind == "smollm":
        if (set(target) != {"kind", "source", "revision", "model", "dtype"}
                or target["revision"] != _REVISION or target["model"] != "HuggingFaceTB/SmolLM2-135M-Instruct"
                or target["dtype"] != "float32" or type(target["source"]) is not str):
            _error("invalid_target", "Unsupported model configuration")
    elif kind == "huggingface":
        if (set(target) != {"kind", "source", "revision", "model", "dtype"}
                or target["dtype"] != "float32" or type(target["source"]) is not str
                or not model_id_ok(target["model"]) or not revision_ok(target["revision"])):
            _error("invalid_target", "Unsupported model configuration")
    else:
        return None
    source = target.pop("source")
    if not os.path.isabs(source):
        _error("invalid_target", "Model source must be absolute")
    mount = os.path.realpath(source)
    if not os.path.isabs(mount) or not os.path.isdir(mount) or "," in mount:
        _error("invalid_target", "Invalid model source")
    if kind == "huggingface":
        huggingface_snapshot(mount, target["revision"], local_files=True)
    target["source"] = "/target"
    return mount


class _WorkerRunner:
    def execute(self, action, target, *, patches=None, edits=None, cases=None, query=None,
                generate=False, purpose="development", timeout=None, on_dispatch=None, forward_limit=None):
        if action not in ("describe", "inspect", "materialize", "evaluate", "infer") or purpose not in ("development", "final") or type(generate) is not bool:
            _error("invalid_request", "Unsupported sandbox operation")
        if type(target) is not dict:
            _error("invalid_target", "Invalid target")
        if forward_limit is not None and (action not in ("evaluate", "infer") or type(forward_limit) is not int or forward_limit < 0):
            _error("invalid_request", "Invalid forward limit")
        target = copy.deepcopy(target)
        mount = _model_mount(target)
        if mount is None and (target.get("kind") != "binary64" or set(target) != {"kind", "checkpoint"}):
            _error("invalid_target", "Unsupported target")
        request = {"action": action, "target": target, "patches": [] if patches is None else patches, "edits": [] if edits is None else edits,
                   "cases": cases, "query": query, "generate": generate, "purpose": purpose}
        if forward_limit is not None:
            request["forward_limit"] = forward_limit
        try:
            payload = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
        except (ValueError, TypeError, RecursionError):
            _error("invalid_request", "Request is not finite JSON")
        raw = self._run(self._command(mount) + ["python", "-m", "wsm.target_worker"], payload, timeout, on_dispatch)
        try:
            response = json.loads(raw)
            if type(response) is not dict or set(response) not in ({"result"}, {"error"}):
                raise ValueError()
            if "error" in response:
                error = response["error"]
                if type(error) is not dict or type(error.get("code")) is not str or type(error.get("message")) is not str:
                    raise ValueError()
                _error(error["code"][:64], error["message"][:256])
            if type(response["result"]) is not dict:
                raise ValueError()
            return response["result"]
        except WeightlessError:
            raise
        except (ValueError, TypeError, UnicodeDecodeError):
            _error("sandbox_invalid_response", "Sandbox returned invalid JSON")

    def probe(self, source, *, timeout=None, input_data=b"", on_dispatch=None):
        """Run a restricted Python probe without target or private mounts."""
        if type(source) is not str or len(source.encode()) > 65536:
            _error("invalid_request", "Invalid probe")
        if type(input_data) is not bytes:
            _error("invalid_request", "Probe input must be bytes")
        return self._run(self._command() + ["python", "-I", "-c", source], input_data, timeout, on_dispatch).decode("utf-8", "replace")


class DockerRunner(_WorkerRunner):
    def __init__(self, image, *, docker="docker", host=None, timeout=60, cpus=2, memory="2g"):
        if (not isinstance(image, str) or not image or not isinstance(docker, str) or not docker
                or (host is not None and (not isinstance(host, str) or not host))
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
                or type(cpus) not in (int, float) or not math.isfinite(cpus) or cpus <= 0
                or not isinstance(memory, str) or not memory):
            _error("invalid_config", "Invalid sandbox configuration")
        self._docker, self._host = docker, host
        self._timeout, self._cpus, self._memory = timeout, cpus, memory
        try:
            result = subprocess.run(self._prefix() + ["image", "inspect", "--format", "{{.Id}}", image],
                                    capture_output=True, timeout=timeout, check=True, text=True)
        except (OSError, subprocess.SubprocessError):
            _error("sandbox_unavailable", "Docker image is unavailable")
        image_id = result.stdout.strip()
        if not (len(image_id) == 71 and image_id.startswith("sha256:") and
                all(c in "0123456789abcdef" for c in image_id[7:])):
            _error("sandbox_unavailable", "Docker returned an invalid image identity")
        self._config = {"mode": "enforced", "image": image_id, "timeout": timeout,
                        "cpus": cpus, "memory": memory, "pids": 128,
                        "uid": "65534:65534", "network": "none", "read_only": True}

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def _prefix(self):
        return [self._docker] + (["--host", self._host] if self._host else [])

    def _run(self, command, payload, timeout=None, on_dispatch=None):
        if len(payload) > _MAX_INPUT:
            _error("invalid_request", "Sandbox input exceeds limit")
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            _error("invalid_request", "Invalid execution deadline")
        limit = self._timeout if timeout is None else min(self._timeout, timeout)
        try:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            _error("sandbox_unavailable", "Docker could not start")
        outputs = [bytearray(), bytearray()]
        overflow = threading.Event()
        def read(stream, result):
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                if len(result) + len(chunk) > _MAX_OUTPUT:
                    overflow.set()
                    proc.kill()
                    break
                result.extend(chunk)
        threads = [threading.Thread(target=read, args=(proc.stdout, outputs[0]), daemon=True),
                   threading.Thread(target=read, args=(proc.stderr, outputs[1]), daemon=True)]
        for thread in threads:
            thread.start()
        def write():
            try:
                proc.stdin.write(payload)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        timed_out = False
        cleanup_unknown = False
        try:
            if on_dispatch is not None:
                on_dispatch()
            proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            raise
        finally:
            name = command[command.index("--name") + 1]
            try:
                removed = subprocess.run(self._prefix() + ["rm", "--force", name], capture_output=True, timeout=10)
                if removed.returncode:
                    detail = removed.stderr.decode(errors='replace').lower()
                    cleanup_unknown = not any(message in detail for message in ('no such container', 'no such object'))
            except (OSError, subprocess.SubprocessError):
                cleanup_unknown = True
            writer.join(timeout=2)
            for thread in threads:
                thread.join(timeout=2)
        if cleanup_unknown or writer.is_alive() or any(thread.is_alive() for thread in threads):
            _error("sandbox_cleanup_unknown", "Sandbox stream cleanup did not finish")
        if timed_out:
            _error("sandbox_timeout", "Sandbox execution timed out")
        if overflow.is_set():
            _error("sandbox_output_limit", "Sandbox output exceeds limit")
        if proc.returncode:
            _error("sandbox_failure", "Sandbox execution failed")
        return bytes(outputs[0])

    def _command(self, mount=None):
        args = self._prefix() + ["run", "--rm", "-i", "--name", "weightless-" + uuid.uuid4().hex, "--network", "none", "--read-only",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
                "--pids-limit", "128", "--memory", self._memory, "--memory-swap", self._memory,
                "--log-driver", "none", "--cpus", str(self._cpus),
                "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=64m"]
        if mount:
            args += ["--mount", "type=bind,src=" + mount + ",dst=/target,readonly"]
        return args + [self._config["image"]]
