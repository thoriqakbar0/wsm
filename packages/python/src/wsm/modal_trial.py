"""One isolated Modal sandbox per coding trial. Credentials remain on the host."""

from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import shutil
import uuid

from .binary64 import WeightlessError

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
INITIAL_CHECKPOINT = f"/target/snapshots/{REVISION}"


class ModalTrialRunner:
    """Transport for agent-controlled execution, never an authoritative grader."""

    def __init__(self, sandbox, *, command_seconds, artifacts, reserve,
                 max_output_bytes=8 * 1024**2):
        self.sandbox = sandbox
        self.command_seconds = command_seconds
        self.max_output_bytes = max_output_bytes
        self.closed = False
        self.artifacts = Path(artifacts)
        self.artifacts.mkdir(mode=0o700)
        self.reserve = reserve
        self.checkpoints = {}

    def _read_stream(self, stream):
        chunks, size = [], 0
        for chunk in stream:
            size += len(chunk.encode("utf-8"))
            if size > self.max_output_bytes:
                self.close()
                raise WeightlessError("output_limit", "Sandbox output exceeded the trial limit")
            chunks.append(chunk)
        return "".join(chunks)

    def _request(self, request):
        if self.closed:
            raise WeightlessError("trial_closed", "Sandbox has terminated")
        token = uuid.uuid4().hex
        request_path = f"/tmp/{token}.request.json"
        response_path = f"/tmp/{token}.response.json"
        self.sandbox.filesystem.write_text(json.dumps(request, allow_nan=False), request_path)
        command = ["python", "-m", "wsm.trial_worker", request_path, response_path]
        process = self.sandbox.exec(*command,
                                    timeout=self.command_seconds, workdir="/trial")
        with ThreadPoolExecutor(max_workers=2) as pool:
            stdout = pool.submit(self._read_stream, process.stdout)
            stderr = pool.submit(self._read_stream, process.stderr)
            process.wait()
            streams = {"stdout": stdout.result(), "stderr": stderr.result()}
        try:
            raw = self.sandbox.filesystem.read_bytes(response_path)
            if len(raw) > self.max_output_bytes:
                raise ValueError("Sandbox response exceeded the trial limit")
            response = json.loads(raw)
            if type(response) is not dict:
                raise ValueError("Expected a sandbox response object")
        except Exception as error:
            from .trial import TrialExecutionError
            raise TrialExecutionError("Sandbox response unavailable", {"returncode": process.returncode,
                                      "command": command, **streams, "error_type": type(error).__name__}) from error
        response.update(streams)
        response["transport_command"] = command
        response["trust"] = "agent-controlled observation"
        if process.returncode or "error" in response:
            from .trial import TrialExecutionError
            raise TrialExecutionError("Sandbox command failed", {"worker": response,
                                      "returncode": process.returncode, **streams})
        return response

    def execute(self, source, checkpoint, cases):
        output = f"/trial/edit-{uuid.uuid4().hex}"
        result = self._request({"action": "execute", "source": source, "checkpoint": checkpoint,
                                "cases": cases, "output": output, "device": "cuda"})
        if result.get("checkpoint") != output + "/checkpoint":
            from .trial import TrialExecutionError
            raise TrialExecutionError("Sandbox returned an unexpected checkpoint path", result)
        artifact = self.artifacts / (Path(output).name + ".safetensors")
        try:
            self._download_weights(result["checkpoint"], artifact)
        except Exception as error:
            from .trial import TrialExecutionError
            raise TrialExecutionError("Checkpoint retention failed", result) from error
        self.checkpoints[result["checkpoint"]] = artifact
        with artifact.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        result["artifact"] = {"path": str(artifact), "sha256": digest, "bytes": artifact.stat().st_size}
        return result

    def evaluate(self, checkpoint, cases):
        return self._request({"action": "evaluate", "checkpoint": checkpoint,
                              "cases": cases, "device": "cuda"})

    def set_history(self, events):
        self.sandbox.filesystem.write_text(json.dumps(events, allow_nan=False), "/trial/history.json")

    def export_weights(self, checkpoint, destination, *, max_bytes=600 * 1024**2):
        if checkpoint in self.checkpoints:
            with self.checkpoints[checkpoint].open("rb") as source, Path(destination).open("xb") as target:
                shutil.copyfileobj(source, target)
            return
        self._download_weights(checkpoint, destination, max_bytes=max_bytes)

    def _download_weights(self, checkpoint, destination, *, max_bytes=600 * 1024**2):
        """Export inert weights only. The trusted grader supplies config and tokenizer."""
        path = Path(checkpoint)
        if (not path.is_absolute() or ".." in path.parts
                or not (str(path).startswith("/trial/edit-") or str(path) == INITIAL_CHECKPOINT)):
            raise WeightlessError("invalid_checkpoint", "Final checkpoint must belong to this trial")
        self.reserve("modal_egress", {"bytes": max_bytes})
        data = self.sandbox.filesystem.read_bytes(str(path / "model.safetensors"))
        if len(data) > max_bytes:
            raise WeightlessError("checkpoint_limit", "Final weights exceed the artifact limit")
        with Path(destination).open("xb") as handle:
            handle.write(data)

    def close(self):
        if not self.closed:
            self.sandbox.terminate(wait=True)
            self.closed = True


def trial_image(source, app):
    """Build only the public runtime and pinned source, never the caller's directory."""
    import modal
    from .sandbox import stage_smollm
    import tempfile

    # stage_smollm validates the revision and removes symlink/cache dependencies.
    with tempfile.TemporaryDirectory(prefix="wsm-trial-source-") as temporary:
        staged = Path(temporary) / "target"
        stage_smollm(source, staged)
        package = Path(__file__).resolve().parent
        image = (modal.Image.debian_slim(python_version="3.12")
                 .uv_pip_install("torch==2.10.0+cu128", index_url="https://download.pytorch.org/whl/cu128")
                 .uv_pip_install("transformers==5.22.2", "safetensors==0.8.0")
                 .env({"PYTHONPATH": "/opt/wsm", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false"})
                 .run_commands("mkdir -p /trial")
                 .add_local_dir(package, "/opt/wsm/wsm", copy=True,
                                ignore=["**/__pycache__/**", "**/*.pyc"])
                 .add_local_dir(staged, "/target", copy=True))
        return image.build(app)
