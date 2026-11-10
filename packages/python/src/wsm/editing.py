"""Experimental live-tensor editing.

`WeightModel` exposes the parameters of a local causal language model as ordinary
PyTorch tensors. Checkpoint publication uses the same lifecycle as program
execution: ready, running, staging, verified, then published. `failure` and
`interrupted` are reachable only from running, staging, or verified. `published`
is terminal.

An operating-system crash can leave a prepublication staging directory and a
journal whose state is not published. Those artifacts are not a completed run.
A later error must not relabel a checkpoint whose directory was already
published.
"""

import contextlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .errors import WeightlessError

_OFFLINE = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
_UNSAFE_SUFFIXES = {".py", ".pyc", ".so", ".bin", ".pt", ".pth", ".pkl", ".pickle"}
_FORWARD = {"ready": "running", "running": "staging", "staging": "verified", "verified": "published"}


class _Lifecycle:
    """One publication statechart for a run journal and for a checkpoint directory."""

    def __init__(self, on_phase=None):
        self.state = "ready"
        self.verified = False
        self.published = False
        self._on_phase = on_phase
        self._notify()

    def _notify(self):
        if self._on_phase is not None:
            self._on_phase(self.state)

    def advance(self, state):
        if _FORWARD.get(self.state) != state:
            raise RuntimeError(f"Cannot move from {self.state} to {state}")
        if state == "verified":
            self.verified = True
        if state == "published":
            if not self.verified:
                raise RuntimeError("Publication requires verification")
            self.published = True
        self.state = state
        self._notify()

    def stop(self, state):
        if self.published or self.state == "published":
            return
        if self.state not in ("running", "staging", "verified") or state not in ("failure", "interrupted"):
            return
        self.state = state
        self._notify()


class _Weights(Mapping):
    """Live parameter views, including tied names, detached from autograd."""

    def __init__(self, model):
        self._model = model

    def _parameters(self):
        return dict(self._model.named_parameters(remove_duplicate=False))

    def __getitem__(self, name):
        try:
            tensor = self._parameters()[name]
        except KeyError:
            raise KeyError(name) from None
        return tensor.detach()

    def __iter__(self):
        return iter(self._parameters())

    def __len__(self):
        return len(self._parameters())


def _libraries():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise WeightlessError("unsupported_target", "Model dependencies are unavailable") from error
    return torch, AutoModelForCausalLM, AutoTokenizer


def _cases(cases):
    if cases is None:
        return []
    if type(cases) is not list:
        raise WeightlessError("invalid_cases", "Cases must be a list of prompt and target strings")
    normalized = []
    for case in cases:
        if type(case) is not dict or set(case) != {"prompt", "target"}:
            raise WeightlessError("invalid_cases", "Each case needs prompt and target strings")
        prompt, target = case["prompt"], case["target"]
        if type(prompt) is not str or type(target) is not str:
            raise WeightlessError("invalid_cases", "Each case needs prompt and target strings")
        normalized.append({"prompt": prompt, "target": target})
    return normalized


@contextlib.contextmanager
def _offline():
    saved = {key: os.environ.get(key) for key in _OFFLINE}
    for key in _OFFLINE:
        os.environ[key] = "1"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _require_finite(model):
    torch, _, _ = _libraries()
    named = list(model.named_parameters(remove_duplicate=False))
    named.extend(model.named_buffers(remove_duplicate=False))
    for name, tensor in named:
        if not tensor.dtype.is_floating_point or tensor.numel() == 0:
            continue
        if not bool(torch.isfinite(tensor).all().item()):
            raise WeightlessError("nonfinite", f"Nonfinite values in {name}")


def _snapshot(model):
    parameters = {}
    groups = {}
    for name, tensor in model.named_parameters(remove_duplicate=False):
        if tensor.dtype != _float32():
            raise WeightlessError("unsupported_target", f"{name} must be float32 before saving")
        parameters[name] = tensor.detach().to(device="cpu").contiguous().clone()
        key = (tensor.untyped_storage().data_ptr(), int(tensor.storage_offset()))
        groups.setdefault(key, []).append(name)
    ties = [names for names in groups.values() if len(names) > 1]
    parameter_names = set(parameters)
    buffers = {}
    for name, tensor in model.state_dict().items():
        if name in parameter_names or not _is_tensor(tensor):
            continue
        buffers[name] = tensor.detach().to(device="cpu").contiguous().clone()
    return parameters, buffers, ties


def _float32():
    torch, _, _ = _libraries()
    return torch.float32


def _is_tensor(value):
    torch, _, _ = _libraries()
    return torch.is_tensor(value)


def _compare(loaded, parameters, buffers, ties):
    torch, _, _ = _libraries()
    found = dict(loaded.named_parameters(remove_duplicate=False))
    if set(found) != set(parameters):
        raise WeightlessError("verification_failed", "Reloaded parameters do not match the saved model")
    for name, expected in parameters.items():
        actual = found[name].detach().to(device="cpu").contiguous()
        if not torch.equal(expected, actual):
            raise WeightlessError("verification_failed", f"Reloaded parameter {name} differs")
    pointers = {}
    for name, tensor in found.items():
        pointers[name] = (tensor.data_ptr(), int(tensor.storage_offset()))
    for names in ties:
        if len({pointers[name] for name in names}) != 1:
            raise WeightlessError("verification_failed", "Tied parameters were not preserved")
    reloaded_buffers = {}
    parameter_names = set(found)
    for name, tensor in loaded.state_dict().items():
        if name in parameter_names or not torch.is_tensor(tensor):
            continue
        reloaded_buffers[name] = tensor.detach().to(device="cpu").contiguous()
    if set(reloaded_buffers) != set(buffers):
        raise WeightlessError("verification_failed", "Reloaded buffers do not match the saved model")
    for name, expected in buffers.items():
        if not torch.equal(expected, reloaded_buffers[name]):
            raise WeightlessError("verification_failed", f"Reloaded buffer {name} differs")


def _audit(root):
    safetensors = False
    for path in root.rglob("*"):
        if path.is_symlink():
            raise WeightlessError("verification_failed", "Checkpoint staging contained a symlink")
        if not path.is_file():
            continue
        if path.suffix in _UNSAFE_SUFFIXES:
            raise WeightlessError("verification_failed", "Checkpoint staging contained an unsupported file")
        if path.suffix == ".safetensors":
            safetensors = True
    if not safetensors:
        raise WeightlessError("verification_failed", "Checkpoint staging has no safetensors weights")
    names = {path.name for path in root.iterdir() if path.is_file()}
    if "config.json" not in names:
        raise WeightlessError("verification_failed", "Checkpoint staging has no model config")
    if "tokenizer_config.json" not in names and "tokenizer.json" not in names:
        raise WeightlessError("verification_failed", "Checkpoint staging has no tokenizer files")


def _fsync_tree(root):
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            descriptor = os.open(os.path.join(directory, name), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _discard(path):
    if path is not None and path.exists():
        shutil.rmtree(path)


def _ids(tokenizer, text, label):
    try:
        values = list(tokenizer.encode(text, add_special_tokens=False))
    except Exception as error:
        raise WeightlessError("invalid_cases", f"{label} could not be tokenized") from error
    if not values or any(type(item) is not int for item in values):
        shown = text if len(text) <= 80 else text[:80]
        raise WeightlessError("invalid_cases", f"{label} produced no tokens: {shown}")
    return values


class WeightModel:
    """Experimental handle for direct tensor edits on a local causal language model.

    Evaluation reports metrics without applying an acceptance policy.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.model.requires_grad_(False)
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration as error:
            raise WeightlessError("invalid_model", "Model has no parameters") from error
        self.weights = _Weights(self.model)

    @classmethod
    def load(cls, local_path, device="cpu"):
        """Load an offline Hugging Face causal LM in float32 and eval mode.

        Repository Python is not executed. Files are read only from `local_path`.
        """
        torch, auto_model, auto_tokenizer = _libraries()
        if type(device) is not str or device == "":
            raise WeightlessError("unsupported_target", "Device must be a Torch device string")
        try:
            requested = torch.device(device)
        except (TypeError, RuntimeError, ValueError) as error:
            raise WeightlessError("unsupported_target", "Unsupported execution device") from error
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise WeightlessError("unsupported_target", "CUDA execution is unavailable")
        path = Path(local_path)
        if not path.is_dir():
            raise WeightlessError("invalid_model", "Local model snapshot could not be loaded")
        try:
            with _offline():
                tokenizer = auto_tokenizer.from_pretrained(
                    path, local_files_only=True, trust_remote_code=False)
                model = auto_model.from_pretrained(
                    path, local_files_only=True, trust_remote_code=False,
                    dtype=torch.float32, attn_implementation="eager")
                model.to(device=requested, dtype=torch.float32)
        except WeightlessError:
            raise
        except (OSError, ValueError, RuntimeError) as error:
            text = str(error).lower()
            if "trust_remote_code" in text or "custom code" in text:
                raise WeightlessError("unsupported_target", "Remote model code is unsupported") from error
            raise WeightlessError("invalid_model", "Local model snapshot could not be loaded") from error
        loaded = cls(model, tokenizer)
        actual = loaded.device
        if actual.type != requested.type or (requested.index is not None and actual.index != requested.index):
            raise WeightlessError("unsupported_target", "Model parameters are not on the requested device")
        if any(parameter.dtype != torch.float32 for parameter in loaded.model.parameters()):
            raise WeightlessError("unsupported_target", "Model parameters are not float32")
        return loaded

    def evaluate(self, cases):
        """Return JSON-ready teacher-forced loss for public prompt and target strings.

        Each case loss is the mean negative log likelihood of the target string's
        tokens, conditioned on the prompt tokens. Prompt and target are encoded
        with special tokens disabled, and no chat template is applied. The result
        loss is the unweighted mean of the case losses. An empty case list returns
        a null loss.
        """
        torch, _, _ = _libraries()
        rows = []
        normalized = _cases(cases)
        with torch.inference_mode():
            for case in normalized:
                prompt_ids = _ids(self.tokenizer, case["prompt"], "Prompt")
                target_ids = _ids(self.tokenizer, case["target"], "Target")
                input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=self.device)
                logits = self.model(input_ids=input_ids).logits
                start = len(prompt_ids) - 1
                chosen = logits[0, start:start + len(target_ids), :]
                labels = torch.tensor(target_ids, dtype=torch.long, device=self.device)
                loss = float(torch.nn.functional.cross_entropy(chosen, labels).item())
                if not math.isfinite(loss):
                    raise WeightlessError("nonfinite", "Nonfinite target loss")
                rows.append({"target_nll": loss})
        if not rows:
            result = {"loss": None, "cases": []}
        else:
            result = {"loss": sum(row["target_nll"] for row in rows) / len(rows), "cases": rows}
        try:
            json.dumps(result, allow_nan=False)
        except ValueError as error:
            raise WeightlessError("nonfinite", "Nonfinite target loss") from error
        return result

    def save(self, destination, *, on_phase=None):
        """Publish a Hugging Face safetensors checkpoint without overwriting `destination`.

        Finite float32 parameters are written to a private staging directory, reloaded
        independently, and compared exactly, including tied storage. The staged directory
        is then renamed into place. Failure or interruption before that rename removes
        the staging directory and leaves an existing destination unchanged.
        `on_phase` receives ready, running, staging, verified, published, failure,
        or interrupted.
        """
        destination = Path(destination).absolute()
        if destination.exists() or destination.is_symlink():
            raise WeightlessError("export_exists", "Destination already exists")
        parent = destination.parent
        if not parent.is_dir():
            raise WeightlessError("invalid_model", "Checkpoint destination parent must be a directory")
        if callable(getattr(self.model, "is_remote_code", None)) and self.model.is_remote_code():
            raise WeightlessError("unsupported_target", "Remote model code is unsupported")
        life = _Lifecycle(on_phase)
        staged = None
        renamed = False
        try:
            life.advance("running")
            _require_finite(self.model)
            parameters, buffers, ties = _snapshot(self.model)
            life.advance("staging")
            staged = Path(tempfile.mkdtemp(prefix=".wsm-stage-", dir=parent))
            self.model.save_pretrained(staged)
            self.tokenizer.save_pretrained(staged)
            _audit(staged)
            reloaded = WeightModel.load(staged, device=str(self.device))
            try:
                _compare(reloaded.model, parameters, buffers, ties)
            finally:
                del reloaded
            _fsync_tree(staged)
            life.advance("verified")
            from ._publication import _publish
            _publish(staged, destination)
            renamed = True
            life.advance("published")
            return {"state": "published", "destination": str(destination)}
        except BaseException as error:
            if staged is not None and not staged.exists() and destination.is_dir():
                renamed = True
            if not renamed and not life.published:
                _discard(staged)
                if life.state in ("running", "staging", "verified"):
                    try:
                        life.stop("interrupted" if isinstance(error, KeyboardInterrupt) else "failure")
                    except Exception:
                        pass
            raise
