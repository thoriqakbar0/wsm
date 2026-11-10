"""Numerical targets and the one-request sandbox worker entry point."""
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys

from . import binary64 as binary

_MANIFEST_PREFIX = {
    "smollm": b"weightless.smollm.manifest.v1\0",
    "huggingface": b"weightless.huggingface.manifest.v1\0",
}
_STATE_PREFIX = {
    "smollm": b"weightless.smollm.state.v1\0",
    "huggingface": b"weightless.huggingface.state.v1\0",
}
_MODEL_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_REVISION = re.compile(r"[0-9a-f]{40}")
_UNSAFE_SUFFIXES = (".py", ".pyc", ".bin", ".pt", ".pth", ".pkl", ".pickle", ".onnx")
_SMOLLM_SIDECARS = (".json", ".txt", ".model", ".jinja")
_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
_TARGET_FIELDS = {"kind", "source", "revision", "model", "dtype"}


def fail(code, message):
    raise binary.WeightlessError(code, message)


def model_id_ok(model):
    if type(model) is not str or not model or len(model) > 192 or model.count("/") > 1:
        return False
    return all(_MODEL_ID.fullmatch(part) and ".." not in part for part in model.split("/"))


def revision_ok(revision):
    return type(revision) is str and _REVISION.fullmatch(revision) is not None


def _contained(root, path):
    try:
        return os.path.commonpath((root, path)) == root
    except ValueError:
        return False


def _regular(path):
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except OSError:
        return False


def resolved_snapshot_file(root, full):
    """Resolve repository files and recognized shared Hugging Face cache blobs."""
    if os.path.islink(full):
        raw = os.readlink(full)
        immediate = raw if os.path.isabs(raw) else os.path.join(os.path.dirname(full), raw)
        if not _contained(root, os.path.normpath(immediate)):
            fail("invalid_target", "Snapshot link escapes the source")
    elif not _contained(root, os.path.realpath(full)):
        fail("invalid_target", "Snapshot link escapes the source")
    try:
        resolved = os.path.realpath(full)
    except OSError:
        fail("invalid_target", "Invalid snapshot entry")
    if not _regular(resolved):
        fail("invalid_target", "Snapshot link escapes the source")
    if not _contained(root, resolved):
        shared = os.path.join(os.path.dirname(root), "blobs")
        relative = os.path.relpath(resolved, shared).split(os.sep)
        if (not os.path.basename(root).startswith("models--") or os.path.realpath(shared) != shared
                or not _contained(shared, resolved) or len(relative) != 2
                or re.fullmatch(r"[0-9a-f]{2}", relative[0]) is None
                or re.fullmatch(r"[0-9a-f]{64}", relative[1]) is None
                or not relative[1].startswith(relative[0])):
            fail("invalid_target", "Snapshot link escapes the source and shared model cache")
    return resolved


def _read_json(path, limit=8 * 1024 * 1024):
    try:
        with open(path, "rb") as handle:
            raw = handle.read(limit + 1)
    except OSError:
        fail("invalid_target", "Snapshot JSON is unreadable")
    if len(raw) > limit:
        fail("invalid_target", "Snapshot JSON exceeds limit")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        fail("invalid_target", "Snapshot JSON is unreadable")


def _reject_index(path, names):
    data = _read_json(path)
    weight_map = data.get("weight_map") if type(data) is dict else None
    if type(weight_map) is not dict or not weight_map:
        fail("invalid_target", "Invalid safetensors index")
    for shard in weight_map.values():
        if (type(shard) is not str or shard != os.path.basename(shard) or not shard.endswith(".safetensors")
                or shard not in names):
            fail("invalid_target", "Safetensors index escapes the snapshot")


def huggingface_snapshot(source, revision, *, local_files=False):
    """Return the real snapshot directory or reject a malformed local tree."""
    if type(source) is not str or not os.path.isabs(source) or not revision_ok(revision):
        fail("invalid_target", "Invalid model snapshot")
    root = os.path.realpath(source)
    if not os.path.isdir(root):
        fail("invalid_target", "Invalid model snapshot")
    snapshot = os.path.realpath(os.path.join(root, "snapshots", revision))
    if not _contained(root, snapshot) or not os.path.isdir(snapshot):
        fail("invalid_target", "Snapshot path escapes the source")
    saw_config = False
    saw_weights = False
    index = None
    root_names = set()
    for dirpath, dirnames, filenames in os.walk(snapshot, followlinks=False):
        for dirname in list(dirnames):
            if dirname in ("", ".", "..") or "/" in dirname or "\\" in dirname or "\x00" in dirname:
                fail("invalid_target", "Invalid snapshot entry")
            full = os.path.join(dirpath, dirname)
            if os.path.islink(full):
                fail("invalid_target", "Snapshot directory links are unsupported")
        for name in filenames:
            if name in ("", ".", "..") or "/" in name or "\\" in name or "\x00" in name:
                fail("invalid_target", "Invalid snapshot entry")
            full = os.path.join(dirpath, name)
            resolved = resolved_snapshot_file(root, full)
            if local_files and not _contained(root, resolved):
                fail("invalid_target", "Snapshot link escapes the source")
            if name.endswith(_UNSAFE_SUFFIXES):
                fail("unsupported_target", "Unsupported model file")
            relative = os.path.relpath(full, snapshot)
            if os.sep not in relative and relative != ".":
                root_names.add(name)
            if name == "config.json" and os.path.dirname(full) == snapshot:
                data = _read_json(resolved)
                if type(data) is not dict:
                    fail("invalid_target", "Snapshot config is unreadable")
                if "auto_map" in data:
                    fail("unsupported_target", "Remote model code is unsupported")
                saw_config = True
            elif name == "tokenizer_config.json" and os.path.dirname(full) == snapshot:
                data = _read_json(resolved)
                if type(data) is dict and "auto_map" in data:
                    fail("unsupported_target", "Remote model code is unsupported")
            if name == "model.safetensors.index.json" and os.path.dirname(full) == snapshot:
                index = resolved
            if name.endswith(".safetensors"):
                saw_weights = True
    if not saw_config or not saw_weights:
        fail("invalid_target", "Snapshot is missing config or safetensors weights")
    if index is not None:
        _reject_index(index, root_names)
    return snapshot


def snapshot_directory(target):
    kind = target.get("kind") if type(target) is dict else None
    if kind == "huggingface":
        return huggingface_snapshot(target["source"], target["revision"], local_files=True)
    if kind == "smollm":
        path = os.path.join(target["source"], "snapshots", target["revision"])
        if not os.path.isdir(path):
            fail("invalid_target", "Invalid model snapshot")
        return path
    fail("unsupported_target", "Unsupported target")


def _sidecars(kind, snapshot):
    files = {}
    if kind == "smollm":
        for name in os.listdir(snapshot):
            if name.endswith(_SMOLLM_SIDECARS):
                with open(os.path.join(snapshot, name), "rb") as handle:
                    files[name] = handle.read()
        return files
    for dirpath, dirnames, filenames in os.walk(snapshot, followlinks=False):
        dirnames[:] = [dirname for dirname in dirnames if not os.path.islink(os.path.join(dirpath, dirname))]
        for name in filenames:
            if name.endswith(".safetensors"):
                continue
            full = os.path.join(dirpath, name)
            relative = os.path.relpath(full, snapshot).replace(os.sep, "/")
            with open(full, "rb") as handle:
                files[relative] = handle.read()
    return files


def manifest_digest(kind, target, torch_version, transformers_version, aliases, parameters, sidecars):
    digest = hashlib.sha256(_MANIFEST_PREFIX[kind])
    digest.update(json.dumps({"model": target["model"], "revision": target["revision"], "dtype": target["dtype"],
                              "torch": torch_version, "transformers": transformers_version,
                              "aliases": aliases, "parameters": parameters}, sort_keys=True).encode())
    for name in sorted(sidecars):
        digest.update(name.encode() + b"\0")
        digest.update(sidecars[name])
    return digest.digest()


def state_id(kind, manifest, tensors, buffers):
    digest = hashlib.sha256(_STATE_PREFIX[kind] + manifest)
    for name, content in tensors:
        digest.update(name.encode() + b"\0")
        digest.update(content)
    for name, dtype_name, shape, content in buffers:
        digest.update(name.encode() + b"\0")
        digest.update(str(dtype_name).encode() + b"\0")
        digest.update(json.dumps(list(shape)).encode() + b"\0")
        digest.update(content)
    return "sha256:" + digest.hexdigest()


@contextlib.contextmanager
def _offline_hub():
    saved = {key: os.environ.get(key) for key in _OFFLINE_ENV}
    for key in _OFFLINE_ENV:
        os.environ[key] = "1"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_causal_lm(path, device):
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        fail("unsupported_target", "Model dependencies are unavailable")
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                 dtype=torch.float32, attn_implementation="eager").to(device).eval()
    return torch, transformers, tokenizer, model


def _default_chat_template(tokenizer):
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        template = template.get("default")
    return template or None


def _eos_id(tokenizer):
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int:
        return None
    return eos


def edits_for(items, parameters, aliases):
    if type(items) is not list or len(items) > 100000:
        fail("invalid_edit", "Invalid edit list")
    result, seen = [], set()
    for item in items:
        if type(item) is not dict or set(item) != {"tensor", "index", "value"}:
            fail("invalid_edit", "Invalid edit fields")
        name, index, value = item["tensor"], item["index"], item["value"]
        if type(name) is not str:
            fail("invalid_edit", "Invalid tensor")
        name = aliases.get(name, name)
        if (name not in parameters or type(index) is not int or index < 0
                or index >= parameters[name]["size"] or (name, index) in seen):
            fail("invalid_edit", "Invalid or duplicate address")
        value = binary._number(value, "invalid_edit")
        if parameters[name]['dtype'] == 'float32':
            try:
                value = struct.unpack('>f', struct.pack('>f', value))[0]
            except (OverflowError, struct.error):
                fail('invalid_edit', 'Value is not representable in float32')
            if not math.isfinite(value):
                fail('invalid_edit', 'Value is not representable in float32')
            if value == 0:
                value = 0.0
        seen.add((name, index))
        result.append({"tensor": name, "index": index, "value": value})
    return result


class BinaryTarget:
    def __init__(self, target):
        self.checkpoint = binary.parse_checkpoint(target["checkpoint"])
        self.original = copy.deepcopy(self.checkpoint)
        self.aliases = {}
        self.parameters = {name: {"shape": tensor["shape"], "dtype": "float64", "size": len(tensor["values"])}
                           for name, tensor in self.checkpoint["tensors"].items()}
        self.target_id = self.identity()
        self.forward_calls = 0

    def value(self, name, index):
        return self.checkpoint["tensors"][name]["values"][index]

    def set(self, name, index, value):
        self.checkpoint["tensors"][name]["values"][index] = value
        return value

    def original_value(self, name, index):
        return self.original["tensors"][name]["values"][index]

    def identity(self):
        return binary.create_checkpoint(self.checkpoint["graph"], self.checkpoint["tensors"])["id"]

    def evaluate(self, cases, generate, query, forward_limit=None):
        if type(cases) is not list or not 1 <= len(cases) <= 10000:
            fail("invalid_task", "Invalid cases")
        checkpoint = binary.create_checkpoint(self.checkpoint["graph"], self.checkpoint["tensors"])
        rows, examples = [], []
        for case in cases:
            if type(case) is not dict or set(case) != {"id", "input", "target"} or type(case["id"]) is not str or len(case["id"]) > 128:
                fail("invalid_task", "Invalid case")
            example = {"input": case["input"], "target": case["target"]}
            examples.append(example)
        examples = binary._examples(examples, self.checkpoint)
        squared_scaled = squared_raw = 0.0
        scalars = 0
        self.forward_calls = 0
        for case, example in zip(cases, examples):
            if forward_limit is not None and self.forward_calls >= forward_limit:
                fail("forward_limit_exceeded", "Target forward limit exhausted")
            self.forward_calls += 1
            prediction = binary.predict(checkpoint, example["input"])
            for actual, target in zip(prediction, example["target"]):
                residual = actual - target
                scaled = residual / max(1.0, abs(target))
                squared_scaled += scaled * scaled
                squared_raw += residual * residual
                scalars += 1
            rows.append({"id": case["id"], "prediction": prediction, "target_nll": None,
                         "exact": prediction == example["target"], "stopped_on_eos": False})
        loss, rmse = 0.5 * (squared_scaled / scalars), math.sqrt(squared_raw / scalars)
        if not math.isfinite(loss) or not math.isfinite(rmse):
            fail("numerical_failure", "Nonfinite metrics")
        return {"loss": loss, "rmse": rmse, "exact": sum(row["exact"] for row in rows), "count": len(rows)}, rows


class ModelTarget:
    def __init__(self, target, *, device="cpu"):
        if device not in ("cpu", "cuda"):
            fail("unsupported_target", "Unsupported execution device")
        if device == "cuda":
            try:
                import torch
            except ImportError:
                fail("unsupported_target", "Model dependencies are unavailable")
            if not torch.cuda.is_available():
                fail("unsupported_target", "CUDA execution is unavailable")
        kind = target.get("kind")
        if kind not in _MANIFEST_PREFIX:
            fail("unsupported_target", "Unsupported target")
        self.kind = kind
        self.device = device
        self.forward_calls = 0
        if kind == "huggingface":
            path = huggingface_snapshot(target["source"], target["revision"], local_files=True)
            try:
                with _offline_hub():
                    torch, transformers, self.tokenizer, self.model = _load_causal_lm(path, device)
            except binary.WeightlessError:
                raise
            except (OSError, ValueError, RuntimeError) as error:
                text = str(error).lower()
                if "trust_remote_code" in text or "custom code" in text:
                    fail("unsupported_target", "Remote model code is unsupported")
                fail("invalid_target", "Local model snapshot could not be loaded")
            attention = getattr(self.model.config, "_attn_implementation", None)
            if attention not in (None, "eager"):
                fail("unsupported_target", "Eager attention is required")
        else:
            path = os.path.join(target["source"], "snapshots", target["revision"])
            torch, transformers, self.tokenizer, self.model = _load_causal_lm(path, device)
        self.torch = torch
        self.execution = {"parameter_devices": sorted({str(parameter.device) for parameter in self.model.parameters()})}
        self.tensors, self.aliases, storage = {}, {}, {}
        pointers = set()
        for name, tensor in self.model.named_parameters(remove_duplicate=False):
            if tensor.dtype != torch.float32 or not tensor.is_contiguous() or tensor.device.type != device:
                fail("unsupported_target", "Unsupported model parameter layout")
            key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
            if key in storage:
                self.aliases[name] = storage[key]
            else:
                if tensor.untyped_storage().data_ptr() in pointers:
                    fail('unsupported_target', 'Overlapping parameter views are unsupported')
                pointers.add(tensor.untyped_storage().data_ptr())
                storage[key] = name
                self.tensors[name] = tensor
        self.buffers = dict(self.model.named_buffers(remove_duplicate=False))
        if any(item.untyped_storage().data_ptr() in pointers for item in self.buffers.values()):
            fail('unsupported_target', 'Parameter-buffer storage overlap is unsupported')
        self.parameters = {name: {"shape": list(tensor.shape), "dtype": "float32", "size": tensor.numel()}
                           for name, tensor in self.tensors.items()}
        self.original_values = {}
        self.manifest = manifest_digest(self.kind, target, torch.__version__, transformers.__version__,
                                        self.aliases, self.parameters, _sidecars(self.kind, path))
        self.target_id = self.identity()

    def value(self, name, index):
        return float(self.tensors[name].view(-1)[index].item())

    def original_value(self, name, index):
        return self.original_values.get((name, index), self.value(name, index))

    def set(self, name, index, value):
        key = (name, index)
        self.original_values.setdefault(key, self.value(name, index))
        with self.torch.no_grad():
            try:
                self.tensors[name].view(-1)[index] = value
            except (RuntimeError, OverflowError):
                fail("invalid_edit", "Value is not representable in float32")
        result = self.value(name, index)
        if not math.isfinite(result):
            fail("invalid_edit", "Value is not representable in float32")
        if result == 0:
            with self.torch.no_grad():
                self.tensors[name].view(-1)[index] = 0.0
            result = 0.0
        return result

    def identity(self):
        tensors = []
        for name in sorted(self.tensors):
            data = self.tensors[name].detach()
            if data.device.type != "cpu":
                data = data.cpu()
            tensors.append((name, data.numpy().tobytes(order="C")))
        buffers = []
        for name in sorted(self.buffers):
            tensor = self.buffers[name]
            data = tensor.detach()
            if data.device.type != "cpu":
                data = data.cpu()
            buffers.append((name, str(tensor.dtype), list(tensor.shape), data.contiguous().numpy().tobytes(order="C")))
        return state_id(self.kind, self.manifest, tensors, buffers)

    def _long(self, values):
        kwargs = {"dtype": self.torch.long}
        if self.device != "cpu":
            kwargs["device"] = self.device
        return self.torch.tensor(values, **kwargs)

    def infer(self, query, forward_limit):
        validate_inference_query(query)
        torch = self.torch
        tokens, cap = query['tokens'], query['max_tokens']
        if any(token >= self.model.config.vocab_size for token in tokens + (query['stop'] or [])):
            fail('invalid_request', 'Token is outside model vocabulary')
        if len(tokens) + cap > min(2048, getattr(self.model.config, 'max_position_embeddings', 2048)):
            fail('invalid_request', 'Inference exceeds model context')
        self.forward_calls = 0

        def count_forward(module, args):
            if forward_limit is not None and self.forward_calls >= forward_limit:
                fail('forward_limit_exceeded', 'Target forward limit exhausted')
            self.forward_calls += 1

        hook = self.model.register_forward_pre_hook(count_forward)
        generated, logprobs, prompt_logprobs = [], [], None
        stop_reason = 'length'
        stops = set(query['stop'] or [])
        if query['stop'] is None:
            eos = _eos_id(self.tokenizer)
            if eos is not None:
                stops.add(eos)
        try:
            with torch.inference_mode():
                for position in range(max(1, cap)):
                    logits = self.model(input_ids=self._long([tokens + generated])).logits[0]
                    if not bool(torch.isfinite(logits).all().item()):
                        fail('numerical_failure', 'Nonfinite inference logits')
                    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
                    if position == 0 and query['prompt_logprobs']:
                        labels = self._long(tokens[1:])
                        scores = probabilities[:-1].gather(1, labels.unsqueeze(1)).squeeze(1)
                        prompt_logprobs = [None, *scores.tolist()]
                    if cap == 0:
                        break
                    token = int(logits[-1].argmax().item())
                    generated.append(token)
                    logprobs.append(float(probabilities[-1, token].item()))
                    if token in stops:
                        stop_reason = 'stop'
                        break
        finally:
            hook.remove()
        sequence_id = hashlib.sha256(json.dumps([tokens, generated]).encode()).hexdigest()
        return {'sequences': [{'tokens': generated, 'logprobs': logprobs,
                              'stop_reason': stop_reason, 'sequence_id': sequence_id}] if cap else [],
                'prompt_logprobs': prompt_logprobs}

    def _prompt_ids(self, text):
        if self.kind == "smollm" or _default_chat_template(self.tokenizer):
            return self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                     tokenize=True, add_generation_prompt=True, return_dict=False)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _label_ids(self, text):
        if self.kind == "smollm":
            return self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        labels = list(self.tokenizer.encode(text, add_special_tokens=False))
        eos = _eos_id(self.tokenizer)
        if eos is not None:
            labels.append(eos)
        return labels

    def evaluate(self, cases, generate, query, forward_limit=None):
        if type(cases) is not list or not 1 <= len(cases) <= 1000:
            fail("invalid_task", "Invalid cases")
        if query is None:
            query = {}
        if type(query) is not dict or set(query) - {"max_new_tokens"}:
            fail("invalid_task", "Invalid generation options")
        cap = query.get("max_new_tokens", 16)
        if type(cap) is not int or not 1 <= cap <= 128:
            fail("invalid_task", "Invalid generation limit")
        self.forward_calls = 0

        def count_forward(module, args):
            if forward_limit is not None and self.forward_calls >= forward_limit:
                fail("forward_limit_exceeded", "Target forward limit exhausted")
            self.forward_calls += 1

        hook = self.model.register_forward_pre_hook(count_forward)
        try:
            with self.torch.inference_mode():
                return self._evaluate_cases(cases, generate, cap)
        finally:
            hook.remove()

    def _evaluate_cases(self, cases, generate, cap):
        torch = self.torch
        rows = []
        placements = []
        for case in cases:
            if (type(case) is not dict or set(case) != {"id", "prompt", "target"}
                    or any(type(case[key]) is not str for key in case)
                    or len(case["id"]) > 128 or len(case["prompt"]) > 16384 or len(case["target"]) > 4096):
                fail("invalid_task", "Invalid case")
            prompt = self._prompt_ids(case["prompt"])
            target_ids = self._label_ids(case["target"])
            if len(prompt) + len(target_ids) > 2048 or not prompt or (self.kind == "huggingface" and not target_ids):
                fail("invalid_task", "Case exceeds token limit")
            ids = self._long([prompt + target_ids])
            logits = self.model(input_ids=ids).logits[0, len(prompt)-1:-1, :]
            labels = self._long(target_ids)
            if self.device == "cuda" and not bool(torch.isfinite(logits).all().item()):
                fail("numerical_failure", "Nonfinite target logits")
            loss = float(torch.nn.functional.cross_entropy(logits, labels).item())
            if not math.isfinite(loss):
                fail("numerical_failure", "Nonfinite target loss")
            prediction, stopped, generation_ids = None, False, None
            if generate:
                eos = self.tokenizer.eos_token_id if self.kind == "smollm" else _eos_id(self.tokenizer)
                options = {"max_new_tokens": cap, "do_sample": False}
                if self.kind == "huggingface":
                    options["eos_token_id"] = eos
                if self.kind == "smollm" or eos is not None:
                    options["pad_token_id"] = eos
                generation_ids = self._long([prompt])
                generated = self.model.generate(generation_ids, **options)[0, len(prompt):].tolist()
                stopped = bool(type(eos) is int and generated and generated[-1] == eos)
                prediction = self.tokenizer.decode(generated, skip_special_tokens=True)
            rows.append({"id": case["id"], "prediction": prediction, "target_nll": loss,
                         "exact": prediction == case["target"] if generate else False,
                         "stopped_on_eos": stopped})
            if self.device == "cuda":
                placements.append({"input_device": str(ids.device), "label_device": str(labels.device),
                                   "logits_device": str(logits.device), "finite_logits": True,
                                   "generation_input_device": str(generation_ids.device) if generate else None})
        if self.device == "cuda":
            self.execution["cases"] = placements
        return {"loss": sum(row["target_nll"] for row in rows) / len(rows),
                "exact": sum(row["exact"] for row in rows), "count": len(rows)}, rows


def isolation_receipt():
    status = {}
    with open("/proc/self/status", encoding="utf-8") as source:
        for line in source:
            key, _, value = line.partition(":")
            if key in ("Uid", "Gid", "CapEff", "NoNewPrivs", "Seccomp"):
                status[key] = value.strip()
    mounts = {}
    with open("/proc/self/mountinfo", encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if fields[4] in ("/", "/target", "/tmp"):
                mounts[fields[4]] = fields[5].split(",")
    return {"status": status, "mounts": mounts, "network_namespace": os.readlink("/proc/self/ns/net")}


def _backend(target, device):
    if type(target) is not dict or type(target.get("kind")) is not str:
        fail("invalid_target", "Invalid target")
    kind = target["kind"]
    if kind == "binary64":
        return BinaryTarget(target)
    if kind == "smollm":
        return ModelTarget(target, device=device)
    if kind == "huggingface":
        if (set(target) != _TARGET_FIELDS or target["dtype"] != "float32" or type(target["source"]) is not str
                or not model_id_ok(target["model"]) or not revision_ok(target["revision"])):
            fail("invalid_target", "Unsupported model configuration")
        return ModelTarget(target, device=device)
    fail("unsupported_target", "Unsupported target")


def validate_inference_query(query):
    fields = {'checkpoint', 'tokens', 'max_tokens', 'temperature', 'prompt_logprobs', 'stop'}
    if (type(query) is not dict or set(query) != fields or type(query['checkpoint']) is not str
            or type(query['tokens']) is not list or not 1 <= len(query['tokens']) <= 2048
            or any(type(token) is not int or token < 0 for token in query['tokens'])
            or type(query['max_tokens']) is not int or not 0 <= query['max_tokens'] <= 128
            or len(query['tokens']) + query['max_tokens'] > 2048
            or type(query['temperature']) not in (int, float) or query['temperature'] != 0
            or type(query['prompt_logprobs']) is not bool
            or (query['stop'] is not None and (type(query['stop']) is not list
                or len(query['stop']) > 128 or any(type(token) is not int or token < 0 for token in query['stop'])))):
        fail('invalid_request', 'Expected bounded token-only greedy inference')


def run(request, *, device="cpu"):
    if device not in ("cpu", "cuda"):
        fail("unsupported_target", "Unsupported execution device")
    forward_limit = request.get("forward_limit")
    if forward_limit is not None and (request.get("action") not in ("evaluate", "infer") or type(forward_limit) is not int or forward_limit < 0):
        fail("invalid_request", "Invalid forward limit")
    target = request["target"]
    backend = _backend(target, device)
    patches = edits_for(request["patches"], backend.parameters, backend.aliases)
    new = edits_for(request["edits"], backend.parameters, backend.aliases)
    addresses = set()
    for item in patches:
        key = (item["tensor"], item["index"])
        addresses.add(key)
        backend.set(*key, item["value"])
    base_checkpoint = backend.identity()
    for item in new:
        key = (item["tensor"], item["index"])
        addresses.add(key)
        backend.set(*key, item["value"])
    checkpoint = backend.identity()
    action = request["action"]
    if action == "describe":
        result = {"target_id": backend.target_id, "checkpoint": checkpoint, "parameters": backend.parameters,
                "aliases": backend.aliases, "capabilities": {"patch": "scalar_replace", "max_edits": 100000,
                "max_inspect": 4096, "dtype": "float64" if target["kind"] == "binary64" else "float32"}}
        if device == "cuda" and isinstance(backend, ModelTarget):
            result["execution"] = backend.execution
        return result
    if action == "inspect":
        query = request["query"]
        if type(query) is not dict or set(query) != {"tensor", "indices"} or type(query["indices"]) is not list or len(query["indices"]) > 4096:
            fail("invalid_query", "Invalid inspection query")
        selected = edits_for([{"tensor": query["tensor"], "index": i, "value": 0} for i in query["indices"]], backend.parameters, backend.aliases)
        return {"checkpoint": checkpoint, "values": [{"tensor": item["tensor"], "index": item["index"], "value": backend.value(item["tensor"], item["index"])} for item in selected]}
    if action == "materialize":
        effective = [{"tensor": name, "index": index, "value": backend.value(name, index)} for name, index in sorted(addresses)
                     if struct.pack(">d", backend.value(name, index)) != struct.pack(">d", backend.original_value(name, index))]
        changes = [{"tensor": item["tensor"], "index": item["index"], "value": backend.value(item["tensor"], item["index"])} for item in new]
        return {"target_id": backend.target_id, "base_checkpoint": base_checkpoint,
                "checkpoint": checkpoint, "patches": effective, "changes": changes}
    if action == "evaluate":
        metrics, rows = backend.evaluate(request["cases"], request["generate"], request["query"], forward_limit)
        usage = {"forward_calls": backend.forward_calls}
        if forward_limit is not None:
            usage["forward_limit"] = forward_limit
        result = {"checkpoint": checkpoint, "metrics": metrics, "cases": rows, "usage": usage}
        if device == "cuda" and isinstance(backend, ModelTarget):
            result["execution"] = backend.execution
        return result
    if action == 'infer':
        if not isinstance(backend, ModelTarget):
            fail('unsupported_target', 'Token inference requires a language model')
        query = request['query']
        validate_inference_query(query)
        if query['checkpoint'] != checkpoint:
            fail('stale_base', 'Inference checkpoint changed')
        result = backend.infer(query, forward_limit)
        result.update(checkpoint=checkpoint, usage={'forward_calls': backend.forward_calls})
        if forward_limit is not None:
            result['usage']['forward_limit'] = forward_limit
        return result
    fail("invalid_request", "Unsupported operation")


def main():
    try:
        request = json.load(sys.stdin)
        value = run(request)
        value["sandbox"] = isolation_receipt()
        result = {"result": value}
    except binary.WeightlessError as error:
        result = {"error": {"code": error.code, "message": str(error)[:256]}}
    except Exception:
        result = {"error": {"code": "target_failure", "message": "Target worker failed"}}
    sys.stdout.write(json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
