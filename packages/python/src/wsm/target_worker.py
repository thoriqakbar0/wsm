"""Numerical targets and the one-request sandbox worker entry point."""
import copy
import hashlib
import json
import math
import os
import struct
import sys

from . import binary64 as binary


def fail(code, message):
    raise binary.WeightlessError(code, message)


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
    def __init__(self, target):
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            fail("unsupported_target", "Model dependencies are unavailable")
        self.torch = torch
        path = os.path.join(target["source"], "snapshots", target["revision"])
        torch.set_num_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                          dtype=torch.float32, attn_implementation="eager").cpu().eval()
        self.tensors, self.aliases, storage = {}, {}, {}
        pointers = set()
        for name, tensor in self.model.named_parameters(remove_duplicate=False):
            if tensor.dtype != torch.float32 or not tensor.is_contiguous():
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
        if any(t.untyped_storage().data_ptr() in pointers for t in self.buffers.values()):
            fail('unsupported_target', 'Parameter-buffer storage overlap is unsupported')
        self.parameters = {name: {"shape": list(t.shape), "dtype": "float32", "size": t.numel()}
                           for name, t in self.tensors.items()}
        self.original_values = {}
        h = hashlib.sha256(b"weightless.smollm.manifest.v1\0")
        h.update(json.dumps({"model": target["model"], "revision": target["revision"], "dtype": target["dtype"],
                             "torch": torch.__version__, "transformers": transformers.__version__,
                             "aliases": self.aliases, "parameters": self.parameters}, sort_keys=True).encode())
        for name in sorted(os.listdir(path)):
            if name.endswith((".json", ".txt", ".model", ".jinja")):
                h.update(name.encode() + b"\0")
                with open(os.path.join(path, name), "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
        self.manifest = h.digest()
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
        h = hashlib.sha256(b"weightless.smollm.state.v1\0" + self.manifest)
        for name, tensor in sorted(self.tensors.items()):
            h.update(name.encode() + b"\0")
            h.update(tensor.detach().numpy().tobytes(order="C"))
        for name, tensor in sorted(self.buffers.items()):
            h.update(name.encode() + b"\0")
            h.update(str(tensor.dtype).encode() + b"\0")
            h.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            h.update(tensor.detach().contiguous().numpy().tobytes(order="C"))
        return "sha256:" + h.hexdigest()

    def evaluate(self, cases, generate, query, forward_limit=None):
        torch = self.torch
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
            with torch.inference_mode():
                return self._evaluate_cases(cases, generate, cap)
        finally:
            hook.remove()

    def _evaluate_cases(self, cases, generate, cap):
        torch = self.torch
        rows = []
        for case in cases:
            if (type(case) is not dict or set(case) != {"id", "prompt", "target"}
                    or any(type(case[k]) is not str for k in case)
                    or len(case["id"]) > 128 or len(case["prompt"]) > 16384 or len(case["target"]) > 4096):
                fail("invalid_task", "Invalid case")
            prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
                                                         tokenize=True, add_generation_prompt=True, return_dict=False)
            target_ids = self.tokenizer.encode(case["target"], add_special_tokens=False) + [self.tokenizer.eos_token_id]
            if len(prompt) + len(target_ids) > 2048 or not prompt:
                fail("invalid_task", "Case exceeds token limit")
            ids = torch.tensor([prompt + target_ids], dtype=torch.long)
            logits = self.model(input_ids=ids).logits[0, len(prompt)-1:-1, :]
            loss = float(torch.nn.functional.cross_entropy(logits, torch.tensor(target_ids)).item())
            if not math.isfinite(loss):
                fail("numerical_failure", "Nonfinite target loss")
            prediction, stopped = None, False
            if generate:
                generated = self.model.generate(torch.tensor([prompt]), max_new_tokens=cap, do_sample=False,
                                                pad_token_id=self.tokenizer.eos_token_id)[0, len(prompt):].tolist()
                stopped = bool(generated and generated[-1] == self.tokenizer.eos_token_id)
                prediction = self.tokenizer.decode(generated, skip_special_tokens=True)
            rows.append({"id": case["id"], "prediction": prediction, "target_nll": loss,
                         "exact": prediction == case["target"] if generate else False,
                         "stopped_on_eos": stopped})
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


def run(request):
    forward_limit = request.get("forward_limit")
    if forward_limit is not None and (request.get("action") != "evaluate" or type(forward_limit) is not int or forward_limit < 0):
        fail("invalid_request", "Invalid forward limit")
    target = request["target"]
    backend = BinaryTarget(target) if target["kind"] == "binary64" else ModelTarget(target)
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
        return {"target_id": backend.target_id, "checkpoint": checkpoint, "parameters": backend.parameters,
                "aliases": backend.aliases, "capabilities": {"patch": "scalar_replace", "max_edits": 100000,
                "max_inspect": 4096, "dtype": "float64" if target["kind"] == "binary64" else "float32"}}
    if action == "inspect":
        query = request["query"]
        if type(query) is not dict or set(query) != {"tensor", "indices"} or type(query["indices"]) is not list or len(query["indices"]) > 4096:
            fail("invalid_query", "Invalid inspection query")
        selected = edits_for([{"tensor": query["tensor"], "index": i, "value": 0} for i in query["indices"]], backend.parameters, backend.aliases)
        return {"checkpoint": checkpoint, "values": [{"tensor": e["tensor"], "index": e["index"], "value": backend.value(e["tensor"], e["index"])} for e in selected]}
    if action == "materialize":
        effective = [{"tensor": name, "index": index, "value": backend.value(name, index)} for name, index in sorted(addresses)
                     if struct.pack(">d", backend.value(name, index)) != struct.pack(">d", backend.original_value(name, index))]
        changes = [{"tensor": e["tensor"], "index": e["index"], "value": backend.value(e["tensor"], e["index"])} for e in new]
        return {"target_id": backend.target_id, "base_checkpoint": base_checkpoint,
                "checkpoint": checkpoint, "patches": effective, "changes": changes}
    if action == "evaluate":
        metrics, rows = backend.evaluate(request["cases"], request["generate"], request["query"], forward_limit)
        usage = {"forward_calls": backend.forward_calls}
        if forward_limit is not None:
            usage["forward_limit"] = forward_limit
        return {"checkpoint": checkpoint, "metrics": metrics, "cases": rows, "usage": usage}
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
