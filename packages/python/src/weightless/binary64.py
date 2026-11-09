import copy
import hashlib
import json
import math
import re
import struct

__version__ = "0.1.0"


class WeightlessError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _fail(code, message):
    raise WeightlessError(code, message)


def _object(value, keys, code):
    if type(value) is not dict or set(value) != set(keys):
        _fail(code, "Expected object fields " + ", ".join(keys))
    return value


def _array(value, code, maximum, minimum=0):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        _fail(code, "Invalid array length")
    return value


def _integer(value, minimum, maximum, code):
    if type(value) not in (int, float) or not minimum <= value <= maximum or int(value) != value:
        _fail(code, "Invalid integer")
    return int(value)


def _number(value, code):
    if type(value) not in (int, float):
        _fail(code, "Expected finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        _fail(code, "Expected finite binary64 number")
    if not math.isfinite(result):
        _fail(code, "Expected finite binary64 number")
    return result if result != 0 else 0.0


def _identifier(value, code):
    if type(value) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value) is None:
        _fail(code, "Invalid identifier")
    return value


def _identity(value, code):
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        _fail(code, "Invalid checkpoint identity")
    return value


def _decode(value, code):
    if isinstance(value, str):
        try:
            def pairs(items):
                result = {}
                for key, item in items:
                    if key in result:
                        _fail(code, "Duplicate JSON field")
                    result[key] = item
                return result
            return json.loads(value, object_pairs_hook=pairs)
        except (ValueError, RecursionError) as error:
            _fail(code, "Invalid JSON: " + str(error))
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _checkpoint(graph, tensors):
    code = "invalid_checkpoint"
    _object(graph, ("inputSize", "nodes", "output"), code)
    input_size = _integer(graph["inputSize"], 1, 1024, code)
    if type(tensors) is not dict or not tensors:
        _fail(code, "Expected tensors")
    clean_tensors = {}
    total = 0
    for name, tensor in tensors.items():
        _identifier(name, code)
        _object(tensor, ("shape", "values"), code)
        shape = [_integer(n, 1, 1024, code) for n in _array(tensor["shape"], code, 2, 1)]
        values = [_number(n, code) for n in _array(tensor["values"], code, 100000, 1)]
        if math.prod(shape) != len(values):
            _fail(code, "Shape does not match stored values")
        total += len(values)
        if total > 100000:
            _fail(code, "Too many stored values")
        clean_tensors[name] = {"shape": shape, "values": values}
    widths = {"input": input_size}
    nodes = []
    used = set()
    for node in _array(graph["nodes"], code, 128, 1):
        if type(node) is not dict or node.get("op") not in ("affine", "relu"):
            _fail(code, "Unsupported operation")
        affine = node["op"] == "affine"
        _object(node, ("id", "op", "input", "weight", "bias") if affine else ("id", "op", "input"), code)
        name = _identifier(node["id"], code)
        source = _identifier(node["input"], code)
        if name in widths or source not in widths:
            _fail(code, "Duplicate node or forward reference")
        width = widths[source]
        if affine:
            weight = _identifier(node["weight"], code)
            bias = _identifier(node["bias"], code)
            if weight not in clean_tensors or bias not in clean_tensors:
                _fail(code, "Unknown parameter")
            ws, bs = clean_tensors[weight]["shape"], clean_tensors[bias]["shape"]
            if len(ws) != 2 or ws[1] != width or bs != [ws[0]]:
                _fail(code, "Incompatible affine shapes")
            width = ws[0]
            used.update((weight, bias))
        widths[name] = width
        nodes.append(dict(node))
    output = _identifier(graph["output"], code)
    if output == "input" or output not in widths or used != set(clean_tensors):
        _fail(code, "Invalid output or unused tensors")
    clean_graph = {"inputSize": input_size, "nodes": nodes, "output": output}
    chunks = [b"weightless.checkpoint.v1\0", _canonical(clean_graph), b"\0"]
    for name in sorted(clean_tensors):
        tensor = clean_tensors[name]
        chunks.extend((name.encode(), b"\0", _canonical(tensor["shape"]), b"\0"))
        chunks.extend(struct.pack(">d", value) for value in tensor["values"])
    digest = "sha256:" + hashlib.sha256(b"".join(chunks)).hexdigest()
    return {"format": "weightless.checkpoint.v1", "graph": clean_graph, "tensors": clean_tensors, "id": digest}


def create_checkpoint(graph, tensors):
    return _checkpoint(graph, tensors)


def parse_checkpoint(value):
    value = _decode(value, "invalid_checkpoint")
    _object(value, ("format", "graph", "tensors", "id"), "invalid_checkpoint")
    checkpoint = _checkpoint(value["graph"], value["tensors"])
    if value["format"] != checkpoint["format"] or value["id"] != checkpoint["id"]:
        _fail("invalid_checkpoint", "Checkpoint identity mismatch")
    return checkpoint


def _vector(value, width, code):
    result = [_number(n, code) for n in _array(value, code, width, width)]
    return result


def _predict(checkpoint, inputs):
    graph, tensors = checkpoint["graph"], checkpoint["tensors"]
    activations = {"input": inputs}
    for node in graph["nodes"]:
        source = activations[node["input"]]
        if node["op"] == "relu":
            result = [max(0.0, n) for n in source]
        else:
            weight, bias = tensors[node["weight"]], tensors[node["bias"]]
            result = []
            for row in range(weight["shape"][0]):
                accumulator = 0.0
                for column, item in enumerate(source):
                    accumulator += weight["values"][row * len(source) + column] * item
                result.append(accumulator + bias["values"][row])
        if not all(math.isfinite(n) for n in result):
            _fail("numerical_failure", "Nonfinite activation")
        activations[node["id"]] = result
    return activations[graph["output"]]


def predict(checkpoint, inputs):
    checkpoint = parse_checkpoint(checkpoint)
    return _predict(checkpoint, _vector(inputs, checkpoint["graph"]["inputSize"], "invalid_task"))


def _examples(examples, checkpoint):
    widths = {"input": checkpoint["graph"]["inputSize"]}
    for node in checkpoint["graph"]["nodes"]:
        widths[node["id"]] = (checkpoint["tensors"][node["weight"]]["shape"][0]
                              if node["op"] == "affine" else widths[node["input"]])
    output_width = widths[checkpoint["graph"]["output"]]
    return [_example(item, checkpoint["graph"]["inputSize"], output_width)
            for item in _array(examples, "invalid_task", 10000, 1)]


def _example(item, input_width, output_width):
    _object(item, ("input", "target"), "invalid_task")
    return {"input": _vector(item["input"], input_width, "invalid_task"),
            "target": _vector(item["target"], output_width, "invalid_task")}


def _evaluate(checkpoint, examples):
    squared_scaled = squared_raw = 0.0
    exact = 0
    scalars = 0
    for example in examples:
        prediction = _predict(checkpoint, example["input"])
        exact += int(prediction == example["target"])
        for actual, target in zip(prediction, example["target"]):
            residual = actual - target
            scaled = residual / max(1.0, abs(target))
            squared_scaled += scaled * scaled
            squared_raw += residual * residual
            scalars += 1
    loss = 0.5 * (squared_scaled / scalars)
    rmse = math.sqrt(squared_raw / scalars)
    if not math.isfinite(loss) or not math.isfinite(rmse):
        _fail("numerical_failure", "Nonfinite metrics")
    return {"loss": loss, "rmse": rmse, "exact": exact, "count": len(examples)}


def evaluate(checkpoint, examples):
    checkpoint = parse_checkpoint(checkpoint)
    return _evaluate(checkpoint, _examples(examples, checkpoint))


def _task(task, checkpoint):
    code = "invalid_task"
    _object(task, ("format", "name", "train", "development", "budget"), code)
    if task["format"] != "weightless.task.v1":
        _fail(code, "Unsupported task format")
    return {"format": task["format"], "name": _identifier(task["name"], code),
            "train": _examples(task["train"], checkpoint),
            "development": _examples(task["development"], checkpoint),
            "budget": _integer(task["budget"], 1, 1000, code)}


def _proposal(proposal, checkpoint):
    code = "invalid_proposal"
    _object(proposal, ("base", "edits", "author"), code)
    base = _identity(proposal["base"], code)
    author = _identifier(proposal["author"], code)
    edits, seen = [], set()
    for edit in _array(proposal["edits"], code, 100000, 1):
        _object(edit, ("tensor", "index", "value"), code)
        tensor = _identifier(edit["tensor"], code)
        if tensor not in checkpoint["tensors"]:
            _fail(code, "Unknown tensor")
        index = _integer(edit["index"], 0, len(checkpoint["tensors"][tensor]["values"]) - 1, code)
        if (tensor, index) in seen:
            _fail(code, "Duplicate edit")
        seen.add((tensor, index))
        edits.append({"tensor": tensor, "index": index, "value": _number(edit["value"], code)})
    if base != checkpoint["id"]:
        _fail("stale_base", "Proposal base is not current incumbent")
    return {"base": base, "edits": edits, "author": author}


def validate_proposal(proposal, checkpoint):
    return _proposal(_decode(proposal, "invalid_proposal"), parse_checkpoint(checkpoint))


def apply_patch(checkpoint, proposal):
    checkpoint = parse_checkpoint(checkpoint)
    proposal = _proposal(proposal, checkpoint)
    tensors = copy.deepcopy(checkpoint["tensors"])
    for edit in proposal["edits"]:
        tensors[edit["tensor"]]["values"][edit["index"]] = edit["value"]
    return _checkpoint(checkpoint["graph"], tensors)


class NumericalEvaluation:
    """Copied public development data, independent of experiment ownership."""

    def __init__(self, task):
        code = "invalid_task"
        task = _decode(task, code)
        _object(task, ("format", "name", "train", "development", "budget"), code)
        if task["format"] != "weightless.task.v1":
            _fail(code, "Unsupported task format")
        _identifier(task["name"], code)
        _integer(task["budget"], 1, 1000, code)
        for split in ("train", "development"):
            for example in _array(task[split], code, 10000, 1):
                _object(example, ("input", "target"), code)
                for field in ("input", "target"):
                    for item in _array(example[field], code, 1024, 1):
                        _number(item, code)
        from .core import evaluator_config
        self._config = evaluator_config("numerical", "development", {
            "arithmetic": "binary64-row-major-v1", "task": copy.deepcopy(task)})

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def __call__(self, request, checkpoint):
        from .core import _json, _request, evaluation_result
        request = _request(_json(request, "invalid_result"))
        checkpoint = parse_checkpoint(checkpoint)
        if (request["checkpoint"] != checkpoint["id"]
                or request["config"] != self._config["id"]
                or request["purpose"] != self._config["purpose"]):
            _fail("invalid_result", "Request differs from the evaluated checkpoint or configuration")
        try:
            task = _task(self._config["settings"]["task"], checkpoint)
            metrics = {split: _evaluate(checkpoint, task[split])
                       for split in ("train", "development")}
            return evaluation_result(request, status="completed", metrics=metrics,
                                     native_status="graded")
        except WeightlessError as error:
            return evaluation_result(request, status="task_failure", metrics={},
                                     native_status="failed",
                                     error={"type": error.code, "message": str(error)})
