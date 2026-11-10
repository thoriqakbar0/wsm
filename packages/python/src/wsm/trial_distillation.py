"""Reward-weighted teacher distillation through Spindle, without student sampling."""

import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit
import uuid


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return float(value)


def validate_student_config(config):
    required = {"provider", "base_url", "model", "rank", "loss_fn", "learning_rate"}
    if (type(config) is not dict or not required <= config.keys()
            or set(config) - required - {"request_seconds", "max_context_tokens"}):
        raise ValueError("Explicit Spindle provider, endpoint, student model, rank, loss, and learning rate are required")
    if config["provider"] != "spindle" or config["loss_fn"] != "cross_entropy":
        raise ValueError("Teacher distillation requires Spindle cross_entropy")
    endpoint = config["base_url"]
    if type(endpoint) is not str or any(character.isspace() for character in endpoint):
        raise ValueError("Student endpoint must be credential-free HTTPS")
    try:
        parsed = urlsplit(endpoint)
        valid = (parsed.scheme == "https" and parsed.hostname and parsed.port != 0
                 and parsed.username is None and parsed.password is None
                 and not parsed.query and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Student endpoint must be credential-free HTTPS")
    if type(config["model"]) is not str or not config["model"].strip():
        raise ValueError("Select an explicit student model")
    if type(config["rank"]) is not int or config["rank"] < 1:
        raise ValueError("rank must be positive")
    if _finite(config["learning_rate"], "learning_rate") <= 0:
        raise ValueError("learning_rate must be positive")
    if _finite(config.get("request_seconds", 600), "request_seconds") <= 0:
        raise ValueError("request_seconds must be positive")
    context = config.get("max_context_tokens", 8192)
    if type(context) is not int or context < 2:
        raise ValueError("max_context_tokens must be an integer of at least two")
    return dict(config)


def _validate_trajectory(trajectory):
    if type(trajectory) is not list or not trajectory:
        raise ValueError("Distillation requires a nonempty teacher trajectory")
    for turn in trajectory:
        if (type(turn) is not dict or turn.get("teacher") != "Codex"
                or any(type(turn.get(key)) is not str or not turn[key].strip() for key in ("prompt", "text"))):
            raise ValueError("Each Codex teacher turn requires literal prompt and response text")


def _token_list(tokens):
    if (type(tokens) is not list or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError("Student tokenizer must produce nonempty integer tokens")
    return list(tokens)


def build_teacher_datums(trajectory, tokenizer, reward, *, max_context_tokens=8192):
    """Retokenize teacher text for the student; mask prompts and weight responses."""
    import tinker

    _validate_trajectory(trajectory)
    if type(max_context_tokens) is not int or max_context_tokens < 2:
        raise ValueError("max_context_tokens must be an integer of at least two")
    weight = math.exp(max(-20.0, min(0.0, _finite(reward, "reward"))))
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise ValueError("Student tokenizer must declare an integer EOS token")
    data = []
    for turn in trajectory:
        prompt = _token_list(tokenizer.apply_chat_template(
            [{"role": "user", "content": turn["prompt"]}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False))
        response = _token_list(tokenizer.encode(turn["text"], add_special_tokens=False))
        if response[-1] != eos:
            response.append(eos)
        tokens = prompt + response
        if len(tokens) > max_context_tokens:
            raise ValueError("Teacher turn exceeds the student context limit; no text was truncated")
        data.append(tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.types.TensorData(data=tokens[1:], dtype="int64"),
                "weights": tinker.types.TensorData(
                    data=[0.0] * (len(prompt) - 1) + [weight] * len(response), dtype="float32"),
            },
        ))
    return data


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class StudentTrainer:
    """Lazily create one training client after the teacher trial has been graded."""

    def __init__(self, config, *, reserve):
        self.config = validate_student_config(config)
        if not callable(reserve):
            raise ValueError("A durable cost reservation callback is required")
        self.reserve = reserve
        self.service = None
        self.training_started = False

    def train(self, trajectory, reward, *, output):
        import tinker
        import jinja2  # Required by the student chat template before allocating a GPU.

        if self.training_started:
            raise ValueError("This student has already attempted a training update")
        _validate_trajectory(trajectory)
        reward = _finite(reward, "reward")
        key = os.environ.get("WSM_SPINDLE_API_KEY")
        if not key:
            raise ValueError("WSM_SPINDLE_API_KEY is required; hosted credentials are not a fallback")
        directory = Path(output)
        directory.mkdir(mode=0o700, exist_ok=False)
        receipt_path = directory / "receipt.json"
        encoded = json.dumps(trajectory, sort_keys=True, allow_nan=False).encode()
        identity = uuid.uuid4().hex
        receipt = {"format": "wsm.teacher-distillation.v1", "state": "prepared", "provider": "spindle",
                   "teacher": "Codex", "student": self.config["model"], "base_url": self.config["base_url"],
                   "algorithm": "reward_weighted_teacher_sft", "loss_fn": "cross_entropy",
                   "reward": reward, "response_weight": math.exp(max(-20.0, min(0.0, reward))),
                   "reward_transform": "exp(clamp(reward, -20, 0))",
                   "trajectory_sha256": hashlib.sha256(encoded).hexdigest(), "checkpoint": None}
        _write(receipt_path, receipt)
        self.reserve("spindle_create", {})
        self.training_started = True
        receipt["state"] = "create_dispatched"
        _write(receipt_path, receipt)
        timeout = self.config.get("request_seconds", 600)
        try:
            self.service = tinker.ServiceClient(base_url=self.config["base_url"], api_key=key)
            training = self.service.create_lora_training_client(
                base_model=self.config["model"], rank=self.config["rank"]).result(timeout=timeout)
            receipt.update(state="client_created", training_run=training.get_info().model_id)
            _write(receipt_path, receipt)
            data = build_teacher_datums(trajectory, training.get_tokenizer(), reward,
                                        max_context_tokens=self.config.get("max_context_tokens", 8192))
            datums = [{"model_input_tokens": datum.model_input.to_ints(),
                       "loss_fn_inputs": {name: {"data": tensor.data, "dtype": tensor.dtype}
                                          for name, tensor in datum.loss_fn_inputs.items()}}
                      for datum in data]
            _write(directory / "datums.json", datums)
            datum_bytes = json.dumps(datums, sort_keys=True, allow_nan=False).encode()
            receipt.update(datums_sha256=hashlib.sha256(datum_bytes).hexdigest(), turns=len(data),
                           training_tokens=sum(datum.model_input.length for datum in data),
                           response_tokens=sum(sum(weight > 0 for weight in datum.loss_fn_inputs["weights"].data)
                                               for datum in data), learning_rate=self.config["learning_rate"])
            receipt["state"] = "data_prepared"
            _write(receipt_path, receipt)
            self.reserve("spindle_train", {"train_tokens": receipt["training_tokens"]})
            receipt["state"] = "training_dispatched"
            _write(receipt_path, receipt)
            backward = training.forward_backward(data, "cross_entropy").result(timeout=timeout)
            receipt.update(state="gradients_computed", backward_metrics=backward.metrics)
            _write(receipt_path, receipt)
            receipt["state"] = "update_dispatched"
            _write(receipt_path, receipt)
            step = training.optim_step(tinker.types.AdamParams(
                learning_rate=self.config["learning_rate"])).result(timeout=timeout)
            receipt.update(state="updated", optimizer_metrics=step.metrics)
            _write(receipt_path, receipt)
            self.reserve("spindle_save", {})
            saved = training.save_state("wsm-student-" + identity, ttl_seconds=3600).result(timeout=timeout)
            receipt.update(state="complete", checkpoint=saved.path)
            _write(receipt_path, receipt)
            return receipt
        except BaseException as error:
            state = receipt["state"]
            receipt.update(last_state=state, state=("failed_before_training" if state in
                           {"client_created", "data_prepared"} else "outcome_unknown"),
                           error_type=type(error).__name__)
            _write(receipt_path, receipt)
            raise

    def close(self, status="success"):
        if self.service is not None:
            self.service.close(status).result(timeout=10)
            self.service = None
