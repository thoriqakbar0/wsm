"""Tinker-compatible policy sampling and one terminal-reward training update.

Only generated response tokens receive advantages. Tool observations remain
conditioning tokens with zero advantage. This module never runs agent Python.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit
import uuid


def _finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _tokens(values, name):
    if (type(values) is not list or not values
            or any(type(value) is not int or value < 0 for value in values)):
        raise ValueError(f"{name} must contain nonnegative integer tokens")
    return values


def build_datums(trajectory, reward, baseline):
    """Build SDK datums from exact sampled turns, with response-only RL loss."""
    import tinker

    advantage = _finite(reward, "reward") - _finite(baseline, "baseline")
    if not math.isfinite(advantage):
        raise ValueError("advantage must be finite")
    if type(trajectory) is not list or not trajectory:
        raise ValueError("Training requires a nonempty sampled trajectory")
    policies = {turn.get("policy") for turn in trajectory if type(turn) is dict}
    if len(policies) != 1 or None in policies:
        raise ValueError("Every turn must identify the same sampled policy")
    data = []
    for turn in trajectory:
        if type(turn) is not dict:
            raise ValueError("Each sampled turn must be an object")
        prompt = _tokens(turn.get("prompt_tokens"), "prompt_tokens")
        response = _tokens(turn.get("tokens"), "tokens")
        logprobs = turn.get("logprobs")
        if type(logprobs) is not list or len(logprobs) != len(response):
            raise ValueError("Each generated token requires its sampling log probability")
        logprobs = [_finite(value, "logprob") for value in logprobs]
        if any(value > 0 for value in logprobs):
            raise ValueError("Sampling log probabilities cannot be positive")
        tokens = prompt + response
        prefix = len(prompt) - 1
        data.append(tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.types.TensorData(data=tokens[1:], dtype="int64"),
                "logprobs": tinker.types.TensorData(data=[0.0] * prefix + logprobs, dtype="float32"),
                "advantages": tinker.types.TensorData(
                    data=[0.0] * prefix + [advantage] * len(response), dtype="float32"),
            },
        ))
    return data


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_config(config):
    required = {"provider", "base_url", "model", "rank", "loss_fn", "learning_rate", "reward_baseline",
                "max_tokens", "max_turns"}
    if (type(config) is not dict or not required <= config.keys()
            or set(config) - required - {"temperature", "request_seconds", "enable_thinking"}):
        raise ValueError("Explicit provider, endpoint, model, training settings, and rollout limits are required")
    if config["provider"] not in ("tinker", "spindle"):
        raise ValueError("Select tinker or spindle explicitly")
    endpoint = config["base_url"]
    if type(endpoint) is not str or not endpoint or any(char.isspace() for char in endpoint):
        raise ValueError("base_url must be an explicit HTTPS endpoint without embedded credentials")
    try:
        parsed = urlsplit(endpoint)
        valid_endpoint = (parsed.scheme == "https" and parsed.hostname and parsed.port != 0
                          and parsed.username is None and parsed.password is None
                          and not parsed.query and not parsed.fragment)
    except ValueError:
        valid_endpoint = False
    if not valid_endpoint:
        raise ValueError("base_url must be an explicit HTTPS endpoint without embedded credentials or query")
    if type(config["model"]) is not str or not config["model"].strip():
        raise ValueError("model must be an explicit supported Tinker model ID")
    for key in ("rank", "max_tokens", "max_turns"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["loss_fn"] not in ("importance_sampling", "ppo"):
        raise ValueError("Select importance_sampling or ppo explicitly")
    if _finite(config["learning_rate"], "learning_rate") <= 0:
        raise ValueError("learning_rate must be positive")
    _finite(config["reward_baseline"], "reward_baseline")
    if _finite(config.get("temperature", 1.0), "temperature") <= 0:
        raise ValueError("Policy-gradient sampling requires positive temperature")
    if _finite(config.get("request_seconds", 120), "request_seconds") <= 0:
        raise ValueError("request_seconds must be positive")
    if type(config.get("enable_thinking", False)) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    return dict(config)


class TinkerPolicy:
    """One sampled editing policy and one explicit training update.

    ``reserve(operation, units)`` must durably reserve the operation's maximum
    cost before returning. Client credentials remain in the host process.
    """

    def __init__(self, config, *, reserve):
        import tinker

        validate_config(config)
        credential_name = "WSM_SPINDLE_API_KEY" if config["provider"] == "spindle" else "TINKER_API_KEY"
        api_key = os.environ.get(credential_name)
        if not api_key:
            raise ValueError(credential_name + " is required for the selected training provider")
        if not callable(reserve):
            raise ValueError("A durable cost reservation callback is required")
        self.config = dict(config)
        self.reserve = reserve
        self.policy_id = str(uuid.uuid4())
        self.turns = 0
        self.training_started = False
        self.service = None
        self._reserve("create", {})
        try:
            self.service = tinker.ServiceClient(base_url=config["base_url"], api_key=api_key)
            self.training = self.service.create_lora_training_client(
                base_model=config["model"], rank=config["rank"]).result(timeout=config.get("request_seconds", 120))
            self.training_run = self.training.get_info().model_id
            self.tokenizer = self.training.get_tokenizer()
            self._reserve("snapshot", {})
            self.sampler = self.training.save_weights_and_get_sampling_client().result(
                timeout=config.get("request_seconds", 120))
        except BaseException:
            if self.service is not None:
                self.service.close("errored").result(timeout=10)
            raise

    def _reserve(self, operation, units):
        self.reserve(self.config["provider"] + "_" + operation, units)

    def sample(self, prompt):
        """Sample a single agent turn and retain exact training token boundaries."""
        import tinker

        if self.training_started:
            raise ValueError("This policy's trial has already entered training")
        if self.turns >= self.config["max_turns"]:
            raise ValueError("Editing-agent turn limit reached")
        if type(prompt) is not str or not prompt:
            raise ValueError("A nonempty public prompt is required")
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            enable_thinking=self.config.get("enable_thinking", False))
        _tokens(tokens, "prompt_tokens")
        self._reserve("sample", {
            "prefill_tokens": len(tokens), "sample_tokens": self.config["max_tokens"],
        })
        self.turns += 1
        output = self.sampler.sample(
            prompt=tinker.types.ModelInput.from_ints(tokens), num_samples=1,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=self.config["max_tokens"], temperature=self.config.get("temperature", 1.0)),
        ).result(timeout=self.config.get("request_seconds", 120))
        if len(output.sequences) != 1:
            raise ValueError("Expected exactly one sampled editing-agent response")
        sequence = output.sequences[0]
        record = {
            "policy": self.policy_id,
            "provider": self.config["provider"],
            "model": self.config["model"],
            "training_run": self.training_run,
            "prompt_tokens": list(tokens),
            "tokens": list(sequence.tokens),
            "logprobs": None if sequence.logprobs is None else list(sequence.logprobs),
            "text": self.tokenizer.decode(sequence.tokens, skip_special_tokens=True),
            "stop_reason": sequence.stop_reason,
        }
        build_datums([record], 0.0, 0.0)
        return record

    def train(self, trajectory, reward, *, output):
        """Persist intent, apply one final-reward update, and save training state.

        An uncertain update is never retried. An existing output directory is
        rejected even if its receipt records failure or uncertainty.
        """
        import tinker

        if self.training_started:
            raise ValueError("This policy has already attempted a training update")
        data = build_datums(trajectory, reward, self.config["reward_baseline"])
        if any(turn["policy"] != self.policy_id for turn in trajectory):
            raise ValueError("Trajectory was not sampled from this policy")
        directory = Path(output)
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        path = directory / "receipt.json"
        encoded = json.dumps(trajectory, sort_keys=True, allow_nan=False).encode()
        receipt = {
            "format": "wsm.trial-training.v1", "state": "prepared",
            "policy": self.policy_id, "model": self.config["model"],
            "provider": self.config["provider"], "base_url": self.config["base_url"],
            "training_run": getattr(self, "training_run", None),
            "trajectory_sha256": hashlib.sha256(encoded).hexdigest(),
            "reward": _finite(reward, "reward"), "baseline": self.config["reward_baseline"],
            "advantage": float(reward) - self.config["reward_baseline"],
            "loss_fn": self.config["loss_fn"], "learning_rate": self.config["learning_rate"],
            "turns": len(data), "response_tokens": sum(len(turn["tokens"]) for turn in trajectory),
            "training_tokens": sum(datum.model_input.length for datum in data),
            "checkpoint": None,
        }
        _write(path, receipt)
        self._reserve("train", {"train_tokens": receipt["training_tokens"]})
        self.training_started = True
        receipt["state"] = "dispatched"
        _write(path, receipt)
        try:
            backward = self.training.forward_backward(data, self.config["loss_fn"]).result(
                timeout=self.config.get("request_seconds", 120))
            receipt["backward_metrics"] = backward.metrics
            receipt["state"] = "gradients_computed"
            _write(path, receipt)
            step = self.training.optim_step(
                tinker.types.AdamParams(learning_rate=self.config["learning_rate"])).result(
                    timeout=self.config.get("request_seconds", 120))
            receipt["optimizer_metrics"] = step.metrics
            receipt["state"] = "updated"
            _write(path, receipt)
            self._reserve("save", {})
            saved = self.training.save_state("wsm-trial-" + self.policy_id, ttl_seconds=3600).result(
                timeout=self.config.get("request_seconds", 120))
            receipt["checkpoint"] = saved.path
            receipt["state"] = "complete"
            _write(path, receipt)
            return receipt
        except BaseException as error:
            receipt["last_state"] = receipt["state"]
            receipt["state"] = "outcome_unknown"
            receipt["error_type"] = type(error).__name__
            _write(path, receipt)
            raise

    def close(self, status="success"):
        if self.service is not None:
            self.service.close(status).result(timeout=10)
            self.service = None
