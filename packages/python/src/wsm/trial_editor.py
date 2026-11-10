"""Codex editing through an explicitly configured Responses-compatible proxy."""

import hashlib
import ipaddress
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from .trial_transport import ResponsesTeacher


def validate_config(config):
    required = {"provider", "base_url", "model_env", "api_key_env", "max_tokens", "max_turns", "request_seconds"}
    if (type(config) is not dict or not required <= config.keys()
            or set(config) - required - {"reasoning_effort"} or config["provider"] != "responses_proxy"):
        raise ValueError("Explicit Responses proxy endpoint, environment names, and turn limits are required")
    endpoint = config["base_url"]
    if type(endpoint) is not str or any(character.isspace() for character in endpoint):
        raise ValueError("Invalid proxy endpoint")
    try:
        parsed = urlsplit(endpoint)
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname == "localhost"
        valid = (parsed.hostname and (parsed.scheme == "https" or parsed.scheme == "http" and loopback)
                 and parsed.port != 0 and parsed.username is None and parsed.password is None
                 and not parsed.query and not parsed.fragment)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Use HTTPS, or HTTP only for a loopback proxy, without URL credentials")
    for name in ("model_env", "api_key_env"):
        if type(config[name]) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config[name]):
            raise ValueError(name + " must name an environment variable")
    for name in ("max_tokens", "max_turns"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + " must be a positive integer")
    if config["max_tokens"] > 4096:
        raise ValueError("max_tokens cannot exceed 4096")
    if type(config["request_seconds"]) not in (int, float) or not 0 < config["request_seconds"] <= 300:
        raise ValueError("request_seconds must be in (0, 300]")
    if config.get("reasoning_effort", "low") not in ("minimal", "low", "medium", "high", "xhigh"):
        raise ValueError("Unsupported reasoning effort")
    return dict(config)


class ProxyEditor:
    """One fresh teacher trajectory. Never train this closed editor through Spindle."""

    def __init__(self, config, *, directory, reserve):
        self.config = validate_config(config)
        model, api_key = os.environ.get(config["model_env"]), os.environ.get(config["api_key_env"])
        if not model or not api_key:
            raise ValueError("Configure the selected editor model and proxy credential environment variables")
        if not callable(reserve):
            raise ValueError("A reservation callback is required")
        directory = Path(directory)
        directory.mkdir(mode=0o700)
        self.transport = ResponsesTeacher(base_url=config["base_url"], model=model, api_key=api_key,
                                          directory=directory, timeout=config["request_seconds"],
                                          max_output_tokens=config["max_tokens"])
        self.route_sha256 = hashlib.sha256(model.encode()).hexdigest()
        self.reserve, self.turns, self.state = reserve, 0, "ready"

    def sample(self, prompt):
        if self.state != "ready":
            raise ValueError("Inspect the uncertain teacher turn before starting another request")
        if self.turns >= self.config["max_turns"]:
            raise ValueError("Editing-agent turn limit reached")
        if type(prompt) is not str or not prompt or len(prompt.encode()) > 262144:
            raise ValueError("A bounded nonempty public prompt is required")
        self.reserve("teacher_request", {"requests": 1, "max_output_tokens": self.config["max_tokens"]})
        self.turns += 1
        self.state = "pending"
        try:
            response = self.transport.complete(prompt, key="turn-" + str(self.turns), display_model="Codex",
                                               reasoning_effort=self.config.get("reasoning_effort", "low"))
        except BaseException:
            self.state = "outcome_unknown"
            raise
        self.state = "ready"
        return {"prompt": prompt, "text": response["text"], "teacher": "Codex", "provider": "responses_proxy",
                "response_id": response.get("response_id"), "usage": response["usage"],
                "route_sha256": self.route_sha256}

    def close(self, status="success"):
        pass
