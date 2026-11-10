"""Budget-bound Responses transport with private, durable send receipts."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit
import uuid

from .binary64 import WeightlessError


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _load(raw):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def _write(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ResponsesTeacher:
    """Only a trusted broker should hold this object or its artifact directory."""

    def __init__(self, *, base_url, model, api_key, directory, timeout=60, max_output_tokens=512):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or type(model) is not str or not model or type(api_key) is not str or not api_key
                or type(timeout) not in (int, float) or not 0 < timeout <= 300
                or type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 4096):
            raise WeightlessError("invalid_teacher", "Invalid teacher configuration")
        self._url, self._model, self._key = parsed, model, api_key
        self._timeout, self._max_tokens = timeout, max_output_tokens
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self._directory.stat().st_mode & 0o077:
            raise WeightlessError("invalid_teacher", "Teacher artifacts require a private directory")

    def ask(self, session, prompt, *, key, arm, purpose="development"):
        """Send once; replay a saved response or refuse an uncertain resend."""
        if type(prompt) is not str or not prompt or len(prompt.encode()) > 262144:
            raise WeightlessError("invalid_teacher", "Teacher prompt exceeds its bound")
        request = {"model": self._model, "input": [{"role": "user", "content": prompt}],
                   "tools": [], "tool_choice": "none", "stream": True,
                   "reasoning": {"effort": "low"}, "max_output_tokens": self._max_tokens}
        reservation = session.reserve_teacher(request, key=key, arm=arm, purpose=purpose)
        identity = {"session": session.observe(arm)["session"], "key": key}
        folder = self._directory / hashlib.sha256(_canonical(identity)).hexdigest()
        folder.mkdir(mode=0o700, exist_ok=True)
        try:
            import fcntl
        except ImportError:
            session.unknown_teacher(key)
            raise WeightlessError("unsupported", "Durable teacher locking requires POSIX") from None
        with (folder / "lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WeightlessError("teacher_busy", "This teacher operation is already active") from None
            saved_request = folder / "request.json"
            if saved_request.exists() and _load(saved_request.read_bytes()) != request:
                raise WeightlessError("idempotency_conflict", "Teacher key has different prompt bytes")
            if not saved_request.exists():
                _write(saved_request, request)
            response_path = folder / "response.json"
            if response_path.exists():
                response = _load(response_path.read_bytes())
                session.complete_teacher(key, response)
                return response
            if (folder / "send-intent.json").exists() or reservation["status"] != "dispatched":
                session.unknown_teacher(key)
                raise WeightlessError("teacher_outcome_unknown", "A possible prior send cannot be repeated")
            timeout = min(self._timeout, session.remaining_seconds())
            _write(folder / "send-intent.json", {"request_sha256": hashlib.sha256(_canonical(request)).hexdigest()})
            try:
                with session.teacher_dispatch(key) as started:
                    response = self._send(request, timeout, folder, on_dispatch=started)
                _write(response_path, response)
                session.complete_teacher(key, response)
                return response
            except Exception:
                session.unknown_teacher(key)
                _write(folder / "failure.json", {"status": "outcome_unknown", "automatic_retry": False})
                raise WeightlessError("teacher_outcome_unknown", "Teacher completion is uncertain; request retained") from None

    def complete(self, prompt, *, key, display_model="Codex", reasoning_effort="low"):
        """Complete one controller-owned turn without coupling it to a Session."""
        if (type(prompt) is not str or not prompt or len(prompt.encode()) > 262144
                or type(key) is not str or not key or type(display_model) is not str or not display_model
                or reasoning_effort not in ("minimal", "low", "medium", "high", "xhigh")):
            raise WeightlessError("invalid_teacher", "Invalid completion prompt or identity")
        request = {"model": self._model, "input": [{"role": "user", "content": prompt}],
                   "tools": [], "tool_choice": "none", "stream": True,
                   "reasoning": {"effort": reasoning_effort}, "max_output_tokens": self._max_tokens}
        folder = self._directory / hashlib.sha256(_canonical({"key": key})).hexdigest()
        folder.mkdir(mode=0o700, exist_ok=True)
        import fcntl
        with (folder / "lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WeightlessError("teacher_busy", "This completion is already active") from None
            record = {"request": {**request, "model": display_model},
                      "wire_request_sha256": hashlib.sha256(_canonical(request)).hexdigest()}
            saved = folder / "request.json"
            if saved.exists() and _load(saved.read_bytes()) != record:
                raise WeightlessError("idempotency_conflict", "Completion key has different request bytes")
            if not saved.exists():
                _write(saved, record)
            response_path = folder / "response.json"
            if response_path.exists():
                return _load(response_path.read_bytes())
            intent = folder / "send-intent.json"
            if intent.exists():
                raise WeightlessError("teacher_outcome_unknown", "A possible prior send cannot be repeated")
            _write(intent, {"request_sha256": record["wire_request_sha256"]})
            try:
                response = self._send(request, self._timeout, folder, display_model=display_model)
                _write(response_path, response)
                return response
            except BaseException:
                _write(folder / "failure.json", {"status": "outcome_unknown", "automatic_retry": False})
                raise WeightlessError("teacher_outcome_unknown", "Completion uncertain; inspect retained records") from None

    def _send(self, request, timeout, folder, on_dispatch=None, display_model=None):
        deadline = time.monotonic() + timeout
        cls = http.client.HTTPSConnection if self._url.scheme == "https" else http.client.HTTPConnection
        connection = cls(self._url.hostname, self._url.port, timeout=timeout)
        guard = timer = None
        events = []
        try:
            connection.connect()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            connection.sock.settimeout(remaining)
            guard = connection.sock.dup()
            def interrupt():
                try:
                    guard.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            timer = threading.Timer(remaining, interrupt)
            timer.daemon = True
            timer.start()
            path = self._url.path.rstrip("/") + "/responses"
            connection.request("POST", path, body=_canonical(request), headers={
                "Authorization": "Bearer " + self._key, "Content-Type": "application/json"})
            if on_dispatch is not None:
                on_dispatch()
            reply = connection.getresponse()
            if reply.status != 200:
                raise ValueError("Non-successful response")
            used = 0
            completed = None
            while True:
                line = reply.readline(2 * 1024 * 1024 + 1)
                used += len(line)
                if used > 2 * 1024 * 1024 or len(events) >= 4096 or time.monotonic() > deadline:
                    raise ValueError("Response bound exceeded")
                if not line:
                    break
                if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                    continue
                event = _load(line[6:])
                if type(event) is not dict:
                    raise ValueError("Invalid event")
                events.append(event)
                if any(word in event.get("type", "") for word in
                       ("tool_call", "function_call", "web_search", "code_interpreter", "mcp_call")):
                    raise ValueError("Unexpected tool event")
                if event.get("type") == "response.completed":
                    completed = event.get("response")
                    break
            _write(folder / "events.json", self._event_record(events, display_model))
            if (type(completed) is not dict or completed.get("status") != "completed"
                    or completed.get("model") != self._model or completed.get("tools") != []
                    or completed.get("tool_choice") != "none"):
                raise ValueError("Unexpected completion identity or tools")
            items = completed.get("output", [])
            if type(items) is not list or any(item.get("type") not in ("message", "reasoning") for item in items):
                raise ValueError("Unexpected output item")
            text = "".join(part["text"] for item in items if item.get("type") == "message"
                           for part in item.get("content", []) if part.get("type") == "output_text")
            if not text:
                raise ValueError("Missing literal response")
            usage = completed.get("usage", {})
            safe_usage = {name: usage[name] for name in ("input_tokens", "output_tokens", "total_tokens")
                          if type(usage.get(name)) is int and usage[name] >= 0}
            response = {"text": text, "model": display_model or completed["model"], "tools": [], "tool_choice": "none",
                    "tool_events": [], "usage": safe_usage, "assurance": "observed CLIProxy/API boundary only"}
            if display_model is not None:
                response["response_id"] = completed.get("id")
            return response
        finally:
            if timer:
                timer.cancel()
            connection.close()
            if guard:
                guard.close()
            if events and not (folder / "events.json").exists():
                _write(folder / "events.json", self._event_record(events, display_model))

    def _event_record(self, value, display_model):
        if display_model is None:
            return value
        if isinstance(value, dict):
            return {key: display_model if key == "model" else self._event_record(item, display_model)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [self._event_record(item, display_model) for item in value]
        return display_model if value == self._model else value
