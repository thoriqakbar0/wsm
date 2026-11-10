"""Experimental execution of a trusted Python weight-editing file.

The file defines `edit(model)`. `execute_program` evaluates the unchanged model,
calls that function, evaluates the same cases again, and publishes a checkpoint
only after `WeightModel.save` verifies it. The run uses the same lifecycle as
checkpoint publication: ready, running, staging, verified, then published.
`failure` and `interrupted` are reachable only before publication.

`output/run.json` is the receipt. A journal whose state is not `published` is
not completion, including when a crash leaves a staging directory or a verified
journal beside a renamed checkpoint. After the checkpoint directory is
published, a later receipt error must not relabel that directory as failed or
interrupted.
"""

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid

from .errors import WeightlessError
from .editing import WeightModel, _Lifecycle, _cases


def execute_program(program_path, *, model_path, output, device="cpu", cases=None):
    """Run a trusted `edit(model)` file and return its publication receipt.

    The receipt includes the source sha256, the baseline evaluation, and the
    candidate evaluation. Nothing in the receipt accepts or rejects the edit.
    `output` is created exclusively. An existing output path is left unchanged.
    """
    program_path = _existing_file(program_path, "invalid_program", "Program file could not be read")
    model_path = _existing_directory(model_path)
    output = _unused_output(output)
    if type(device) is not str or device == "":
        raise WeightlessError("unsupported_target", "Device must be a Torch device string")
    normalized = _cases(cases)
    source = _read_source(program_path)
    digest = "sha256:" + hashlib.sha256(source).hexdigest()
    try:
        text = source.decode("utf-8")
    except UnicodeError as error:
        raise WeightlessError("invalid_program", "Program file must be UTF-8") from error
    os.mkdir(output)
    journal = output / "run.json"
    receipt = {
        "format": "wsm.program-run.v1",
        "experimental": True,
        "state": "ready",
        "phases": [],
        "program": str(program_path),
        "source_sha256": digest,
        "model": str(model_path),
        "device": device,
        "case_count": len(normalized),
        "baseline": None,
        "candidate": None,
        "checkpoint": None,
        "stdout": None,
        "stderr": None,
        "error": None,
    }
    life = None
    checkpoint_published = False

    def record(state):
        receipt["state"] = state
        receipt["phases"].append(state)
        _write_json(journal, receipt)

    try:
        life = _Lifecycle(record)
        life.advance("running")
        code = compile(text, str(program_path), "exec")
        model = WeightModel.load(model_path, device=device)
        receipt["baseline"] = model.evaluate(normalized)
        _write_json(journal, receipt)
        stdout_path = output / "stdout.txt"
        stderr_path = output / "stderr.txt"
        try:
            _run_edit(program_path, code, model, stdout_path, stderr_path)
        finally:
            if stdout_path.is_file():
                receipt["stdout"] = _stream(stdout_path, "stdout.txt")
            if stderr_path.is_file():
                receipt["stderr"] = _stream(stderr_path, "stderr.txt")
        receipt["candidate"] = model.evaluate(normalized)
        _write_json(journal, receipt)
        checkpoint = output / "checkpoint"

        def on_save(state):
            if state == "staging" and life.state == "running":
                life.advance("staging")
            elif state == "verified" and life.state == "staging":
                life.advance("verified")

        try:
            model.save(checkpoint, on_phase=on_save)
        except BaseException:
            if checkpoint.is_dir() and not checkpoint.is_symlink():
                checkpoint_published = True
            raise
        checkpoint_published = True
        receipt["checkpoint"] = "checkpoint"
        life.advance("published")
        return copy.deepcopy(receipt)
    except BaseException as error:
        if checkpoint_published or (life is not None and life.published):
            raise
        if life is None:
            if output.is_dir() and not any(output.iterdir()):
                output.rmdir()
            raise
        _record_failure(life, receipt, journal, error)
        raise


def _existing_file(value, code, message):
    path = _path(value, code, message)
    if not path.is_file():
        raise WeightlessError(code, message)
    return path


def _existing_directory(value):
    path = _path(value, "invalid_model", "Local model snapshot could not be loaded")
    if not path.is_dir():
        raise WeightlessError("invalid_model", "Local model snapshot could not be loaded")
    return path


def _unused_output(value):
    path = _path(value, "output_exists", "Output directory must be unused")
    if path.exists() or path.is_symlink():
        raise WeightlessError("output_exists", "Output directory must be unused")
    if not path.parent.is_dir():
        raise WeightlessError("output_exists", "Output parent must be a directory")
    return path


def _path(value, code, message):
    if type(value) is not str and not isinstance(value, os.PathLike):
        raise WeightlessError(code, message)
    return Path(value).absolute()


def _read_source(path):
    try:
        return path.read_bytes()
    except OSError as error:
        raise WeightlessError("invalid_program", "Program file could not be read") from error


def _run_edit(program_path, code, model, stdout_path, stderr_path):
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            namespace = {"__file__": str(program_path), "__name__": "<run_path>"}
            exec(code, namespace)
            edit = namespace.get("edit")
            if not callable(edit):
                raise WeightlessError("invalid_program", "Program must define edit(model)")
            edit(model)


def _stream(path, relative):
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    record = {"path": relative, "bytes": len(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}
    if len(text) <= 4096:
        record["text"] = text
    else:
        record["text"] = text[:4096]
        record["truncated"] = True
    return record


def _record_failure(life, receipt, journal, error):
    receipt["error"] = {"type": type(error).__name__, "message": str(error)[:4000]}
    label = "interrupted" if isinstance(error, KeyboardInterrupt) else "failure"
    try:
        if life.state in ("running", "staging", "verified"):
            life.stop(label)
        else:
            _write_json(journal, receipt)
    except Exception:
        return


def _write_json(path, value):
    try:
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as error:
        raise WeightlessError("nonfinite", "JSON evidence contains a nonfinite number") from error
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
