"""Trusted operator export of one frozen selected checkpoint.

The caller supplies a complete evidence bundle and the same trusted target
descriptor the session hashed. binary64 writes checkpoint.json. The pinned
SmolLM2-135M-Instruct revision writes snapshots/<revision>/ with the original
metadata bytes and edited weights. receipt.json stays outside that snapshot.

max_bytes bounds the published directory, including the receipt. The binary64
encoder is rejected before it writes when the checkpoint exceeds that budget.
SmolLM preflight sums float32 tensor bytes from the safetensors header, so a
narrower source file does not understate the float32 export. Serialization
runs in a child process with RLIMIT_FSIZE set to max_bytes, which caps each
file. A serializer can briefly write one extra weight file before cleanup.
CPU, memory, and wall time are not set here. The operator applies those with
a supervisor such as Docker or ulimit.
A failed export removes only its temporary sibling. An existing destination is
left untouched. Retry after a failure reuses the same destination. Retry after
success needs a new destination.
"""
import ctypes
import errno
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .binary64 import WeightlessError, create_checkpoint, parse_checkpoint
from .session import _hash, replay_session
from .target_worker import BinaryTarget, edits_for

_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
_HASHED_SUFFIXES = (".json", ".txt", ".model", ".jinja")
_OVERHEAD = 1024 * 1024


def _fail(code, message):
    raise WeightlessError(code, message)


def _write_bytes(path, data, exclusive=False):
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_tree(root):
    root = Path(root)
    if root.is_symlink() or root.is_file():
        root.unlink(missing_ok=True)
        return
    if not root.exists():
        return
    for directory, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for name in filenames:
            Path(directory, name).unlink(missing_ok=True)
        for name in dirnames:
            path = Path(directory, name)
            if path.is_symlink():
                path.unlink()
            else:
                path.rmdir()
    root.rmdir()


def _metadata(snapshot):
    files = {}
    for entry in Path(snapshot).iterdir():
        if entry.is_symlink() or not entry.is_file():
            _fail("invalid_target", "Snapshot entries must be regular files")
        if entry.name.endswith(_HASHED_SUFFIXES):
            files[entry.name] = entry.read_bytes()
    return files


def _apply(backend, patches):
    for item in edits_for(list(patches), backend.parameters, backend.aliases):
        backend.set(item["tensor"], item["index"], item["value"])


def _export_binary(target, initial, selected, patches, directory, max_bytes):
    if set(target) != {"kind", "checkpoint"}:
        _fail("invalid_target", "binary64 target fields are kind and checkpoint")
    backend = BinaryTarget(target)
    if backend.identity() != initial:
        _fail("identity_mismatch", "Loaded target does not match the evidence initial checkpoint")
    _apply(backend, patches)
    if backend.identity() != selected:
        _fail("identity_mismatch", "Edited target does not match the frozen checkpoint")
    checkpoint = create_checkpoint(backend.checkpoint["graph"], backend.checkpoint["tensors"])
    if checkpoint["id"] != selected:
        _fail("identity_mismatch", "Checkpoint encoding does not match the frozen checkpoint")
    payload = (json.dumps(checkpoint, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(payload) > max_bytes:
        _fail("export_too_large", "Checkpoint exceeds max_bytes")
    _write_bytes(directory / "checkpoint.json", payload, exclusive=True)
    loaded = parse_checkpoint((directory / "checkpoint.json").read_text())
    if loaded["id"] != selected:
        _fail("identity_mismatch", "Reloaded checkpoint does not match the frozen checkpoint")
    return "weightless.checkpoint.v1"


def _limit_files(max_bytes):
    import resource
    import signal
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, max_bytes))
    except (OSError, ValueError):
        _fail("export_failed", "The process file-size limit could not be set")


def _restore_metadata(snapshot, original):
    snapshot = Path(snapshot)
    for entry in list(snapshot.iterdir()):
        if entry.is_symlink():
            _fail("export_failed", "Serializer produced a symlink")
        if entry.name.endswith(_HASHED_SUFFIXES) and entry.name not in original:
            entry.unlink()
    for name, data in original.items():
        _write_bytes(snapshot / name, data)
    allowed = set(original) | {"model.safetensors"}
    for entry in list(snapshot.iterdir()):
        if entry.name in allowed:
            continue
        if entry.is_symlink() or not entry.is_file():
            _fail("export_failed", "Serializer produced an unexpected entry")
        entry.unlink()
    if {entry.name for entry in snapshot.iterdir()} != allowed:
        _fail("export_failed", "Exported snapshot file set does not match the pinned model")


def _serialize_smollm(request):
    from .target_worker import ModelTarget
    target = request["target"]
    backend = ModelTarget(target)
    if backend.identity() != request["initial"]:
        _fail("identity_mismatch", "Loaded target does not match the evidence initial checkpoint")
    _apply(backend, request["patches"])
    selected = backend.identity()
    if selected != request["selected"]:
        _fail("identity_mismatch", "Edited target does not match the frozen checkpoint")
    source = Path(target["source"]) / "snapshots" / target["revision"]
    original = _metadata(source)
    _limit_files(request["max_bytes"])
    snapshot = Path(request["snapshot"])
    try:
        backend.model.save_pretrained(snapshot, safe_serialization=True)
        backend.tokenizer.save_pretrained(snapshot)
    except OSError as error:
        if error.errno == errno.EFBIG:
            _fail("export_too_large", "Serialization exceeded max_bytes")
        raise
    except Exception as error:
        if "file too large" in str(error).lower():
            _fail("export_too_large", "Serialization exceeded max_bytes")
        raise
    _restore_metadata(snapshot, original)
    return {"identity": selected, "aliases": backend.aliases}


def _verify_smollm(request):
    from .target_worker import ModelTarget
    backend = ModelTarget(request["target"])
    return {"identity": backend.identity(), "aliases": backend.aliases}


def _child_environment(temp):
    home = temp / "hf-home"
    home.mkdir(mode=0o700, exist_ok=True)
    env = os.environ.copy()
    for name in list(env):
        upper = name.upper()
        if any(word in upper for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
            env.pop(name)
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", HF_HOME=str(home), HUGGINGFACE_HUB_CACHE=str(home),
               TRANSFORMERS_CACHE=str(home), PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    return env


def _child(request, temp):
    result_path = temp / "child-result.json"
    request = {**request, "result": str(result_path)}
    proc = subprocess.run([sys.executable, "-m", "wsm.checkpoint_export"], input=json.dumps(request, allow_nan=False).encode(),
                          cwd=temp, env=_child_environment(temp), capture_output=True, start_new_session=True)
    payload = None
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text())
        except ValueError:
            payload = None
        result_path.unlink()
    if type(payload) is dict and payload.get("ok") is True and proc.returncode == 0:
        return payload
    if type(payload) is dict and type(payload.get("code")) is str and type(payload.get("message")) is str:
        _fail(payload["code"], payload["message"])
    detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
    suffix = detail[-1][:200] if detail else "Serialization failed"
    _fail("export_failed", suffix)


def _check_smollm_target(target):
    if (set(target) != {"kind", "source", "revision", "model", "dtype"} or target["revision"] != _REVISION
            or target["model"] != _MODEL or target["dtype"] != "float32"
            or type(target["source"]) is not str or not os.path.isabs(target["source"])):
        _fail("unsupported_target", "Export supports binary64 and the pinned SmolLM2 checkpoint only")
    snapshot = Path(target["source"]) / "snapshots" / target["revision"]
    if snapshot.is_symlink() or not snapshot.is_dir():
        _fail("invalid_target", "Pinned snapshot directory is required")
    return snapshot


def _stored_bytes(path):
    if path.suffix != ".safetensors":
        return path.stat().st_size
    with path.open("rb") as stream:
        header_len = int.from_bytes(stream.read(8), "little")
        if header_len <= 0 or header_len > 100_000_000:
            _fail("invalid_target", "Weight file header is invalid")
        try:
            header = json.loads(stream.read(header_len))
        except (ValueError, RecursionError):
            _fail("invalid_target", "Weight file header is invalid")
    if type(header) is not dict:
        _fail("invalid_target", "Weight file header is invalid")
    total = header_len + 8
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape") if type(meta) is dict else None
        if type(shape) is not list or not shape:
            _fail("invalid_target", "Weight file header is invalid")
        numel = 1
        for dim in shape:
            if type(dim) is not int or isinstance(dim, bool) or dim < 1:
                _fail("invalid_target", "Weight file header is invalid")
            numel *= dim
        total += numel * 4
    return total


def _estimate(snapshot, max_bytes):
    total = largest = 0
    for entry in snapshot.iterdir():
        if entry.is_symlink() or not entry.is_file():
            _fail("invalid_target", "Snapshot entries must be regular files")
        size = _stored_bytes(entry)
        total += size
        largest = max(largest, size)
    if largest > max_bytes or total + _OVERHEAD > max_bytes:
        _fail("export_too_large", "Trusted snapshot plus export overhead exceeds max_bytes")


def _same_metadata(source, exported):
    left, right = _metadata(source), _metadata(exported)
    if set(left) != set(right) or any(left[name] != right[name] for name in left):
        _fail("export_failed", "Exported metadata bytes differ from the trusted snapshot")


def _export_smollm(target, initial, selected, patches, temp, max_bytes):
    source = _check_smollm_target(target)
    _estimate(source, max_bytes)
    snapshot = temp / "snapshots" / target["revision"]
    snapshot.mkdir(parents=True, mode=0o755)
    produced = _child({"op": "serialize", "target": target, "patches": patches, "snapshot": str(snapshot),
                       "initial": initial, "selected": selected, "max_bytes": max_bytes}, temp)
    _remove_tree(temp / "hf-home")
    if produced.get("identity") != selected:
        _fail("identity_mismatch", "Serialized model does not match the frozen checkpoint")
    _same_metadata(source, snapshot)
    exported = {"kind": "smollm", "source": str(temp.resolve()), "revision": target["revision"],
                "model": target["model"], "dtype": "float32"}
    verified = _child({"op": "verify", "target": exported}, temp)
    _remove_tree(temp / "hf-home")
    if verified.get("identity") != selected or verified.get("aliases") != produced.get("aliases"):
        _fail("identity_mismatch", "Reloaded model does not match the frozen checkpoint")
    for entry in list(temp.iterdir()):
        if entry.name != "snapshots":
            _remove_tree(entry)
    revision_root = temp / "snapshots"
    if {entry.name for entry in revision_root.iterdir()} != {target["revision"]}:
        _fail("export_failed", "Export snapshot revision changed")
    return "huggingface-snapshot"


def _artifacts(root):
    records = []
    total = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            if Path(directory, name).is_symlink():
                _fail("export_failed", "Export produced a symlink")
        for name in filenames:
            path = Path(directory, name)
            if path.is_symlink() or not path.is_file():
                _fail("export_failed", "Export produced a symlink")
            size = path.stat().st_size
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append({"path": path.relative_to(root).as_posix(), "bytes": size,
                            "sha256": "sha256:" + digest})
            total += size
    if not records:
        _fail("export_failed", "Export produced no files")
    records.sort(key=lambda item: item["path"])
    return records, total


def _fsync_tree(root):
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            descriptor = os.open(os.path.join(directory, name), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(directory)


def _publish(source, destination):
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        _fail("unsupported_platform", "Atomic exclusive export requires renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            _fail("export_exists", "Destination already exists")
        raise OSError(code, os.strerror(code), str(destination))


def export_checkpoint(bundle, *, target, arm, destination, max_bytes):
    if type(max_bytes) is not int or isinstance(max_bytes, bool) or max_bytes < 1:
        _fail("invalid_request", "max_bytes must be a positive integer")
    if type(bundle) is not dict or type(target) is not dict or type(arm) is not str:
        _fail("invalid_request", "Export inputs are invalid")
    try:
        frozen = replay_session(bundle)
    except WeightlessError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, RecursionError):
        _fail("invalid_artifact", "Malformed session evidence")
    if frozen.get("complete") is not True:
        _fail("incomplete_freeze", "Complete freeze required")
    if arm not in frozen["arms"]:
        _fail("unknown_arm", "Frozen evidence has no such arm")
    if frozen.get("target_identity") != _hash(target):
        _fail("identity_mismatch", "Trusted target does not match the frozen descriptor")
    selected = frozen["arms"][arm]["checkpoint"]
    patches = frozen["arms"][arm]["patches"]
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        _fail("export_exists", "Destination already exists")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        _fail("invalid_request", "Destination parent must be an existing directory")
    if target.get("kind") == "smollm":
        _check_smollm_target(target)
        source = Path(target["source"]).resolve()
        resolved_parent = parent.resolve()
        if resolved_parent == source or source in resolved_parent.parents:
            _fail("invalid_request", "Destination must not be inside the trusted source")
    temp = parent / f".wsm-export-{uuid.uuid4().hex}"
    temp.mkdir(mode=0o700)
    published = False
    try:
        if target.get("kind") == "binary64":
            serialization = _export_binary(target, bundle["initial"], selected, patches, temp, max_bytes)
        elif target.get("kind") == "smollm":
            serialization = _export_smollm(target, bundle["initial"], selected, patches, temp, max_bytes)
        else:
            _fail("unsupported_target", "Export supports binary64 and the pinned SmolLM2 checkpoint only")
        records, total = _artifacts(temp)
        runtime = {"python": sys.version.split()[0]}
        if target["kind"] == "smollm":
            runtime.update({name: version(name) for name in ("torch", "transformers", "safetensors")})
        receipt = {"format": "weightless.checkpoint-export.v1", "evidence": bundle["id"], "frozen": frozen["id"],
                   "arm": arm, "initial": bundle["initial"], "selected": selected, "serialization": serialization,
                   "runner": bundle["runner"], "runtime": runtime,
                   "target": {key: target[key] for key in ("kind", "model", "revision", "dtype") if key in target},
                   "total_bytes": total, "files": records}
        payload = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if total + len(payload) > max_bytes:
            _fail("export_too_large", "Published checkpoint exceeds max_bytes")
        _write_bytes(temp / "receipt.json", payload, exclusive=True)
        _fsync_tree(temp)
        _publish(temp, destination)
        _fsync_directory(parent)
        published = True
        return receipt
    finally:
        if not published:
            _remove_tree(temp)


def _child_main():
    request = {}
    try:
        request = json.load(sys.stdin)
        if request.get("op") == "serialize":
            value = _serialize_smollm(request)
        elif request.get("op") == "verify":
            value = _verify_smollm(request)
        else:
            raise WeightlessError("invalid_request", "Unknown export operation")
        _write_bytes(Path(request["result"]), json.dumps({"ok": True, **value}, sort_keys=True).encode(), exclusive=True)
        return 0
    except WeightlessError as error:
        payload = {"ok": False, "code": error.code, "message": str(error)}
    except Exception as error:
        payload = {"ok": False, "code": "export_failed", "message": f"{type(error).__name__}: {error}".splitlines()[0][:200]}
    try:
        _write_bytes(Path(request["result"]), json.dumps(payload).encode(), exclusive=True)
    except Exception:
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(_child_main())
