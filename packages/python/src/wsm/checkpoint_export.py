"""Write the selected numerical state as a standalone artifact."""
import ctypes
import errno
import hashlib
from importlib.metadata import version
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from .binary64 import WeightlessError, apply_patch, parse_checkpoint
from .session import _hash, replay_session
from .target_worker import ModelTarget, snapshot_directory


def _fail(code, message):
    raise WeightlessError(code, message)


def _selected(bundle, target, arm):
    try:
        frozen = replay_session(bundle)
    except WeightlessError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, RecursionError):
        _fail("invalid_artifact", "Malformed session evidence")
    if frozen.get("complete") is not True:
        _fail("incomplete_freeze", "Complete freeze required")
    arms = frozen.get("arms")
    if type(arms) is not dict or arm not in arms:
        _fail("unknown_arm", "Frozen evidence has no such arm")
    if frozen.get("target_identity") != _hash(target):
        _fail("identity_mismatch", "Trusted target does not match the frozen descriptor")
    selected = arms[arm]
    patches = selected.get("patches")
    checkpoint = selected.get("checkpoint")
    if type(patches) is not list or type(checkpoint) is not str:
        _fail("invalid_artifact", "Selected arm has no checkpoint")
    return checkpoint, patches


def _proposal(base, patches):
    return {"base": base, "author": "export", "edits": patches}


def _bounded(destination, max_bytes):
    total = 0
    count = 0
    for dirpath, _dirnames, filenames in os.walk(destination):
        for name in filenames:
            count += 1
            total += os.path.getsize(os.path.join(dirpath, name))
            if total > max_bytes:
                _fail("invalid_artifact", "Exported checkpoint exceeds max_bytes")
    return total, count


def _export_binary(target, checkpoint, patches, destination, max_bytes):
    initial = parse_checkpoint(target["checkpoint"])
    edited = apply_patch(target["checkpoint"], _proposal(initial["id"], patches)) if patches else initial
    if edited["id"] != checkpoint:
        _fail("invalid_artifact", "Exported state does not match the selected checkpoint")
    path = destination / "checkpoint.json"
    data = (json.dumps(edited, indent=2, allow_nan=False) + "\n").encode()
    if len(data) > max_bytes:
        _fail("invalid_artifact", "Exported checkpoint exceeds max_bytes")
    path.write_bytes(data)
    return edited["id"]


def _rewrite_weights(snapshot, model, max_bytes):
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save
    except ImportError:
        _fail("unsupported_target", "Model dependencies are unavailable")
    total, _ = _bounded(snapshot, max_bytes)
    live = dict(model.model.named_parameters(remove_duplicate=False))
    seen = set()
    for path in sorted(Path(snapshot).rglob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata()
            loaded = {key: handle.get_tensor(key) for key in keys}
        changed = False
        for key in keys:
            if key not in live:
                continue
            current = live[key].detach()
            if current.device.type != "cpu":
                current = current.cpu()
            current = current.contiguous()
            if not torch.equal(loaded[key], current):
                loaded[key] = current
                changed = True
            seen.add(key)
        if changed:
            data = save(loaded, metadata=metadata)
            total += len(data) - path.stat().st_size
            if total > max_bytes:
                _fail("invalid_artifact", "Exported checkpoint exceeds max_bytes")
            path.write_bytes(data)
    missing = set(model.tensors) - seen
    if missing:
        _fail("invalid_artifact", "Selected parameters are not in the snapshot weights")


def _export_model(target, initial, checkpoint, patches, destination, device, max_bytes):
    model = ModelTarget(target, device=device)
    if model.identity() != initial:
        _fail("identity_mismatch", "Loaded target does not match the initial checkpoint")
    for item in patches:
        model.set(item["tensor"], item["index"], item["value"])
    if model.identity() != checkpoint:
        _fail("invalid_artifact", "Exported state does not match the selected checkpoint")
    snapshot = Path(snapshot_directory(target))
    copied = destination / "snapshots" / target["revision"]
    remaining = max_bytes

    def copy_bounded(source, output):
        nonlocal remaining
        with open(source, "rb") as reader, open(output, "xb") as writer:
            while chunk := reader.read(min(1024 * 1024, remaining + 1)):
                if len(chunk) > remaining:
                    _fail("invalid_artifact", "Exported checkpoint exceeds max_bytes")
                writer.write(chunk)
                remaining -= len(chunk)
        return output

    shutil.copytree(snapshot, copied, symlinks=False, copy_function=copy_bounded)
    _rewrite_weights(copied, model, max_bytes)
    reloaded = ModelTarget({**target, "source": str(destination)}, device=device)
    if reloaded.identity() != checkpoint:
        _fail("invalid_artifact", "Reloaded export does not match the selected checkpoint")
    return checkpoint


def _publish(source, destination):
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename is None:
        _fail("unsupported_platform", "Atomic exclusive export requires renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            _fail("export_exists", "Destination already exists")
        raise OSError(code, os.strerror(code), str(destination))


def _file_hashes(root):
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            _fail("invalid_artifact", "Export produced a symlink")
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                            "sha256": "sha256:" + digest})
    return records


def export_checkpoint(bundle, *, target, arm, destination, max_bytes, device="cpu"):
    """Write one selected arm so a later process can load it without the original source."""
    if type(max_bytes) is not int or max_bytes < 1:
        _fail("invalid_artifact", "max_bytes is required")
    if type(target) is not dict:
        _fail("invalid_target", "Invalid target")
    checkpoint, patches = _selected(bundle, target, arm)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        _fail("export_exists", "Export destination must be unused")
    if target.get("kind") in ("smollm", "huggingface"):
        snapshot = snapshot_directory(target)
        source = Path(target["source"]).resolve()
        if destination.resolve().is_relative_to(source):
            _fail("invalid_artifact", "Export destination must be outside the source model tree")
        _bounded(snapshot, max_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".wsm-export-", dir=destination.parent))
    try:
        kind = target.get("kind")
        if kind == "binary64":
            if parse_checkpoint(target["checkpoint"])["id"] != bundle["initial"]:
                _fail("identity_mismatch", "Loaded target does not match the initial checkpoint")
            checkpoint = _export_binary(target, checkpoint, patches, staged, max_bytes)
        elif kind in ("smollm", "huggingface"):
            checkpoint = _export_model(target, bundle["initial"], checkpoint, patches, staged, device, max_bytes)
        else:
            _fail("unsupported_target", "Unsupported target")
        total, count = _bounded(staged, max_bytes)
        runtime = {"python": sys.version.split()[0]}
        if kind != "binary64":
            runtime.update({name: version(name) for name in ("torch", "transformers", "safetensors")})
        receipt = {"format": "wsm.checkpoint-export.v1", "arm": arm, "checkpoint": checkpoint,
                   "bytes": total, "files": count, "selected": checkpoint, "initial": bundle["initial"],
                   "evidence": bundle["id"], "frozen": bundle["frozen"]["id"], "runner": bundle["runner"],
                   "serialization": "weightless.checkpoint.v1" if kind == "binary64" else "huggingface-snapshot",
                   "total_bytes": total, "file_hashes": _file_hashes(staged), "runtime": runtime,
                   "target": {key: target[key] for key in ("kind", "model", "revision", "dtype") if key in target}}
        (staged / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        _bounded(staged, max_bytes)
        _publish(staged, destination)
        return receipt
    finally:
        if staged.exists():
            shutil.rmtree(staged)
