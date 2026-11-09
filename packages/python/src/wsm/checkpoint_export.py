"""Write the selected numerical state as a standalone artifact."""
import json
import os
import shutil
from pathlib import Path

from .binary64 import WeightlessError, apply_patch, parse_checkpoint
from .session import _hash, replay_session
from .target_worker import ModelTarget, snapshot_directory


def _fail(code, message):
    raise WeightlessError(code, message)


def _selected(bundle, target, arm):
    frozen = replay_session(bundle)
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


def _export_binary(target, checkpoint, patches, destination):
    initial = parse_checkpoint(target["checkpoint"])
    edited = apply_patch(target["checkpoint"], _proposal(initial["id"], patches)) if patches else initial
    if edited["id"] != checkpoint:
        _fail("invalid_artifact", "Exported state does not match the selected checkpoint")
    path = destination / "checkpoint.json"
    path.write_text(json.dumps(edited, indent=2, allow_nan=False) + "\n")
    return edited["id"]


def _rewrite_weights(snapshot, model):
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        _fail("unsupported_target", "Model dependencies are unavailable")
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
            temporary = path.with_name(path.name + ".tmp")
            save_file(loaded, str(temporary), metadata=metadata)
            os.replace(temporary, path)
    missing = set(model.tensors) - seen
    if missing:
        _fail("invalid_artifact", "Selected parameters are not in the snapshot weights")


def _export_model(target, checkpoint, patches, destination, device):
    model = ModelTarget(target, device=device)
    for item in patches:
        model.set(item["tensor"], item["index"], item["value"])
    if model.identity() != checkpoint:
        _fail("invalid_artifact", "Exported state does not match the selected checkpoint")
    snapshot = Path(snapshot_directory(target))
    copied = destination / "snapshots" / target["revision"]
    shutil.copytree(snapshot, copied, symlinks=False)
    _rewrite_weights(copied, model)
    reloaded = ModelTarget({**target, "source": str(destination)}, device=device)
    if reloaded.identity() != checkpoint:
        _fail("invalid_artifact", "Reloaded export does not match the selected checkpoint")
    return checkpoint


def export_checkpoint(bundle, *, target, arm, destination, max_bytes, device="cpu"):
    """Write one selected arm so a later process can load it without the original source."""
    if type(max_bytes) is not int or max_bytes < 1:
        _fail("invalid_artifact", "max_bytes is required")
    if type(target) is not dict:
        _fail("invalid_target", "Invalid target")
    checkpoint, patches = _selected(bundle, target, arm)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        _fail("invalid_artifact", "Export destination must be unused")
    destination.mkdir(parents=True)
    try:
        kind = target.get("kind")
        if kind == "binary64":
            checkpoint = _export_binary(target, checkpoint, patches, destination)
        elif kind in ("smollm", "huggingface"):
            checkpoint = _export_model(target, checkpoint, patches, destination, device)
        else:
            _fail("unsupported_target", "Unsupported target")
        total, count = _bounded(destination, max_bytes)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {"format": "wsm.checkpoint-export.v1", "arm": arm, "checkpoint": checkpoint,
            "bytes": total, "files": count}
