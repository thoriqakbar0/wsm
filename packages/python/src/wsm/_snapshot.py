"""Content identity and the local hash cache for Hugging Face snapshots."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import NamedTuple

from .errors import WeightlessError

_MANIFEST_FORMAT = "wsm.source-snapshot.v1"
_WEIGHT_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


class _FileStat(NamedTuple):
    size: int
    mtime_ns: int
    inode: int

    @classmethod
    def read(cls, info: os.stat_result) -> "_FileStat":
        if not stat.S_ISREG(info.st_mode):
            raise WeightlessError("invalid_model", "Snapshot entries must resolve to regular files")
        return cls(info.st_size, info.st_mtime_ns, info.st_ino)


def _raise_walk_error(error: OSError) -> None:
    raise error


def _inventory(path: Path) -> dict[str, _FileStat]:
    files = {}
    try:
        for directory, _, names in path.walk(on_error=_raise_walk_error):
            for name in names:
                entry = directory / name
                files[entry.relative_to(path).as_posix()] = _FileStat.read(entry.stat())
    except OSError as error:
        raise WeightlessError("invalid_model", f"Could not inspect source snapshot: {path}") from error
    return files


def _changed(path: Path) -> WeightlessError:
    return WeightlessError("source_snapshot_changed", f"Source snapshot changed: {path}")


def _hash_file(path: Path, expected: _FileStat) -> str:
    try:
        with path.open("rb") as stream:
            if _FileStat.read(os.fstat(stream.fileno())) != expected:
                raise _changed(path)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if _FileStat.read(os.fstat(stream.fileno())) != expected:
                raise _changed(path)
    except OSError as error:
        raise WeightlessError("invalid_model", f"Could not hash snapshot file: {path}") from error
    return digest


def _manifest_path(path: Path) -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    key = hashlib.sha256(os.fsencode(path)).hexdigest()
    return cache.absolute() / "wsm" / "snapshots" / f"{key}.json"


def _read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}
    if isinstance(manifest, dict) and manifest.get("format") == _MANIFEST_FORMAT:
        files = manifest.get("files")
        if isinstance(files, dict):
            return files
    return {}


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    inventory: dict[str, _FileStat]
    digests: dict[str, str]
    sha256: str

    @classmethod
    def scan(cls, path: Path, *, previous: dict | None = None) -> "_Snapshot":
        inventory = _inventory(path)
        digests = {}
        for name, metadata in inventory.items():
            record = previous.get(name) if previous is not None else None
            digest = record.get("sha256") if isinstance(record, dict) else None
            if (isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                    and record.get("stat") == metadata._asdict()):
                digests[name] = digest
            else:
                digests[name] = _hash_file(path / name, metadata)
        payload = json.dumps(digests, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        snapshot = cls(path, inventory, digests, "sha256:" + hashlib.sha256(payload.encode()).hexdigest())
        snapshot.check_metadata()
        return snapshot

    @classmethod
    def load(cls, path: Path, *, verify_full: bool) -> "_Snapshot":
        manifest = _manifest_path(path)
        previous = None if verify_full else _read_manifest(manifest)
        snapshot = cls.scan(path, previous=previous)
        snapshot.write_manifest(manifest)
        return snapshot

    def write_manifest(self, path: Path) -> None:
        files = {
            name: {"stat": metadata._asdict(), "sha256": self.digests[name]}
            for name, metadata in self.inventory.items()
        }
        manifest = {"format": _MANIFEST_FORMAT, "files": files}

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, sort_keys=True)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def check_metadata(self) -> None:
        if _inventory(self.path) != self.inventory:
            raise _changed(self.path)

    def check_shards(self) -> None:
        for name in _WEIGHT_INDEXES:
            if name not in self.inventory:
                continue
            try:
                index = json.loads((self.path / name).read_text(encoding="utf-8"))
                shards = index["weight_map"].values()
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                raise WeightlessError("invalid_model", f"Invalid checkpoint index: {name}") from error
            if any(not isinstance(shard, str) or Path(shard).as_posix() not in self.inventory
                   for shard in shards):
                message = f"Checkpoint index refers outside the source snapshot: {name}"
                raise WeightlessError("invalid_model", message)

    def verify(self) -> None:
        if self.scan(self.path).sha256 != self.sha256:
            raise _changed(self.path)
