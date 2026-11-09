
import argparse
import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if type(value) is int:
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(canonical(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            canonical(key) + ":" + canonical(value[key]) for key in sorted(value)
        ) + "}"
    raise ValueError("The identity metadata must contain only integers and ASCII strings")


def golden():
    fixture = json.loads((ROOT / "fixtures/addition.json").read_text())
    identity = bytearray(b"weightless.checkpoint.v1\x00")
    identity.extend(canonical(fixture["graph"]).encode("ascii") + b"\x00")
    for name in sorted(fixture["tensors"]):
        tensor = fixture["tensors"][name]
        identity.extend(name.encode("ascii") + b"\x00")
        identity.extend(canonical(tensor["shape"]).encode("ascii") + b"\x00")
        for number in tensor["values"]:
            identity.extend(struct.pack("!d", float(number) if number else 0.0))
    return {
        "id": "sha256:" + hashlib.sha256(identity).hexdigest(),
        "identityHex": identity.hex(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the addition fixture's identity without importing the SDK.")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = ROOT / "fixtures/golden.json"
    expected = golden()
    if args.write:
        path.write_text(json.dumps(expected, indent=2) + "\n")
    else:
        actual = json.loads(path.read_text())
        if actual != expected:
            raise SystemExit("Golden identity differs from the independent contract encoding")
    print(expected["id"])
