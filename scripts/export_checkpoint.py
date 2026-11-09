"""Export a frozen selected checkpoint. This is a trusted operator command.

Usage:
  python scripts/export_checkpoint.py --bundle EVIDENCE.json --target TARGET.json \\
      --arm ARM --destination DIRECTORY --max-bytes N

TARGET.json is the trusted descriptor hashed into the session. The destination
must not exist. The receipt is DIRECTORY/receipt.json and does not include the
host source path, credentials, transcripts, or private cases.
"""
import argparse
import json
import sys
from pathlib import Path

from wsm.binary64 import WeightlessError
from wsm.checkpoint_export import export_checkpoint


def _load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise WeightlessError("invalid_request", "Duplicate JSON field")
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    except WeightlessError:
        raise
    except (OSError, ValueError, RecursionError):
        raise WeightlessError("invalid_request", "JSON input is invalid") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export one frozen selected checkpoint.")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--max-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        receipt = export_checkpoint(_load(args.bundle), target=_load(args.target), arm=args.arm,
                                    destination=args.destination, max_bytes=args.max_bytes)
    except WeightlessError as error:
        sys.stderr.write(f"{error.code}: {error}\n")
        return 1
    sys.stdout.write(json.dumps({"selected": receipt["selected"], "total_bytes": receipt["total_bytes"],
                                 "serialization": receipt["serialization"]}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
