#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/packages/python/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m unittest discover -s packages/python/tests -v
python3 scripts/package_smoke.py
