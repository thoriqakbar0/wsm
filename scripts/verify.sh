#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/packages/python/src${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/golden_identity.py
python3 -m unittest discover -s packages/python/tests -v
"${INTEGRATION_PYTHON:-python3}" scripts/harbor_smoke.py
