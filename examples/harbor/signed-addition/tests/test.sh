#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python3}" "${WEIGHTLESS_TESTS:-/tests}/grade.py"
