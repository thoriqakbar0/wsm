#!/usr/bin/env bash
set -euo pipefail
"${PYTHON:-python3}" - <<'PY'
import json
import os
from pathlib import Path
from wsm import MetricPolicy, NumericalEvaluation, SearchRun
from wsm.binary64 import create_checkpoint

app = Path(os.environ.get("WEIGHTLESS_APP", "/app"))
public = json.loads((app / "public.json").read_text())
run = SearchRun(create_checkpoint(public["graph"], public["tensors"]),
                evaluator=NumericalEvaluation(public["task"]),
                policy=MetricPolicy(("development.loss", "minimize"), ("train.loss", "minimize")),
                budget=public["task"]["budget"])
run.submit({"base": run.inspect()["checkpoint"]["id"], "author": "scripted-oracle",
            "edits": [{"tensor": "output.weight", "index": 0, "value": 1},
                      {"tensor": "output.weight", "index": 1, "value": -1}]})
(app / "submission.json").write_text(json.dumps(run.finish(), allow_nan=False) + "\n")
PY
