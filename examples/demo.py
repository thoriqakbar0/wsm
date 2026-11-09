import argparse
import json
from pathlib import Path

from weightless import MetricPolicy, NumericalEvaluation, SearchRun
from weightless.binary64 import create_checkpoint

parser = argparse.ArgumentParser(description="Offline scripted teacher. No model call is made.")
parser.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parents[1] / "fixtures/addition.json")
parser.add_argument("--proposals", type=Path, default=Path(__file__).with_name("teacher-proposals.json"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
fixture = json.loads(args.fixture.read_text())
run = SearchRun(create_checkpoint(fixture["graph"], fixture["tensors"]),
                evaluator=NumericalEvaluation(fixture["task"]),
                policy=MetricPolicy(("development.loss", "minimize"), ("train.loss", "minimize")),
                budget=fixture["task"]["budget"])
for teacher in json.loads(args.proposals.read_text()):
    event = run.submit(dict(teacher, base=run.inspect()["checkpoint"]["id"]))
    print(json.dumps({"accepted": event["accepted"], "reason": event["reason"]}))
args.output.write_text(json.dumps(run.finish(), allow_nan=False, indent=2) + "\n")
