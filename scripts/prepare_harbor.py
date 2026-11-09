import argparse
import json
from pathlib import Path
import shutil

from wsm import MetricPolicy
from wsm.binary64 import create_checkpoint
from wsm.core import _digest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "examples/harbor/signed-addition"
POLICY = MetricPolicy(("development.loss", "minimize"), ("train.loss", "minimize"))


def prepare(destination, purpose, public):
    if destination != TASK and destination.exists() and any(destination.iterdir()):
        marker = destination / "tests/manifest.json"
        if not marker.is_file() or json.loads(marker.read_text()).get("purpose") != purpose:
            raise ValueError("Refusing to overwrite a destination that is not a prepared task")
    destination.mkdir(parents=True, exist_ok=True)
    if destination != TASK:
        for name in ("task.toml", "instruction.md"):
            shutil.copyfile(TASK / name, destination / name)
        shutil.copytree(TASK / "environment", destination / "environment", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("generated"))
        (destination / "tests").mkdir(exist_ok=True)
        for name in ("grade.py", "test.sh"):
            shutil.copyfile(TASK / "tests" / name, destination / "tests" / name)
        shutil.copytree(TASK / "solution", destination / "solution", dirs_exist_ok=True)
    generated = destination / "environment/generated"
    if generated.exists():
        shutil.rmtree(generated)
    generated.mkdir(parents=True)
    shutil.copytree(ROOT / "packages/python/src/wsm", generated / "wsm",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    encoded = json.dumps(public, indent=2) + "\n"
    (generated / "public.json").write_text(encoded)
    (destination / "tests/expected.json").write_text(encoded)
    if purpose == "development":
        split = {key: public["task"][key] for key in ("train", "development")}
        if (destination / "tests/heldout.json").exists():
            raise ValueError("Development destination contains heldout data")
    else:
        split = json.loads((TASK / "tests/heldout.json").read_text())
    manifest = {"format": "weightless.harbor-task.v1", "purpose": purpose,
                "problem": _digest("weightless.problem.v1", public),
                "initial": create_checkpoint(public["graph"], public["tensors"])["id"],
                "policy": POLICY, "split": _digest("weightless.split.v1", split)}
    (destination / "tests/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared", purpose, destination)


def main():
    parser = argparse.ArgumentParser(description="Prepare final and optional development Harbor tasks without leaking heldout data.")
    parser.add_argument("--development", type=Path)
    args = parser.parse_args()
    public = json.loads((ROOT / "fixtures/addition.json").read_text())
    public = {name: public[name] for name in ("graph", "tensors", "task")}
    prepare(TASK, "final", public)
    if args.development:
        destination = args.development.resolve()
        if destination == TASK:
            parser.error("Development destination must differ from final task")
        prepare(destination, "development", public)


if __name__ == "__main__":
    main()
