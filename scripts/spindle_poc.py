"""Run one WSM proof against a task-owned Spindle deployment, then stop its apps."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/python/src"))

from wsm.trial_poc import Budget, load_config, resume_student, run_poc

FRONTEND = "wsm-spindle-poc"


def bounded_command(command, *, deadline, log, env=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("The proof's overall deadline expired")
    with Path(log).open("a") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True, cwd=ROOT)
        try:
            code = process.wait(timeout=remaining)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if code:
            raise RuntimeError(f"Command exited {code}. Inspect {log}")


def run_owner(args):
    os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2026.10"
    from spindle_watchdog import owned_apps, process_identity

    output = args.output.resolve()
    output.mkdir(mode=0o700)
    prices = {"max_usd": 10, "prior_reserved_usd": args.prior_reserved_usd,
              "modal_hourly_usd": 5, "modal_build_usd": 0.5, "modal_egress_per_gib": 0.04,
              "spindle_hourly_usd": args.student_hourly_usd, "spindle_seconds": args.student_seconds}
    budget = Budget(prices, output / "deployment-budget.jsonl")
    if any(row["state"] != "stopped" or int(row["tasks"]) for row in owned_apps(FRONTEND)):
        raise RuntimeError("The task frontend already has live apps. Inspect them before a new proof.")
    budget.reserve("modal_build", {})
    deadline = time.monotonic() + 1800
    with (output / "watchdog.log").open("a") as stream:
        subprocess.Popen([sys.executable, str(ROOT / "scripts/spindle_watchdog.py"),
                          "--owner", str(os.getpid()), "--identity", process_identity(os.getpid()),
                          "--deadline", str(time.time() + 1800), "--output", str(output)],
                         stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                         start_new_session=True)
    def interrupted(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    report = {"state": "starting", "provider": "spindle", "teacher": "Codex", "student": "Qwen/Qwen3-4B",
              "algorithm": "reward_weighted_teacher_sft", "scope": "One teacher rollout and one student update; no learning efficacy claim.",
              "max_usd": 10, "prior_reserved_usd": args.prior_reserved_usd}
    try:
        bounded_command([sys.executable, "-m", "spindle.deployment_cli", "deploy",
                         str(ROOT / "examples/spindle/qwen3_4b.py")],
                        deadline=deadline, log=output / "deployment.log")
        work_seconds = min(840, args.student_seconds - 360)
        if work_seconds <= 0:
            raise ValueError("Student envelope must include more than 360 seconds for cleanup")
        deadline = time.monotonic() + work_seconds
        (output / "training-deadline.json").write_text(json.dumps({"deadline": time.time() + work_seconds}))
        prices["prior_reserved_usd"] = budget.reserved
        config = {
            "model_source": str(args.model_source.resolve()),
            "public_cases": str(ROOT / "examples/coding/cases.json"),
            "heldout_cases": str(output / "heldout.json"),
            "instruction": "Improve short English completions on the public cases by editing the target weights. "
                           "Before finishing, run at least one Python edit. You may inspect tensor names and "
                           "values, evaluate, and revise. Keep changes finite. Be concise and return only action JSON.",
            "scoring": "negative_loss",
            "policy": {"provider": "responses_proxy", "base_url": "http://127.0.0.1:8317/v1",
                       "model_env": "WSM_EDITOR_MODEL", "api_key_env": "WSM_EDITOR_API_KEY",
                       "max_tokens": 4096, "max_turns": 3, "request_seconds": 180, "reasoning_effort": "medium"},
            "student": {"provider": "spindle", "base_url": "https://endpoint-resolved-in-child.modal.run",
                        "model": "Qwen/Qwen3-4B", "rank": 8, "loss_fn": "cross_entropy",
                        "learning_rate": 1e-5, "request_seconds": 600, "max_context_tokens": 16384},
            "budget": prices, "sandbox_seconds": 600, "command_seconds": 120,
            "grader_python": str(args.grader_python.absolute()), "grader_seconds": 120,
        }
        if args.resume_trial:
            config = json.loads((args.resume_trial.parent / "config.json").read_text())
            config["budget"] = prices
            config["grader_python"] = str(args.grader_python.absolute())
        else:
            (output / "heldout.json").write_text(json.dumps([
                {"prompt": "Say good morning.", "target": "Good morning!"},
                {"prompt": "The capital of Spain is", "target": " Madrid."},
            ]))
        config_path = output / "config.json"
        config_path.write_text(json.dumps(config, indent=2))
        child = [sys.executable, str(Path(__file__).resolve()), "--child", str(config_path),
                 "--output", str(output / "run"), "--api-key-file", str(args.api_key_file.resolve()),
                 "--prior-reserved-usd", str(prices["prior_reserved_usd"])]
        if args.resume_trial:
            child += ["--resume-trial", str(args.resume_trial.absolute())]
            if args.retry_pretraining:
                child += ["--retry-pretraining"]
        else:
            child += ["--editor-config-file", str(args.editor_config_file.resolve())]
        bounded_command(child, deadline=deadline, log=output / "run.log")
        manifest = json.loads((output / "run/poc.json").read_text())
        report.update(state=manifest["state"], reserved_usd=manifest["reserved_usd"], training=manifest["training"])
        history = (args.resume_trial or output / "run") / "trial"
        report["trial_directory"] = str(history)
        events = [json.loads(line) for line in (history / "events.jsonl").read_text().splitlines()]
        report["published_edits"] = sum(event["kind"] == "result" and event["data"]["operation"] == "execute"
                                        for event in events)
        grade = json.loads((history / "grade.json").read_text())["result"]
        report["changed_tensors"] = grade["changed_tensors"]
        if not report["published_edits"] or not report["changed_tensors"]:
            report["state"] = "incomplete_no_edits"
    except BaseException as error:
        report.update(state="failed", error_type=type(error).__name__, message=str(error))
        raise
    finally:
        cleanup = {"confirmed": False}
        try:
            bounded_command([sys.executable, str(ROOT / "scripts/spindle_watchdog.py"), "--cleanup",
                             "--output", str(output)], deadline=time.monotonic() + 330,
                            log=output / "cleanup.log")
            cleanup = json.loads((output / "cleanup-confirmed.json").read_text())
        finally:
            report["cleanup"] = cleanup
            report.setdefault("reserved_usd", budget.reserved)
            ledger = output / "run/budget.jsonl"
            if ledger.is_file():
                for line in ledger.read_text().splitlines():
                    try:
                        report["reserved_usd"] = max(report["reserved_usd"], json.loads(line)["reserved_usd"])
                    except (ValueError, KeyError):
                        continue
            (output / "proof.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if not cleanup["confirmed"]:
            raise RuntimeError("Spindle cleanup was not confirmed. Inspect the owned apps.")
    return report


def run_child(args):
    import modal
    config = load_config(args.child)
    endpoint = modal.Function.from_name(FRONTEND, "server", environment_name="main").get_web_url()
    config["student"]["base_url"] = endpoint
    key = json.loads(args.api_key_file.read_text())["TINKER_API_KEY"]
    os.environ["WSM_SPINDLE_API_KEY"] = key
    if args.resume_trial:
        return resume_student(config, args.resume_trial, args.output, retry_pretraining=args.retry_pretraining)
    editor = json.loads(args.editor_config_file.read_text())
    os.environ["WSM_EDITOR_MODEL"] = editor["model"]
    os.environ["WSM_EDITOR_API_KEY"] = editor["api_key"]
    return run_poc(config, args.output, app_name=FRONTEND + "-target")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--grader-python", type=Path)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--editor-config-file", type=Path)
    parser.add_argument("--resume-trial", type=Path)
    parser.add_argument("--retry-pretraining", action="store_true")
    parser.add_argument("--student-hourly-usd", type=float, default=8)
    parser.add_argument("--student-seconds", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-reserved-usd", type=float, required=True)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if not args.child and (args.model_source is None or args.grader_python is None):
        parser.error("Owner requires --model-source and --grader-python")
    if args.editor_config_file is None and args.resume_trial is None:
        parser.error("Supply --editor-config-file or a frozen --resume-trial")
    print(json.dumps(run_child(args) if args.child else run_owner(args), indent=2, allow_nan=False))
