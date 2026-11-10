"""Operator commands. Session remains the owner of experiment policy."""
import argparse
from contextlib import ExitStack
import copy
import hashlib
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import sys
import uuid

from . import BubblewrapRunner, DockerRunner, ResponsesTeacher, Session, Verifier, WeightlessError
from .binary64 import __version__, create_checkpoint, parse_checkpoint
from .client import Experiment, addition_config, create_example_verifier, read_json, write_json


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise WeightlessError("usage", message)


def _options(parser, *, root=False):
    default = {} if root else {"default": argparse.SUPPRESS}
    parser.add_argument("--workspace", **({"default": "wsm-experiment"} if root else default),
                        help="Experiment directory, default ./wsm-experiment")
    parser.add_argument("--verifier", **default, help="Separate private verifier JSON file")
    parser.add_argument("--arm", **default, help="Arm name for multi-arm experiments")
    parser.add_argument("--json", action="store_true", **default, help="Emit machine-readable JSON")


def parser():
    root = Parser(prog="wsm", description="Run and inspect weight-editing experiments. No command opens the TUI.")
    root.add_argument("--version", action="version", version=f"wsm {__version__}")
    _options(root, root=True)
    commands = root.add_subparsers(dest="command", parser_class=Parser)

    def command(name, help):
        item = commands.add_parser(name, help=help)
        _options(item)
        return item

    init = command("init", "Create a durable experiment")
    source = init.add_mutually_exclusive_group()
    source.add_argument("--config", help="wsm.workspace.v1 JSON configuration")
    source.add_argument("--example", choices=["addition"], default="addition")
    init.add_argument("--private-dir", help="Create separate synthetic addition-demo verifier data")
    command("observe", "Show the current arm observation")
    inspect = command("inspect", "Read permitted numerical weights")
    inspect.add_argument("--tensor", required=True)
    inspect.add_argument("--indices", type=int, nargs="+", required=True)
    inspect.add_argument("--checkpoint", help="Expected checkpoint, defaults to current checkpoint")
    inspect.add_argument("--key", required=True)
    propose = command("propose", "Bind an edit list to the current observation")
    propose.add_argument("--edits", required=True, help="JSON file containing an edit list")
    propose.add_argument("--output", help="Create a proposal file without overwriting")
    submit = command("submit", "Evaluate a saved, fully bound proposal")
    submit.add_argument("proposal")
    submit.add_argument("--key", required=True)
    status = command("status", "Inspect one operation, or read a prompt-harness report directory")
    status.add_argument("ref", help="Operation key, or a prompt-harness run directory")
    for name in ("diagnose", "freeze", "finalize", "cancel"):
        action = command(name, f"{name.capitalize()} the current experiment")
        action.add_argument("--key", required=True)
    export = command("export", "Export frozen experiment evidence")
    export.add_argument("--output", help="Create a bundle file without overwriting")
    exported = command("export-model", "Write the selected checkpoint so it reloads without the original source")
    exported.add_argument("--destination", required=True)
    exported.add_argument("--max-bytes", type=int, required=True)
    replay = command("replay", "Replay recorded evidence without numerical execution")
    replay.add_argument("bundle")
    verify = command("verify", "Reconstruct evidence against this workspace's numerical target")
    verify.add_argument("bundle")
    resume = command("resume", "Retry a saved command, or resume a prompt-harness run directory")
    resume.add_argument("ref", help="Saved command file, or a prompt-harness run directory")
    teacher = command("teacher", "Ask an explicitly configured teacher; never execute its reply")
    teacher.add_argument("--config", required=True, help="Private teacher JSON configuration")
    teacher.add_argument("--prompt", required=True, help="Literal UTF-8 prompt file; reuse unchanged for a retry")
    teacher.add_argument("--key", required=True)
    command("models", "Describe teacher and editable-target support")
    stage = command("stage-model", "Copy a pinned local model snapshot into a self-contained target directory")
    stage.add_argument("source")
    stage.add_argument("destination")
    stage.add_argument("--model", required=True)
    stage.add_argument("--revision", required=True)
    start = command("run", "Start a prompt-driven weight modification experiment")
    start.add_argument("prompt", help="Describe the desired behavior change")
    start.add_argument("--task", required=True, help="Task profile JSON path, or addition for the bundled example")
    start.add_argument("--teacher-url", default=os.environ.get("WSM_TEACHER_URL"),
                       help="Responses-compatible base URL (or WSM_TEACHER_URL)")
    start.add_argument("--teacher-model", default=os.environ.get("WSM_TEACHER_MODEL"),
                       help="Exact teacher model ID (or WSM_TEACHER_MODEL)")
    start.add_argument("--output", type=Path, required=True, help="New private run directory")
    start.add_argument("--attempts", type=_positive, default=20, help="Maximum candidate edits (default: 20)")
    start.add_argument("--turns", type=_positive, default=64, help="Maximum teacher requests (default: 64)")
    start.add_argument("--code-executions", type=_positive, default=0,
                       help="Maximum isolated teacher Python executions (default: disabled)")
    start.add_argument("--forward-calls", type=_positive_budget, default=10000, help="Maximum total model forward calls, including final grading")
    start.add_argument("--final-forward-calls", type=_positive_budget, default=1000, help="Private grading envelope charged to the total limit")
    start.add_argument("--seconds", type=float, default=900, help="Total wall deadline in seconds (default: 900)")
    start.add_argument("--runner", choices=("bubblewrap", "docker"), default="bubblewrap")
    start.add_argument("--python", default="/usr/bin/python3", help="Python runtime for bubblewrap")
    start.add_argument("--image", help="Existing image for the Docker runner")
    command("tui", "Open the interactive terminal")
    return root


def _positive_budget(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def _positive(value):
    number = int(value)
    if not 1 <= number <= 1000:
        raise argparse.ArgumentTypeError("Expected an integer from 1 to 1000")
    return number


def _harness_command(args, argv):
    if args.command == "run":
        return True
    if args.command not in ("status", "resume") or any(arg == "--workspace" or arg.startswith("--workspace=") for arg in argv):
        return False
    path = Path(args.ref)
    return path.is_dir() and ((path / "run.json").is_file() or (path / "harness" / "report.json").is_file())


def _json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _write(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _task(name):
    if name == "addition":
        raw = files("wsm").joinpath("tasks/addition.json").read_text()
        base = Path.cwd()
    else:
        path = Path(name).resolve()
        raw, base = path.read_text(), path.parent
    task = _json(raw)
    allowed = {"format", "target", "editable", "development", "context", "max_edits",
               "max_inspect", "development_retention", "development_retention_loss_max",
               "final", "criteria"}
    if (type(task) is not dict or task.get("format") != "wsm.task.v1"
            or not {"target", "editable", "development"} <= task.keys() or set(task) - allowed):
        raise WeightlessError("invalid_task", "Expected a wsm.task.v1 task profile")
    target = task["target"]
    if type(target) is not dict or target.get("kind") not in ("binary64", "smollm", "huggingface"):
        raise WeightlessError("invalid_task", "Use a binary64, pinned smollm, or local huggingface target")
    if target["kind"] == "binary64":
        if set(target) != {"kind", "checkpoint"}:
            raise WeightlessError("invalid_task", "binary64 target requires checkpoint")
        checkpoint = target["checkpoint"]
        if type(checkpoint) is dict and set(checkpoint) == {"graph", "tensors"}:
            checkpoint = create_checkpoint(checkpoint["graph"], checkpoint["tensors"])
        target["checkpoint"] = parse_checkpoint(checkpoint)
    elif type(target.get("source")) is str:
        target["source"] = str((base / target["source"]).resolve())
    if (type(task["editable"]) is not list or not task["editable"]
            or any(type(t) is not str or not t for t in task["editable"])
            or len(set(task["editable"])) != len(task["editable"])):
        raise WeightlessError("invalid_task", "Declare distinct editable tensor names")
    if "final" in task:
        final = task["final"]
        if (type(final) is not dict or set(final) != {"cases", "registry", "dataset"}
                or any(type(v) is not str or not v for v in final.values())):
            raise WeightlessError("invalid_task", "final requires cases, registry, and dataset")
        for key in ("cases", "registry"):
            final[key] = str((base / final[key]).resolve())
    if "criteria" in task and "final" not in task:
        raise WeightlessError("invalid_task", "Confirmatory criteria require a private final dataset")
    return task


def _context(task):
    target = task["target"]
    return {"task": task.get("context", ""), "target_kind": target["kind"],
            **({"graph": target["checkpoint"]["graph"]} if target["kind"] == "binary64" else {})}


def _declaration(task, prompt, attempts, turns, seconds):
    diagnostics = min(attempts, 10)
    retention = bool(task.get("development_retention"))
    declaration = {"hypothesis": prompt, "arms": ["teacher"], "editable": task["editable"],
                   "max_edits": task.get("max_edits", 16), "max_inspect": task.get("max_inspect", 64),
                   "wall_seconds": seconds, "claim": "confirmatory" if "criteria" in task else "exploratory",
                   "budgets": {"attempts": attempts, "evaluations": attempts * (1 + retention) + 1 + diagnostics,
                               "inspections": turns, "diagnostics": diagnostics, "teacher_requests": turns,
                               "final_evaluations": 2 if "final" in task else 0}}
    for field in ("criteria", "development_retention_loss_max"):
        if field in task:
            declaration[field] = task[field]
    return declaration


def _private_paths(task, output):
    if task["target"]["kind"] == "binary64":
        return
    mount = Path(task["target"]["source"]).resolve()
    protected = [output]
    if task.get("final"):
        protected += [Path(task["final"]["cases"]), Path(task["final"]["registry"])]
    if any(p.resolve().is_relative_to(mount) for p in protected):
        raise WeightlessError("invalid_task", "Run output and private verifier files must be outside the model mount")


def _checkpoint(output, target, report):
    if report.get("selected_checkpoint") is None:
        return
    bundle = _json((output / "harness" / "bundle.json").read_text())
    frozen = bundle.get("frozen")
    if not frozen:
        return
    selected = frozen["arms"]["teacher"]
    mod = {"format": "wsm.mod.v1", "target": target, "checkpoint": selected["checkpoint"],
           "patches": selected["patches"], "evidence": "harness/bundle.json"}
    _write(output / "mod.json", mod)
    if target["kind"] == "binary64":
        tensors = copy.deepcopy(target["checkpoint"]["tensors"])
        for edit in selected["patches"]:
            tensors[edit["tensor"]]["values"][edit["index"]] = edit["value"]
        checkpoint = create_checkpoint(target["checkpoint"]["graph"], tensors)
        if checkpoint["id"] != selected["checkpoint"]:
            raise WeightlessError("identity_mismatch", "Selected checkpoint does not match exported patches")
        _write(output / "checkpoint.json", checkpoint)
    elif target["kind"] in ("smollm", "huggingface"):
        from .checkpoint_export import export_checkpoint
        export_checkpoint(bundle, target=target, arm="teacher", destination=output / "selected", max_bytes=1024 ** 3)


def _execute(config, output, stack):
    from .harness import run

    task = config["task"]
    _private_paths(task, output)
    runner_spec = config["runner"]
    runner = (BubblewrapRunner(python=runner_spec["python"], timeout=runner_spec["timeout"])
              if runner_spec["kind"] == "bubblewrap" else
              DockerRunner(runner_spec["image"], timeout=runner_spec["timeout"]))
    verifier, final_dataset = None, None
    if task.get("final"):
        final = task["final"]
        raw = Path(final["cases"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != config["final_sha256"]:
            raise WeightlessError("identity_mismatch", "Private final dataset changed")
        final_dataset = final["dataset"]
        verifier = Verifier(final["registry"], runner=runner, datasets={final_dataset: _json(raw)})
        stack.callback(verifier.close)
    kwargs = {"runner": runner, "target": task["target"], "verifier": verifier, "final_dataset": final_dataset}
    database = output / "session.db"
    try:
        session = Session.open(database, **kwargs)
    except WeightlessError as error:
        if error.code != "missing_session":
            raise
        session = Session.create(database, declaration=config["declaration"], development=task["development"],
                                 development_retention=task.get("development_retention"), **kwargs)
    stack.callback(session.close)
    key = os.environ.get("WSM_TEACHER_API_KEY")
    if not key:
        raise WeightlessError("invalid_teacher", "Set WSM_TEACHER_API_KEY for the configured teacher endpoint")
    teacher = ResponsesTeacher(base_url=config["teacher"]["url"], model=config["teacher"]["model"],
                               api_key=key, directory=output / "teacher-private", max_output_tokens=2048)
    report = run(session, teacher, prompt=config["prompt"], context=_context(task), output=output / "harness")
    _checkpoint(output, task["target"], report)
    return report


def _run_harness(args):
    output = args.output.resolve() if args.command == "run" else Path(args.ref).resolve()
    if args.command == "status":
        return _json((output / "harness" / "report.json").read_text())
    if args.command == "run":
        if not args.prompt.strip() or len(args.prompt.encode()) > 65536:
            raise WeightlessError("invalid_request", "Use a nonempty prompt of at most 64 KiB")
        if not math.isfinite(args.seconds) or args.seconds <= 0:
            raise WeightlessError("invalid_request", "Use a positive finite wall deadline")
        if args.runner == "docker" and not args.image:
            raise WeightlessError("invalid_config", "Docker execution requires --image")
        if not args.teacher_url or not args.teacher_model:
            raise WeightlessError("invalid_teacher", "Set --teacher-url and --teacher-model, or WSM_TEACHER_URL and WSM_TEACHER_MODEL")
        if not os.environ.get("WSM_TEACHER_API_KEY"):
            raise WeightlessError("invalid_teacher", "Set WSM_TEACHER_API_KEY for the configured teacher endpoint")
        task = _task(args.task)
        _private_paths(task, output)
        config = {"format": "wsm.run.v1", "prompt": args.prompt, "task": task,
                  "teacher": {"url": args.teacher_url, "model": args.teacher_model},
                  "runner": {"kind": args.runner, "python": args.python, "image": args.image, "timeout": 60},
                  "declaration": _declaration(task, args.prompt, args.attempts, args.turns, args.seconds)}
        config["declaration"]["budgets"]["forward_calls"] = args.forward_calls
        if args.code_executions:
            config["declaration"]["budgets"]["code_executions"] = args.code_executions
        if task.get("final"):
            config["declaration"]["final_forward_calls"] = args.final_forward_calls
            config["final_sha256"] = hashlib.sha256(Path(task["final"]["cases"]).read_bytes()).hexdigest()
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        _write(output / "run.json", config)
    else:
        config = _json((output / "run.json").read_text())
    if output.stat().st_mode & 0o077:
        raise WeightlessError("invalid_config", "Run directory must be private (mode 0700)")
    import fcntl
    with ExitStack() as stack:
        lock = stack.enter_context((output / "run.lock").open("a"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WeightlessError("run_busy", "Another process owns this run") from None
        return _execute(config, output, stack)


def save_command(experiment, command, arm):
    """Persist the exact request before dispatch so a restart needs no rebinding."""
    if (type(command) is not dict or set(command) != {"action", "key", "payload"}
            or type(command["action"]) is not str or type(command["payload"]) is not dict):
        raise WeightlessError("invalid_request", "Expected a command with action, key, and object payload")
    key = command.get("key")
    if type(key) is not str or not 0 < len(key) <= 200:
        raise WeightlessError("invalid_request", "Operation keys must contain 1 to 200 characters")
    directory = experiment.directory / "requests"
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    record = {"arm": experiment.arm(arm), "command": command}
    try:
        write_json(path, record)
    except FileExistsError:
        if read_json(path) != record:
            raise WeightlessError("idempotency_conflict", "Saved operation key already has a different request") from None
    return path


def dispatch(experiment, action, payload, *, key, arm=None):
    command = {"action": action, "key": key, "payload": payload}
    save_command(experiment, command, arm)
    return experiment.execute(command, arm=arm)


def ask_teacher(experiment, config_path, prompt_path, *, key, arm=None):
    config_file = Path(config_path).resolve()
    target_source = experiment.config["target"].get("source")
    protected_roots = [experiment.directory]
    if target_source:
        protected_roots.append(Path(target_source).resolve())
    if any(config_file.is_relative_to(root) for root in protected_roots):
        raise WeightlessError("invalid_teacher", "Keep teacher configuration outside the workspace and model source")
    config = read_json(config_path)
    required = {"base_url", "model", "api_key_env", "directory"}
    optional = {"timeout", "max_output_tokens"}
    if (type(config) is not dict or not required <= config.keys()
            or set(config) - required - optional
            or any(type(config[name]) is not str or not config[name] for name in required)):
        raise WeightlessError("invalid_teacher", "Teacher config needs base_url, model, api_key_env, and private directory")
    directory = (Path(config_path).resolve().parent / config["directory"]).resolve()
    if any(directory.is_relative_to(root) for root in protected_roots):
        raise WeightlessError("invalid_teacher", "Keep teacher artifacts outside the workspace and model source")
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise WeightlessError("invalid_teacher", "The configured API-key environment variable is unset")
    teacher = ResponsesTeacher(base_url=config["base_url"], model=config["model"], api_key=api_key,
                               directory=directory, **{name: config[name] for name in optional if name in config})
    try:
        previous = experiment.session.status(key, arm=experiment.arm(arm))
    except WeightlessError as error:
        if error.code != "unknown_operation":
            raise
        purpose = "development" if experiment.session.observe(experiment.arm(arm))["phase"] == "developing" else "report"
    else:
        if previous["kind"] not in ("teacher", "teacher_report"):
            raise WeightlessError("idempotency_conflict", "This operation key belongs to a different action")
        purpose = "report" if previous["kind"] == "teacher_report" else "development"
    return teacher.ask(experiment.session, Path(prompt_path).read_text(), key=key,
                       arm=experiment.arm(arm), purpose=purpose)


def models():
    return {"teacher": {"selection": "explicit model ID and Responses-compatible base_url",
                        "automatic_fallback": False, "tools": False,
                        "hosted_product": "open; no live hosted integration is claimed"},
            "targets": [{"kind": "binary64", "profile": "fixed affine/ReLU checkpoints"},
                        {"kind": "smollm", "profile": "pinned SmolLM2 compatibility profile; CPU by default"},
                        {"kind": "huggingface", "profile": "local pinned causal-LM snapshots; offline float32; CPU by default",
                         "limits": ["No remote model code", "Supported tensor layouts only",
                                    "CUDA only when the caller passes device=cuda", "Explicit resource ceilings"]}]}


def run(args):
    if args.command == "models":
        return models()
    if args.command == "stage-model":
        from .sandbox import stage_model
        return stage_model(args.source, args.destination, model=args.model, revision=args.revision)
    if args.command == "replay":
        from .session import replay_session
        return replay_session(read_json(args.bundle))
    if args.command == "init":
        if Path(args.workspace).exists():
            raise WeightlessError("workspace_exists", "Choose a new workspace directory; existing files are preserved")
        if args.private_dir and args.config:
            raise WeightlessError("usage", "--private-dir creates addition-demo data; use --verifier with a custom config")
        verifier = args.verifier
        if args.private_dir:
            if verifier:
                raise WeightlessError("usage", "Choose --private-dir or --verifier")
            private = Path(args.private_dir).resolve()
            if private.is_relative_to(Path(args.workspace).resolve()):
                raise WeightlessError("invalid_config", "Keep private demo data outside the public workspace")
            verifier = create_example_verifier(private)
        config = read_json(args.config) if args.config else addition_config()
        with Experiment.create(args.workspace, config, verifier_config=verifier) as experiment:
            result = experiment.session.observe(experiment.arm(args.arm))
            result["workspace"] = str(experiment.directory)
            if verifier:
                result["verifier_config"] = str(Path(verifier).resolve())
            return result
    with Experiment.open(args.workspace, verifier_config=args.verifier) as experiment:
        arm = None if args.command == "resume" else experiment.arm(args.arm)
        if args.command == "observe":
            return experiment.session.observe(arm)
        if args.command == "propose":
            result = experiment.proposal(read_json(args.edits), arm=arm)
        elif args.command == "verify":
            from .session import verify_session
            return verify_session(read_json(args.bundle), runner=experiment.runner, target=experiment.config["target"])
        elif args.command == "teacher":
            return ask_teacher(experiment, args.config, args.prompt, key=args.key, arm=arm)
        elif args.command == "resume":
            record = read_json(args.ref)
            if type(record) is not dict or set(record) != {"arm", "command"}:
                raise WeightlessError("invalid_request", "Expected a saved WSM command envelope")
            if args.arm is not None and args.arm != record["arm"]:
                raise WeightlessError("invalid_request", "Saved request belongs to a different arm")
            save_command(experiment, record["command"], record["arm"])
            return experiment.execute(record["command"], arm=record["arm"])
        elif args.command == "export":
            result = experiment.execute({"action": "export", "key": "export", "payload": {}}, arm=arm)
        elif args.command == "export-model":
            from .checkpoint_export import export_checkpoint
            bundle = experiment.session.export()
            return export_checkpoint(bundle, target=experiment.config["target"], arm=experiment.arm(args.arm),
                                     destination=args.destination, max_bytes=args.max_bytes)
        elif args.command == "status":
            return experiment.session.status(args.ref, arm=arm)
        else:
            payload = {}
            if args.command == "inspect":
                checkpoint = args.checkpoint or experiment.session.observe(arm)["base"]
                payload = {"checkpoint": checkpoint, "tensor": args.tensor, "indices": args.indices}
            elif args.command == "submit":
                payload = read_json(args.proposal)
            result = dispatch(experiment, args.command, payload, key=args.key, arm=arm)
        if getattr(args, "output", None):
            write_json(args.output, result)
        return result


def result_exit_code(result):
    if type(result) is dict:
        if result.get("status") in ("task_failure", "invalid_response", "outcome_unknown", "dispatched"):
            return 1
        if result.get("complete") is False:
            return 1
    return 0


def _text_error(error):
    code = error.code if isinstance(error, WeightlessError) else "invalid_config"
    message = str(error) if isinstance(error, WeightlessError) else "Check task files and run configuration."
    print("wsm: " + code + ". " + message, file=sys.stderr)
    return 2


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    harness = False
    try:
        args = parser().parse_args(argv)
        if args.command in (None, "tui"):
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                raise WeightlessError("terminal_required", "The TUI requires a terminal; use --help for scriptable commands")
            from .tui import launch
            return launch(args.workspace, verifier_config=args.verifier, arm=args.arm)
        harness = _harness_command(args, argv)
        if harness:
            result = _run_harness(args)
            print(json.dumps(result, allow_nan=False, indent=2))
            return 0 if result.get("stop") in ("finished", "closed", "complete") else 1
        result = run(args)
        print(json.dumps(result, indent=None if args.json else 2, allow_nan=False))
        return result_exit_code(result)
    except KeyboardInterrupt:
        print(json.dumps({"error": {"code": "interrupted", "message": "Interrupted. Reopen the workspace and inspect the saved operation before retrying."}}), file=sys.stderr)
        return 130
    except WeightlessError as error:
        if harness:
            return _text_error(error)
        print(json.dumps({"error": {"code": error.code, "message": str(error)}}), file=sys.stderr)
        return 1 if error.code in ("teacher_outcome_unknown", "sandbox_timeout", "sandbox_cleanup_unknown") else 2
    except OSError as error:
        if harness:
            return _text_error(error)
        code = "file_exists" if isinstance(error, FileExistsError) else "io_error"
        print(json.dumps({"error": {"code": code, "message": "File operation failed; check paths, permissions, and whether the destination exists"}}), file=sys.stderr)
        return 2
    except (ValueError, TypeError, KeyError):
        if harness:
            return _text_error(WeightlessError("invalid_config", "Check task files and run configuration."))
        print(json.dumps({"error": {"code": "invalid_input", "message": "Input does not match the command's expected format"}}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
