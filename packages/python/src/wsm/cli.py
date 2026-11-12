"""Commands for trusted Python weight editing."""
import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .errors import WeightlessError


def parser():
    root = argparse.ArgumentParser(prog="wsm", description="Edit model weights with trusted Python. No command opens the TUI.")
    root.add_argument('--version', action='version', version=f'wsm {__version__}')
    commands = root.add_subparsers(dest='command')
    execute = commands.add_parser('execute', help='Execute a trusted Python file defining edit(model)')
    execute.add_argument('program', type=Path)
    execute.add_argument('--model', type=Path, required=True)
    execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--cases', type=Path, required=True)
    execute.add_argument('--device', default='cpu')
    commands.add_parser('tui', help='Open the Python editing terminal')
    trial = commands.add_parser("trial", help="Run or watch an experimental Modal coding trial")
    trial_commands = trial.add_subparsers(dest="trial_command", required=True)
    preflight = trial_commands.add_parser("preflight", help="Validate explicit task, policy, and price configuration offline")
    preflight.add_argument("--config", type=Path, required=True)
    trial_run = trial_commands.add_parser("run", help="Run one fixed-model Modal trial and hosted Tinker update")
    trial_run.add_argument("--config", type=Path, required=True)
    trial_run.add_argument("--output", type=Path, required=True)
    trial_view = trial_commands.add_parser("view", help="Read the public trial journal without control access")
    trial_view.add_argument("directory", type=Path)
    trial_view.add_argument("--follow", action="store_true")
    weights = trial_commands.add_parser("weights", help="Read retained tensor shapes or numerical values")
    weights.add_argument("checkpoint", type=Path, help="Retained safetensors file")
    weights.add_argument("--tensor")
    weights.add_argument("--indices", type=int, nargs="+")
    benchmark = commands.add_parser("benchmark", help="Run frozen evaluation campaigns without student training")
    benchmarks = benchmark.add_subparsers(dest="benchmark_command", required=True)
    for name in ("preflight", "run"):
        command = benchmarks.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--local-trusted", action="store_true",
                                 help="Execute reviewed synthetic fixtures with your host permissions")
    for name in ("report", "regrade"):
        command = benchmarks.add_parser(name)
        command.add_argument("directory", type=Path)
        if name == "report":
            command.add_argument("--markdown", action="store_true")
        else:
            command.add_argument("--trial-id", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "benchmark":
            from .benchmark import preflight, regrade, run
            from .benchmark_report import markdown, report
            if args.benchmark_command == "preflight":
                result = preflight(args.config)
            elif args.benchmark_command == "run":
                run(args.config, args.output, local_trusted=args.local_trusted)
                result = report(args.output)
            elif args.benchmark_command == "regrade":
                result = regrade(args.directory, args.trial_id)
            else:
                result = report(args.directory)
                if args.markdown:
                    print(markdown(result))
                    return 0
            print(json.dumps(result, indent=2, allow_nan=False))
            return 1 if args.benchmark_command == "run" and not result["complete"] else 0
        if args.command == "trial":
            from .trial_poc import inspect_weights, load_config, run_poc, watch
            if args.trial_command == "view":
                watch(args.directory, follow=args.follow)
            elif args.trial_command == "weights":
                print(json.dumps(inspect_weights(args.checkpoint, tensor=args.tensor, indices=args.indices),
                                 indent=2, allow_nan=False))
            else:
                config = load_config(args.config)
                result = ({"valid": True, "cloud_calls": 0, "budget_usd": config["budget"]["max_usd"]}
                          if args.trial_command == "preflight" else run_poc(config, args.output))
                print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        if args.command in (None, 'tui'):
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                raise WeightlessError('terminal_required', 'The TUI requires a terminal; use --help for scriptable commands')
            from .tui import launch
            return launch()
        from .program import execute_program
        cases = json.loads(args.cases.read_text())
        result = execute_program(args.program, model_path=args.model, output=args.output,
                                 device=args.device, cases=cases)
        print(json.dumps(result, allow_nan=False, indent=2))
        return 0 if result.get('state') == 'published' else 1
    except KeyboardInterrupt:
        message = (f"Interrupted. Inspect {args.output / 'run.json'} and its checkpoint before retrying."
                   if args.command == 'execute' else 'Interrupted.')
        print(json.dumps({'error': {'code': 'interrupted', 'message': message}}), file=sys.stderr)
        return 130
    except (Exception, SystemExit) as error:
        print(json.dumps({'error': {'code': getattr(error, 'code', 'execution_error'),
                                    'type': type(error).__name__, 'message': str(error)}}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
