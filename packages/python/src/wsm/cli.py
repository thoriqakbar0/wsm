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
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
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
