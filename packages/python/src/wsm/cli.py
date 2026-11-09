"""Prompt-first command line for Weight Space Modifiers."""
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
from .binary64 import create_checkpoint, parse_checkpoint


def _json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _write(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex)
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write('\n')
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


def _positive(value):
    number = int(value)
    if not 1 <= number <= 1000:
        raise argparse.ArgumentTypeError('Expected an integer from 1 to 1000')
    return number


def _task(name):
    if name == 'addition':
        raw = files('wsm').joinpath('tasks/addition.json').read_text()
        base = Path.cwd()
    else:
        path = Path(name).resolve()
        raw, base = path.read_text(), path.parent
    task = _json(raw)
    allowed = {'format', 'target', 'editable', 'development', 'context', 'max_edits',
               'max_inspect', 'development_retention', 'development_retention_loss_max',
               'final', 'criteria'}
    if (type(task) is not dict or task.get('format') != 'wsm.task.v1'
            or not {'target', 'editable', 'development'} <= task.keys() or set(task) - allowed):
        raise WeightlessError('invalid_task', 'Expected a wsm.task.v1 task profile')
    target = task['target']
    if type(target) is not dict or target.get('kind') not in ('binary64', 'smollm'):
        raise WeightlessError('invalid_task', 'Use a binary64 or pinned smollm target')
    if target['kind'] == 'binary64':
        if set(target) != {'kind', 'checkpoint'}:
            raise WeightlessError('invalid_task', 'binary64 target requires checkpoint')
        checkpoint = target['checkpoint']
        if type(checkpoint) is dict and set(checkpoint) == {'graph', 'tensors'}:
            checkpoint = create_checkpoint(checkpoint['graph'], checkpoint['tensors'])
        target['checkpoint'] = parse_checkpoint(checkpoint)
    elif type(target.get('source')) is str:
        target['source'] = str((base / target['source']).resolve())
    if (type(task['editable']) is not list or not task['editable']
            or any(type(t) is not str or not t for t in task['editable'])
            or len(set(task['editable'])) != len(task['editable'])):
        raise WeightlessError('invalid_task', 'Declare distinct editable tensor names')
    if 'final' in task:
        final = task['final']
        if (type(final) is not dict or set(final) != {'cases', 'registry', 'dataset'}
                or any(type(v) is not str or not v for v in final.values())):
            raise WeightlessError('invalid_task', 'final requires cases, registry, and dataset')
        for key in ('cases', 'registry'):
            final[key] = str((base / final[key]).resolve())
    if 'criteria' in task and 'final' not in task:
        raise WeightlessError('invalid_task', 'Confirmatory criteria require a private final dataset')
    return task


def _context(task):
    target = task['target']
    return {'task': task.get('context', ''), 'target_kind': target['kind'],
            **({'graph': target['checkpoint']['graph']} if target['kind'] == 'binary64' else {})}


def _declaration(task, prompt, attempts, turns, seconds):
    diagnostics = min(attempts, 10)
    retention = bool(task.get('development_retention'))
    declaration = {'hypothesis': prompt, 'arms': ['teacher'], 'editable': task['editable'],
                   'max_edits': task.get('max_edits', 16), 'max_inspect': task.get('max_inspect', 64),
                   'wall_seconds': seconds, 'claim': 'confirmatory' if 'criteria' in task else 'exploratory',
                   'budgets': {'attempts': attempts, 'evaluations': attempts * (1 + retention) + 1 + diagnostics,
                               'inspections': turns, 'diagnostics': diagnostics, 'teacher_requests': turns,
                               'final_evaluations': 2 if 'final' in task else 0}}
    for field in ('criteria', 'development_retention_loss_max'):
        if field in task:
            declaration[field] = task[field]
    return declaration


def _private_paths(task, output):
    if task['target']['kind'] != 'smollm':
        return
    mount = Path(task['target']['source']).resolve()
    protected = [output]
    if task.get('final'):
        protected += [Path(task['final']['cases']), Path(task['final']['registry'])]
    if any(p.resolve().is_relative_to(mount) for p in protected):
        raise WeightlessError('invalid_task', 'Run output and private verifier files must be outside the model mount')


def _checkpoint(output, target, report):
    if report.get('selected_checkpoint') is None:
        return
    bundle = _json((output / 'harness' / 'bundle.json').read_text())
    frozen = bundle.get('frozen')
    if not frozen:
        return
    selected = frozen['arms']['teacher']
    mod = {'format': 'wsm.mod.v1', 'target': target, 'checkpoint': selected['checkpoint'],
           'patches': selected['patches'], 'evidence': 'harness/bundle.json'}
    _write(output / 'mod.json', mod)
    if target['kind'] == 'binary64':
        tensors = copy.deepcopy(target['checkpoint']['tensors'])
        for edit in selected['patches']:
            tensors[edit['tensor']]['values'][edit['index']] = edit['value']
        checkpoint = create_checkpoint(target['checkpoint']['graph'], tensors)
        if checkpoint['id'] != selected['checkpoint']:
            raise WeightlessError('identity_mismatch', 'Selected checkpoint does not match exported patches')
        _write(output / 'checkpoint.json', checkpoint)


def _execute(config, output, stack):
    from .harness import run

    task = config['task']
    _private_paths(task, output)
    runner_spec = config['runner']
    runner = (BubblewrapRunner(python=runner_spec['python'], timeout=runner_spec['timeout'])
              if runner_spec['kind'] == 'bubblewrap' else
              DockerRunner(runner_spec['image'], timeout=runner_spec['timeout']))
    verifier, final_dataset = None, None
    if task.get('final'):
        final = task['final']
        raw = Path(final['cases']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != config['final_sha256']:
            raise WeightlessError('identity_mismatch', 'Private final dataset changed')
        final_dataset = final['dataset']
        verifier = Verifier(final['registry'], runner=runner, datasets={final_dataset: _json(raw)})
        stack.callback(verifier.close)
    kwargs = {'runner': runner, 'target': task['target'], 'verifier': verifier, 'final_dataset': final_dataset}
    database = output / 'session.db'
    try:
        session = Session.open(database, **kwargs)
    except WeightlessError as error:
        if error.code != 'missing_session':
            raise
        session = Session.create(database, declaration=config['declaration'], development=task['development'],
                                 development_retention=task.get('development_retention'), **kwargs)
    stack.callback(session.close)
    key = os.environ.get('WSM_TEACHER_API_KEY')
    if not key:
        raise WeightlessError('invalid_teacher', 'Set WSM_TEACHER_API_KEY for the configured teacher endpoint')
    teacher = ResponsesTeacher(base_url=config['teacher']['url'], model=config['teacher']['model'],
                               api_key=key, directory=output / 'teacher-private', max_output_tokens=2048)
    report = run(session, teacher, prompt=config['prompt'], context=_context(task), output=output / 'harness')
    _checkpoint(output, task['target'], report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='wsm', description="Weight Space Modifiers: a harness for editing a model's weights.")
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('run', help='Start a prompt-driven weight modification experiment')
    start.add_argument('prompt', help='Describe the desired behavior change')
    start.add_argument('--task', required=True, help='Task profile JSON path, or addition for the bundled example')
    start.add_argument('--teacher-url', default=os.environ.get('WSM_TEACHER_URL'),
                       help='Responses-compatible base URL (or WSM_TEACHER_URL)')
    start.add_argument('--teacher-model', default=os.environ.get('WSM_TEACHER_MODEL'),
                       help='Exact teacher model ID (or WSM_TEACHER_MODEL)')
    start.add_argument('--output', type=Path, required=True, help='New private run directory')
    start.add_argument('--attempts', type=_positive, default=20, help='Maximum candidate edits (default: 20)')
    start.add_argument('--turns', type=_positive, default=64, help='Maximum teacher requests (default: 64)')
    start.add_argument('--seconds', type=float, default=900, help='Total wall deadline in seconds (default: 900)')
    start.add_argument('--runner', choices=('bubblewrap', 'docker'), default='bubblewrap')
    start.add_argument('--python', default='/usr/bin/python3', help='Python runtime for bubblewrap')
    start.add_argument('--image', help='Existing image for the Docker runner')
    resume = commands.add_parser('resume', help='Resume the same prompt, route, limits, and saved operations')
    resume.add_argument('output', type=Path)
    status = commands.add_parser('status', help='Read a saved report without running a teacher or target')
    status.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if args.command == 'status':
            report = _json((output / 'harness' / 'report.json').read_text())
        else:
            if args.command == 'run':
                if not args.prompt.strip() or len(args.prompt.encode()) > 65536:
                    raise WeightlessError('invalid_request', 'Use a nonempty prompt of at most 64 KiB')
                if not math.isfinite(args.seconds) or args.seconds <= 0:
                    raise WeightlessError('invalid_request', 'Use a positive finite wall deadline')
                if args.runner == 'docker' and not args.image:
                    raise WeightlessError('invalid_config', 'Docker execution requires --image')
                if not args.teacher_url or not args.teacher_model:
                    raise WeightlessError('invalid_teacher', 'Set --teacher-url and --teacher-model, or WSM_TEACHER_URL and WSM_TEACHER_MODEL')
                if not os.environ.get('WSM_TEACHER_API_KEY'):
                    raise WeightlessError('invalid_teacher', 'Set WSM_TEACHER_API_KEY for the configured teacher endpoint')
                task = _task(args.task)
                _private_paths(task, output)
                config = {'format': 'wsm.run.v1', 'prompt': args.prompt, 'task': task,
                          'teacher': {'url': args.teacher_url, 'model': args.teacher_model},
                          'runner': {'kind': args.runner, 'python': args.python, 'image': args.image, 'timeout': 60},
                          'declaration': _declaration(task, args.prompt, args.attempts, args.turns, args.seconds)}
                if task.get('final'):
                    config['final_sha256'] = hashlib.sha256(Path(task['final']['cases']).read_bytes()).hexdigest()
                output.mkdir(mode=0o700, parents=True, exist_ok=False)
                _write(output / 'run.json', config)
            else:
                config = _json((output / 'run.json').read_text())
            if output.stat().st_mode & 0o077:
                raise WeightlessError('invalid_config', 'Run directory must be private (mode 0700)')
            import fcntl
            with ExitStack() as stack:
                lock = stack.enter_context((output / 'run.lock').open('a'))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise WeightlessError('run_busy', 'Another process owns this run') from None
                report = _execute(config, output, stack)
        print(json.dumps(report, allow_nan=False, indent=2))
        return 0 if report.get('stop') in ('finished', 'closed', 'complete') else 1
    except (WeightlessError, OSError, ValueError, KeyError, TypeError) as error:
        code = error.code if isinstance(error, WeightlessError) else 'invalid_config'
        print('wsm: ' + code + '. ' + (str(error) if isinstance(error, WeightlessError) else 'Check task files and run configuration.'), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
