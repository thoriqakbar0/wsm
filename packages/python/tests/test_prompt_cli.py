import http.server
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from wsm.binary64 import predict
from wsm import replay_session


class CommandTests(unittest.TestCase):
    def command(self, *args, env=None):
        return subprocess.run([sys.executable, '-m', 'wsm.cli', *map(str, args)],
                              capture_output=True, text=True, env=env, timeout=60)

    def test_prompt_requires_explicit_credentials_before_creating_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            env = {k: v for k, v in os.environ.items() if k != 'WSM_TEACHER_API_KEY'}
            result = self.command('run', 'Improve addition', '--task', 'addition', '--teacher-url',
                                  'http://localhost/v1', '--teacher-model', 'test-teacher', '--output', output, env=env)
            self.assertEqual(result.returncode, 2)
            self.assertIn('WSM_TEACHER_API_KEY', result.stderr)
            self.assertFalse(output.exists())

    def test_explicit_workspace_equals_prevents_harness_routing(self):
        from wsm.cli import parser, _harness_command
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run.json").write_text("{}")
            for option in (["--workspace=/explicit"], ["--workspace", "/explicit"]):
                argv = option + ["status", directory]
                self.assertFalse(_harness_command(parser().parse_args(argv), argv))

    def test_status_reads_a_saved_report_without_teacher_or_sandbox(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'harness').mkdir()
            report = {'stop': 'finished', 'final_conclusion': 'inconclusive'}
            (root / 'harness/report.json').write_text(json.dumps(report))
            result = self.command('status', root)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(json.loads(result.stdout), report)

    def test_resume_recovers_creation_interrupted_before_session_state(self):
        from wsm.cli import main
        from wsm import WeightlessError
        from test_harness import WorkerRunner, ScriptedTeacher

        runner = WorkerRunner()
        execute = runner.execute
        interrupted = True

        def describe_once(action, target, **kwargs):
            nonlocal interrupted
            if action == 'describe' and interrupted:
                interrupted = False
                raise WeightlessError('sandbox_timeout', 'Initial metadata operation interrupted')
            return execute(action, target, **kwargs)

        runner.execute = describe_once
        teacher = ScriptedTeacher([{'action': 'finish'}])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with patch('wsm.cli.BubblewrapRunner', return_value=runner), \
                    patch('wsm.cli.ResponsesTeacher', return_value=teacher), \
                    patch.dict(os.environ, {'WSM_TEACHER_API_KEY': 'synthetic-secret'}), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(['run', 'Improve addition', '--task', 'addition',
                                       '--teacher-url', 'http://localhost/v1', '--teacher-model', 'test-teacher',
                                       '--output', str(output)]), 2)
                self.assertTrue((output / 'session.db').exists())
                self.assertEqual(main(['resume', str(output)]), 1)
            report = json.loads((output / 'harness/report.json').read_text())
            self.assertEqual(report['stop'], 'baseline_absent')
            self.assertEqual(report['used']['teacher_requests'], 1)
            self.assertEqual(report['used']['evaluations'], 0)


@unittest.skipUnless(os.environ.get('WEIGHTLESS_LOCAL_SANDBOX') == '1',
                     'Set WEIGHTLESS_LOCAL_SANDBOX=1 for the native CLI experiment')
class NativeCommandTests(unittest.TestCase):
    command = CommandTests.command

    def test_prompt_to_checkpoint_private_grade_and_resume(self):
        from importlib.resources import files
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = json.loads(files('wsm').joinpath('tasks/addition.json').read_text())
            task['final'] = {'cases': 'private.json', 'registry': 'verifier.db', 'dataset': 'confirmation'}
            task['criteria'] = {'final_exact_min': 1, 'retention_loss_max': 0}
            (root / 'task.json').write_text(json.dumps(task))
            (root / 'private.json').write_text(json.dumps({
                'task': [{'id': 'private-canary', 'input': [-8, 3], 'target': [-5]}],
                'retention': [{'id': 'private-retention', 'input': [0, 0], 'target': [0]}]}))
            commands = [
                {'action': 'inspect', 'tensor': 'output.weight', 'indices': [0, 1]},
                {'action': 'edit', 'edits': [{'tensor': 'output.weight', 'index': 0, 'value': -1}]},
                {'action': 'edit', 'edits': [{'tensor': 'output.weight', 'index': 0, 'value': 1},
                                             {'tensor': 'output.weight', 'index': 1, 'value': -1}]},
                {'action': 'finish'},
            ]
            requests = []
            class Handler(http.server.BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    command = commands[len(requests)]
                    requests.append(request)
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    event = {'type': 'response.completed', 'response': {'status': 'completed',
                             'model': request['model'], 'tools': [], 'tool_choice': 'none',
                             'output': [{'type': 'message', 'content': [
                                 {'type': 'output_text', 'text': json.dumps(command)}]}]}}
                    self.wfile.write(b'data: ' + json.dumps(event).encode() + b'\n\n')

            server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                output = root / 'run'
                env = {**os.environ, 'WSM_TEACHER_API_KEY': 'synthetic-cli-secret'}
                result = self.command('run', 'Improve signed addition', '--task', root / 'task.json',
                                      '--teacher-url', f'http://127.0.0.1:{server.server_port}/v1',
                                      '--teacher-model', 'test-teacher', '--attempts', '2', '--turns', '4',
                                      '--output', output, env=env)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                report = json.loads(result.stdout)
                self.assertEqual(report['final_conclusion'], 'supported')
                self.assertEqual(report['used']['attempts'], 2)
                self.assertEqual(report['used']['evaluations'], 3)
                self.assertEqual(report['used']['teacher_requests'], 4)
                checkpoint = json.loads((output / 'checkpoint.json').read_text())
                self.assertEqual(predict(checkpoint, [-8, 3]), [-5])
                bundle = json.loads((output / 'harness/bundle.json').read_text())
                self.assertEqual(replay_session(bundle)['arms']['teacher']['checkpoint'], checkpoint['id'])
                self.assertEqual([op['selection']['accepted'] for op in bundle['operations']
                                  if op['kind'] == 'submit'], [False, True])
                sent = json.dumps(requests)
                self.assertNotIn('private-canary', sent)
                self.assertNotIn('synthetic-cli-secret', sent)
                for path in [output / 'run.json', output / 'harness/report.json', output / 'harness/bundle.json']:
                    self.assertNotIn('synthetic-cli-secret', path.read_text())
                resumed = self.command('resume', output, env=env)
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertEqual(json.loads(resumed.stdout), report)
                self.assertEqual(len(requests), 4)
                (root / 'private.json').write_text('[]')
                changed = self.command('resume', output, env=env)
                self.assertEqual(changed.returncode, 2)
                self.assertIn('identity_mismatch', changed.stderr)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
