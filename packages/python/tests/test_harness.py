import copy
import http.server
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from wsm import ResponsesTeacher, Session, Verifier, WeightlessError, replay_session
from wsm.binary64 import create_checkpoint, predict
from wsm.harness import run
from wsm.target_worker import run as worker_run


class WorkerRunner:
    config = {'mode': 'controlled-arithmetic-test'}

    def __init__(self):
        self.calls = []
        self.unknown = False

    def execute(self, action, target, **kwargs):
        started = kwargs.pop('on_dispatch', None)
        kwargs.pop('timeout', None)
        if started:
            started()
        self.calls.append(action)
        if self.unknown and action == 'evaluate':
            raise RuntimeError('private worker detail')
        return worker_run({'action': action, 'target': copy.deepcopy(target), 'patches': kwargs.get('patches', []),
                           'edits': kwargs.get('edits', []), 'cases': kwargs.get('cases'),
                           'query': kwargs.get('query'), 'generate': kwargs.get('generate', False)})


class ScriptedTeacher:
    def __init__(self, commands):
        self.commands = iter(commands)
        self.prompts = []
        self.responses = {}
        self.unknown = False

    def ask(self, session, prompt, *, key, arm):
        request = {'prompt': prompt, 'model': 'scripted-test'}
        operation = session.reserve_teacher(request, key=key, arm=arm)
        if key in self.responses:
            session.complete_teacher(key, self.responses[key])
            return self.responses[key]
        if operation['status'] != 'dispatched' or self.unknown:
            session.unknown_teacher(key)
            raise WeightlessError('teacher_outcome_unknown', 'Possible prior send')
        self.prompts.append(prompt)
        command = next(self.commands)
        response = {'text': json.dumps(command) if type(command) is dict else command}
        self.responses[key] = response
        session.complete_teacher(key, response)
        return response


def edit(value, tensor='w'):
    return {'action': 'edit', 'edits': [{'tensor': tensor, 'index': 0, 'value': value}]}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runner = WorkerRunner()
        graph = {'inputSize': 1, 'nodes': [{'id': 'out', 'op': 'affine', 'input': 'input',
                                         'weight': 'w', 'bias': 'b'}], 'output': 'out'}
        self.initial = create_checkpoint(graph, {'w': {'shape': [1, 1], 'values': [0]},
                                                 'b': {'shape': [1], 'values': [0]}})
        self.target = {'kind': 'binary64', 'checkpoint': self.initial}
        self.verifier = Verifier(self.root/'private.db', runner=self.runner,
                                 datasets={'final': {'task': [{'id': 'PRIVATE-CANARY', 'input': [7], 'target': [14]}],
                                                     'retention': [{'id': 'PRIVATE-RETENTION', 'input': [0], 'target': [0]}]}})
        self.sessions = []
        self.session = self.create('session.db')
        self.output = self.root/'harness'

    def create(self, name, requests=8):
        declaration = {'hypothesis': 'Learn double', 'arms': ['teacher'], 'editable': ['w'],
                       'budgets': {'attempts': 4, 'evaluations': 8, 'inspections': 4, 'diagnostics': 2,
                                   'teacher_requests': requests, 'final_evaluations': 2},
                       'max_edits': 2, 'max_inspect': 2, 'wall_seconds': 60,
                       'claim': 'confirmatory', 'criteria': {'final_exact_min': 1, 'retention_loss_max': 0}}
        session = Session.create(self.root/name, runner=self.runner, target=self.target,
                                 declaration=declaration, development=[{'id': 'public', 'input': [2], 'target': [4]}],
                                 verifier=self.verifier, final_dataset='final')
        self.sessions.append(session)
        return session

    def tearDown(self):
        for session in self.sessions:
            session.close()
        self.verifier.close()
        self.tmp.cleanup()

    def execute(self, commands, session=None, **kwargs):
        teacher = ScriptedTeacher(commands)
        report = run(session or self.session, teacher, prompt='Learn double', output=self.output, **kwargs)
        return report, teacher

    def test_inspect_reject_improve_finish_and_private_grade(self):
        report, teacher = self.execute([{'action': 'inspect', 'tensor': 'w', 'indices': [0]},
                                        edit(-2), edit(2), {'action': 'finish'}], context={'graph': self.initial['graph']})
        self.assertEqual(report['stop'], 'finished')
        self.assertEqual(report['final_conclusion'], 'supported')
        self.assertEqual(report['used']['attempts'], 2)
        self.assertFalse(report['no_edit'])
        bundle = json.loads((self.output/'bundle.json').read_text())
        frozen = replay_session(bundle)
        materialized = self.runner.execute('materialize', self.target, patches=frozen['arms']['teacher']['patches'])
        checkpoint = copy.deepcopy(self.initial)
        for item in materialized['patches']:
            checkpoint['tensors'][item['tensor']]['values'][item['index']] = item['value']
        checkpoint = create_checkpoint(checkpoint['graph'], checkpoint['tensors'])
        self.assertEqual(predict(checkpoint, [-8]), [-16])
        self.assertEqual([op['selection']['accepted'] for op in bundle['operations'] if op['kind'] == 'submit'], [False, True])
        self.assertIn('"value":0.0', teacher.prompts[1])
        self.assertNotIn('PRIVATE', ''.join(teacher.prompts))
        self.assertNotIn('PRIVATE', json.dumps(bundle))
        self.assertNotIn('Learn double', json.dumps(report))

    def test_malformed_output_never_dispatches_and_feedback_allows_recovery(self):
        invalid = ['{"action":"finish","action":"edit"}', '```json\n{}\n```',
                   '{"action":"finish","extra":1}', '{"action":"edit","edits":[{"tensor":"w","index":0,"value":NaN}]}']
        report, teacher = self.execute(invalid + [edit(2), {'action': 'finish'}])
        self.assertEqual(report['used']['attempts'], 1)
        self.assertEqual(report['used']['teacher_requests'], 6)
        self.assertEqual(report['final_conclusion'], 'supported')
        self.assertIn('invalid_command', teacher.prompts[1])

    def test_forbidden_tensor_is_rejected_without_attempt_and_no_final_before_baseline(self):
        report, teacher = self.execute([edit(2, tensor='b'), {'action': 'finish'}])
        self.assertEqual(report['used']['attempts'], 0)
        self.assertEqual(report['stop'], 'baseline_absent')
        self.assertFalse(report['baseline_established'])
        self.assertFalse(report['frozen_complete'])
        self.assertIsNone(report['final_status'])
        self.assertIn('invalid_proposal', teacher.prompts[1])

    def test_zero_teacher_budget_freezes_without_sending_or_finalizing(self):
        other = self.create('zero.db', requests=0)
        report, teacher = self.execute([], session=other)
        self.assertEqual(report['stop'], 'budget_exhausted')
        self.assertEqual(teacher.prompts, [])
        self.assertFalse(report['frozen_complete'])

    def test_unknown_numerical_operation_is_never_repeated(self):
        self.runner.unknown = True
        teacher = ScriptedTeacher([edit(2)])
        first = run(self.session, teacher, prompt='Learn double', output=self.output)
        calls = list(self.runner.calls)
        again = run(self.session, teacher, prompt='Learn double', output=self.output)
        self.assertEqual(first, again)
        self.assertEqual(self.runner.calls, calls)
        self.assertEqual(first['stop'], 'operation_unknown')
        self.assertTrue(first['obligations'])
        self.assertFalse(first['frozen_complete'])

    def test_unknown_teacher_stops_without_new_key(self):
        teacher = ScriptedTeacher([])
        teacher.unknown = True
        first = run(self.session, teacher, prompt='Learn double', output=self.output)
        self.assertEqual(first['stop'], 'teacher_outcome_unknown')
        self.assertEqual(first['used']['teacher_requests'], 1)
        self.assertEqual(run(self.session, teacher, prompt='Learn double', output=self.output), first)
        self.assertEqual(self.session.export()['used']['teacher_requests'], 1)

    def test_response_survives_cursor_write_crash_without_second_teacher_send(self):
        import wsm.harness as harness
        teacher = ScriptedTeacher([edit(2), {'action': 'finish'}])
        write = harness._write
        crashed = False
        def crash(path, value):
            nonlocal crashed
            if path.name == 'cursor.json' and value.get('pending') and value['pending'].get('response') and not crashed:
                crashed = True
                raise RuntimeError('process crash after response')
            return write(path, value)
        with patch.object(harness, '_write', crash), self.assertRaises(RuntimeError):
            run(self.session, teacher, prompt='Learn double', output=self.output)
        report = run(self.session, teacher, prompt='Learn double', output=self.output)
        self.assertEqual(report['final_conclusion'], 'supported')
        self.assertEqual(len(teacher.prompts), 2)
        self.assertEqual(report['used']['attempts'], 1)
        self.assertEqual(report['used']['teacher_requests'], 2)

    def test_completed_dispatch_recovery_has_no_second_evaluation(self):
        import wsm.harness as harness
        teacher = ScriptedTeacher([edit(2), {'action': 'finish'}])
        dispatch = harness._dispatch
        def crash(*args, **kwargs):
            dispatch(*args, **kwargs)
            raise RuntimeError('process crash after committed selection')
        with patch.object(harness, '_dispatch', crash), self.assertRaises(RuntimeError):
            run(self.session, teacher, prompt='Learn double', output=self.output)
        report = run(self.session, teacher, prompt='Learn double', output=self.output)
        self.assertEqual(report['used']['attempts'], 1)
        self.assertEqual(report['used']['evaluations'], 2)
        self.assertEqual(report['final_conclusion'], 'supported')

    def test_bound_prompt_and_session_cannot_change_on_resume(self):
        teacher = ScriptedTeacher([{'action': 'finish'}])
        run(self.session, teacher, prompt='Learn double', output=self.output)
        for session, prompt in [(self.session, 'Different request'), (self.create('different.db'), 'Learn double')]:
            with self.assertRaises(WeightlessError) as error:
                run(session, teacher, prompt=prompt, output=self.output)
            self.assertEqual(error.exception.code, 'identity_mismatch')

    def test_deadline_and_interrupt_preserve_partial_artifacts(self):
        teacher = ScriptedTeacher([])
        with patch.object(self.session, 'remaining_seconds', side_effect=WeightlessError('deadline_exhausted', 'Expired')):
            report = run(self.session, teacher, prompt='Learn double', output=self.output)
        self.assertEqual(report['stop'], 'deadline_exhausted')
        self.assertTrue((self.output/'bundle.json').is_file())
        other = self.create('interrupt.db')
        with patch.object(teacher, 'ask', side_effect=KeyboardInterrupt):
            report = run(other, teacher, prompt='Learn double', output=self.root/'interrupt')
        self.assertEqual(report['stop'], 'interrupted')
        self.assertFalse(report['frozen_complete'])

    def test_no_edit_and_exhausted_grammar_are_honest(self):
        report, _ = self.execute([edit(0), {'action': 'finish'}])
        self.assertTrue(report['no_edit'])
        self.assertTrue(report['baseline_established'])
        self.assertEqual(report['final_conclusion'], 'bounded_negative')
        other = self.create('malformed.db', requests=1)
        report = run(other, ScriptedTeacher(['garbage']), prompt='Learn double', output=self.root/'malformed')
        self.assertEqual(report['stop'], 'budget_exhausted')
        self.assertFalse(report['baseline_established'])

    def test_edit_is_bound_to_the_observation_the_teacher_received(self):
        teacher = ScriptedTeacher([edit(3), {'action': 'finish'}])
        ask = teacher.ask
        changed = False
        def concurrent_edit(session, prompt, **kwargs):
            nonlocal changed
            response = ask(session, prompt, **kwargs)
            if not changed:
                changed = True
                view = session.observe('teacher')
                proposal = {k: view[k] for k in ('session', 'arm', 'declaration', 'observation', 'base')}
                proposal['edits'] = edit(2)['edits']
                session.submit(proposal, key='another-editor')
            return response
        teacher.ask = concurrent_edit
        report = run(self.session, teacher, prompt='Learn double', output=self.output)
        self.assertEqual(report['used']['attempts'], 1)
        self.assertEqual(report['development_loss'], 0)
        self.assertIn('stale_base', teacher.prompts[1])

    def test_oversized_public_context_stops_without_sending(self):
        report, teacher = self.execute([], context='x' * 262144)
        self.assertEqual(report['stop'], 'prompt_bound')
        self.assertEqual(teacher.prompts, [])
        self.assertTrue((self.output/'bundle.json').is_file())

    def test_terminal_artifacts_are_restored_without_teacher_or_target_work(self):
        report, teacher = self.execute([edit(2), {'action': 'finish'}])
        calls = list(self.runner.calls)
        for name in ('bundle.json', 'report.json'):
            (self.output/name).unlink()
        self.assertEqual(run(self.session, teacher, prompt='Learn double', output=self.output), report)
        self.assertEqual(self.runner.calls, calls)
        self.assertEqual(len(teacher.prompts), 2)
        self.assertTrue((self.output/'bundle.json').is_file())
        self.assertEqual(json.loads((self.output/'report.json').read_text()), report)

    def test_uncertain_final_is_not_reported_as_success(self):
        execute = self.runner.execute
        def fail_final(action, target, **kwargs):
            if kwargs.get('purpose') == 'final':
                raise RuntimeError('private grading failure')
            return execute(action, target, **kwargs)
        self.runner.execute = fail_final
        report, _ = self.execute([edit(2), {'action': 'finish'}])
        self.assertEqual(report['stop'], 'final_outcome_unknown')
        self.assertEqual(report['final_status'], 'outcome_unknown')
        self.assertEqual(report['final_conclusion'], 'inconclusive')
        self.assertTrue(report['frozen_complete'])
        self.assertNotIn('private grading failure', (self.output/'report.json').read_text())

    def test_real_responses_transport_selects_weights_without_key_leak(self):
        received = []
        owner = self
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append(payload)
                command = edit(2) if len(received) == 1 else {'action': 'finish'}
                response = {'status': 'completed', 'model': payload['model'], 'tools': [], 'tool_choice': 'none',
                            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(command)}]}]}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'data: ' + json.dumps({'type': 'response.completed', 'response': response}).encode() + b'\n\n')
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            teacher = ResponsesTeacher(base_url=f'http://127.0.0.1:{server.server_port}/v1', model='test-route',
                                       api_key='test-only-secret', directory=self.root/'teacher-private')
            report = run(owner.session, teacher, prompt='Learn double', output=self.output)
            self.assertEqual(report['final_conclusion'], 'supported')
            self.assertEqual(len(received), 2)
            self.assertTrue(all(request['model'] == 'test-route' for request in received))
            self.assertNotIn('test-only-secret', ''.join(path.read_text() for path in self.output.glob('*.json')))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
