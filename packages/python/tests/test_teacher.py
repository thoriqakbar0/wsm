import http.server
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from wsm import ResponsesTeacher, Session, WeightlessError
from test_session import Runner, declaration, proposal


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.requests = []
        self.mode = 'normal'
        owner = self
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append(payload)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                if owner.mode == 'stall':
                    time.sleep(2)
                    return
                output = [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"action":"observe"}'}]}]
                if owner.mode == 'tool':
                    output = [{'type': 'function_call', 'name': 'shell'}]
                event = {'type': 'response.completed', 'response': {'status': 'completed',
                    'model': payload['model'], 'tools': [], 'tool_choice': 'none', 'output': output,
                    'usage': {'input_tokens': 12, 'output_tokens': 5, 'total_tokens': 17}}}
                try:
                    self.wfile.write(b'data: ' + json.dumps(event).encode() + b'\n\n')
                except BrokenPipeError:
                    pass
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.sessions = []

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        for session in self.sessions:
            session.close()
        self.temporary.cleanup()

    def session(self, name):
        session = Session.create(self.root/name, runner=Runner(), target={'kind': 'test'},
                                 declaration=declaration(), development=[{'id': 'public'}])
        self.sessions.append(session)
        return session

    def broker(self, timeout=2):
        return ResponsesTeacher(base_url=f'http://127.0.0.1:{self.server.server_port}/v1',
            model='test-model', api_key='test-only-key', directory=self.root/'private', timeout=timeout)

    def test_literal_response_replay_and_session_scoped_keys(self):
        first, second = self.session('first'), self.session('second')
        teacher = self.broker()
        result = teacher.ask(first, 'Public prompt', key='same-key', arm='teacher')
        self.assertEqual(result['text'], '{"action":"observe"}')
        self.assertEqual(teacher.ask(first, 'Public prompt', key='same-key', arm='teacher'), result)
        teacher.ask(second, 'Public prompt', key='same-key', arm='teacher')
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(first.export()['used']['teacher_requests'], 1)
        self.assertEqual(second.export()['used']['teacher_requests'], 1)
        self.assertTrue(all(request['tools'] == [] and request['tool_choice'] == 'none' for request in self.requests))
        artifacts = ''.join(path.read_text() for path in (self.root/'private').rglob('*.json'))
        self.assertNotIn('test-only-key', artifacts)
        self.assertNotIn('Public prompt', json.dumps(first.export()))

    def test_unexpected_tool_output_is_preserved_but_never_retried(self):
        session, teacher = self.session('one'), self.broker()
        self.mode = 'tool'
        for _ in range(2):
            with self.assertRaises(WeightlessError) as error:
                teacher.ask(session, 'Public prompt', key='bad', arm='teacher')
            self.assertEqual(error.exception.code, 'teacher_outcome_unknown')
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(session.export()['used']['teacher_requests'], 1)
        self.assertFalse(session.freeze(key='stop')['complete'])

    def test_total_stream_deadline_interrupts_stalled_server(self):
        session, teacher = self.session('one'), self.broker(timeout=.1)
        self.mode = 'stall'
        started = time.monotonic()
        with self.assertRaises(WeightlessError):
            teacher.ask(session, 'Public prompt', key='stall', arm='teacher')
        self.assertLess(time.monotonic()-started, 1.5)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(session.status('stall', arm='teacher')['status'], 'outcome_unknown')

    def test_freeze_blocks_a_reserved_but_not_started_request(self):
        session, teacher = self.session('one'), self.broker()
        original = session.teacher_dispatch
        def freeze_before_send(key):
            session.freeze(key='barrier')
            return original(key)
        session.teacher_dispatch = freeze_before_send
        with self.assertRaises(WeightlessError):
            teacher.ask(session, 'Public prompt', key='reserved', arm='teacher')
        self.assertEqual(len(self.requests), 0)

    def test_frozen_orchestration_is_charged_without_reopening_development(self):
        session, teacher = self.session('frozen'), self.broker()
        session.submit(proposal(session), key='edit')
        frozen = session.freeze(key='freeze')
        teacher.ask(session, 'Choose final grading', key='orchestrate', arm='teacher', purpose='report')
        self.assertEqual(session.export()['used']['teacher_requests'], 1)
        self.assertEqual(session.export()['frozen'], frozen)
        with self.assertRaises(WeightlessError):
            teacher.ask(session, 'New development', key='forbidden', arm='teacher')
        self.assertEqual(len(self.requests), 1)
