import copy
import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from wsm.binary64 import WeightlessError
from wsm.session import Session, _hash
from wsm.tinker_backend import TinkerServer, main
from wsm.tinker_client import WeightEditor

API_KEY = 'tml-local-test-key'
ARM = 'teacher'


class Runner:
    config = {'mode': 'controlled-test', 'image': 'fixed'}

    def execute(self, action, target, **kwargs):
        started = kwargs.pop('on_dispatch', None)
        if started:
            started()
        patches = kwargs.get('patches') or []
        value = patches[0]['value'] if patches else 0
        if action == 'describe':
            return {'target_id': 'target', 'checkpoint': _hash(0), 'parameters': {'w': {'shape': [1], 'size': 1, 'dtype': 'float64'}},
                    'aliases': {}, 'capabilities': {}}
        if action == 'materialize':
            edited = kwargs['edits'][0]['value']
            return {'target_id': 'target', 'base_checkpoint': _hash(value),
                    'checkpoint': _hash(edited), 'patches': kwargs['edits'], 'changes': kwargs['edits']}
        if action == 'inspect':
            return {'checkpoint': _hash(value), 'values': [{'tensor': 'w', 'index': 0, 'value': value}]}
        cases = kwargs['cases']
        return {'checkpoint': _hash(value), 'metrics': {'loss': abs(1 - value), 'exact': int(value == 1) * len(cases), 'count': len(cases)},
                'cases': [{'id': case['id'], 'prediction': [value], 'target_nll': abs(1 - value), 'exact': value == 1, 'stopped_on_eos': True} for case in cases]}


def declaration():
    return {'hypothesis': 'A scalar replacement solves the finite task', 'arms': [ARM], 'editable': ['w'],
            'budgets': {'attempts': 4, 'evaluations': 8, 'inspections': 4, 'teacher_requests': 2, 'final_evaluations': 1},
            'max_edits': 2, 'max_inspect': 4, 'wall_seconds': 60}


class FakeSession:
    def __init__(self):
        self.checkpoint = 'ckpt-a'
        self.calls = []
        self.unknown = False
        self.boom = False
        self.run_calls = []

    def reopen(self):
        return self

    def close(self):
        pass

    def observe(self, arm):
        if arm != ARM:
            raise WeightlessError('unauthorized', 'Unknown arm')
        return {'session': 'sess', 'arm': arm, 'declaration': 'decl', 'observation': 3, 'base': self.checkpoint,
                'parameters': {'w': {'shape': [1], 'size': 1}}}

    def infer(self, query, *, key, arm):
        self.calls.append({'query': copy.deepcopy(query), 'key': key, 'arm': arm})
        if self.boom:
            raise RuntimeError('SECRET-/tmp/private-traceback')
        if self.unknown:
            return {'status': 'outcome_unknown', 'result': None, 'key': key}
        count = query['max_tokens']
        prompt = query['tokens']
        prompt_logprobs = ([None] + [-1.5] * (len(prompt) - 1)) if query['prompt_logprobs'] else None
        sequences = [] if count == 0 else [{
            'tokens': [1000 + index for index in range(count)],
            'logprobs': [-0.5] * count,
            'stop_reason': 'length',
            'sequence_id': 'seq-fixed',
        }]
        operation = {'status': 'completed', 'key': key, 'result': {
            'checkpoint': query['checkpoint'],
            'sequences': sequences,
            'prompt_logprobs': prompt_logprobs,
            'usage': {'forward_calls': 1},
        }}
        self.last_result = operation['result']
        return operation

    def run_code(self, source, inspections, *, key, arm, base, observation):
        self.run_calls.append({'source': source, 'inspections': inspections, 'key': key, 'arm': arm, 'base': base, 'observation': observation})
        return {'status': 'completed', 'submitted': True, 'base': base, 'observation': observation}

    def submit(self, proposal, *, key):
        return {'status': 'committed', 'key': key, 'selection': {'accepted': True}}

    def inspect(self, query, *, key, arm):
        self.calls.append({'inspect': copy.deepcopy(query), 'key': key, 'arm': arm})
        if query['checkpoint'] != self.checkpoint:
            raise WeightlessError('stale_base', 'Inspection checkpoint changed')
        return {'status': 'completed', 'result': {'checkpoint': query['checkpoint'], 'values': [0]}}


def request(server, path, body=None, *, key=API_KEY, bearer=False, method=None):
    headers = {'Connection': 'close', 'Accept': 'application/json'}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers['Content-Type'] = 'application/json'
    if bearer:
        headers['Authorization'] = 'Bearer ' + key
    elif key is not None:
        headers['X-API-Key'] = key
    req = urllib.request.Request(server.base_url + path, data=data, headers=headers, method=method or ('POST' if body is not None else 'GET'))
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            raw = response.read()
            return response.status, response.headers, raw
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def loads(raw):
    return json.loads(raw.decode())


def sample_body(sampling_session_id, seq_id, tokens=(11, 22, 33), **params):
    sampling = {'max_tokens': 4, 'temperature': 0, 'top_k': -1, 'top_p': 1}
    sampling.update(params)
    return {
        'type': 'sample',
        'num_samples': 1,
        'sampling_session_id': sampling_session_id,
        'seq_id': seq_id,
        'prompt': {'chunks': [{'type': 'encoded_text', 'tokens': list(tokens)}]},
        'sampling_params': sampling,
        'prompt_logprobs': True,
        'topk_prompt_logprobs': 0,
        'topk_sample_logprobs': 0,
        'prompt_alt_tokens_k': 0,
        'record_stability_info': False,
    }


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.server = TinkerServer(self.session, arm=ARM, api_key=API_KEY).__enter__()

    def tearDown(self):
        self.server.close()

    def sdk_session(self):
        status, _headers, raw = request(self.server, '/api/v1/create_session', {'sdk_version': '0.39.0', 'tags': [], 'user_metadata': {}})
        self.assertEqual(status, 200)
        return loads(raw)['session_id']

    def sampling_session(self, sdk_session, seq_id=0, **spec):
        body = {'type': 'create_sampling_session', 'session_id': sdk_session, 'sampling_session_seq_id': seq_id, 'base_model': 'wsm'}
        body.update(spec)
        status, _headers, raw = request(self.server, '/api/v1/create_sampling_session', body)
        self.assertEqual(status, 200, raw)
        return loads(raw)['sampling_session_id']

    def test_auth_and_connection_close(self):
        status, headers, raw = request(self.server, '/api/v1/healthz', method='GET', key=None)
        self.assertEqual(status, 401)
        self.assertEqual(loads(raw)['code'], 'unauthorized')
        self.assertNotIn(API_KEY, raw.decode())
        self.assertEqual(headers['Connection'], 'close')
        status, _headers, raw = request(self.server, '/api/v1/healthz', method='GET', key='nope')
        self.assertEqual(status, 401)
        status, headers, raw = request(self.server, '/api/v1/healthz', method='GET', bearer=True)
        self.assertEqual((status, loads(raw)), (200, {'status': 'ok'}))
        self.assertEqual(headers['Connection'], 'close')
        status, _headers, raw = request(self.server, '/api/v1/get_server_capabilities', method='GET')
        self.assertEqual(loads(raw)['supported_models'], [{'model_name': 'wsm', 'sampleable': True, 'trainable': False}])

    def test_unsupported_training_and_sample_options(self):
        status, _headers, raw = request(self.server, '/api/v1/optim_step', {'type': 'optim_step'})
        self.assertEqual((status, loads(raw)['code']), (400, 'unsupported'))
        self.assertNotIn('Traceback', raw.decode())
        sdk = self.sdk_session()
        sampling = self.sampling_session(sdk)
        status, _headers, raw = request(self.server, '/api/v1/asample', sample_body(sampling, 0, temperature=0.7))
        self.assertEqual((status, loads(raw)['code']), (400, 'unsupported'))
        self.assertEqual(self.session.calls, [])
        hot = sample_body(sampling, 1)
        hot['num_samples'] = 2
        status, _headers, raw = request(self.server, '/api/v1/asample', hot)
        self.assertEqual(loads(raw)['code'], 'unsupported')
        image = sample_body(sampling, 2)
        image['prompt'] = {'chunks': [{'type': 'image', 'data': 'abc'}]}
        status, _headers, raw = request(self.server, '/api/v1/asample', image)
        self.assertEqual(loads(raw)['code'], 'unsupported')
        self.assertEqual(self.session.calls, [])

    def test_rejects_filesystem_model_paths(self):
        sdk = self.sdk_session()
        for path in ('/etc/passwd', 'tinker://wsm/../../etc', '../wsm', 'file:///tmp/model'):
            status, _headers, raw = request(self.server, '/api/v1/create_sampling_session', {
                'session_id': sdk, 'sampling_session_seq_id': 0, 'model_path': path,
            })
            self.assertEqual(status, 400, path)
            self.assertIn(loads(raw)['code'], ('unsupported', 'invalid_request'))

    def test_sample_idempotency_stale_checkpoint_and_unknown_future(self):
        sdk = self.sdk_session()
        sampling = self.sampling_session(sdk)
        body = sample_body(sampling, 7, stop=[99])
        status, _headers, raw = request(self.server, '/api/v1/asample', body)
        self.assertEqual(status, 200, raw)
        first = loads(raw)
        self.assertEqual(first['sample_sequence_ids'], [first['request_id'] + ':0'])
        status, _headers, raw = request(self.server, '/api/v1/asample', body)
        self.assertEqual(loads(raw)['request_id'], first['request_id'])
        self.assertEqual(len(self.session.calls), 1)
        query = self.session.calls[0]['query']
        self.assertEqual(query, {
            'checkpoint': 'ckpt-a', 'tokens': [11, 22, 33], 'max_tokens': 4,
            'temperature': 0, 'prompt_logprobs': True, 'stop': [99],
        })
        self.assertEqual(self.session.calls[0]['arm'], ARM)
        self.assertTrue(self.session.calls[0]['key'].startswith('tinker-sample:'))
        changed = sample_body(sampling, 7, tokens=(1,))
        status, _headers, raw = request(self.server, '/api/v1/asample', changed)
        self.assertEqual((status, loads(raw)['code']), (422, 'idempotency_conflict'))
        self.assertEqual(len(self.session.calls), 1)
        status, headers, raw = request(self.server, '/api/v1/retrieve_future', {'request_id': first['request_id']}, method='POST')
        self.assertEqual(status, 406)
        self.assertNotIn(b'SECRET', raw)
        retrieve = urllib.request.Request(
            self.server.base_url + '/api/v1/retrieve_future',
            data=json.dumps({'request_id': first['request_id']}).encode(),
            headers={'Content-Type': 'application/json', 'Accept': 'application/x-protobuf', 'X-API-Key': API_KEY, 'Connection': 'close'},
        )
        with urllib.request.urlopen(retrieve, timeout=10) as response:
            self.assertIn('application/x-protobuf', response.headers['Content-Type'])
            encoded = response.read()
        self.assertGreater(len(encoded), 0)
        self.session.last_result['sequences'][0]['tokens'][0] = 9999
        with urllib.request.urlopen(retrieve, timeout=10) as response:
            self.assertEqual(response.read(), encoded)
        status, _headers, raw = request(self.server, '/api/v1/retrieve_future', {'request_id': 'missing'})
        self.assertEqual(status, 404)
        self.assertEqual(len(self.session.calls), 1)
        self.session.checkpoint = 'ckpt-b'
        status, _headers, raw = request(self.server, '/api/v1/asample', sample_body(sampling, 8))
        self.assertEqual((status, loads(raw)['code']), (422, 'stale_base'))
        self.assertEqual(len(self.session.calls), 1)
        rebound = self.sampling_session(sdk, seq_id=1)
        status, _headers, raw = request(self.server, '/api/v1/asample', sample_body(rebound, 0))
        self.assertEqual(status, 200, raw)
        self.assertEqual(self.session.calls[-1]['query']['checkpoint'], 'ckpt-b')

    def test_unknown_outcome_is_not_rerun_and_errors_hide_tracebacks(self):
        sdk = self.sdk_session()
        sampling = self.sampling_session(sdk)
        self.session.unknown = True
        status, _headers, raw = request(self.server, '/api/v1/asample', sample_body(sampling, 1))
        request_id = loads(raw)['request_id']
        status, _headers, raw = request(self.server, '/api/v1/asample', sample_body(sampling, 1))
        self.assertEqual(loads(raw)['request_id'], request_id)
        self.assertEqual(len(self.session.calls), 1)
        status, _headers, raw = request(self.server, '/api/v1/retrieve_future', {'request_id': request_id})
        self.assertEqual(loads(raw)['category'], 'server')
        self.session.boom = True
        status, _headers, raw = request(self.server, '/api/v1/asample', sample_body(sampling, 2))
        text = raw.decode()
        self.assertEqual(status, 500)
        self.assertNotIn('SECRET', text)
        self.assertNotIn('Traceback', text)
        self.assertNotIn('private-traceback', text)

    def test_weight_editor_refreshes_inspection_and_keeps_explicit_edit_views(self):
        editor = WeightEditor(self.server.base_url, API_KEY)
        view = editor.observe()
        self.assertEqual(view['base'], 'ckpt-a')
        inspected = editor.inspect('w', [0], key='inspect-1')
        self.assertEqual(inspected['result']['checkpoint'], 'ckpt-a')
        self.session.checkpoint = 'ckpt-b'
        refreshed = editor.observe()
        self.assertEqual(refreshed['base'], 'ckpt-b')
        editor.inspect('w', [0], key='inspect-2')
        self.assertEqual(self.session.calls[-1]['inspect']['checkpoint'], 'ckpt-b')
        edited = editor.submit([{'tensor': 'w', 'index': 0, 'value': 1}], view=view, key='edit-1')
        self.assertEqual(edited['selection']['accepted'], True)
        ran = editor.run_code('weights[0][0] = 1', [{'tensor': 'w', 'indices': [0]}], view=view, key='code-1')
        self.assertEqual(ran['submitted'], True)
        self.assertEqual(self.session.run_calls[0]['base'], 'ckpt-a')
        self.assertEqual(self.session.run_calls[0]['observation'], 3)
        self.assertEqual(self.session.run_calls[0]['inspections'], [{'tensor': 'w', 'indices': [0]}])

    def test_loopback_and_credential_url_are_rejected(self):
        with self.assertRaises(WeightlessError):
            TinkerServer(self.session, arm=ARM, api_key=API_KEY, host='0.0.0.0')
        with self.assertRaises(WeightlessError):
            WeightEditor('http://example.com:9', API_KEY)
        with self.assertRaises(WeightlessError):
            WeightEditor('http://user:secret@127.0.0.1:9', API_KEY)
        with self.assertRaises(WeightlessError):
            WeightEditor('https://127.0.0.1:9', API_KEY)

    def test_code_rejects_foreign_experiment_views_before_dispatch(self):
        view = self.session.observe(ARM)
        for field in ('session', 'declaration'):
            body = {'source': "weights['w'][0]=1", 'inspections': [{'tensor': 'w', 'indices': [0]}],
                    'view': {name: view[name] for name in ('session', 'arm', 'declaration', 'observation', 'base')},
                    'key': 'foreign-' + field}
            body['view'][field] = 'another-experiment'
            status, _, raw = request(self.server, '/wsm/v1/run_code', body)
            self.assertEqual((status, loads(raw)['code']), (400, 'invalid_proposal'))
        self.assertEqual(self.session.run_calls, [])


class SessionAdapterTests(unittest.TestCase):
    def test_connection_factory_failure_stops_server_startup(self):
        class UnavailableSession(FakeSession):
            def reopen(self):
                raise RuntimeError('private connection failure')

        with self.assertRaises(WeightlessError) as error:
            with TinkerServer(UnavailableSession(), arm=ARM, api_key=API_KEY):
                self.fail('Server started without a session connection')
        self.assertEqual(error.exception.code, 'target_failure')
        self.assertNotIn('private', str(error.exception))

    def test_real_session_inspect_and_submit_from_the_server_thread(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        session = Session.create(Path(tmp.name) / 'session.db', runner=Runner(), target={'kind': 'test'},
                                  declaration=declaration(), development=[{'id': 'public'}])
        self.addCleanup(session.close)
        with TinkerServer(session, arm=ARM, api_key=API_KEY) as server:
            editor = WeightEditor(server.base_url, API_KEY)
            view = editor.observe()
            self.assertEqual(view['arm'], ARM)
            inspected = editor.inspect('w', [0], key='read')
            self.assertEqual(inspected['result']['values'][0]['value'], 0)
            submitted = editor.submit([{'tensor': 'w', 'index': 0, 'value': 1}], view=view, key='write')
            self.assertTrue(submitted['selection']['accepted'])
            moved = editor.observe()
            self.assertNotEqual(moved['base'], view['base'])
            self.assertEqual(editor.inspect('w', [0], key='new-read')['result']['values'][0]['value'], 1)
            with self.assertRaises(WeightlessError) as stale:
                editor.submit([{'tensor': 'w', 'index': 0, 'value': 0.5}], view=view, key='stale-write')
            self.assertEqual(stale.exception.code, 'stale_base')
            rejected = editor.submit([{'tensor': 'w', 'index': 0, 'value': 0.5}], view=moved, key='next')
            self.assertFalse(rejected['selection']['accepted'])
            self.assertEqual(session.export()['used']['attempts'], 2)


class SdkTests(unittest.TestCase):
    def test_explicit_empty_stop_disables_eos_in_the_real_worker(self):
        try:
            import tinker
            import torch
        except ImportError:
            self.skipTest('tinker and torch are required')
        from types import SimpleNamespace
        from wsm.target_worker import ModelTarget

        class EosModel(torch.nn.Module):
            config = SimpleNamespace(vocab_size=3, max_position_embeddings=2048)

            def forward(self, input_ids):
                logits = torch.zeros((*input_ids.shape, 3))
                logits[..., 2] = 10
                return SimpleNamespace(logits=logits)

        target = ModelTarget.__new__(ModelTarget)
        target.torch = torch
        target.device = 'cpu'
        target.tokenizer = SimpleNamespace(eos_token_id=2)
        target.model = EosModel()

        class WorkerSession(FakeSession):
            def infer(self, query, *, key, arm):
                result = target.infer(query, query['max_tokens'] + 1)
                result.update(checkpoint=query['checkpoint'], usage={'forward_calls': target.forward_calls})
                return {'status': 'completed', 'result': result}

        with TinkerServer(WorkerSession(), arm=ARM, api_key=API_KEY) as server:
            client = tinker.ServiceClient(base_url=server.base_url, api_key=API_KEY)
            try:
                sampler = client.create_sampling_client(base_model='wsm').result()
                prompt = tinker.types.ModelInput.from_ints([1])
                for stop, tokens, reason in ((None, [2], 'stop'), ([], [2, 2, 2], 'length'),
                                             ([1], [2, 2, 2], 'length'), ([2], [2], 'stop')):
                    with self.subTest(stop=stop):
                        result = sampler.sample(prompt, num_samples=1,
                            sampling_params=tinker.types.SamplingParams(
                                max_tokens=3, temperature=0, stop=stop)).result(timeout=30)
                        self.assertEqual(result.sequences[0].tokens, tokens)
                        self.assertEqual(result.sequences[0].stop_reason, reason)
            finally:
                client.close('success').result(timeout=10)

    def test_real_sdk_samples_and_scores_prompt_logprobs(self):
        try:
            import tinker
        except ImportError:
            self.skipTest('tinker SDK not installed')
        session = FakeSession()
        with TinkerServer(session, arm=ARM, api_key=API_KEY) as server:
            saved = {name: os.environ.get(name) for name in ('TINKER_API_KEY', 'TINKER_BASE_URL')}
            os.environ['TINKER_API_KEY'] = API_KEY
            os.environ['TINKER_BASE_URL'] = server.base_url
            client = tinker.ServiceClient(base_url=server.base_url, api_key=API_KEY)
            try:
                capabilities = client.get_server_capabilities()
                self.assertEqual(capabilities.supported_models[0].model_name, 'wsm')
                self.assertTrue(capabilities.supported_models[0].sampleable)
                self.assertFalse(capabilities.supported_models[0].trainable)
                sampler = client.create_sampling_client(base_model='wsm').result()
                prompt = tinker.types.ModelInput.from_ints([11, 22, 33])
                sampled = sampler.sample(
                    prompt,
                    num_samples=1,
                    sampling_params=tinker.types.SamplingParams(max_tokens=4, temperature=0, stop=[99]),
                    include_prompt_logprobs=True,
                ).result()
                self.assertEqual(sampled.sequences[0].tokens, [1000, 1001, 1002, 1003])
                self.assertEqual(sampled.sequences[0].logprobs, [-0.5, -0.5, -0.5, -0.5])
                self.assertEqual(sampled.sequences[0].stop_reason, 'length')
                first_sequence_id = sampled.sequences[0].sequence_id
                self.assertTrue(first_sequence_id.endswith(':0'))
                self.assertEqual(sampled.prompt_logprobs, [None, -1.5, -1.5])
                self.assertEqual(session.calls[0]['query']['stop'], [99])
                self.assertEqual(session.calls[0]['query']['temperature'], 0)
                scored = sampler.compute_logprobs(prompt).result()
                self.assertEqual(scored, [None, -1.5, -1.5])
                self.assertEqual(session.calls[-1]['query']['temperature'], 0)
                self.assertEqual(session.calls[-1]['query']['max_tokens'], 0)
                self.assertTrue(session.calls[-1]['query']['prompt_logprobs'])
                repeated = sampler.sample(prompt, num_samples=1,
                    sampling_params=tinker.types.SamplingParams(max_tokens=4, temperature=0)).result()
                self.assertEqual(repeated.sequences[0].tokens, sampled.sequences[0].tokens)
                self.assertNotEqual(repeated.sequences[0].sequence_id, first_sequence_id)
                silent = sampler.sample(
                    prompt,
                    num_samples=1,
                    sampling_params=tinker.types.SamplingParams(max_tokens=0, temperature=0),
                    include_prompt_logprobs=True,
                ).result()
                self.assertEqual(list(silent.sequences), [])
                self.assertEqual(silent.prompt_logprobs, [None, -1.5, -1.5])
                self.assertEqual(session.calls[-1]['query']['max_tokens'], 0)
            finally:
                client.close('success').result()
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value


class CliTests(unittest.TestCase):
    def test_missing_api_key_exits_without_printing_a_token(self):
        previous = os.environ.pop('WSM_TINKER_API_KEY', None)
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(['/no/such/workspace', '--arm', ARM])
        finally:
            if previous is not None:
                os.environ['WSM_TINKER_API_KEY'] = previous
        self.assertEqual(code, 2)
        self.assertIn('WSM_TINKER_API_KEY is required', stderr.getvalue())
        self.assertNotIn('tml-', stdout.getvalue() + stderr.getvalue())

    def test_workspace_error_does_not_echo_the_api_key(self):
        previous = os.environ.get('WSM_TINKER_API_KEY')
        os.environ['WSM_TINKER_API_KEY'] = 'tml-do-not-print'
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(WeightlessError):
                    main(['/no/such/workspace', '--arm', ARM])
        finally:
            if previous is None:
                os.environ.pop('WSM_TINKER_API_KEY', None)
            else:
                os.environ['WSM_TINKER_API_KEY'] = previous
        self.assertNotIn('tml-do-not-print', stdout.getvalue() + stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
