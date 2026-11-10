import json
import os
import math
import io
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from wsm import binary64
from wsm.sandbox import DockerRunner, forward_reservation
from wsm.target_worker import edits_for, run


class WorkerTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), '../../../fixtures/addition.json')) as f:
            fixture = json.load(f)
        checkpoint = binary64.create_checkpoint(fixture['graph'], fixture['tensors'])
        self.request = {'target': {'kind': 'binary64', 'checkpoint': checkpoint}, 'action': 'describe',
                        'patches': [], 'edits': [], 'cases': None, 'query': None, 'generate': False}

    def test_materialize_full_identity_and_reversion(self):
        initial = run(self.request)
        name = next(iter(initial['parameters']))
        self.request.update(action='inspect', query={'tensor': name, 'indices': [0]})
        old = run(self.request)['values'][0]['value']
        self.request.update(action='materialize', edits=[{'tensor': name, 'index': 0, 'value': old + 1}])
        changed = run(self.request)
        self.assertNotEqual(initial['checkpoint'], changed['checkpoint'])
        self.request.update(patches=changed['patches'], edits=[{'tensor': name, 'index': 0, 'value': old}])
        restored = run(self.request)
        self.assertEqual(initial['checkpoint'], restored['checkpoint'])
        self.assertEqual(restored['patches'], [])

    def test_aliases_reject_duplicate_canonical_addresses(self):
        parameters = {'embedding': {'size': 2, 'dtype': 'float32'}}
        aliases = {'head': 'embedding'}
        edits = [{'tensor': 'embedding', 'index': 0, 'value': 1},
                 {'tensor': 'head', 'index': 0, 'value': 2}]
        with self.assertRaises(binary64.WeightlessError):
            edits_for(edits, parameters, aliases)
        self.assertEqual(edits_for(edits[1:], parameters, aliases)[0]['tensor'], 'embedding')

    def test_float32_rounding_has_replayable_positive_zero(self):
        parameters = {'w': {'size': 1, 'dtype': 'float32'}}
        value = edits_for([{'tensor':'w','index':0,'value':-1e-50}], parameters, {})[0]['value']
        self.assertEqual(value, 0.0)
        self.assertEqual(math.copysign(1.0, value), 1.0)
        with self.assertRaises(binary64.WeightlessError):
            edits_for([{'tensor':'w','index':0,'value':1e100}], parameters, {})

    def test_worker_process_returns_observed_receipt(self):
        result = subprocess.run([sys.executable, '-m', 'wsm.target_worker'],
                                input=json.dumps(self.request), text=True, capture_output=True, check=True)
        response = json.loads(result.stdout)['result']
        self.assertIn('Uid', response['sandbox']['status'])
        self.assertIn('/', response['sandbox']['mounts'])

    def test_rejects_duplicate_edits_and_evaluates(self):
        name = next(iter(run(self.request)['parameters']))
        edit = {'tensor': name, 'index': 0, 'value': 1}
        self.request.update(action='materialize', edits=[edit, edit])
        with self.assertRaises(binary64.WeightlessError):
            run(self.request)
        self.request.update(action='evaluate', edits=[], cases=[{'id': 'one', 'input': [1, 2], 'target': [3]}])
        result = run(self.request)
        self.assertEqual(result['metrics']['count'], 1)
        self.assertEqual(result['cases'][0]['id'], 'one')

    def test_binary_measures_each_prediction_once_and_enforces_limit(self):
        cases = [{'id': 'positive', 'input': [1, 2], 'target': [3]},
                 {'id': 'negative', 'input': [-1, -2], 'target': [-3]}]
        self.request.update(action='evaluate', cases=cases, forward_limit=2)
        original = binary64.predict
        with patch('wsm.target_worker.binary.predict', wraps=original) as predict:
            result = run(self.request)
            self.assertEqual(predict.call_count, 2)
        self.assertEqual(result['metrics'], {'loss': .5, 'rmse': 3., 'exact': 0, 'count': 2})
        self.assertEqual(result['usage'], {'forward_calls': 2, 'forward_limit': 2})
        self.request['forward_limit'] = 1
        with patch('wsm.target_worker.binary.predict', wraps=original) as predict:
            with self.assertRaises(binary64.WeightlessError) as error:
                run(self.request)
            self.assertEqual(error.exception.code, 'forward_limit_exceeded')
            self.assertEqual(predict.call_count, 1)

    def test_forward_reservations_separate_generation_from_prediction(self):
        self.assertEqual(forward_reservation({'kind': 'smollm'}, 2, generate=True), 34)
        self.assertEqual(forward_reservation({'kind': 'smollm'}, 2, generate=True, max_new_tokens=3), 8)
        self.assertEqual(forward_reservation({'kind': 'smollm'}, 2), 2)
        self.assertEqual(forward_reservation({'kind': 'binary64'}, 2, generate=True), 2)
        self.assertEqual(forward_reservation({'kind': 'huggingface'}, 2, generate=True), 34)


class RunnerTests(unittest.TestCase):
    def runner(self):
        with patch('wsm.sandbox.subprocess.run') as inspect:
            inspect.return_value.stdout = 'sha256:' + 'a' * 64
            return DockerRunner('trusted')

    def test_restrictions_and_probe_never_mount_target(self):
        runner = self.runner()
        commands = []
        def capture(command, payload, timeout=None, on_dispatch=None):
            commands.append(command)
            return b'probe'
        with patch.object(runner, '_run', side_effect=capture):
            self.assertEqual(runner.probe('print(1)'), 'probe')
        command = commands[0]
        for option in ['--network', '--read-only', '--cap-drop', '--security-opt', '--user', '--pids-limit', '--memory', '--cpus', '--tmpfs']:
            self.assertIn(option, command)
        self.assertNotIn('--mount', command)
        self.assertIn('sha256:' + 'a' * 64, command)

    def test_unavailable_docker_fails_closed(self):
        with patch('wsm.sandbox.subprocess.run', side_effect=FileNotFoundError):
            with self.assertRaises(binary64.WeightlessError) as error:
                DockerRunner('missing')
        self.assertEqual(error.exception.code, 'sandbox_unavailable')

    def test_worker_error_is_preserved(self):
        runner = self.runner()
        with patch.object(runner, '_run', return_value=b'{"error":{"code":"invalid_edit","message":"bad"}}'):
            with self.assertRaises(binary64.WeightlessError) as error:
                runner.execute('describe', {'kind': 'binary64', 'checkpoint': {}})
        self.assertEqual(error.exception.code, 'invalid_edit')

    def test_forward_limit_is_validated_and_transported(self):
        runner = self.runner()
        target = {'kind': 'binary64', 'checkpoint': {}}
        with patch.object(runner, '_run', return_value=b'{"result":{}}') as execute:
            for limit in (True, -1, 1.5):
                with self.assertRaises(binary64.WeightlessError):
                    runner.execute('evaluate', target, forward_limit=limit)
            self.assertFalse(execute.called)
            runner.execute('evaluate', target, forward_limit=3)
            request = json.loads(execute.call_args.args[1])
            self.assertEqual(request['forward_limit'], 3)

    def test_failed_container_cleanup_is_not_reported_as_success(self):
        runner = self.runner()
        process = Mock(stdin=io.BytesIO(), stdout=io.BytesIO(b'{}'), stderr=io.BytesIO(), returncode=0)
        process.wait.return_value = 0
        cleanup = subprocess.CompletedProcess([], 1, b'', b'Cannot connect to Docker daemon')
        with patch('wsm.sandbox.subprocess.Popen', return_value=process), \
                patch('wsm.sandbox.subprocess.run', return_value=cleanup):
            with self.assertRaises(binary64.WeightlessError) as error:
                runner._run(['docker', 'run', '--name', 'owned-container'], b'{}')
        self.assertEqual(error.exception.code, 'sandbox_cleanup_unknown')


class DeviceTests(unittest.TestCase):
    def test_execution_device_is_explicit(self):
        from wsm.target_worker import ModelTarget

        with self.assertRaises(binary64.WeightlessError) as automatic:
            ModelTarget({}, device='automatic')
        self.assertEqual(automatic.exception.code, 'unsupported_target')
        torch = Mock()
        torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {'torch': torch}):
            with self.assertRaises(binary64.WeightlessError) as missing:
                ModelTarget({}, device='cuda')
        self.assertEqual(missing.exception.code, 'unsupported_target')


if __name__ == '__main__':
    unittest.main()
