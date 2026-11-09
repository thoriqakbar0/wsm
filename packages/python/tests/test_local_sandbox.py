"""Run with WEIGHTLESS_LOCAL_SANDBOX=1 for real namespace and cgroup checks."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from wsm import BubblewrapRunner, Session, Verifier, WeightlessError, replay_session, verify_session
from wsm.binary64 import create_checkpoint


class LocalConfigurationTests(unittest.TestCase):
    def test_unavailable_launcher_fails_closed(self):
        with self.assertRaises(WeightlessError) as raised:
            BubblewrapRunner(bwrap='/no-such-weightless-bwrap')
        self.assertEqual(raised.exception.code, 'sandbox_unavailable')

    def test_invalid_limits_are_rejected_before_dispatch(self):
        for options in ({'timeout': float('nan')}, {'timeout': True}, {'cpus': 0},
                        {'memory_bytes': -1}, {'pids': 0}, {'tmp_bytes': 0}):
            with self.subTest(options=options), self.assertRaises(WeightlessError) as raised:
                BubblewrapRunner(**options)
            self.assertEqual(raised.exception.code, 'invalid_config')

    def test_cleanup_rejects_unacknowledged_or_still_active_workloads(self):
        runner = object.__new__(BubblewrapRunner)
        runner._systemctl = 'systemctl'
        for stop_code, state in [(5, 'LoadState=not-found\nActiveState=inactive\n'),
                                 (0, 'LoadState=loaded\nActiveState=active\n')]:
            replies = [subprocess.CompletedProcess([], stop_code, b'', b''),
                       subprocess.CompletedProcess([], 0, state, '')]
            with self.subTest(state=state), patch('wsm.local_sandbox.subprocess.run', side_effect=replies):
                with self.assertRaises(WeightlessError) as raised:
                    runner._cleanup('unacknowledged.service', settled=False)
            self.assertEqual(raised.exception.code, 'sandbox_cleanup_unknown')


@unittest.skipUnless(os.environ.get('WEIGHTLESS_LOCAL_SANDBOX') == '1',
                     'Set WEIGHTLESS_LOCAL_SANDBOX=1 to exercise bubblewrap and systemd')
class NativeLocalSandboxTests(unittest.TestCase):
    def setUp(self):
        self.runner = BubblewrapRunner(timeout=10, cpus=.5, memory_bytes=256 * 1024**2)
        self.units = []
        popen = subprocess.Popen

        def observe(command, *args, **kwargs):
            self.units.extend(arg.split('=', 1)[1] for arg in command if arg.startswith('--unit='))
            return popen(command, *args, **kwargs)

        observer = patch('wsm.local_sandbox.subprocess.Popen', side_effect=observe)
        observer.start()
        self.addCleanup(observer.stop)

    def tearDown(self):
        for unit in self.units:
            state = subprocess.run(['systemctl', 'show', unit, '--property=LoadState', '--value'],
                                   capture_output=True, text=True, timeout=5)
            self.assertEqual(state.stdout.strip(), 'not-found', 'Owned unit survived teardown')
            self.assertFalse((Path('/sys/fs/cgroup/system.slice') / unit).exists(),
                             'Owned workload cgroup survived teardown')

    def test_private_files_environment_and_network_are_unreachable(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            private = Path(directory) / 'private-verifier-canary'
            private.write_text('synthetic-private-canary')
            listener.bind(('0.0.0.0', 0))
            listener.listen()
            port = listener.getsockname()[1]
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                accepted, _ = listener.accept()
                accepted.close()
            addresses = sorted({'127.0.0.1', *socket.gethostbyname_ex(socket.gethostname())[2]})
            source = '''import json,os,socket,sys
data=json.load(sys.stdin)
blocked=[]
for path in [data['private'],'/proc/1/root'+data['private'],'/var/run/docker.sock','/target','/run/dbus/system_bus_socket']:
    try:
        with open(path,'rb') as stream: stream.read(1)
        blocked.append(False)
    except OSError: blocked.append(True)
reachable=[]
for address in data['addresses']:
    try:
        with socket.create_connection((address,data['port']),timeout=.2): pass
        reachable.append(address)
    except OSError: pass
print(json.dumps({'blocked':blocked,'reachable':reachable,'credentials':os.environ.get('WEIGHTLESS_PRIVATE_CANARY'),
    'network':os.readlink('/proc/self/ns/net'),'pid':os.getpid(),'uid':os.getuid(),
    'status':dict(line.split(':',1) for line in open('/proc/self/status') if ':' in line)}))'''
            before = os.environ.get('WEIGHTLESS_PRIVATE_CANARY')
            os.environ['WEIGHTLESS_PRIVATE_CANARY'] = 'synthetic-secret'
            try:
                result = json.loads(self.runner.probe(source, input_data=json.dumps({
                    'private': str(private), 'addresses': addresses, 'port': port}).encode()))
            finally:
                if before is None:
                    os.environ.pop('WEIGHTLESS_PRIVATE_CANARY')
                else:
                    os.environ['WEIGHTLESS_PRIVATE_CANARY'] = before
            self.assertTrue(all(result['blocked']))
            self.assertEqual(result['reachable'], [])
            self.assertIsNone(result['credentials'])
            self.assertNotEqual(result['network'], os.readlink('/proc/self/ns/net'))
            self.assertLess(result['pid'], 10)
            self.assertEqual(result['uid'], 65534)
            self.assertEqual(result['status']['CapEff'].strip(), '0000000000000000')
            self.assertEqual(result['status']['NoNewPrivs'].strip(), '1')

    def test_root_and_runtime_are_read_only(self):
        result = json.loads(self.runner.probe('''import json
from pathlib import Path
blocked=[]
for path in ['/escape-canary','/usr/escape-canary']:
    try: Path(path).write_text('bad')
    except OSError: blocked.append(True)
    else: blocked.append(False)
Path('/tmp/public').write_text('allowed')
print(json.dumps({'blocked':blocked,'tmp':Path('/tmp/public').read_text()}))'''))
        self.assertEqual(result, {'blocked': [True, True], 'tmp': 'allowed'})

    def test_probe_source_is_not_expanded_by_service_manager(self):
        self.assertEqual(self.runner.probe("print('${PATH}')").strip(), '${PATH}')

    def test_model_mount_is_read_only_and_does_not_expose_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model'
            model.mkdir(mode=0o755)
            (model / 'public.json').write_text('public-model-canary')
            (Path(directory) / 'private.json').write_text('synthetic-private-canary')
            source = '''from pathlib import Path
assert Path('/target/public.json').read_text()=='public-model-canary'
assert not Path('/target/private.json').exists()
assert not Path('/private.json').exists()
try: Path('/target/public.json').write_text('changed')
except OSError: print('read-only')
else: raise AssertionError('Writable model mount')'''
            output = self.runner._run(self.runner._command(str(model)) + ['python', '-I', '-c', source], b'')
            self.assertEqual(output.strip(), b'read-only')
            self.assertEqual((model / 'public.json').read_text(), 'public-model-canary')

    def test_input_is_complete_and_output_streams_do_not_deadlock(self):
        payload = b'x' * (1024**2)
        source = '''import os,sys
data=sys.stdin.buffer.read()
os.write(2,b'e'*65536)
print(len(data))'''
        self.assertEqual(self.runner.probe(source, input_data=payload).strip(), str(len(payload)))
        with self.assertRaises(WeightlessError) as raised:
            self.runner.probe('print(1)', input_data=b'x' * (4 * 1024**2 + 1))
        self.assertEqual(raised.exception.code, 'invalid_request')

    def test_explicit_virtual_environment_remains_available_under_tmp(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = Path(directory) / 'public runtime'
            subprocess.run(['/usr/bin/python3', '-m', 'venv', '--without-pip', str(environment)],
                           check=True, timeout=10)
            site, = (environment / 'lib').glob('python*/site-packages')
            (site / 'weightless_runtime_canary.py').write_text('value = 42\n')
            runner = BubblewrapRunner(python=str(environment / 'bin/python'))
            self.assertEqual(runner.probe('import weightless_runtime_canary; print(weightless_runtime_canary.value)').strip(), '42')

    def test_each_output_stream_is_bounded_before_deadline(self):
        for descriptor in (1, 2):
            with self.subTest(descriptor=descriptor), self.assertRaises(WeightlessError) as raised:
                self.runner.probe(f'''import os,time
data=memoryview(b'x'*(4*1024*1024+1))
while data: data=data[os.write({descriptor},data):]
time.sleep(30)''', timeout=5)
            self.assertEqual(raised.exception.code, 'sandbox_output_limit')

    def test_pids_limit_stops_forking(self):
        runner = BubblewrapRunner(timeout=10, pids=24)
        result = json.loads(runner.probe('''import errno,json,subprocess,sys
children=[]
try:
    for _ in range(40):
        children.append(subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']))
except OSError as error: print(json.dumps({'errno':error.errno,'children':len(children)}),flush=True)
else: raise AssertionError('Process ceiling not enforced')
finally:
    for child in children: child.kill()
    for child in children: child.wait()'''))
        self.assertEqual(result['errno'], 11)
        self.assertLess(result['children'], 24)

    def test_tmpfs_limit_stops_writes(self):
        runner = BubblewrapRunner(timeout=10, tmp_bytes=8 * 1024**2)
        result = json.loads(runner.probe('''import json
try:
    with open('/tmp/fill','wb',buffering=0) as stream:
        for _ in range(10): stream.write(b'x'*(1024*1024))
except OSError as error: print(json.dumps({'errno':error.errno}))
else: raise AssertionError('Temporary filesystem ceiling not enforced')'''))
        self.assertEqual(result['errno'], 28)

    def test_memory_exhaustion_and_nonzero_exit_fail(self):
        for source, runner in [('raise SystemExit(7)', self.runner),
                               ('data=bytearray(256*1024*1024)', BubblewrapRunner(memory_bytes=96*1024**2))]:
            with self.subTest(source=source), self.assertRaises(WeightlessError) as raised:
                runner.probe(source)
            self.assertEqual(raised.exception.code, 'sandbox_failure')

    def test_timeout_and_success_remove_detached_descendants(self):
        code = '''import subprocess,sys,time
subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True,
                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
'''
        self.assertEqual(self.runner.probe(code + "print('done')").strip(), 'done')
        with self.assertRaises(WeightlessError) as raised:
            self.runner.probe(code + 'time.sleep(30)', timeout=2)
        self.assertEqual(raised.exception.code, 'sandbox_timeout')
        self.assertEqual(self.runner.probe("print('fresh')").strip(), 'fresh')

    def test_dispatch_cancellation_cleans_execution(self):
        def cancel():
            raise KeyboardInterrupt('synthetic cancellation')

        with self.assertRaises(KeyboardInterrupt):
            self.runner._run(self.runner._command() + ['python', '-I', '-c', 'import time; time.sleep(30)'],
                             b'', on_dispatch=cancel)
        self.assertEqual(self.runner.probe("print('fresh')").strip(), 'fresh')

    def test_kernel_cgroup_limits_and_complete_cleanup(self):
        groups = []

        def dispatched():
            unit, = self.units
            state = subprocess.run(['systemctl', 'show', unit, '--property=ControlGroup', '--value'],
                                   capture_output=True, text=True, check=True, timeout=5)
            group = Path('/sys/fs/cgroup') / state.stdout.strip().lstrip('/')
            self.assertTrue(group.is_dir())
            self.assertEqual((group / 'memory.max').read_text().strip(), str(256 * 1024**2))
            self.assertEqual((group / 'memory.swap.max').read_text().strip(), '0')
            self.assertEqual((group / 'pids.max').read_text().strip(), '128')
            quota, period = map(int, (group / 'cpu.max').read_text().split())
            self.assertEqual(quota / period, .5)
            groups.append(group)

        output = self.runner._run(self.runner._command() + ['python', '-I', '-c', "print('bounded')"],
                                  b'', on_dispatch=dispatched)
        self.assertEqual(output.strip(), b'bounded')
        for group in groups:
            self.assertFalse(group.exists(), 'Workload cgroup survived cleanup')

    def test_private_final_grading_and_replay_preserve_numerical_contract(self):
        graph = {'inputSize': 1, 'nodes': [{'id': 'out', 'op': 'affine', 'input': 'input', 'weight': 'w', 'bias': 'b'}], 'output': 'out'}
        checkpoint = create_checkpoint(graph, {'w': {'shape': [1, 1], 'values': [0]}, 'b': {'shape': [1], 'values': [0]}})
        target = {'kind': 'binary64', 'checkpoint': checkpoint}
        declaration = {'hypothesis': 'Scalar edit solves the finite task', 'arms': ['teacher'], 'editable': ['w'],
                       'budgets': {'attempts': 2, 'evaluations': 4, 'inspections': 1, 'teacher_requests': 0, 'final_evaluations': 1},
                       'max_edits': 1, 'max_inspect': 1, 'wall_seconds': 60}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verifier = Verifier(root / 'private.db', runner=self.runner,
                                datasets={'final': [{'id': 'PRIVATE-CASE', 'input': [7], 'target': [14]}]})
            session = Session.create(root / 'session.db', runner=self.runner, target=target,
                                     declaration=declaration, development=[{'id': 'public', 'input': [3], 'target': [6]}],
                                     verifier=verifier)
            try:
                view = session.observe('teacher')
                proposal = {k: view[k] for k in ('session', 'arm', 'declaration', 'observation', 'base')}
                proposal['edits'] = [{'tensor': 'w', 'index': 0, 'value': 2}]
                self.assertTrue(session.submit(proposal, key='good')['selection']['accepted'])
                frozen = session.freeze(key='freeze')
                receipt = session.finalize(key='final', dataset='final')
                self.assertEqual(receipt['status'], 'completed')
                self.assertEqual(receipt['arms']['teacher']['task']['exact'], 1)
                self.assertNotIn('PRIVATE-CASE', json.dumps(receipt))
                self.assertEqual(replay_session(session.export()), frozen)
                self.assertIsNotNone(verify_session(session.export(), runner=self.runner, target=target))
            finally:
                session.close()
                verifier.close()


if __name__ == '__main__':
    unittest.main()
