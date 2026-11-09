"""Local Linux isolation through bubblewrap and systemd resource accounting."""
import copy
import hashlib
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import tempfile
import time
import uuid

from .sandbox import _MAX_INPUT, _MAX_OUTPUT, _WorkerRunner, _error

_READY = b'weightless.sandbox.ready.v1\n'
_WORKER = "import runpy,sys;sys.path.insert(0,'/opt/wsm');runpy.run_module('wsm.target_worker',run_name='__main__')"


class BubblewrapRunner(_WorkerRunner):
    """One-shot workers on a root-authorized Linux systemd host, without Docker."""

    def __init__(self, *, python='/usr/bin/python3', bwrap='bwrap',
                 systemd_run='systemd-run', systemctl='systemctl', timeout=60,
                 cpus=2, memory_bytes=2 * 1024 ** 3, pids=128,
                 tmp_bytes=64 * 1024 ** 2):
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
                or type(cpus) not in (int, float) or not math.isfinite(cpus) or cpus <= 0
                or any(type(value) is not int or value <= 0 for value in (memory_bytes, pids, tmp_bytes))
                or any(type(value) is not str or not value for value in (python, bwrap, systemd_run, systemctl))):
            _error('invalid_config', 'Invalid local sandbox configuration')
        if os.name != 'posix' or not Path('/run/systemd/system').is_dir() or os.geteuid() != 0:
            _error('sandbox_unavailable', 'A root-authorized Linux systemd host is required')
        self._python = os.path.abspath(python)
        self._runtime = None
        if not self._python.startswith('/usr/'):
            prefix = Path(self._python).parent.parent
            if Path(self._python).parent.name != 'bin' or not (prefix / 'pyvenv.cfg').is_file():
                _error('invalid_config', 'Python must be under /usr or an explicit virtual environment')
            self._runtime = str(prefix)
        self._timeout = timeout
        self._limits = {'cpus': cpus, 'memory_bytes': memory_bytes, 'pids': pids, 'tmp_bytes': tmp_bytes}
        self._bwrap, self._systemd_run, self._systemctl = [shutil.which(tool) for tool in (bwrap, systemd_run, systemctl)]
        if not all((self._bwrap, self._systemd_run, self._systemctl)):
            _error('sandbox_unavailable', 'bubblewrap and systemd tools are required')
        try:
            versions = [subprocess.run([tool, '--version'], capture_output=True, text=True,
                                      check=True, timeout=5).stdout.splitlines()[0]
                        for tool in (self._bwrap, self._systemd_run)]
            python_digest = hashlib.sha256(Path(self._python).read_bytes()).hexdigest()
            package = Path(__file__).parent
            self._worker = {name: (package / name).read_bytes()
                            for name in ('binary64.py', 'target_worker.py')}
            self._worker['__init__.py'] = b''
        except (OSError, subprocess.SubprocessError, IndexError):
            _error('sandbox_unavailable', 'Local sandbox runtime is unavailable')
        digest = hashlib.sha256()
        for name, value in sorted(self._worker.items()):
            digest.update(name.encode() + b'\0' + value)
        self._config = {'mode': 'enforced', 'backend': 'bubblewrap-systemd-v1',
                        'timeout': timeout, **self._limits, 'network': 'none', 'read_only': True,
                        'uid': '65534:65534', 'python': self._python, 'python_sha256': python_digest,
                        'worker_sha256': digest.hexdigest(), 'bubblewrap': versions[0],
                        'systemd': versions[1], 'runtime': self._runtime or '/usr'}

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def _command(self, mount=None):
        command = [self._bwrap, '--unshare-all', '--unshare-user', '--disable-userns',
                   '--uid', '65534', '--gid', '65534', '--die-with-parent', '--new-session',
                   '--cap-drop', 'ALL', '--clearenv', '--ro-bind', '/usr', '/usr']
        for path in ('/lib', '/lib64', '/bin', '/sbin'):
            if Path(path).is_symlink():
                command += ['--symlink', os.readlink(path), path]
            elif Path(path).exists():
                command += ['--ro-bind', path, path]
        for key, value in {'PATH': str(Path(self._python).parent) + ':/usr/bin', 'HOME': '/tmp',
                           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                           'TRANSFORMERS_OFFLINE': '1', 'LANG': 'C.UTF-8'}.items():
            command += ['--setenv', key, value]
        command += ['--proc', '/proc', '--dev', '/dev', '--size', str(self._limits['tmp_bytes']),
                    '--tmpfs', '/tmp', '--chdir', '/tmp']
        if self._runtime:
            command += ['--ro-bind', self._runtime, self._runtime]
        if mount:
            command += ['--ro-bind', mount, '/target']
        return command + ['--']

    def _cleanup(self, unit, settled):
        try:
            stopped = subprocess.run([self._systemctl, '--no-ask-password', 'stop', unit],
                                     capture_output=True, timeout=5)
            state = subprocess.run([self._systemctl, 'show', unit, '--property=LoadState',
                                    '--property=ActiveState', '--property=ControlGroup', '--property=Result'],
                                   capture_output=True, text=True, timeout=5)
            fields = dict(line.split('=', 1) for line in state.stdout.splitlines() if '=' in line)
            absent = fields.get('LoadState') == 'not-found'
            inactive = fields.get('ActiveState') in ('inactive', 'failed')
            if (not absent and (state.returncode or not inactive)) or (stopped.returncode and not (absent and settled)):
                _error('sandbox_cleanup_unknown', 'Local sandbox termination could not be confirmed')
            group = fields.get('ControlGroup')
            if group:
                events = Path('/sys/fs/cgroup') / group.lstrip('/') / 'cgroup.events'
                if events.exists() and 'populated 1' in events.read_text():
                    _error('sandbox_cleanup_unknown', 'Local sandbox still has live processes')
            subprocess.run([self._systemctl, 'reset-failed', unit], capture_output=True, timeout=5)
            return fields.get('Result')
        except (OSError, subprocess.SubprocessError):
            _error('sandbox_cleanup_unknown', 'Local sandbox termination could not be confirmed')

    def _run(self, command, payload, timeout=None, on_dispatch=None):
        if type(payload) is not bytes or len(payload) > _MAX_INPUT:
            _error('invalid_request', 'Sandbox input exceeds limit')
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            _error('invalid_request', 'Invalid execution deadline')
        limit = self._timeout if timeout is None else min(self._timeout, timeout)
        deadline = time.monotonic() + limit
        unit = 'weightless-local-' + uuid.uuid4().hex + '.service'
        boundary = command.index('--')
        program = command[boundary + 1:]
        if program == ['python', '-m', 'wsm.target_worker']:
            program = [self._python, '-I', '-c', _WORKER]
        else:
            program = [self._python, *program[1:]]
        gate = (f'import os;os.write(1,{_READY!r});'
                f'ack=os.read(0,1);'
                f'os._exit(125) if ack!=b"\\0" else os.execv({self._python!r},{program!r})')
        properties = {'NoNewPrivileges': 'yes', 'MemoryMax': str(self._limits['memory_bytes']),
                      'MemorySwapMax': '0', 'TasksMax': str(self._limits['pids']),
                      'CPUQuota': str(self._limits['cpus'] * 100) + '%',
                      'RuntimeMaxSec': str(limit) + 's', 'TimeoutStopSec': '1s',
                      'KillMode': 'control-group', 'SendSIGKILL': 'yes', 'LimitCORE': '0',
                      'RestrictRealtime': 'yes', 'SystemCallArchitectures': 'native',
                      'SystemCallFilter': '~@reboot @swap @raw-io @obsolete @keyring'}
        launch = [self._systemd_run, '--quiet', '--wait', '--pipe', '--no-ask-password',
                  '--expand-environment=no', '--unit=' + unit]
        for key, value in properties.items():
            launch += ['--property=' + key + '=' + value]
        outputs = [bytearray(), bytearray()]
        ready = False
        proc = None
        cleanup_result = None
        with tempfile.TemporaryDirectory(prefix='weightless-worker-') as staging:
            package = Path(staging) / 'wsm'
            package.mkdir(mode=0o755)
            for name, value in self._worker.items():
                (package / name).write_bytes(value)
                (package / name).chmod(0o444)
            launch += command[:boundary] + ['--ro-bind', staging, '/opt/wsm', '--remount-ro', '/',
                                            '--', self._python, '-I', '-c', gate]
            try:
                try:
                    proc = subprocess.Popen(launch, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, start_new_session=True,
                                            env={'PATH': os.defpath, 'LANG': 'C.UTF-8'})
                except OSError:
                    _error('sandbox_unavailable', 'Local sandbox could not start')
                with selectors.DefaultSelector() as selector:
                    for index, stream in enumerate((proc.stdout, proc.stderr)):
                        os.set_blocking(stream.fileno(), False)
                        selector.register(stream, selectors.EVENT_READ, index)
                    os.set_blocking(proc.stdin.fileno(), False)
                    pending = memoryview(b'\0' + payload)
                    while selector.get_map() or proc.poll() is None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            _error('sandbox_timeout', 'Sandbox execution timed out')
                        for key, _ in selector.select(min(remaining, .1)):
                            if key.data == 'input':
                                try:
                                    written = os.write(key.fd, pending)
                                except BlockingIOError:
                                    continue
                                except OSError:
                                    _error('sandbox_failure', 'Sandbox input transport failed')
                                pending = pending[written:]
                                if not pending:
                                    selector.unregister(key.fileobj)
                                    proc.stdin.close()
                                continue
                            try:
                                chunk = os.read(key.fd, 65536)
                            except BlockingIOError:
                                continue
                            except OSError:
                                _error('sandbox_failure', 'Sandbox output transport failed')
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            output = outputs[key.data]
                            output.extend(chunk)
                            if key.data == 0 and not ready:
                                if not _READY.startswith(output) and not output.startswith(_READY):
                                    _error('sandbox_failure', 'Invalid sandbox startup acknowledgement')
                                if len(output) >= len(_READY):
                                    del output[:len(_READY)]
                                    ready = True
                                    if on_dispatch is not None:
                                        on_dispatch()
                                    selector.register(proc.stdin, selectors.EVENT_WRITE, 'input')
                            if len(output) > _MAX_OUTPUT:
                                _error('sandbox_output_limit', 'Sandbox output exceeds limit')
                    proc.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                _error('sandbox_timeout', 'Sandbox execution timed out')
            finally:
                if proc is not None:
                    settled = ready or proc.poll() is not None
                    if proc.poll() is None:
                        proc.kill()
                    try:
                        cleanup_result = self._cleanup(unit, settled)
                    finally:
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            _error('sandbox_cleanup_unknown', 'Sandbox control process did not terminate')
                        finally:
                            for stream in (proc.stdin, proc.stdout, proc.stderr):
                                stream.close()
        if not ready:
            _error('sandbox_unavailable', 'Local sandbox did not acknowledge startup')
        if cleanup_result == 'timeout':
            _error('sandbox_timeout', 'Sandbox execution timed out')
        if proc.returncode:
            _error('sandbox_failure', 'Sandbox execution failed')
        return bytes(outputs[0])
