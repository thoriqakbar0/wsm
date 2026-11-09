"""Build wl, then prove the pipe, a stale retry, a duplicate, and a killed run.

The follower process is not given the database path. Its current directory
contains this script's public instructions and nothing from the session.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/python/tests'))
sys.path.insert(0, str(ROOT / 'packages/python/src'))

from wsm.broker import Broker, accept_once, parse_command  # noqa: E402
from wsm.session import Session  # noqa: E402
from wsm.verifier import Verifier  # noqa: E402
from test_session import Runner, declaration  # noqa: E402

GRANT = ('observe', 'inspect', 'submit', 'diagnose', 'status', 'freeze', 'cancel', 'finalize', 'export')


def clean_env():
    env = os.environ.copy()
    env['PATH'] = '/usr/bin:/bin'
    env.pop('PYTHONPATH', None)
    return env


def cargo_on_clean_path():
    probe = subprocess.run(['sh', '-c', 'command -v cargo || true'], env=clean_env(), text=True, capture_output=True, check=True)
    return probe.stdout.strip()


def build_wl():
    cargo = Path(shutil.which('cargo') or Path.home() / '.cargo/bin/cargo')
    env = clean_env()
    env['PATH'] = str(cargo.parent) + ':/usr/bin:/bin'
    subprocess.run([str(cargo), 'build', '--quiet', '--manifest-path', str(ROOT / 'clients/wl/Cargo.toml')],
                   cwd=ROOT, env=env, check=True)
    return ROOT / 'clients/wl/target/debug/wl'


def new_stack(root):
    runner = Runner()
    config = declaration()
    config['budgets'] = dict(config['budgets'], diagnostics=1)
    verifier = Verifier(root / 'private.db', runner=runner, datasets={
        'final': {'task': [{'id': 'CANARY-PRIVATE', 'target': [1]}], 'retention': []}})
    os.chmod(root, 0o700)
    session = Session.create(root / 'session.db', runner=runner, target={'kind': 'test'}, declaration=config,
                             development=[{'id': 'public'}], verifier=verifier, final_dataset='final')
    os.chmod(root / 'session.db', 0o600)
    broker = Broker(session, arm='teacher', allowed_actions=GRANT)
    return runner, verifier, session, broker


class StaleOnce:
    def __init__(self, broker):
        self.broker = broker
        self.armed = True
        self.injected = False

    def handle(self, envelope):
        command = parse_command(envelope) if type(envelope) is bytes else envelope
        if self.armed and command['action'] == 'submit':
            self.armed = False
            self.injected = True
            command = json.loads(json.dumps(command))
            command['payload']['base'] = 'injected-stale-base'
            return self.broker.handle(command)
        return self.broker.handle(envelope)


def run_follower(broker, wl, agent_dir):
    env = clean_env()
    proc = subprocess.Popen([sys.executable, str(ROOT / 'scripts/agent_follow.py')], cwd=agent_dir,
                            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    sent, seen = [], []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        sent.append(json.loads(line))
        if any(private in line for private in ('session.db', 'private.db', 'CANARY-PRIVATE')):
            raise AssertionError('agent command named private state')
        raw = accept_once(broker, line.encode(), wl=str(wl))
        if not raw:
            raise AssertionError('wl returned no response')
        response = json.loads(raw)
        seen.append(response)
        if 'CANARY-PRIVATE' in json.dumps(response):
            raise AssertionError('private case reached the agent')
        if sent[-1]['action'] == 'export':
            break
        proc.stdin.write(json.dumps(response) + '\n')
        proc.stdin.flush()
    proc.stdin.close()
    code = proc.wait(timeout=5)
    if code != 0:
        raise AssertionError('follower exited ' + str(code))
    return sent, seen


def interrupt_once(root, wl):
    root.mkdir(mode=0o700)
    runner, verifier, session, broker = new_stack(root)
    try:
        original = runner.execute
        def crash(action, target, **kwargs):
            if action == 'evaluate':
                runner.calls.append((action, kwargs))
                raise KeyboardInterrupt()
            return original(action, target, **kwargs)
        runner.execute = crash
        observed = broker.handle({'format': 'weightless.command.v1', 'action': 'observe', 'key': 'wl:observe', 'payload': {}})
        draft = json.loads(json.dumps(next(item['command'] for item in observed['next'] if item['command']['action'] == 'submit')))
        draft['key'] = 'crash'
        draft['payload']['edits'] = [{'tensor': 'w', 'index': 0, 'value': 1}]
        try:
            accept_once(broker, json.dumps(draft).encode(), wl=str(wl))
            raise AssertionError('expected the runner interrupt to produce no response')
        except KeyboardInterrupt:
            pass
        calls = len(runner.calls)
        runner.execute = original
        replayed = json.loads(accept_once(broker, json.dumps(draft).encode(), wl=str(wl)))
        if replayed['outcome'] != 'unknown' or len(runner.calls) != calls:
            raise AssertionError('interrupted submit was not replayed as unknown')
        frozen = json.loads(accept_once(broker, json.dumps(replayed['next'][0]['command']).encode(), wl=str(wl)))
        if frozen['result']['complete'] or frozen['result']['obligations'] != ['crash']:
            raise AssertionError('freeze did not keep the interrupted operation')
    finally:
        session.close()
        verifier.close()


def install_package(root):
    """Install into an empty directory. python3-venv is not installed, so this does not create a venv."""
    dest = root / 'site'
    dest.mkdir(parents=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--disable-pip-version-check',
                    '--target', str(dest), str(ROOT / 'packages/python')],
                   check=True, env=clean_env(), stdout=subprocess.DEVNULL)
    env = clean_env()
    env['PYTHONPATH'] = str(dest)
    probe = subprocess.run([sys.executable, '-c', 'from wsm.broker import Broker; print(Broker.__module__)'],
                           cwd='/', env=env, check=True, text=True, capture_output=True)
    return probe.stdout.strip()


def main():
    default_cargo = cargo_on_clean_path()
    wl = build_wl()
    class LargeBroker:
        def handle(self, _command):
            return {'payload': 'x' * 262144}
    large = json.loads(accept_once(LargeBroker(), b'{}', wl=str(wl)))
    if len(large['payload']) != 262144:
        raise AssertionError('large response was truncated')
    refused = subprocess.run([str(wl), '--db', '/tmp/not-a-session'], capture_output=True)
    extra = subprocess.run([str(wl), '--help'], capture_output=True)
    unattached = subprocess.run([str(wl)], input=b'{}', capture_output=True, timeout=5)
    if unattached.returncode != 1 or json.loads(unattached.stdout)['error']['code'] != 'broker_unavailable':
        raise SystemExit('missing broker pipes did not produce a structured error')
    if refused.returncode != 2 or json.loads(refused.stdout)['error']['code'] != 'unauthorized':
        raise SystemExit('privileged flag was not refused')
    if extra.returncode != 0 or b'Usage: wl' not in extra.stdout:
        raise SystemExit('help did not describe the CLI')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        private = root / 'operator'
        private.mkdir(mode=0o700)
        agent = root / 'agent'
        agent.mkdir()
        shutil.copy(ROOT / 'docs/agent.md', agent / 'AGENT.md')
        runner, verifier, session, broker = new_stack(private)
        try:
            raw = json.dumps({'format': 'weightless.command.v1', 'action': 'observe', 'key': 'wl:observe', 'payload': {}}).encode()
            direct = broker.handle(raw)
            piped = json.loads(accept_once(broker, raw, wl=str(wl)))
            if piped != direct:
                raise AssertionError('wl response did not match the kernel response')
            inspection = next(item['command'] for item in direct['next'] if item['command']['action'] == 'inspect')
            inspection['key'] = 'inspect-1'
            inspection['payload']['indices'] = [0]
            expected_inspection = broker.handle(inspection)
            if json.loads(accept_once(broker, json.dumps(inspection).encode(), wl=str(wl))) != expected_inspection:
                raise AssertionError('inspection response differed between API and CLI')
            wrapper = StaleOnce(broker)
            before = len(runner.calls)
            sent, seen = run_follower(wrapper, wl, agent)
            if not wrapper.injected:
                raise AssertionError('stale fault was not injected')
            errors = [item for item in seen if item['outcome'] == 'error']
            if not errors or errors[0]['error']['code'] != 'stale_base':
                raise AssertionError('stale submit was not refused')
            exports = [item for item in seen if item['command']['action'] == 'export']
            if len(exports) != 1 or exports[0]['result']['final']['status'] != 'completed':
                raise AssertionError('follower did not export a final receipt')
            if session.export()['used']['attempts'] != 1:
                raise AssertionError('stale or duplicate submit changed the attempt count')
            calls = len(runner.calls)
            submit = next(item for item in sent if item['action'] == 'submit' and item['payload']['base'] != 'injected-stale-base')
            duplicate = json.loads(accept_once(broker, json.dumps(submit).encode(), wl=str(wl)))
            if duplicate['result']['status'] != 'committed' or len(runner.calls) != calls:
                raise AssertionError('duplicate submit ran again')
            if before == calls:
                raise AssertionError('the accepted submit did not reach the runner')
            for request, response in zip(sent, seen):
                if request['action'] != 'observe' and response['outcome'] == 'ok':
                    if broker.handle(request) != response:
                        raise AssertionError('API and CLI receipt differed for ' + request['action'])
        finally:
            session.close()
            verifier.close()
        interrupt_once(root / 'interrupt', wl)
        installed = install_package(root / 'install')
    rustc_path = shutil.which('rustc') or str(Path.home() / '.cargo/bin/rustc')
    rustc = subprocess.run([rustc_path, '--version'], check=True, text=True, capture_output=True).stdout.strip()
    report = {
        'passed': True,
        'rustc': rustc,
        'cargo_on_default_path': bool(default_cargo),
        'wl': str(wl),
        'privileged_flag': 'unauthorized',
        'install': 'pip target directory; python3-venv is not installed',
        'installed_module': installed,
        'stale_recovered': True,
        'duplicate_submit_runner_calls_unchanged': True,
        'interrupted_submit_replayed_unknown': True,
        'large_response_bytes': len(large['payload']),
        'missing_broker': 'broker_unavailable',
        'api_cli_parity': ['observe', 'inspect', 'submit', 'freeze', 'finalize', 'export'],
        'agent_cwd_had_no_database': True,
        'private_case_absent': True,
        'scope': 'One scripted follower through the wl pipe using a test runner. Not an autonomy rate or model-backend validation.',
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
