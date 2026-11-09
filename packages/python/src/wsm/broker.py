"""One command in, one response out. Session.dispatch is the only policy call."""
import json
import os
import threading

from .binary64 import WeightlessError
from .session import _json

FORMAT_COMMAND = 'weightless.command.v1'
FORMAT_RESPONSE = 'weightless.response.v1'
RESERVED_KEY_PREFIX = 'wl:'
COMMAND_FD = 3
RESPONSE_FD = 4
COMMAND_BYTE_CAP = 70000
RESPONSE_BYTE_CAP = 16 * 1024 * 1024


def _loads(raw):
    def pairs(seq):
        seen, out = set(), {}
        for key, value in seq:
            if key in seen:
                raise WeightlessError('invalid_request', 'Duplicate JSON key')
            seen.add(key)
            out[key] = value
        return out

    def reject_constant(_name):
        raise WeightlessError('invalid_request', 'Expected finite JSON data')

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
    except WeightlessError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError, TypeError):
        raise WeightlessError('invalid_request', 'Expected one JSON object') from None


def parse_command(document):
    if type(document) is not bytes:
        raise WeightlessError('invalid_request', 'Expected command bytes')
    if document.startswith(b'\xef\xbb\xbf'):
        raise WeightlessError('invalid_request', 'Command must not start with a BOM')
    if len(document) > COMMAND_BYTE_CAP:
        raise WeightlessError('invalid_request', 'Command exceeds byte bound')
    try:
        text = document.decode('utf-8')
    except UnicodeDecodeError:
        raise WeightlessError('invalid_request', 'Expected UTF-8') from None
    value = _loads(text)
    if type(value) is not dict or set(value) != {'format', 'action', 'key', 'payload'}:
        raise WeightlessError('invalid_request', 'Unexpected fields')
    if value['format'] != FORMAT_COMMAND:
        raise WeightlessError('invalid_request', 'Unknown command format')
    if type(value['action']) is not str or type(value['key']) is not str:
        raise WeightlessError('invalid_request', 'Action and key must be strings')
    return value


def _echo(command):
    return {'action': command.get('action', ''), 'key': command.get('key', '')}


def _ready(action, payload):
    role = ACTIONS_ROLES[action]
    return {'role': role, 'fill': [], 'command': {
        'format': FORMAT_COMMAND, 'action': action, 'key': RESERVED_KEY_PREFIX + action, 'payload': payload}}


def build_observe(_view=None):
    return (_ready('observe', {}),)


def build_submit(view):
    return ({'role': 'act', 'fill': ['key', 'payload.edits'], 'command': {
        'format': FORMAT_COMMAND, 'action': 'submit', 'key': '',
        'payload': {'session': view['session'], 'arm': view['arm'], 'declaration': view['declaration'],
                    'observation': view['observation'], 'base': view['base'], 'edits': []}}},)


def build_inspect(view):
    templates = []
    for name in view['parameters']:
        templates.append({'role': 'act', 'fill': ['key', 'payload.indices'], 'command': {
            'format': FORMAT_COMMAND, 'action': 'inspect', 'key': '',
            'payload': {'checkpoint': view['base'], 'tensor': name, 'indices': []}}})
    return tuple(templates)


def build_diagnose(_view=None):
    return ({'role': 'act', 'fill': ['key'], 'command': {
        'format': FORMAT_COMMAND, 'action': 'diagnose', 'key': '', 'payload': {}}},)


def build_status(view):
    if not view:
        return ()
    keys = list(view.get('blocking') or [])
    pending = view.get('pending')
    if type(pending) is str and pending not in keys:
        keys.append(pending)
    return tuple({'role': 'read', 'fill': [], 'command': {
        'format': FORMAT_COMMAND, 'action': 'status', 'key': RESERVED_KEY_PREFIX + 'status',
        'payload': {'key': key}}} for key in keys)


def build_freeze(_view=None):
    return (_ready('freeze', {}),)


def build_finalize(_view=None):
    return (_ready('finalize', {}),)


def build_export(_view=None):
    return (_ready('export', {}),)


def build_nothing(_view=None):
    return ()


ACTIONS = (
    ('inspect', 'act', build_inspect),
    ('submit', 'act', build_submit),
    ('diagnose', 'act', build_diagnose),
    ('status', 'read', build_status),
    ('freeze', 'stop', build_freeze),
    ('cancel', 'stop', build_nothing),
    ('finalize', 'finish', build_finalize),
    ('export', 'finish', build_export),
    ('observe', 'read', build_observe),
)
ACTIONS_ROLES = {name: role for name, role, _builder in ACTIONS}


def templates_from_view(view, allowed_actions):
    allowed, names, produced = set(allowed_actions), set(view.get('actions') or []), []
    for name, _role, builder in ACTIONS:
        if name == 'observe' or name not in names or name not in allowed:
            continue
        produced.extend(builder(view))
    if not any(item['role'] == 'act' for item in produced):
        produced = [item for item in produced if item['role'] == 'stop'] + [item for item in produced if item['role'] != 'stop']
    return produced


def _error(command, code, message, nxt):
    return {'format': FORMAT_RESPONSE, 'command': _echo(command), 'outcome': 'error',
            'result': None, 'error': {'code': code, 'message': message}, 'next': list(nxt)}


def _ok(command, result, nxt, outcome='ok'):
    return {'format': FORMAT_RESPONSE, 'command': _echo(command), 'outcome': outcome,
            'result': result, 'error': None, 'next': list(nxt)}


def _encode(value):
    body = _json(value).encode() + b'\n'
    if len(body) > RESPONSE_BYTE_CAP:
        body = _json(_error({'action': '', 'key': ''}, 'invalid_request', 'Response exceeds byte bound', [])).encode() + b'\n'
    return body


class Broker:
    def __init__(self, session, *, arm, allowed_actions):
        self.session = session
        self.arm = arm
        self.allowed_actions = tuple(allowed_actions)

    def handle(self, envelope):
        refresh = build_observe() if 'observe' in self.allowed_actions else ()
        try:
            raw = envelope if type(envelope) is bytes else _json(envelope).encode()
            command = parse_command(raw)
        except WeightlessError as error:
            return _error({'action': '', 'key': ''}, error.code, str(error), [])
        if command['action'] in ('submit', 'inspect', 'diagnose') and command['key'].startswith(RESERVED_KEY_PREFIX):
            return _error(command, 'invalid_request', 'Reserved key prefix', refresh)
        try:
            result = self.session.dispatch(
                {'action': command['action'], 'key': command['key'], 'payload': command['payload']},
                arm=self.arm, allowed_actions=self.allowed_actions)
        except WeightlessError as error:
            return _error(command, error.code, str(error), refresh)
        if type(result) is dict and result.get('status') == 'outcome_unknown' and 'kind' in result:
            nxt = list(build_freeze()) if 'freeze' in self.allowed_actions else []
            return _ok(command, result, nxt, outcome='unknown')
        if command['action'] == 'observe':
            return _ok(command, result, templates_from_view(result, self.allowed_actions))
        nxt = list(build_observe()) if 'observe' in self.allowed_actions else []
        if 'status' in self.allowed_actions and type(result) is dict and 'kind' in result and 'key' in result:
            nxt.append({'role': 'read', 'fill': [], 'command': {
                'format': FORMAT_COMMAND, 'action': 'status', 'key': RESERVED_KEY_PREFIX + 'status',
                'payload': {'key': result['key']}}})
        return _ok(command, result, nxt)


def serve_once(broker, stdin, stdout):
    raw = stdin.read(COMMAND_BYTE_CAP + 1)
    if isinstance(raw, str):
        raw = raw.encode()
    if len(raw) > COMMAND_BYTE_CAP:
        stdout.write(_encode(_error({'action': '', 'key': ''}, 'invalid_request', 'Command exceeds byte bound', [])))
        return
    stdout.write(_encode(broker.handle(raw)))


def _write_all(fd, data):
    try:
        view = memoryview(data)
        while view:
            wrote = os.write(fd, view)
            view = view[wrote:]
    except OSError:
        pass
    finally:
        os.close(fd)


def _read_capped(fd, limit):
    chunks, size = [], 0
    while size < limit:
        try:
            block = os.read(fd, min(65536, limit - size))
        except InterruptedError:
            continue
        if not block:
            break
        chunks.append(block)
        size += len(block)
    os.close(fd)
    return b''.join(chunks)


def accept_once(broker, source, wl='wl'):
    """Run `wl` once. Command bytes travel on fd 3 and the response on fd 4."""
    if type(source) is str:
        source = source.encode()
    stdin_r, stdin_w = os.pipe()
    stdout_r, stdout_w = os.pipe()
    cmd_r, cmd_w = os.pipe()
    resp_r, resp_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.dup2(stdin_r, 0)
            os.dup2(stdout_w, 1)
            os.dup2(cmd_w, COMMAND_FD)
            os.dup2(resp_r, RESPONSE_FD)
            for fd in (stdin_r, stdin_w, stdout_r, stdout_w, cmd_r, cmd_w, resp_r, resp_w):
                if fd > RESPONSE_FD:
                    os.close(fd)
            os.execv(wl, [wl])
        finally:
            os._exit(127)
    for fd in (stdin_r, stdout_w, cmd_w, resp_r):
        os.close(fd)
    writer = threading.Thread(target=_write_all, args=(stdin_w, source))
    writer.start()
    command = _read_capped(cmd_r, COMMAND_BYTE_CAP + 1)
    writer.join()
    exited, _status = os.waitpid(pid, os.WNOHANG)
    if command == b'' and exited:
        os.close(resp_w)
        return _read_capped(stdout_r, RESPONSE_BYTE_CAP + 1)
    try:
        if command == b'' or len(command) > COMMAND_BYTE_CAP:
            response = _error({'action': '', 'key': ''}, 'invalid_request', 'Command exceeds byte bound', [])
        else:
            response = broker.handle(command)
    except BaseException:
        os.close(resp_w)
        if not exited:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
        os.close(stdout_r)
        raise
    _write_all(resp_w, _encode(response))
    output = _read_capped(stdout_r, RESPONSE_BYTE_CAP + 1)
    if not exited:
        os.waitpid(pid, 0)
    return output
