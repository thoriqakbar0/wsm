"""Local client for the WSM endpoints on TinkerServer.

observe() explicitly refreshes the view used for inspection. Edits always
carry the caller's supplied view; no operation retries or refreshes it.
"""
import json
import math
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .binary64 import WeightlessError

_LOOPBACK = {'127.0.0.1', 'localhost', '::1'}
_VIEW_FIELDS = ('session', 'arm', 'declaration', 'observation', 'base')


def _origin(base_url):
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != 'http' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise WeightlessError('invalid_request', 'WeightEditor accepts a loopback HTTP URL')
    if parsed.path not in ('', '/'):
        raise WeightlessError('invalid_request', 'WeightEditor accepts a loopback HTTP URL')
    host = parsed.hostname
    if host not in _LOOPBACK or parsed.port is None:
        raise WeightlessError('invalid_request', 'WeightEditor accepts a loopback HTTP URL')
    return 'http://' + ('[' + host + ']' if ':' in host else host) + ':' + str(parsed.port)


class WeightEditor:
    """Edit one arm through the local server with explicit checkpoint bindings."""

    def __init__(self, base_url, api_key, *, timeout=300):
        if type(api_key) is not str or not api_key:
            raise WeightlessError('invalid_request', 'API key is required')
        self._origin = _origin(base_url)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise WeightlessError('invalid_request', 'A positive finite timeout is required')
        self._timeout = timeout
        self._api_key = api_key
        self._view = None

    def observe(self):
        view = self._post('/wsm/v1/observe', {})
        self._view = {name: view[name] for name in _VIEW_FIELDS}
        return view

    def inspect(self, tensor, indices, key=None):
        view = self._bound()
        return self._post('/wsm/v1/inspect', {
            'checkpoint': view['base'],
            'tensor': tensor,
            'indices': indices,
            'key': _key(key),
        })

    def submit(self, edits, *, view, key=None):
        return self._post('/wsm/v1/submit', {'view': _identity(view), 'edits': edits, 'key': _key(key)})

    def run_code(self, source, inspections, *, view, key=None):
        return self._post('/wsm/v1/run_code', {
            'source': source,
            'inspections': inspections,
            'view': _identity(view),
            'key': _key(key),
        })

    def _bound(self):
        if self._view is None:
            raise WeightlessError('invalid_state', 'Call observe() before using the bound view')
        return self._view

    def _post(self, path, body):
        payload = json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(
            self._origin + path,
            data=payload,
            method='POST',
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-API-Key': self._api_key,
                'Connection': 'close',
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raw = error.read()
            _raise_http(error.code, raw)
        try:
            return json.loads(raw.decode())
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            raise WeightlessError('invalid_result', 'Expected a JSON response') from None


def _identity(view):
    if type(view) is not dict or any(name not in view for name in _VIEW_FIELDS):
        raise WeightlessError('invalid_request', 'An observed view is required')
    return {name: view[name] for name in _VIEW_FIELDS}


def _key(value):
    if value is None:
        return uuid.uuid4().hex
    if type(value) is not str or not 0 < len(value) <= 200:
        raise WeightlessError('invalid_request', 'Invalid operation key')
    return value


def _raise_http(status, raw):
    code = 'invalid_request'
    message = 'Request failed'
    try:
        body = json.loads(raw.decode())
        if type(body) is dict:
            if type(body.get('code')) is str:
                code = body['code']
            if type(body.get('error')) is str and len(body['error']) <= 500:
                message = body['error']
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        pass
    raise WeightlessError(code, message)
