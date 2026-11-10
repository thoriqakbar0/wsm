"""Loopback HTTP adapter that lets the Tinker SDK sample one WSM session arm.

The server keeps no model registry. Each sampling session is bound to the
arm checkpoint returned by Session.observe at creation. After an accepted
edit that checkpoint changes, and a sample on the old session is rejected.
Create a new sampling client to sample the new checkpoint.

tinker 0.39.0 accepts SampleResponse only as protobuf. This server emits
that encoding. Futures and sampling sessions live in memory. Restarting
the process invalidates those handles. The session journal is unchanged,
and old request ids are not replayed after a restart.

The server thread owns a separate Session connection. The caller's
connection stays on its original thread.
"""
import argparse
import hashlib
import hmac
import json
import os
import socket
import struct
import sys
import threading
import urllib.parse
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

from .binary64 import WeightlessError

MAX_BODY = 1024 * 1024
MAX_PROMPT_TOKENS = 2048
MAX_STOP_TOKENS = 32
MAX_FUTURES = 256
MAX_SAMPLING_SESSIONS = 64
MAX_SDK_SESSIONS = 32
MAX_OUTPUT_TOKENS = 128
MODEL_NAME = 'wsm'
MODEL_PATH = 'tinker://wsm'
_LOOPBACK = ('127.0.0.1', '::1')
_UNSUPPORTED = (
    '/api/v1/optim_step',
    '/api/v1/forward_backward',
    '/api/v1/forward',
    '/api/v1/create_model',
    '/api/v1/load_weights',
    '/api/v1/save_weights',
    '/api/v1/save_weights_for_sampler',
    '/api/v1/save_weights_external',
    '/api/v1/create_custom_loss',
    '/api/v1/copy_weights',
    '/api/v1/get_info',
    '/api/v1/cancel_future',
    '/api/v1/join_sampling_session',
    '/api/v1/retrieve_futures',
    '/api/v1/auth/token',
)
_SAMPLE_FIELDS = {
    'num_samples', 'prompt', 'sampling_params', 'base_model', 'model_path',
    'sampling_session_id', 'seq_id', 'prompt_logprobs', 'topk_prompt_logprobs',
    'topk_sample_logprobs', 'target_prompt_logprobs', 'prompt_alt_tokens_k',
    'prompt_logprobs_last_n', 'record_stability_info', 'type',
}
_PARAM_FIELDS = {'max_tokens', 'seed', 'stop', 'temperature', 'top_k', 'top_p'}
_VIEW_FIELDS = ('session', 'arm', 'declaration', 'observation', 'base')


class _HttpError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code


def _eq(left, right):
    if type(left) is not str or type(right) is not str:
        return False
    if len(left) != len(right):
        hmac.compare_digest(right, right)
        return False
    return hmac.compare_digest(left, right)


def _loads(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if type(key) is not str or key in result:
                raise _HttpError(400, 'invalid_request', 'Duplicate JSON field')
            result[key] = value
        return result

    def nonfinite(_value):
        raise _HttpError(400, 'invalid_request', 'Nonfinite JSON number')

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    except _HttpError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise _HttpError(400, 'invalid_request', 'Expected JSON') from None


def _object(value):
    if type(value) is not dict:
        raise _HttpError(400, 'invalid_request', 'Expected a JSON object')
    return value


def _key(value):
    if type(value) is not str or not 0 < len(value) <= 200:
        raise _HttpError(400, 'invalid_request', 'Invalid operation key')
    return value


def _status_for(error):
    return {
        'stale_base': 409,
        'idempotency_conflict': 409,
        'pending_operation': 409,
        'invalid_state': 409,
        'unauthorized': 403,
        'unknown_operation': 404,
        'budget_exhausted': 429,
        'deadline_exhausted': 429,
    }.get(error.code, 400)


def _http(error):
    if isinstance(error, _HttpError):
        return error
    if isinstance(error, WeightlessError):
        return _HttpError(_status_for(error), error.code, str(error))
    return _HttpError(500, 'internal', 'Internal error')


def _varint(value):
    out = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        out.append(bits | 0x80 if value else bits)
        if not value:
            return bytes(out)


def _field_bytes(field, payload):
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _field_varint(field, value):
    return _varint((field << 3) | 0) + _varint(value)


def _i32(values):
    return struct.pack('<' + 'i' * len(values), *values)


def _f32(values):
    values = [float('nan') if item is None else float(item) for item in values]
    if not values:
        return b''
    return struct.pack('<' + 'f' * len(values), *values)


def _encode_sequence(sequence):
    reason = sequence['stop_reason']
    body = _field_varint(1, 1) if reason == 'length' else b''
    body += _field_bytes(2, _i32(sequence['tokens']))
    logprobs = sequence.get('logprobs')
    if logprobs:
        body += _field_bytes(3, _f32(logprobs))
    return body


def _encode_sample(result):
    """Serialize the Session.infer result as tinker 0.39.0 SampleResponse protobuf."""
    body = b''.join(_field_bytes(1, _encode_sequence(sequence)) for sequence in result['sequences'])
    prompt = result.get('prompt_logprobs')
    if prompt is not None:
        body += _field_bytes(2, _f32(prompt))
    return body


def _int(value, minimum, maximum, message):
    if type(value) is not int or isinstance(value, bool) or not minimum <= value <= maximum:
        raise _HttpError(400, 'invalid_request', message)
    return value


def _tokens(prompt):
    prompt = _object(prompt)
    if set(prompt) != {'chunks'}:
        raise _HttpError(400, 'unsupported', 'Only token prompts are supported')
    chunks = prompt['chunks']
    if type(chunks) is not list or not chunks:
        raise _HttpError(400, 'invalid_request', 'Prompt chunks are required')
    tokens = []
    for chunk in chunks:
        chunk = _object(chunk)
        if chunk.get('type') != 'encoded_text' or set(chunk) != {'type', 'tokens'}:
            raise _HttpError(400, 'unsupported', 'Only encoded text prompts are supported')
        part = chunk['tokens']
        if type(part) is not list:
            raise _HttpError(400, 'invalid_request', 'Tokens must be a list')
        for token in part:
            tokens.append(_int(token, 0, 2 ** 31 - 1, 'Invalid token'))
    if not tokens or len(tokens) > MAX_PROMPT_TOKENS:
        raise _HttpError(400, 'invalid_request', 'Prompt token count is outside the supported range')
    return tokens


def _stop(value):
    if value is None:
        return None
    if type(value) is not list or len(value) > MAX_STOP_TOKENS:
        raise _HttpError(400, 'unsupported', 'Only integer stop tokens are supported')
    return [_int(item, 0, 2 ** 31 - 1, 'Invalid stop token') for item in value]


def _is(value, expected):
    return type(value) in (int, float) and not isinstance(value, bool) and value == expected


def _flag(body, name, default):
    value = body.get(name, default)
    if type(value) is not bool:
        raise _HttpError(400, 'invalid_request', 'Invalid ' + name)
    return value


def _normalize_sample(body):
    body = _object(body)
    if set(body) - _SAMPLE_FIELDS:
        raise _HttpError(400, 'unsupported', 'Unsupported sample field')
    if body.get('type', 'sample') != 'sample':
        raise _HttpError(400, 'unsupported', 'Unsupported sample type')
    if 'base_model' in body or 'model_path' in body:
        raise _HttpError(400, 'unsupported', 'Sample by sampling session, not by model path')
    if 'sampling_session_id' not in body or 'seq_id' not in body or 'prompt' not in body:
        raise _HttpError(400, 'invalid_request', 'Sampling session, seq_id, and prompt are required')
    params = _object(body.get('sampling_params', {}))
    if set(params) - _PARAM_FIELDS:
        raise _HttpError(400, 'unsupported', 'Unsupported sampling parameter')
    if 'seed' in params and params['seed'] is not None:
        raise _HttpError(400, 'unsupported', 'Seeded sampling is not supported')
    if 'top_k' in params and params['top_k'] != -1:
        raise _HttpError(400, 'unsupported', 'top_k is not supported')
    if 'top_p' in params and not _is(params['top_p'], 1):
        raise _HttpError(400, 'unsupported', 'top_p is not supported')
    for name in ('topk_prompt_logprobs', 'topk_sample_logprobs', 'prompt_alt_tokens_k'):
        if body.get(name, 0) != 0:
            raise _HttpError(400, 'unsupported', name + ' is not supported')
    if body.get('target_prompt_logprobs') is not None or body.get('prompt_logprobs_last_n') is not None:
        raise _HttpError(400, 'unsupported', 'Partial prompt scoring is not supported')
    if _flag(body, 'record_stability_info', False):
        raise _HttpError(400, 'unsupported', 'Stability recording is not supported')
    num_samples = body.get('num_samples', 1)
    if num_samples != 1:
        raise _HttpError(400, 'unsupported', 'Only num_samples=1 is supported')
    if 'max_tokens' not in params or 'temperature' not in params:
        raise _HttpError(400, 'invalid_request', 'max_tokens and temperature are required')
    max_tokens = _int(params['max_tokens'], 0, MAX_OUTPUT_TOKENS, 'max_tokens must be from 0 through 128')
    prompt_logprobs = _flag(body, 'prompt_logprobs', False)
    temperature = params['temperature']
    # compute_logprobs always sends max_tokens 1 and the default temperature
    # of 1, then reads only prompt_logprobs. Session.infer accepts temperature 0.
    logprobs_only = (
        prompt_logprobs and max_tokens == 1 and _is(temperature, 1) and params.get('stop') is None
    )
    if not logprobs_only and not _is(temperature, 0):
        raise _HttpError(400, 'unsupported', 'Only greedy sampling and compute_logprobs are supported')
    stop = None if logprobs_only else _stop(params.get('stop'))
    session_id = body['sampling_session_id']
    if type(session_id) is not str or not session_id or len(session_id) > 200 or '/' in session_id:
        raise _HttpError(400, 'invalid_request', 'Invalid sampling session')
    return {
        'sampling_session_id': session_id,
        'seq_id': _int(body['seq_id'], 0, 2 ** 63 - 1, 'Invalid seq_id'),
        'tokens': _tokens(body['prompt']),
        'max_tokens': 0 if logprobs_only else max_tokens,
        'prompt_logprobs': prompt_logprobs,
        'stop': stop,
    }


def _fingerprint(sample):
    payload = {key: sample[key] for key in ('tokens', 'max_tokens', 'prompt_logprobs', 'stop')}
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _view(value):
    value = _object(value)
    if set(value) != set(_VIEW_FIELDS):
        raise _HttpError(400, 'invalid_request', 'View must carry the five identity fields')
    for name in ('session', 'arm', 'declaration', 'base'):
        if type(value[name]) is not str or not value[name] or len(value[name]) > 500:
            raise _HttpError(400, 'invalid_request', 'Invalid view field')
    observation = value['observation']
    if type(observation) is not int or isinstance(observation, bool) or observation < 0:
        raise _HttpError(400, 'invalid_request', 'Invalid observation')
    return {name: value[name] for name in _VIEW_FIELDS}


def _inspections(value):
    if type(value) is not list or not value or len(value) > 64:
        raise _HttpError(400, 'invalid_request', 'Invalid inspections')
    clean = []
    for item in value:
        item = _object(item)
        if set(item) != {'tensor', 'indices'}:
            raise _HttpError(400, 'invalid_request', 'Invalid inspection')
        if type(item['tensor']) is not str or not item['tensor']:
            raise _HttpError(400, 'invalid_request', 'Invalid tensor')
        indices = item['indices']
        if type(indices) is not list or not indices or len(indices) > 10000:
            raise _HttpError(400, 'invalid_request', 'Invalid indices')
        clean.append({
            'tensor': item['tensor'],
            'indices': [_int(index, 0, 2 ** 63 - 1, 'Invalid index') for index in indices],
        })
    return clean


def _model_choice(body):
    base = body.get('base_model')
    path = body.get('model_path')
    if base is None and path is None:
        raise _HttpError(400, 'invalid_request', 'base_model or model_path is required')
    if base is not None and base != MODEL_NAME:
        raise _HttpError(400, 'unsupported', 'Only the wsm model is served')
    if path is not None and (type(path) is not str or path != MODEL_PATH or path.startswith(('/', '.'))):
        raise _HttpError(400, 'unsupported', 'Only tinker://wsm is served')
    if type(base) not in (str, type(None)) or (base is not None and (not base or '/' in base or base.startswith('.'))):
        raise _HttpError(400, 'invalid_request', 'Invalid base model')
    return base, path


@dataclass(frozen=True)
class _Future:
    request_id: str
    fingerprint: str
    sequence_ids: tuple[str, ...]
    status: str
    payload: bytes | None = None
    error: str | None = None
    category: str | None = None


class _Sampling:
    def __init__(self, sampling_session_id, sdk_session_id, seq_id, checkpoint, spec):
        self.sampling_session_id = sampling_session_id
        self.sdk_session_id = sdk_session_id
        self.seq_id = seq_id
        self.checkpoint = checkpoint
        self.spec = spec


class _ServerState:
    def __init__(self, session, arm, api_key):
        if type(arm) is not str or not arm:
            raise WeightlessError('invalid_request', 'Arm is required')
        if type(api_key) is not str or not api_key:
            raise WeightlessError('invalid_request', 'API key is required')
        self.session = session
        self.arm = arm
        self.api_key = api_key
        self.lock = threading.Lock()
        self.sdk_sessions = {}
        self.sdk_by_idem = {}
        self.sampling = {}
        self.sampling_by_seq = {}
        self.futures = {}

    def authorize(self, headers):
        bearer = None
        authorization = headers.get('Authorization')
        if authorization is not None:
            kind, _, token = authorization.partition(' ')
            if kind != 'Bearer' or not token or ' ' in token:
                return False
            bearer = token
        presented = headers.get('X-API-Key')
        if bearer is None and presented is None:
            return False
        if bearer is not None and not _eq(bearer, self.api_key):
            return False
        if presented is not None and not _eq(presented, self.api_key):
            return False
        return True

    def dispatch(self, method, path, body, headers):
        with self.lock:
            if method == 'GET' and path == '/api/v1/healthz':
                return 200, {'status': 'ok'}, 'application/json'
            if method == 'GET' and path == '/api/v1/get_server_capabilities':
                return 200, self._capabilities(), 'application/json'
            if method != 'POST':
                raise _HttpError(405, 'unsupported', 'Method is not supported')
            if path in _UNSUPPORTED:
                raise _HttpError(400, 'unsupported', 'This Tinker method is not supported')
            handler = {
                '/api/v1/client/config': self._client_config,
                '/api/v1/client/dynamic_config': self._dynamic_config,
                '/api/v1/create_session': self._create_session,
                '/api/v1/session_heartbeat': self._heartbeat,
                '/api/v1/create_sampling_session': self._create_sampling,
                '/api/v1/asample': self._asample,
                '/api/v1/retrieve_future': self._retrieve,
                '/api/v1/telemetry': self._telemetry,
                '/wsm/v1/observe': self._observe,
                '/wsm/v1/inspect': self._inspect,
                '/wsm/v1/submit': self._submit,
                '/wsm/v1/run_code': self._run_code,
                '/wsm/v1/freeze': self._freeze,
            }.get(path)
            if handler is None:
                session_id = _finish_session_id(path)
                if session_id is None:
                    raise _HttpError(404, 'unknown_operation', 'Unknown route')
                return 200, self._finish(session_id), 'application/json'
            status, payload, content = handler(body, headers)
            return status, payload, content

    def _capabilities(self):
        return {'supported_models': [{'model_name': MODEL_NAME, 'sampleable': True, 'trainable': False}]}

    def _client_config(self, _body, _headers):
        return 200, {
            'use_pyqwest_transport': False,
            'pjwt_auth_enabled': False,
            'sample_use_retrieve_futures': False,
            'sample_join_sampling_session': False,
            'custom_loss_server_side': False,
            'credential_default_source': 'api_key',
            'sample_no_retries': True,
        }, 'application/json'

    def _dynamic_config(self, _body, _headers):
        return 200, {'refresh_interval_sec': 300, 'sample_cancel_enabled': False}, 'application/json'

    def _create_session(self, _body, headers):
        idem = headers.get('X-Idempotency-Key')
        if idem is not None and (type(idem) is not str or not idem or len(idem) > 200):
            raise _HttpError(400, 'invalid_request', 'Invalid idempotency key')
        if idem and idem in self.sdk_by_idem:
            return 200, {'session_id': self.sdk_by_idem[idem], 'type': 'create_session'}, 'application/json'
        if len(self.sdk_sessions) >= MAX_SDK_SESSIONS:
            raise _HttpError(429, 'budget_exhausted', 'SDK session limit reached')
        session_id = 'wsm-' + uuid.uuid4().hex
        self.sdk_sessions[session_id] = {'finished': False}
        if idem:
            self.sdk_by_idem[idem] = session_id
        return 200, {'session_id': session_id, 'type': 'create_session'}, 'application/json'

    def _require_sdk(self, session_id):
        if type(session_id) is not str or session_id not in self.sdk_sessions:
            raise _HttpError(404, 'unknown_operation', 'Unknown SDK session')
        record = self.sdk_sessions[session_id]
        if record['finished']:
            raise _HttpError(410, 'invalid_state', 'SDK session is finished')
        return record

    def _heartbeat(self, body, _headers):
        body = _object(body)
        session_id = body.get('session_id')
        if session_id not in self.sdk_sessions:
            raise _HttpError(404, 'unknown_operation', 'Unknown SDK session')
        if self.sdk_sessions[session_id]['finished']:
            raise _HttpError(410, 'invalid_state', 'SDK session is finished')
        return 200, {'type': 'session_heartbeat'}, 'application/json'

    def _finish(self, session_id):
        if session_id not in self.sdk_sessions:
            raise _HttpError(404, 'unknown_operation', 'Unknown SDK session')
        self.sdk_sessions[session_id]['finished'] = True
        return None

    def _checkpoint(self):
        view = self.session.observe(self.arm)
        if type(view) is not dict or type(view.get('base')) is not str or not view['base']:
            raise _HttpError(500, 'internal', 'Observation has no checkpoint')
        return view['base']

    def _create_sampling(self, body, _headers):
        body = _object(body)
        allowed = {'session_id', 'sampling_session_seq_id', 'base_model', 'model_path', 'type'}
        if set(body) - allowed or body.get('type', 'create_sampling_session') != 'create_sampling_session':
            raise _HttpError(400, 'unsupported', 'Unsupported sampling session field')
        sdk = self._require_sdk(body.get('session_id'))
        del sdk
        seq_id = _int(body.get('sampling_session_seq_id'), 0, 2 ** 63 - 1, 'Invalid sampling session seq')
        spec = _model_choice(body)
        existing = self.sampling_by_seq.get((body['session_id'], seq_id))
        if existing is not None:
            if existing.spec != spec:
                raise _HttpError(422, 'idempotency_conflict', 'Sampling session seq was already used')
            return 200, {'sampling_session_id': existing.sampling_session_id, 'type': 'create_sampling_session'}, 'application/json'
        if len(self.sampling) >= MAX_SAMPLING_SESSIONS:
            raise _HttpError(429, 'budget_exhausted', 'Sampling session limit reached')
        sampling_session_id = uuid.uuid4().hex
        record = _Sampling(sampling_session_id, body['session_id'], seq_id, self._checkpoint(), spec)
        self.sampling[sampling_session_id] = record
        self.sampling_by_seq[(body['session_id'], seq_id)] = record
        return 200, {'sampling_session_id': sampling_session_id, 'type': 'create_sampling_session'}, 'application/json'

    def _asample(self, body, _headers):
        sample = _normalize_sample(body)
        sampling = self.sampling.get(sample['sampling_session_id'])
        if sampling is None:
            raise _HttpError(404, 'unknown_operation', 'Unknown sampling session')
        self._require_sdk(sampling.sdk_session_id)
        fingerprint = _fingerprint(sample)
        request_id = sampling.sampling_session_id + ':' + str(sample['seq_id'])
        existing = self.futures.get(request_id)
        if existing is not None:
            if existing.fingerprint != fingerprint:
                raise _HttpError(422, 'idempotency_conflict', 'Sampling seq_id was already used')
            return 200, _promise(existing), 'application/json'
        if self._checkpoint() != sampling.checkpoint:
            # The SDK retries HTTP 409 for several minutes. 422 lets the caller
            # create a new sampling client immediately.
            raise _HttpError(422, 'stale_base', 'Sampling session checkpoint changed')
        if len(self.futures) >= MAX_FUTURES:
            raise _HttpError(429, 'budget_exhausted', 'Future limit reached')
        op_key = 'tinker-sample:' + request_id
        if len(op_key) > 200:
            raise _HttpError(400, 'invalid_request', 'Sampling identity exceeds the operation key limit')
        record = self._run_infer(sample, sampling.checkpoint, request_id, fingerprint, op_key)
        self.futures[request_id] = record
        return 200, _promise(record), 'application/json'

    def _run_infer(self, sample, checkpoint, request_id, fingerprint, op_key):
        query = {
            'checkpoint': checkpoint,
            'tokens': sample['tokens'],
            'max_tokens': sample['max_tokens'],
            'temperature': 0,
            'prompt_logprobs': sample['prompt_logprobs'],
            'stop': sample['stop'],
        }
        try:
            operation = self.session.infer(query, key=op_key, arm=self.arm)
        except WeightlessError as error:
            if error.code in ('stale_base', 'idempotency_conflict', 'budget_exhausted',
                              'deadline_exhausted', 'pending_operation', 'invalid_state'):
                raise _HttpError(422, error.code, str(error)) from None
            raise
        return _future_from_operation(operation, request_id, fingerprint)

    def _retrieve(self, body, headers):
        body = _object(body)
        if set(body) - {'request_id', 'allow_metadata_only'}:
            raise _HttpError(400, 'unsupported', 'Unsupported retrieve field')
        request_id = body.get('request_id')
        record = self.futures.get(request_id) if type(request_id) is str else None
        if record is None:
            raise _HttpError(404, 'unknown_operation', 'Unknown future')
        accept = headers.get('Accept', '')
        if record.status == 'ready':
            if 'application/x-protobuf' not in accept:
                raise _HttpError(406, 'unsupported', 'Sample results are protobuf')
            return 200, record.payload, 'application/x-protobuf'
        return 200, {'error': record.error, 'category': record.category}, 'application/json'

    def _telemetry(self, body, _headers):
        _object(body)
        return 200, {'status': 'accepted'}, 'application/json'

    def _observe(self, body, _headers):
        if _object(body) != {}:
            raise _HttpError(400, 'invalid_request', 'observe takes an empty object')
        return 200, self.session.observe(self.arm), 'application/json'

    def _inspect(self, body, _headers):
        body = _object(body)
        if set(body) != {'checkpoint', 'tensor', 'indices', 'key'}:
            raise _HttpError(400, 'invalid_request', 'Unexpected inspect fields')
        checkpoint = body['checkpoint']
        tensor = body['tensor']
        if type(checkpoint) is not str or not checkpoint or type(tensor) is not str or not tensor:
            raise _HttpError(400, 'invalid_request', 'Invalid inspect target')
        indices = body['indices']
        if type(indices) is not list or not indices:
            raise _HttpError(400, 'invalid_request', 'Invalid indices')
        query = {
            'checkpoint': checkpoint,
            'tensor': tensor,
            'indices': [_int(index, 0, 2 ** 63 - 1, 'Invalid index') for index in indices],
        }
        return 200, self.session.inspect(query, key=_key(body['key']), arm=self.arm), 'application/json'

    def _submit(self, body, _headers):
        body = _object(body)
        if set(body) != {'view', 'edits', 'key'}:
            raise _HttpError(400, 'invalid_request', 'Unexpected submit fields')
        view = _view(body['view'])
        if view['arm'] != self.arm:
            raise _HttpError(403, 'unauthorized', 'Arm mismatch')
        if type(body['edits']) is not list:
            raise _HttpError(400, 'invalid_request', 'Edits must be a list')
        proposal = {name: view[name] for name in _VIEW_FIELDS}
        proposal['edits'] = body['edits']
        return 200, self.session.submit(proposal, key=_key(body['key'])), 'application/json'

    def _run_code(self, body, _headers):
        body = _object(body)
        if set(body) != {'source', 'inspections', 'view', 'key'}:
            raise _HttpError(400, 'invalid_request', 'Unexpected run_code fields')
        view = _view(body['view'])
        if view['arm'] != self.arm:
            raise _HttpError(403, 'unauthorized', 'Arm mismatch')
        if type(body['source']) is not str or not body['source'] or len(body['source'].encode()) > 65536:
            raise _HttpError(400, 'invalid_request', 'Invalid source')
        current = self.session.observe(self.arm)
        if any(view[field] != current[field] for field in ('session', 'declaration')):
            raise _HttpError(400, 'invalid_proposal', 'Code belongs to another experiment declaration')
        code_key = _key(body['key'])
        if len(code_key) > 160:
            raise _HttpError(400, 'invalid_request', 'Invalid operation key')
        result = self.session.run_code(
            body['source'],
            _inspections(body['inspections']),
            key=code_key,
            arm=self.arm,
            base=view['base'],
            observation=view['observation'],
        )
        return 200, result, 'application/json'

    def _freeze(self, body, _headers):
        body = _object(body)
        if set(body) != {'key'}:
            raise _HttpError(400, 'invalid_request', 'Unexpected freeze fields')
        return 200, self.session.freeze(key=_key(body['key'])), 'application/json'


def _promise(record):
    return {'request_id': record.request_id, 'sample_sequence_ids': list(record.sequence_ids)}


def _sequence_ids(result, request_id):
    if not result['sequences']:
        return ()
    return (request_id + ':0',)


def _future_from_operation(operation, request_id, fingerprint):
    placeholder = (request_id + ':0',)
    status = operation.get('status')
    if status == 'outcome_unknown':
        return _Future(request_id, fingerprint, placeholder, 'unknown', error='Outcome unknown', category='server')
    result = operation.get('result')
    if status == 'completed':
        sequence_ids = _sequence_ids(result, request_id)
        try:
            payload = _encode_sample(result)
        except (struct.error, ValueError, OverflowError):
            return _Future(request_id, fingerprint, sequence_ids or placeholder, 'failed', error='Infer result is not encodable', category='server')
        return _Future(request_id, fingerprint, sequence_ids, 'ready', payload=payload)
    code = operation.get('error_code') or status or 'failed'
    return _Future(request_id, fingerprint, placeholder, 'failed', error='Infer ' + str(code), category='user' if status == 'task_failure' else 'server')


def _finish_session_id(path):
    prefix = '/api/v1/sessions/'
    suffix = '/finish'
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    session_id = urllib.parse.unquote(path[len(prefix):-len(suffix)])
    if not session_id or '/' in session_id:
        return None
    return session_id


class _HTTPServer(HTTPServer):
    request_queue_size = 16

    def __init__(self, address, handler):
        if ':' in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, handler)

    def handle_error(self, request, client_address):
        return


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    timeout = 30

    def handle(self):
        self.close_connection = True
        self.handle_one_request()

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        self._handle('GET')

    def do_POST(self):
        self._handle('POST')

    def _handle(self, method):
        self.close_connection = True
        state = self.server.state
        try:
            if not state.authorize(self.headers):
                self._reply(401, {'error': 'Unauthorized', 'code': 'unauthorized'}, 'application/json')
                return
            body = b''
            headers = self.headers
            if method == 'POST':
                body = _read_body(self)
                try:
                    parsed = _loads(body.decode())
                except UnicodeError:
                    raise _HttpError(400, 'invalid_request', 'Expected JSON') from None
            else:
                parsed = None
            path = urllib.parse.urlsplit(self.path).path
            status, payload, content = state.dispatch(method, path, parsed, headers)
            if payload is None:
                self._reply(status, b'', 'application/json')
            elif content == 'application/x-protobuf':
                self._reply(status, payload, content)
            else:
                self._reply(status, json.dumps(payload, allow_nan=False).encode(), content)
        except _HttpError as error:
            self._reply(error.status, {'error': str(error), 'code': error.code}, 'application/json')
        except WeightlessError as error:
            mapped = _http(error)
            self._reply(mapped.status, {'error': mapped.args[0] if mapped.args else 'Request failed', 'code': mapped.code}, 'application/json')
        except Exception:
            self._reply(500, {'error': 'Internal error', 'code': 'internal'}, 'application/json')

    def _reply(self, status, body, content_type):
        if type(body) is not bytes:
            try:
                body = json.dumps(body, allow_nan=False).encode()
            except (TypeError, ValueError):
                status = 500
                body = b'{"error":"Internal error","code":"internal"}'
                content_type = 'application/json'
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)


def _read_body(handler):
    if handler.headers.get('Transfer-Encoding'):
        raise _HttpError(400, 'invalid_request', 'Chunked bodies are not accepted')
    length = handler.headers.get('Content-Length')
    if length is None:
        raise _HttpError(400, 'invalid_request', 'Content-Length is required')
    try:
        size = int(length)
    except ValueError:
        raise _HttpError(400, 'invalid_request', 'Invalid Content-Length') from None
    if size < 0 or size > MAX_BODY:
        raise _HttpError(413, 'invalid_request', 'Body exceeds limit')
    data = handler.rfile.read(size)
    if len(data) != size:
        raise _HttpError(400, 'invalid_request', 'Truncated body')
    return data


class TinkerServer:
    """Serve one Session arm on loopback for the Tinker SDK and WeightEditor."""

    def __init__(self, session, arm, api_key, host='127.0.0.1', port=0):
        if host not in _LOOPBACK:
            raise WeightlessError('invalid_request', 'TinkerServer binds loopback only')
        if type(port) is not int or isinstance(port, bool) or not 0 <= port <= 65535:
            raise WeightlessError('invalid_request', 'Invalid port')
        self.session = session
        self.arm = arm
        self.host = host
        self.port = port
        if type(arm) is not str or not arm or type(api_key) is not str or not api_key:
            raise WeightlessError('invalid_request', 'Arm and API key are required')
        self._api_key = api_key
        self.state = None
        self._httpd = None
        self._thread = None
        self._closed = False
        self._started = threading.Event()
        self._startup_error = None
        self.base_url = None

    def __enter__(self):
        self._httpd = _HTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(target=self._serve, name='wsm-tinker')
        self._thread.start()
        if not self._started.wait(5) or self._startup_error is not None:
            self.close()
            raise WeightlessError('target_failure', 'Tinker server did not start')
        host, port = self._httpd.server_address[:2]
        self.base_url = 'http://' + ('[' + host + ']' if ':' in host else host) + ':' + str(port)
        return self

    def _serve(self):
        owned = None
        try:
            owned = self.session.reopen()
            self.state = _ServerState(owned, self.arm, self._api_key)
            self._httpd.state = self.state
            self._started.set()
            self._httpd.serve_forever(poll_interval=0.05)
        except Exception as error:
            self._startup_error = error
            self._started.set()
        finally:
            if owned is not None:
                owned.close()

    def serve_until_stopped(self):
        try:
            self._thread.join()
        except KeyboardInterrupt:
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        httpd = self._httpd
        if httpd is not None and self._startup_error is None and self._started.is_set():
            httpd.shutdown()
        if httpd is not None:
            httpd.server_close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return 'TinkerServer(base_url=%r, arm=%r)' % (self.base_url, self.arm)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='wsm.tinker_backend')
    parser.add_argument('workspace')
    parser.add_argument('--arm', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args(argv)
    api_key = os.environ.get('WSM_TINKER_API_KEY')
    if not api_key:
        print('WSM_TINKER_API_KEY is required', file=sys.stderr)
        return 2
    from .client import Experiment
    experiment = Experiment.open(args.workspace)
    try:
        with TinkerServer(experiment.session, arm=args.arm, api_key=api_key, host=args.host, port=args.port) as server:
            print(server.base_url, flush=True)
            server.serve_until_stopped()
    finally:
        experiment.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
