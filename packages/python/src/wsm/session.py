"""Durable exploratory experiment sessions. The host kernel is a trusted component."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import contextmanager

from .binary64 import WeightlessError
from .sandbox import forward_reservation


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        raise WeightlessError('invalid_request', 'Expected finite JSON data') from None


def _hash(value):
    return 'sha256:' + hashlib.sha256(_json(value).encode()).hexdigest()


def _seal(value):
    return {**value, 'id': _hash(value)}


def _check(condition, code, message):
    if not condition:
        raise WeightlessError(code, message)


def _fields(value, fields):
    _check(type(value) is dict and set(value) == set(fields.split()), 'invalid_request', 'Unexpected fields')


def _connect(path):
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    return db


class Session:
    """One SQLite-owned session; expose dispatch, never this object, to untrusted agents."""

    @classmethod
    def create(cls, path, *, runner, target, declaration, development, verifier=None, development_retention=None, final_dataset=None):
        declaration = json.loads(_json(declaration))
        required = {'hypothesis', 'arms', 'editable', 'budgets', 'max_edits', 'max_inspect', 'wall_seconds'}
        _check(required <= declaration.keys(), 'invalid_declaration', 'Missing required declaration fields')
        _check(type(declaration['wall_seconds']) in (int, float) and math.isfinite(declaration['wall_seconds'])
               and declaration['wall_seconds'] > 0, 'invalid_declaration', 'Positive finite wall_seconds required')
        arms = declaration['arms']
        _check(type(arms) is list and arms and all(type(a) is str and a for a in arms)
               and len(set(arms)) == len(arms), 'invalid_declaration', 'Invalid arms')
        limits = declaration['budgets']
        _check(type(limits) is dict and {'attempts', 'evaluations', 'inspections', 'teacher_requests', 'final_evaluations'} <= set(limits)
               and set(limits) <= {'attempts', 'evaluations', 'inspections', 'teacher_requests', 'diagnostics', 'final_evaluations', 'forward_calls'}
               and all(type(n) is int and n >= 0 for n in limits.values()), 'invalid_declaration', 'Explicit budgets required')
        if 'forward_calls' in limits and verifier is not None:
            _check(type(declaration.get('final_forward_calls')) is int and declaration['final_forward_calls'] >= 0,
                   'invalid_declaration', 'Predeclared final_forward_calls envelope required')
        if 'final_forward_calls' in declaration:
            _check(type(declaration['final_forward_calls']) is int and declaration['final_forward_calls'] >= 0,
                   'invalid_declaration', 'Invalid final forward envelope')
        _check(all(type(declaration[k]) is int and 0 < declaration[k] <= 10000
                   for k in ('max_edits', 'max_inspect')), 'invalid_declaration', 'Invalid bounds')
        _check(declaration.get('selection', 'loss') == 'loss', 'unsupported', 'Only strict loss minimization is supported')
        _check(declaration.get('claim', 'exploratory') in ('exploratory', 'confirmatory'), 'unsupported', 'Unknown claim profile')
        criteria = declaration.get('criteria', {})
        _check(type(criteria) is dict and set(criteria) <= {'final_exact_min', 'retention_loss_max'}, 'invalid_declaration', 'Invalid criteria')
        for name, value in criteria.items():
            _check(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0
                                    and (name != 'final_exact_min' or value <= 1)), 'invalid_declaration', 'Invalid threshold')
        if declaration.get('claim') == 'confirmatory':
            _check(all(criteria.get(k) is not None for k in ('final_exact_min', 'retention_loss_max')),
                   'invalid_declaration', 'Confirmatory criteria must be predeclared')
        _check(type(development) is list and development, 'invalid_declaration', 'Public development cases required')
        retention_max = declaration.get('development_retention_loss_max')
        _check(retention_max is None or (type(retention_max) in (int, float) and math.isfinite(retention_max) and retention_max >= 0
                                        and development_retention), 'invalid_declaration', 'Development retention gate needs public cases and a finite threshold')
        obj = cls(path, runner=runner, target=target, verifier=verifier, final_dataset=final_dataset)
        with obj._transaction():
            _check(obj.db.execute('SELECT 1 FROM state').fetchone() is None, 'already_exists', 'Session exists')
            manifest = runner.execute('describe', target)
            _check(set(declaration['editable']) <= set(manifest['parameters']), 'invalid_declaration', 'Unknown editable tensor')
            state = {'format': 'weightless.session.v1', 'session': str(uuid.uuid4()),
                     'origin': time.time(), 'declaration': _seal(declaration), 'manifest': manifest, 'runner': copy.deepcopy(runner.config),
                     'target_identity': _hash(target), 'development': copy.deepcopy(development),
                     'development_retention': copy.deepcopy(development_retention or []),
                     'phase': 'developing', 'used': dict.fromkeys(limits, 0),
                     'arms': {a: {'checkpoint': manifest['checkpoint'], 'patches': [], 'loss': None,
                                  'version': 0, 'pending': None, 'history': []} for a in arms},
                     'frozen': None, 'final': None}
            obj.db.execute('INSERT INTO state VALUES (?)', (_json(state),))
        return obj

    @classmethod
    def open(cls, path, *, runner, target, verifier=None, final_dataset=None):
        obj = cls(path, runner=runner, target=target, verifier=verifier, final_dataset=final_dataset)
        state = obj._state()
        _check(state['target_identity'] == _hash(target) and state['runner'] == runner.config,
               'identity_mismatch', 'Target or runner identity changed')
        return obj

    def __init__(self, path, *, runner, target, verifier=None, final_dataset=None):
        self._dispatch_path = str(Path(path).resolve()) + '.dispatch'
        self.db = _connect(path)
        self.final_dataset = final_dataset
        self.runner, self.target, self.verifier = runner, copy.deepcopy(target), verifier
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS state (body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS operations (key TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS journal (seq INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL);
        ''')

    def close(self):
        self.db.close()

    @contextmanager
    def _transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def _state(self):
        row = self.db.execute('SELECT body FROM state').fetchone()
        _check(row is not None, 'missing_session', 'Session does not exist')
        return json.loads(row[0])

    def _save(self, state):
        self.db.execute('UPDATE state SET body=?', (_json(state),))

    def _event(self, event):
        return self.db.execute('INSERT INTO journal(body) VALUES (?)', (_json(event),)).lastrowid

    def _op(self, key):
        row = self.db.execute('SELECT body FROM operations WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, op):
        self.db.execute('INSERT OR REPLACE INTO operations VALUES (?,?)', (op['key'], _json(op)))

    def _existing(self, key, request):
        _check(type(key) is str and 0 < len(key) <= 200, 'invalid_request', 'Invalid operation key')
        op = self._op(key)
        if op:
            _check(op['request_hash'] == _hash(request), 'idempotency_conflict', 'Key has different content')
        return op

    @staticmethod
    def _remaining(state):
        now = time.time()
        _check(now >= state['origin'], 'clock_regressed', 'Clock is before persisted origin')
        remaining = state['origin'] + state['declaration']['wall_seconds'] - now
        _check(remaining > 0, 'deadline_exhausted', 'Session wall deadline exhausted')
        return remaining

    def remaining_seconds(self):
        return self._remaining(self._state())

    def _reserve(self, state, key, request, costs, arm):
        self._remaining(state)
        for dimension, amount in costs.items():
            _check(state['used'][dimension] + amount <= state['declaration']['budgets'][dimension],
                   'budget_exhausted', 'Budget exhausted: ' + dimension)
        for dimension, amount in costs.items():
            state['used'][dimension] += amount
        op = {'key': key, 'request_hash': _hash(request), 'kind': request['action'], 'arm': arm,
              'status': 'dispatched', 'costs': costs, 'result': None, 'selection': None}
        op['dispatch_seq'] = self._event({'type': 'dispatch', 'key': key, 'request_hash': op['request_hash'], 'costs': costs})
        self._put(op)
        self._save(state)
        return op

    def status(self, key, *, arm):
        op = self._op(key)
        _check(op is not None and op['arm'] == arm, 'unknown_operation', 'Operation unavailable')
        result = copy.deepcopy(op)
        if result['status'] == 'dispatched':
            result['status'] = 'outcome_unknown'
        return result

    def observe(self, arm):
        with self._transaction():
            state = self._state()
            _check(arm in state['arms'], 'unauthorized', 'Unknown arm')
            record = state['arms'][arm]
            actions = ['status']
            if state['phase'] == 'developing':
                if not record['pending']:
                    actions += ['inspect', 'submit']
                    if 'diagnostics' in state['used']:
                        actions.append('diagnose')
                if record['loss'] is not None:
                    actions += ['freeze', 'cancel']
            elif state['phase'] in ('frozen', 'finalizing', 'closed'):
                actions.append('export')
                if state['phase'] == 'frozen' and state['frozen']['complete']:
                    actions.append('finalize')
            view = {'projection': 'weightless.agent-view.v1', 'session': state['session'], 'arm': arm,
                    'declaration': state['declaration']['id'], 'observation': record['version'],
                    'base': record['checkpoint'], 'phase': state['phase'],
                    'parameters': {k: state['manifest']['parameters'][k] for k in state['declaration']['editable']},
                    'public_cases': [{k: c[k] for k in ('id', 'input', 'prompt', 'target') if k in c} for c in state['development'][:100]],
                    'public_cases_truncated': len(state['development']) > 100,
                    'pending': record['pending'], 'history': record['history'],
                    'metric': {'selection': 'loss', 'behavior': 'exact generation is separate from target-conditioned loss'},
                    'remaining': {k: state['declaration']['budgets'][k] - v for k, v in state['used'].items()},
                    'retention_acceptability': 'undetermined',
                    'actions': actions}
            self._event({'type': 'disclosure', 'arm': arm, 'projection': view['projection'], 'digest': _hash(view)})
            return copy.deepcopy(view)

    def _edits(self, state, edits):
        _check(type(edits) is list and 0 < len(edits) <= state['declaration']['max_edits'], 'invalid_proposal', 'Invalid edit count')
        seen, clean = set(), []
        for edit in edits:
            _fields(edit, 'tensor index value')
            name = state['manifest'].get('aliases', {}).get(edit['tensor'], edit['tensor'])
            _check(name in state['declaration']['editable'], 'invalid_proposal', 'Forbidden tensor')
            index, value = edit['index'], edit['value']
            _check(type(index) is int and 0 <= index < state['manifest']['parameters'][name]['size'], 'invalid_proposal', 'Invalid index')
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            _check(finite, 'invalid_proposal', 'Nonfinite edit')
            _check((name, index) not in seen, 'invalid_proposal', 'Duplicate canonical edit')
            seen.add((name, index))
            clean.append({'tensor': name, 'index': index, 'value': value})
        return clean

    def inspect(self, query, *, key, arm):
        _fields(query, 'checkpoint tensor indices')
        request = {'action': 'inspect', 'query': query, 'arm': arm}
        with self._transaction():
            existing = self._existing(key, request)
            if existing:
                return self.status(key, arm=arm)
            state = self._state()
            _check(arm in state['arms'] and state['phase'] == 'developing', 'invalid_state', 'Inspection unavailable')
            record = state['arms'][arm]
            _check(query['checkpoint'] == record['checkpoint'], 'stale_base', 'Inspection checkpoint changed')
            indices = query['indices']
            _check(type(indices) is list and 0 < len(indices) <= state['declaration']['max_inspect'], 'invalid_request', 'Inspection exceeds bound')
            edits = self._edits({**state, 'declaration': {**state['declaration'], 'max_edits': state['declaration']['max_inspect']}},
                                [{'tensor': query['tensor'], 'index': i, 'value': 0} for i in indices])
            op = self._reserve(state, key, request, {'inspections': 1}, arm)
        try:
            result = self._development_execute('inspect', patches=record['patches'],
                                         query={'tensor': edits[0]['tensor'], 'indices': indices})
            _check(result['checkpoint'] == record['checkpoint'], 'invalid_result', 'Inspection binding mismatch')
            result = {'checkpoint': result['checkpoint'], 'values': result['values']}
            return self._complete(op, result)
        except WeightlessError as error:
            if error.code in ('invalid_result', 'numerical_failure', 'invalid_task', 'unsupported', 'target_failure'):
                return self._failed(op, 'invalid_response' if error.code == 'invalid_result' else 'task_failure', error.code)
            return self._unknown(op, error.code)
        except Exception:
            return self._unknown(op)

    def _development_execute(self, action, **kwargs):
        with self._dispatch_window(action) as started:
            return self.runner.execute(action, self.target, timeout=self.remaining_seconds(),
                                       on_dispatch=started, **kwargs)

    def _forward_cost(self, state, cases):
        return forward_reservation(self.target, len(cases), generate=True) if 'forward_calls' in state['used'] else None

    def _evaluate_development(self, state, *, patches, cases, checkpoint):
        limit = self._forward_cost(state, cases)
        kwargs = {} if limit is None else {'forward_limit': limit}
        result = self._development_execute('evaluate', patches=patches, cases=cases, generate=True, **kwargs)
        self._validate_evaluation(result, checkpoint, len(cases), forward_limit=limit)
        return result

    @contextmanager
    def _dispatch_window(self, action, phases=('developing',)):
        import fcntl
        descriptor = os.open(self._dispatch_path, os.O_CREAT | os.O_RDWR, 0o600)
        released = False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            with self._transaction():
                state = self._state()
                self._remaining(state)
                _check(state['phase'] in phases, 'invalid_state', 'Session phase prevents dispatch')
                self._event({'type': 'execution_intent', 'action': action})
            def started():
                nonlocal released
                if not released:
                    with self._transaction():
                        self._event({'type': 'execution_started', 'action': action})
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    released = True
            yield started
        finally:
            if not released:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def teacher_dispatch(self, key):
        op = self._op(key)
        _check(op is not None and op['kind'] in ('teacher', 'teacher_report') and op['status'] == 'dispatched',
               'invalid_state', 'Teacher operation is not dispatchable')
        phases = ('frozen', 'closed') if op['kind'] == 'teacher_report' else ('developing',)
        return self._dispatch_window('teacher:' + key, phases)

    def submit(self, proposal, *, key):
        _fields(proposal, 'session arm declaration observation base edits')
        request = {'action': 'submit', 'proposal': proposal}
        arm = proposal['arm']
        existing = self._existing(key, request)
        if existing:
            if existing['status'] == 'completed':
                self._select(key)
            return self.status(key, arm=arm)
        with self._transaction():
            existing = self._existing(key, request)
            if existing:
                return self.status(key, arm=arm)
            state = self._state()
            _check(state['phase'] == 'developing' and arm in state['arms'], 'invalid_state', 'Development unavailable')
            record = state['arms'][arm]
            _check(not record['pending'], 'pending_operation', 'Arm already has pending work')
            _check(proposal['session'] == state['session'] and proposal['declaration'] == state['declaration']['id'],
                   'invalid_proposal', 'Proposal belongs to another declaration')
            _check(proposal['base'] == record['checkpoint'] and proposal['observation'] == record['version'],
                   'stale_base', 'Proposal observation changed')
            edits = self._edits(state, proposal['edits'])
            count = (2 if record['loss'] is None else 1) + bool(state['development_retention'])
            record['pending'] = key
            costs = {'attempts': 1, 'evaluations': count}
            if 'forward_calls' in state['used']:
                costs['forward_calls'] = self._forward_cost(state, state['development']) * (2 if record['loss'] is None else 1)
                costs['forward_calls'] += self._forward_cost(state, state['development_retention'])
            op = self._reserve(state, key, request, costs, arm)
            op['base'] = record['checkpoint']
            op['edits'] = edits
            self._put(op)
        try:
            baseline = None
            if record['loss'] is None:
                baseline = self._evaluate_development(state, patches=record['patches'],
                                                     cases=state['development'], checkpoint=record['checkpoint'])
            materialized = self._development_execute('materialize', patches=record['patches'], edits=edits)
            _check(materialized.get('base_checkpoint') == record['checkpoint']
                   and materialized.get('target_id') == state['manifest']['target_id'],
                   'invalid_result', 'Materialization source or base changed')
            result = self._evaluate_development(state, patches=materialized['patches'],
                                                cases=state['development'], checkpoint=materialized['checkpoint'])
            retention, retained = None, None
            if state['development_retention']:
                retained = self._evaluate_development(state, patches=materialized['patches'],
                                                      cases=state['development_retention'], checkpoint=materialized['checkpoint'])
                retention = retained['metrics']['loss']
            feedback = {'retention_loss': retention, 'checkpoint': materialized['checkpoint'], 'patches': materialized['patches'],
                        'baseline': self._feedback(baseline) if baseline else None, **self._feedback(result)}
            if retained is not None and 'usage' in retained:
                feedback['retention_usage'] = self._feedback(retained)['usage']
            self._complete(op, feedback)
            self._select(key)
            return self.status(key, arm=arm)
        except WeightlessError as error:
            if error.code in ('invalid_result', 'numerical_failure', 'invalid_task', 'unsupported', 'target_failure'):
                return self._failed(op, 'invalid_response' if error.code == 'invalid_result' else 'task_failure', error.code)
            return self._unknown(op, error.code)
        except Exception:
            return self._unknown(op)

    @staticmethod
    def _validate_evaluation(result, checkpoint, count, *, forward_limit=None):
        metrics = result.get('metrics', {})
        _check(result.get('checkpoint') == checkpoint and type(metrics.get('loss')) in (int, float)
               and math.isfinite(metrics['loss']) and type(metrics.get('exact')) is int
               and 0 <= metrics['exact'] <= count and metrics.get('count') == count,
               'invalid_result', 'Invalid bound evaluation')
        usage = result.get('usage')
        if forward_limit is not None or usage is not None:
            _check(type(usage) is dict and type(usage.get('forward_calls')) is int and usage['forward_calls'] >= 0,
                   'invalid_result', 'Missing measured forward usage')
        if forward_limit is not None:
            reported_limit = usage.get('forward_limit', forward_limit)
            _check(usage['forward_calls'] <= forward_limit and type(reported_limit) is int and reported_limit == forward_limit,
                   'invalid_result', 'Forward usage exceeds reservation or has a different limit')

    @staticmethod
    def _feedback(result):
        cases = []
        for case in result.get('cases', [])[:100]:
            row = {k: case[k] for k in ('id', 'prediction', 'target_nll', 'exact', 'stopped_on_eos') if k in case}
            prediction = row.get('prediction')
            row['prediction_truncated'] = isinstance(prediction, str) and len(prediction) > 4096
            if row['prediction_truncated']:
                row['prediction'] = prediction[:4096]
            cases.append(row)
        feedback = {'metrics': {k: result['metrics'][k] for k in ('loss', 'exact', 'count')},
                    'cases': cases, 'truncated': len(result.get('cases', [])) > 100}
        if 'usage' in result:
            feedback['usage'] = {k: result['usage'][k] for k in ('forward_calls', 'forward_limit') if k in result['usage']}
        return feedback

    def _complete(self, op, result):
        with self._transaction():
            current = self._op(op['key'])
            if current['status'] in ('completed', 'committed'):
                return current
            current.update(status='completed', result=result)
            current['result_seq'] = self._event({'type': 'result', 'key': op['key'], 'result': result})
            self._put(current)
            return copy.deepcopy(current)

    def _failed(self, op, status, code=None):
        with self._transaction():
            current = self._op(op['key'])
            if current['status'] not in ('completed', 'committed'):
                current['status'] = status
                current['error_code'] = code
                self._event({'type': 'failure', 'key': op['key'], 'status': status, 'error_code': code})
                state = self._state()
                if current['kind'] == 'submit':
                    state['arms'][current['arm']]['pending'] = None
                    self._save(state)
                self._put(current)
        return self.status(op['key'], arm=op['arm'])

    def _unknown(self, op, code=None):
        with self._transaction():
            current = self._op(op['key'])
            if current['status'] not in ('completed', 'committed'):
                current['status'] = 'outcome_unknown'
                current['error_code'] = code
                self._event({'type': 'unknown', 'key': op['key'], 'error_code': code})
                self._put(current)
        return self.status(op['key'], arm=op['arm'])

    def _select(self, key, *, cutoff=None):
        with self._transaction():
            state, op = self._state(), self._op(key)
            if op['selection'] is not None or op['status'] != 'completed':
                return
            if state['phase'] != 'developing' and (cutoff is None or op['result_seq'] >= cutoff):
                return
            record = state['arms'][op['arm']]
            result = op['result']
            baseline = result['baseline']['metrics']['loss'] if result['baseline'] else record['loss']
            retention_max = state['declaration'].get('development_retention_loss_max')
            retention_ok = retention_max is None or (result['retention_loss'] is not None and result['retention_loss'] <= retention_max)
            accepted = result['metrics']['loss'] < baseline and retention_ok
            decision = {'accepted': accepted, 'before': record['checkpoint'], 'after': result['checkpoint'] if accepted else record['checkpoint']}
            if accepted:
                record.update(checkpoint=result['checkpoint'], patches=result['patches'], loss=result['metrics']['loss'])
            else:
                record['loss'] = baseline
            record['pending'] = None
            record['version'] += 1
            record['history'].append({'key': key, 'checkpoint': result['checkpoint'], 'metrics': result['metrics'], 'cases': result['cases'], 'selection': decision})
            for field in ('usage', 'retention_usage'):
                if field in result:
                    record['history'][-1][field] = copy.deepcopy(result[field])
            op.update(status='committed', selection=decision)
            self._event({'type': 'selection', 'key': key, 'decision': decision})
            self._put(op)
            self._save(state)

    def reserve_teacher(self, request, *, key, arm, purpose='development'):
        _check(purpose in ('development', 'report'), 'invalid_request', 'Unknown teacher purpose')
        request = {'action': 'teacher_report' if purpose == 'report' else 'teacher', 'arm': arm, 'digest': _hash(request)}
        with self._transaction():
            existing = self._existing(key, request)
            if existing:
                return self.status(key, arm=arm)
            state = self._state()
            phases = ('frozen', 'closed') if purpose == 'report' else ('developing',)
            _check(state['phase'] in phases and arm in state['arms'], 'invalid_state', 'Teacher dispatch unavailable')
            return self._reserve(state, key, request, {'teacher_requests': 1}, arm)

    def complete_teacher(self, key, response):
        op = self._op(key)
        _check(op is not None and op['kind'] in ('teacher', 'teacher_report'), 'unknown_operation', 'Unknown teacher operation')
        return self._complete(op, {'response_digest': _hash(response)})

    def unknown_teacher(self, key):
        op = self._op(key)
        _check(op is not None and op['kind'] in ('teacher', 'teacher_report'), 'unknown_operation', 'Unknown teacher operation')
        return self._unknown(op)

    def freeze(self, *, key):
        import fcntl
        descriptor = os.open(self._dispatch_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            return self._freeze(key=key)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _freeze(self, *, key):
        request = {'action': 'freeze'}
        with self._transaction():
            state = self._state()
            self._existing(key, request)
            if state['frozen']:
                return state['frozen']
            if state['phase'] == 'developing':
                freeze_op = {'key': key, 'request_hash': _hash(request), 'kind': 'freeze', 'arm': None,
                             'status': 'committed', 'costs': {}, 'result': None, 'selection': None}
                freeze_op['dispatch_seq'] = self._event({'type': 'dispatch', 'key': key, 'request_hash': freeze_op['request_hash'], 'costs': {}})
                self._put(freeze_op)
                state['cutoff'] = self._event({'type': 'barrier', 'key': key})
                state['phase'] = 'freezing'
                self._save(state)
            _check(state['phase'] == 'freezing', 'invalid_state', 'Cannot freeze')
            cutoff = state['cutoff']
        for row in self.db.execute('SELECT body FROM operations ORDER BY rowid').fetchall():
            op = json.loads(row[0])
            if op['kind'] == 'submit' and op['status'] == 'completed' and op['result_seq'] < cutoff:
                self._select(op['key'], cutoff=cutoff)
        with self._transaction():
            state = self._state()
            if state['frozen']:
                return state['frozen']
            obligations = [json.loads(r[0])['key'] for r in self.db.execute('SELECT body FROM operations')
                           if json.loads(r[0])['status'] in ('dispatched', 'outcome_unknown')]
            complete = not obligations and all(r['loss'] is not None and r['pending'] is None for r in state['arms'].values())
            state['frozen'] = _seal({'format': 'weightless.frozen.v1', 'session': state['session'],
                                     'declaration': state['declaration']['id'], 'cutoff': cutoff,
                                     'initial': state['manifest']['checkpoint'], 'runner': _hash(state['runner']),
                                     'criteria': state['declaration'].get('criteria', {}),
                                     'target_identity': state['target_identity'],
                                     **({'final_forward_calls': state['declaration']['final_forward_calls']}
                                        if 'forward_calls' in state['used'] and 'final_forward_calls' in state['declaration'] else {}),
                                     'complete': complete, 'obligations': obligations,
                                     'arms': {a: {k: r[k] for k in ('checkpoint', 'patches', 'loss')} for a, r in state['arms'].items()}})
            state['phase'] = 'frozen'
            self._event({'type': 'frozen', 'id': state['frozen']['id']})
            self._save(state)
            return copy.deepcopy(state['frozen'])

    def finalize(self, *, key, dataset=None):
        dataset = self.final_dataset if dataset is None else dataset
        _check(dataset is not None, 'unsupported', 'No final dataset configured')
        _check(self.verifier is not None, 'unsupported', 'No verifier configured')
        with self._transaction():
            state = self._state()
            _check(state['frozen'] and state['frozen']['complete'], 'incomplete_freeze', 'Complete freeze required')
            if state['final']:
                _check(state['final_key'] == key and state['final_dataset'] == dataset, 'idempotency_conflict', 'Final claim already fixed')
                return state['final']
            if state['phase'] == 'finalizing':
                _check(state['final_key'] == key and state['final_dataset'] == dataset, 'idempotency_conflict', 'Final claim already fixed')
            request = {'action': 'finalize', 'dataset': dataset, 'frozen': state['frozen']['id']}
            existing = self._existing(key, request)
            if not existing:
                costs = {'final_evaluations': self.verifier.evaluation_cost(dataset, state['frozen'])}
                if 'forward_calls' in state['used']:
                    _check('final_forward_calls' in state['frozen'], 'invalid_declaration',
                           'Predeclared final_forward_calls envelope required')
                    costs['forward_calls'] = state['declaration']['final_forward_calls']
                self._reserve(state, key, request, costs, None)
            state.update(phase='finalizing', final_key=key, final_dataset=dataset)
            self._save(state)
        receipt = self.verifier.grade(state['frozen'], target=self.target, key=key, dataset=dataset, timeout=self.remaining_seconds())
        self._complete(self._op(key), receipt)
        with self._transaction():
            state = self._state()
            if state['final'] is not None:
                self._event({'type': 'late_final', 'digest': _hash(receipt)})
                return state['final']
            state.update(final=receipt, phase='closed')
            self._save(state)
        self.verifier.ack(receipt['claim'])
        return receipt

    def export(self):
        with self._transaction():
            state = self._state()
            return _seal({'format': 'weightless.evidence.v1', 'session': state['session'],
                          'declaration': state['declaration'], 'initial': state['manifest']['checkpoint'],
                          'runner': _hash(state['runner']), 'used': state['used'], 'origin': state['origin'],
                          'operations': [json.loads(r[0]) for r in self.db.execute('SELECT body FROM operations ORDER BY rowid')],
                          'journal': [{'seq': r[0], **json.loads(r[1])} for r in self.db.execute('SELECT seq,body FROM journal ORDER BY seq')],
                          'frozen': state['frozen'], 'final': state['final'],
                          'limitations': ['unsigned evidence does not authenticate scores', 'strict loss selection only',
                                         'finite-case conclusions only; no population confidence rule',
                                         'no basis coefficients, logits, gradients, activation probes or protocol migration',
                                         'replay checks journal and selection, not numerical state hashes',
                                         'no cache or stochastic replication', 'no provider billing guarantee']})

    def diagnose(self, *, key, arm):
        request = {'action': 'diagnose', 'arm': arm}
        with self._transaction():
            if self._existing(key, request):
                return self.status(key, arm=arm)
            state = self._state()
            _check(state['phase'] == 'developing' and arm in state['arms'], 'invalid_state', 'Diagnostics unavailable')
            _check('diagnostics' in state['used'], 'unsupported', 'Diagnostic budget not declared')
            record = state['arms'][arm]
            costs = {'diagnostics': 1, 'evaluations': 1}
            if 'forward_calls' in state['used']:
                costs['forward_calls'] = self._forward_cost(state, state['development'])
            op = self._reserve(state, key, request, costs, arm)
        try:
            result = self._evaluate_development(state, patches=record['patches'], cases=state['development'], checkpoint=record['checkpoint'])
            return self._complete(op, {'checkpoint': record['checkpoint'], **self._feedback(result)})
        except WeightlessError as error:
            if error.code in ('invalid_result', 'numerical_failure', 'invalid_task', 'unsupported', 'target_failure'):
                return self._failed(op, 'invalid_response' if error.code == 'invalid_result' else 'task_failure', error.code)
            return self._unknown(op, error.code)
        except Exception:
            return self._unknown(op)

    def cancel(self, *, key):
        frozen = self.freeze(key=key)
        return {'status': 'cancel_requested', 'frozen': frozen, 'termination_confirmed': False}

    def dispatch(self, command, *, arm, allowed_actions=('observe', 'inspect', 'submit', 'status'), final_dataset=None):
        _check(len(_json(command).encode()) <= 65536, 'invalid_request', 'Command exceeds byte bound')
        _fields(command, 'action key payload')
        action, key, payload = command['action'], command['key'], command['payload']
        _check(action in allowed_actions, 'unauthorized', 'Command not granted by operator')
        if action in ('freeze', 'cancel', 'export', 'finalize'):
            _fields(payload, '')
            if action == 'export':
                _check(self._state()['phase'] in ('frozen', 'finalizing', 'closed'),
                       'unauthorized', 'Agent comparison export requires freeze')
                return self.export()
            if action == 'finalize':
                final_dataset = self.final_dataset if final_dataset is None else final_dataset
                _check(final_dataset is not None, 'unauthorized', 'Final dataset not configured')
                return self.finalize(key=key, dataset=final_dataset)
            return self.freeze(key=key) if action == 'freeze' else self.cancel(key=key)
        if action == 'observe':
            _fields(payload, '')
            return self.observe(arm)
        if action == 'diagnose':
            _fields(payload, '')
            return self.diagnose(key=key, arm=arm)
        if action == 'inspect':
            return self.inspect(payload, key=key, arm=arm)
        if action == 'submit':
            _check(type(payload) is dict and payload.get('arm') == arm, 'unauthorized', 'Arm mismatch')
            return self.submit(payload, key=key)
        if action == 'status':
            _fields(payload, 'key')
            return self.status(payload['key'], arm=arm)
        raise WeightlessError('unauthorized', 'Command unavailable to agent')


def replay_session(bundle):
    """Verify internal evidence and selection consistency without executing a target."""
    bundle = copy.deepcopy(bundle)
    def sealed(value):
        _check(type(value) is dict and value.get('id') == _hash({k: v for k, v in value.items() if k != 'id'}),
               'invalid_artifact', 'Artifact digest mismatch')
    sealed(bundle)
    sealed(bundle['declaration'])
    frozen = bundle['frozen']
    _check(frozen is not None, 'invalid_artifact', 'Frozen record required')
    sealed(frozen)
    _check(frozen.get('criteria', {}) == bundle['declaration'].get('criteria', {}), 'invalid_artifact', 'Criteria changed')
    if 'forward_calls' in bundle['declaration']['budgets']:
        _check(frozen.get('final_forward_calls') == bundle['declaration'].get('final_forward_calls'),
               'invalid_artifact', 'Final forward envelope changed')
    _check(frozen['session'] == bundle['session'] and frozen['declaration'] == bundle['declaration']['id']
           and frozen['initial'] == bundle['initial'] and frozen['runner'] == bundle['runner'], 'invalid_artifact', 'Freeze binding mismatch')
    journal = bundle['journal']
    _check([e['seq'] for e in journal] == list(range(1, len(journal) + 1)), 'invalid_artifact', 'Journal sequence gap')
    barriers = [e for e in journal if e['type'] == 'barrier']
    _check(len(barriers) == 1 and barriers[0]['seq'] == frozen['cutoff'], 'invalid_artifact', 'Invalid cutoff')
    used = dict.fromkeys(bundle['used'], 0)
    arms = {a: {'checkpoint': bundle['initial'], 'patches': [], 'loss': None} for a in bundle['declaration']['arms']}
    for op in bundle['operations']:
        if op['kind'] == 'finalize' and 'forward_calls' in bundle['declaration']['budgets']:
            _check(op['costs'].get('forward_calls') == bundle['declaration'].get('final_forward_calls'),
                   'invalid_artifact', 'Final forward reservation differs from declaration')
        for dimension, cost in op['costs'].items():
            used[dimension] += cost
        dispatch = journal[op['dispatch_seq'] - 1]
        _check(dispatch['type'] == 'dispatch' and dispatch['key'] == op['key'] and dispatch['costs'] == op['costs']
               and dispatch['request_hash'] == op['request_hash'], 'invalid_artifact', 'Dispatch mismatch')
        if op.get('result_seq'):
            event = journal[op['result_seq'] - 1]
            _check(event['type'] == 'result' and event['key'] == op['key'] and event['result'] == op['result'], 'invalid_artifact', 'Result mismatch')
        if op['selection'] is None:
            continue
        selections = [e for e in journal if e['type'] == 'selection' and e['key'] == op['key']]
        _check(len(selections) == 1 and selections[0]['decision'] == op['selection']
               and selections[0]['seq'] > op['result_seq'], 'invalid_artifact', 'Selection journal mismatch')
        _check(op['kind'] == 'submit' and op['result_seq'] < frozen['cutoff'], 'invalid_artifact', 'Ineligible selection')
        record, result = arms[op['arm']], op['result']
        _check(op['base'] == record['checkpoint'], 'invalid_artifact', 'Stale selection')
        baseline = result['baseline']['metrics']['loss'] if result['baseline'] else record['loss']
        retention_max = bundle['declaration'].get('development_retention_loss_max')
        retention_ok = retention_max is None or (result['retention_loss'] is not None and result['retention_loss'] <= retention_max)
        accepted = result['metrics']['loss'] < baseline and retention_ok
        expected = {'accepted': accepted, 'before': record['checkpoint'], 'after': result['checkpoint'] if accepted else record['checkpoint']}
        _check(op['selection'] == expected, 'invalid_artifact', 'Selection differs from policy')
        if accepted:
            record.update(checkpoint=result['checkpoint'], patches=result['patches'], loss=result['metrics']['loss'])
        else:
            record['loss'] = baseline
    _check(used == bundle['used'] and all(v <= bundle['declaration']['budgets'][k] for k, v in used.items()), 'invalid_artifact', 'Budget mismatch')
    selected_keys = {op['key'] for op in bundle['operations'] if op['selection'] is not None}
    _check(all(e['key'] in selected_keys for e in journal if e['type'] == 'selection'),
           'invalid_artifact', 'Unowned selection event')
    _check(arms == frozen['arms'], 'invalid_artifact', 'Frozen selection mismatch')
    pending = {op['key']: 'dispatched' for op in bundle['operations']
               if op['kind'] != 'freeze' and op['dispatch_seq'] < frozen['cutoff']}
    for event in journal[:frozen['cutoff'] - 1]:
        key = event.get('key')
        if key in pending:
            if event['type'] == 'result':
                pending[key] = 'completed'
            elif event['type'] == 'failure':
                pending[key] = event['status']
            elif event['type'] == 'unknown':
                pending[key] = 'outcome_unknown'
    obligations = [key for key, status in pending.items() if status in ('dispatched', 'outcome_unknown')]
    complete = not obligations and all(record['loss'] is not None for record in arms.values())
    _check(frozen.get('obligations') == obligations and type(frozen.get('complete')) is bool
           and frozen['complete'] == complete, 'invalid_artifact', 'Freeze completeness differs from recorded execution')
    return frozen


def verify_session(bundle, *, runner, target):
    """Reconstruct full states with the declared runner, independent of recorded scores."""
    frozen = replay_session(bundle)
    _check(_hash(runner.config) == bundle['runner'], 'identity_mismatch', 'Runner identity differs')
    manifest = runner.execute('describe', target)
    _check(manifest['checkpoint'] == bundle['initial'], 'identity_mismatch', 'Initial target differs')
    patches = {arm: [] for arm in bundle['declaration']['arms']}
    checked = []
    for op in bundle['operations']:
        if op['kind'] != 'submit' or op['result'] is None:
            continue
        rebuilt = runner.execute('materialize', target, patches=patches[op['arm']], edits=op['edits'])
        _check(rebuilt['checkpoint'] == op['result']['checkpoint'] and rebuilt['patches'] == op['result']['patches'],
               'invalid_artifact', 'Materialized candidate differs from recorded full state')
        checked.append(rebuilt['checkpoint'])
        if op['selection'] and op['selection']['accepted']:
            patches[op['arm']] = rebuilt['patches']
    return {'frozen': frozen['id'], 'verified_candidates': checked,
            'verification': 'full numerical state reconstruction; scores not authenticated'}
