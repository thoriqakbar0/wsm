"""Private final datasets and durable, cross-session disclosure claims."""
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import time

from .session import Session, _check, _connect, _hash, _json, _seal
from .sandbox import forward_reservation


class Verifier:
    """Keep this registry and its datasets outside every untrusted agent sandbox."""

    def __init__(self, path, *, runner, datasets):
        self._grade_path = str(Path(path).resolve()) + '.grade'
        self.db = _connect(path)
        self.runner = runner
        self.datasets = copy.deepcopy(datasets)
        self.db.execute('''CREATE TABLE IF NOT EXISTS claims (
            dataset TEXT PRIMARY KEY, body TEXT NOT NULL)''')
        self.db.execute('CREATE TABLE IF NOT EXISTS authority (id INTEGER PRIMARY KEY CHECK(id=1), secret BLOB NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS session_bindings (session TEXT PRIMARY KEY, identity TEXT NOT NULL)')
        self.db.execute('INSERT OR IGNORE INTO authority VALUES (1,?)', (os.urandom(32),))
        self._secret = self.db.execute('SELECT secret FROM authority WHERE id=1').fetchone()[0]

    def _identity(self, cases):
        return hmac.new(self._secret, _json(cases).encode(), hashlib.sha256).hexdigest()

    def close(self):
        self.db.close()

    def bind_session(self, session, dataset, *, create=False):
        """Pin a workspace's final authority and cases without publishing their digest."""
        _check(dataset in self.datasets, 'unknown_dataset', 'Dataset unavailable')
        identity = self._identity({'dataset': dataset, 'cases': self.datasets[dataset]})
        if create:
            self.db.execute('INSERT OR IGNORE INTO session_bindings VALUES (?,?)', (session, identity))
        row = self.db.execute('SELECT identity FROM session_bindings WHERE session=?', (session,)).fetchone()
        _check(row is not None and hmac.compare_digest(row[0], identity),
               'verifier_mismatch', 'Reopen with the verifier authority, dataset, and cases bound at workspace creation')

    def _read(self, identity):
        row = self.db.execute('SELECT body FROM claims WHERE dataset=?', (identity,)).fetchone()
        return json.loads(row[0]) if row else None

    def _write(self, identity, record):
        self.db.execute('INSERT OR REPLACE INTO claims VALUES (?,?)', (identity, _json(record)))

    def evaluation_cost(self, dataset, frozen):
        _check(dataset in self.datasets, 'unknown_dataset', 'Dataset unavailable')
        cases = self.datasets[dataset]
        return len(frozen['arms']) * (1 + int(type(cases) is dict and bool(cases.get('retention'))))

    def grade(self, frozen, *, target, key, dataset, timeout=None):
        import fcntl
        descriptor = os.open(self._grade_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                _check(False, 'verifier_in_progress', 'Final execution is already active')
            return self._grade(frozen, target=target, key=key, dataset=dataset, timeout=timeout)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _grade(self, frozen, *, target, key, dataset, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else None
        def remaining():
            if deadline is None:
                return None
            value = deadline - time.monotonic()
            _check(value > 0, 'deadline_exhausted', 'Final deadline exhausted')
            return value
        _check(frozen.get('complete') is True and frozen.get('id') == _hash({k: v for k, v in frozen.items() if k != 'id'}),
               'incomplete_freeze', 'Complete intact freeze required')
        _check(frozen.get('runner') == _hash(self.runner.config), 'identity_mismatch', 'Final runner identity differs')
        _check(frozen.get('target_identity') == _hash(target), 'identity_mismatch', 'Final target identity differs')
        _check(dataset in self.datasets, 'unknown_dataset', 'Dataset is unavailable')
        cases = self.datasets[dataset]
        cases = {'task': cases, 'retention': []} if type(cases) is list else cases
        envelope = frozen.get('final_forward_calls')
        limits = {name: forward_reservation(target, len(cases.get(name, [])), generate=True)
                  for name in ('task', 'retention')}
        if envelope is not None:
            _check(type(envelope) is int and envelope >= 0
                   and len(frozen['arms']) * sum(limits.values()) <= envelope,
                   'budget_exhausted', 'Final forward envelope is insufficient')
        identity = self._identity(cases)
        claim_id = _hash({'dataset': identity, 'frozen': frozen['id'], 'key': key})
        self.db.execute('BEGIN IMMEDIATE')
        try:
            claim = self._read(identity)
            if claim:
                _check(claim['claim'] == claim_id, 'dataset_spent', 'Dataset already claimed by another comparison')
                if claim['receipt'] is None:
                    claim['operation'] = 'outcome_unknown'
                    claim['receipt'] = _seal({'format': 'weightless.final.v1', 'claim': claim_id, 'frozen': frozen['id'],
                                               'status': 'outcome_unknown', 'arms': {}, 'conclusion': 'inconclusive'})
                    self._write(identity, claim)
                self.db.execute('COMMIT')
                return self._release(identity, claim)
            claim = {'claim': claim_id, 'frozen': frozen['id'], 'state': 'claimed',
                     'operation': 'dispatched', 'receipt': None, 'usage': {}}
            self._write(identity, claim)
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:
                self.db.execute('ROLLBACK')
            raise
        try:
            results = {}
            for arm, candidate in frozen['arms'].items():
                kwargs = {} if envelope is None else {'forward_limit': limits['task']}
                task = self.runner.execute('evaluate', target, patches=candidate['patches'],
                                           cases=cases['task'], generate=True, purpose='final', timeout=remaining(), **kwargs)
                Session._validate_evaluation(task, candidate['checkpoint'], len(cases['task']),
                                             forward_limit=None if envelope is None else limits['task'])
                if 'usage' in task:
                    claim['usage'][arm] = {'task': copy.deepcopy(task['usage'])}
                    self._write(identity, claim)
                task['metrics']['exact'] = self._exact(task, cases['task'])
                retention = None
                if cases.get('retention'):
                    kwargs = {} if envelope is None else {'forward_limit': limits['retention']}
                    measured = self.runner.execute('evaluate', target, patches=candidate['patches'],
                                                   cases=cases['retention'], generate=True, purpose='final', timeout=remaining(), **kwargs)
                    Session._validate_evaluation(measured, candidate['checkpoint'], len(cases['retention']),
                                                 forward_limit=None if envelope is None else limits['retention'])
                    if 'usage' in measured:
                        claim['usage'].setdefault(arm, {})['retention'] = copy.deepcopy(measured['usage'])
                        self._write(identity, claim)
                    measured['metrics']['exact'] = self._exact(measured, cases['retention'])
                    retention = {k: measured['metrics'][k] for k in ('loss', 'exact', 'count')}
                criteria = frozen.get('criteria', {})
                exact_min = criteria.get('final_exact_min')
                retention_max = criteria.get('retention_loss_max')
                exact = task['metrics']['exact'] / task['metrics']['count']
                acceptable = None if retention is None or retention_max is None else retention['loss'] <= retention_max
                conclusion = 'inconclusive'
                if exact_min is not None and retention_max is not None and retention is not None and (exact < exact_min or acceptable is False):
                    conclusion = 'bounded_negative'
                elif exact_min is not None and exact >= exact_min and acceptable is True:
                    conclusion = 'supported'
                results[arm] = {'checkpoint': candidate['checkpoint'],
                                'task': {k: task['metrics'][k] for k in ('loss', 'exact', 'count')},
                                'retention': retention,
                                'retention_acceptability': 'undetermined' if acceptable is None else ('acceptable' if acceptable else 'unacceptable'),
                                'conclusion': conclusion,
                                'scope': 'This frozen candidate on the declared finite final cases; no population generalization.'}
            claim['operation'] = 'completed'
            claim['state'] = 'evaluated_private'
            claim['receipt'] = _seal({'format': 'weightless.final.v1', 'claim': claim_id, 'frozen': frozen['id'],
                                       'status': 'completed', 'arms': results, 'assurance': 'trusted verifier; unsigned receipt'})
        except Exception:
            claim['operation'] = 'outcome_unknown'
            claim['receipt'] = _seal({'format': 'weightless.final.v1', 'claim': claim_id, 'frozen': frozen['id'],
                                       'status': 'outcome_unknown', 'arms': {}, 'conclusion': 'inconclusive'})
        self._write(identity, claim)
        return self._release(identity, claim)

    @staticmethod
    def _exact(result, cases):
        rows = result.get('cases', [])
        _check(len(rows) == len(cases) and len({r.get('id') for r in rows}) == len(cases),
               'invalid_result', 'Missing or duplicate final predictions')
        by_id = {r['id']: r for r in rows}
        _check(set(by_id) == {c['id'] for c in cases} and all('target' in c for c in cases),
               'invalid_result', 'Final case binding mismatch')
        for case in cases:
            row = by_id[case['id']]
            target = case['target']
            _check('prediction' in row and ((type(target) is str and type(row['prediction']) is str)
                   or (type(target) is list and type(row['prediction']) is list and len(target) == len(row['prediction']))),
                   'invalid_result', 'Missing or malformed literal final prediction')
        return sum(by_id[c['id']]['prediction'] == c['target'] for c in cases)

    def _release(self, identity, claim):
        if claim['receipt'] is None:
            return _seal({'format': 'weightless.final.v1', 'claim': claim['claim'], 'frozen': claim['frozen'],
                           'status': 'outcome_unknown', 'arms': {}, 'conclusion': 'inconclusive'})
        if claim['state'] != 'exposed':
            claim['state'] = 'disclosure_unknown'
            self._write(identity, claim)
        return copy.deepcopy(claim['receipt'])

    def ack(self, claim_id):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for row in self.db.execute('SELECT dataset,body FROM claims').fetchall():
                claim = json.loads(row[1])
                if claim['claim'] == claim_id and claim['state'] == 'disclosure_unknown':
                    claim['state'] = 'exposed'
                    self._write(row[0], claim)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def status(self, dataset):
        cases = self.datasets[dataset]
        identity = self._identity({'task': cases, 'retention': []} if type(cases) is list else cases)
        claim = self._read(identity)
        return {'state': 'unused', 'operation': None} if claim is None else {k: claim[k] for k in ('state', 'operation')}
