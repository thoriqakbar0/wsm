import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wsm.binary64 import WeightlessError
from wsm.session import Session, replay_session, verify_session, _hash, _seal


class Runner:
    config = {'mode': 'controlled-test', 'image': 'fixed'}

    def __init__(self):
        self.calls = []
        self.fail = None
        self.hook = None

    def execute(self, action, target, **kwargs):
        started = kwargs.pop('on_dispatch', None)
        if started:
            started()
        self.calls.append((action, copy.deepcopy(kwargs)))
        if action == self.fail:
            raise RuntimeError('private error should never escape')
        patches = kwargs.get('patches') or []
        value = patches[0]['value'] if patches else 0
        if action == 'describe':
            return {'target_id': 'target', 'checkpoint': _hash(0), 'parameters': {'w': {'shape': [1], 'size': 1, 'dtype': 'float64'}},
                    'aliases': {'alias': 'w'}, 'capabilities': {}}
        if action == 'materialize':
            base = _hash(value)
            value = kwargs['edits'][0]['value']
            return {'target_id': 'target', 'base_checkpoint': base,
                    'checkpoint': _hash(value), 'patches': kwargs['edits'], 'changes': kwargs['edits']}
        if action == 'inspect':
            return {'checkpoint': _hash(value), 'values': [{'tensor': 'w', 'index': 0, 'value': value}]}
        if self.hook:
            self.hook()
        cases = kwargs['cases']
        return {'checkpoint': _hash(value), 'metrics': {'loss': abs(1-value), 'exact': int(value == 1) * len(cases), 'count': len(cases)},
                'cases': [{'id': c['id'], 'prediction': [value], 'target_nll': abs(1-value), 'exact': value == 1,
                           'stopped_on_eos': True, 'private_log': 'do not expose'} for c in cases]}


def declaration(**kwargs):
    return {'hypothesis': 'A scalar replacement solves the finite task', 'arms': ['teacher'], 'editable': ['w'],
            'budgets': {'attempts': 4, 'evaluations': 8, 'inspections': 3, 'teacher_requests': 2, 'final_evaluations': 4},
            'max_edits': 2, 'max_inspect': 2, 'wall_seconds': 60, **kwargs}


def proposal(session, value=1, arm='teacher'):
    view = session.observe(arm)
    return {k: view[k] for k in ('session', 'arm', 'declaration', 'observation', 'base')} | {'edits': [{'tensor': 'w', 'index': 0, 'value': value}]}


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'session.db'
        self.runner = Runner()
        self.session = Session.create(self.path, runner=self.runner, target={'kind': 'test'}, declaration=declaration(), development=[{'id': 'public'}])

    def tearDown(self):
        self.session.close()
        self.tmp.cleanup()

    def test_idempotency_restart_budget_and_replay(self):
        p = proposal(self.session)
        first = self.session.submit(p, key='one')
        calls = len(self.runner.calls)
        self.assertTrue(first['selection']['accepted'])
        self.session.close()
        self.session = Session.open(self.path, runner=self.runner, target={'kind': 'test'})
        self.assertEqual(first, self.session.submit(p, key='one'))
        self.assertEqual(calls, len(self.runner.calls))
        self.assertEqual(self.session.export()['used']['evaluations'], 2)
        frozen = self.session.freeze(key='stop')
        self.assertEqual(replay_session(self.session.export()), frozen)
        self.assertEqual(self.session.freeze(key='other'), frozen)
        with self.assertRaises(WeightlessError):
            self.session.submit({**p, 'edits': [{'tensor': 'w', 'index': 0, 'value': 2}]}, key='one')

    def test_invalid_admission_never_executes_or_charges(self):
        p = proposal(self.session)
        mutations = [dict(p, surprise=1), dict(p, base='stale'), dict(p, edits=[{'tensor': 'w', 'index': 0, 'value': 1}, {'tensor': 'alias', 'index': 0, 'value': 2}]),
                     dict(p, edits=[{'tensor': 'w', 'index': 0, 'value': float('nan')}])]
        for i, invalid in enumerate(mutations):
            with self.assertRaises(WeightlessError):
                self.session.submit(invalid, key=str(i))
        self.assertEqual(len(self.runner.calls), 1)
        self.assertEqual(self.session.export()['used']['attempts'], 0)

    def test_same_key_recovers_durable_result_before_another_proposal(self):
        original = self.session._select
        self.session._select = lambda *a, **k: None
        request = proposal(self.session)
        self.session.submit(request, key='saved')
        self.session._select = original
        before = len(self.runner.calls)
        result = self.session.submit(request, key='saved')
        self.assertTrue(result['selection']['accepted'])
        self.assertEqual(before, len(self.runner.calls))
        self.assertIsNone(self.session.observe('teacher')['pending'])
        self.session.submit(proposal(self.session, .5), key='next')

    def test_changed_materialization_source_is_not_evaluated_or_selected(self):
        self.session.submit(proposal(self.session), key='initial')
        original = self.runner.execute
        def changed(action, target, **kwargs):
            value = original(action, target, **kwargs)
            if action == 'materialize':
                value['base_checkpoint'] = 'different-source'
            return value
        self.runner.execute = changed
        before = sum(action == 'evaluate' for action, _ in self.runner.calls)
        result = self.session.submit(proposal(self.session, .5), key='changed')
        self.assertEqual(result['status'], 'invalid_response')
        self.assertEqual(before, sum(action == 'evaluate' for action, _ in self.runner.calls))
        self.assertEqual(self.session.observe('teacher')['base'], _hash(1))

    def test_contradictory_selection_journal_is_rejected(self):
        self.session.submit(proposal(self.session), key='one')
        self.session.freeze(key='stop')
        bundle = self.session.export()
        for event in bundle['journal']:
            if event['type'] == 'selection':
                event['decision']['accepted'] = False
        bundle = _seal({k:v for k,v in bundle.items() if k != 'id'})
        with self.assertRaises(WeightlessError):
            replay_session(bundle)

    def test_diagnostic_cannot_bypass_total_evaluation_ceiling(self):
        self.session.close()
        config = declaration()
        config['budgets'].update(evaluations=2, diagnostics=1)
        self.session = Session.create(Path(self.tmp.name)/'bounded', runner=self.runner,
            target={'kind':'test'}, declaration=config, development=[{'id':'public'}])
        self.session.submit(proposal(self.session), key='one')
        before = len(self.runner.calls)
        with self.assertRaises(WeightlessError):
            self.session.diagnose(key='probe', arm='teacher')
        self.assertEqual(before, len(self.runner.calls))
        self.assertEqual(self.session.export()['used']['diagnostics'], 0)

    def test_completed_worker_failure_keeps_safe_diagnostic_and_allows_recovery(self):
        original = self.runner.execute
        def failed(action, target, **kwargs):
            if action == 'evaluate':
                raise WeightlessError('target_failure', 'Private worker detail')
            return original(action, target, **kwargs)
        self.runner.execute = failed
        receipt = self.session.submit(proposal(self.session), key='failed-worker')
        self.assertEqual(receipt['status'], 'task_failure')
        self.assertEqual(receipt['error_code'], 'target_failure')
        self.assertNotIn('Private worker detail', str(self.session.export()))
        self.runner.execute = original
        recovered = self.session.submit(proposal(self.session), key='recovered')
        self.assertTrue(recovered['selection']['accepted'])

    def test_unknown_dispatch_never_resends(self):
        self.runner.fail = 'evaluate'
        p = proposal(self.session)
        receipt = self.session.submit(p, key='lost')
        self.assertEqual(receipt['status'], 'outcome_unknown')
        calls = len(self.runner.calls)
        self.runner.fail = None
        self.session.close()
        self.session = Session.open(self.path, runner=self.runner, target={'kind': 'test'})
        self.assertEqual(self.session.submit(p, key='lost')['status'], 'outcome_unknown')
        self.assertEqual(len(self.runner.calls), calls)
        frozen = self.session.freeze(key='stop')
        self.assertFalse(frozen['complete'])
        self.assertEqual(frozen['obligations'], ['lost'])

    def test_result_durable_before_selection_is_drained_at_freeze(self):
        original = self.session._select
        self.session._select = lambda *a, **k: None
        self.session.submit(proposal(self.session), key='one')
        self.session._select = original
        calls = len(self.runner.calls)
        frozen = self.session.freeze(key='stop')
        self.assertTrue(frozen['complete'])
        self.assertEqual(frozen['arms']['teacher']['checkpoint'], _hash(1))
        self.assertEqual(calls, len(self.runner.calls))
        replay_session(self.session.export())

    def test_result_after_barrier_is_supplement_only(self):
        frozen = []
        self.runner.hook = lambda: frozen.append(self.session.freeze(key='stop')) if not frozen else None
        self.session.submit(proposal(self.session), key='one')
        self.assertFalse(frozen[0]['complete'])
        self.assertEqual(self.session.freeze(key='stop'), frozen[0])
        self.assertEqual(frozen[0]['arms']['teacher']['checkpoint'], _hash(0))
        replay_session(self.session.export())

    def test_arm_views_and_scope_are_allowlisted(self):
        with self.session._transaction():
            state = self.session._state()
            state['arms']['secret-control'] = copy.deepcopy(state['arms']['teacher'])
            state['arms']['secret-control']['history'] = ['CANARY']
            self.session._save(state)
        self.session.submit(proposal(self.session), key='one')
        self.assertNotIn('CANARY', str(self.session.observe('teacher')))
        self.assertNotIn('private_log', str(self.session.observe('teacher')))
        with self.assertRaises(WeightlessError):
            self.session.dispatch({'action': 'status', 'key': 'x', 'payload': {'key': 'one'}}, arm='secret-control')
        with self.assertRaises(WeightlessError):
            self.session.dispatch({'action': 'export', 'key': 'x', 'payload': {}}, arm='teacher')

    def test_teacher_reservation_unknown_survives_and_cancel_preserves_it(self):
        self.session.reserve_teacher({'prompt': 'private'}, key='teacher-1', arm='teacher')
        self.assertEqual(self.session.reserve_teacher({'prompt': 'private'}, key='teacher-1', arm='teacher')['status'], 'outcome_unknown')
        result = self.session.cancel(key='cancel')
        self.assertFalse(result['termination_confirmed'])
        self.assertFalse(result['frozen']['complete'])
        self.assertEqual(self.session.export()['used']['teacher_requests'], 1)
        self.assertNotIn('private', str(self.session.export()))

    def test_crash_after_dispatch_retains_reservation_without_resend(self):
        original = self.runner.execute
        def crash(action, target, **kwargs):
            if action == 'evaluate':
                self.runner.calls.append((action, kwargs))
                raise KeyboardInterrupt()
            return original(action, target, **kwargs)
        self.runner.execute = crash
        p = proposal(self.session)
        with self.assertRaises(KeyboardInterrupt):
            self.session.submit(p, key='crash')
        calls = len(self.runner.calls)
        self.session.close()
        self.session = Session.open(self.path, runner=self.runner, target={'kind': 'test'})
        self.assertEqual(self.session.submit(p, key='crash')['status'], 'outcome_unknown')
        self.assertEqual(calls, len(self.runner.calls))
        self.assertEqual(self.session.export()['used']['evaluations'], 2)

    def test_diagnostics_and_retention_gate_have_observable_costs(self):
        self.session.close()
        self.path = Path(self.tmp.name) / 'retention.db'
        config = declaration(development_retention_loss_max=.1)
        config['budgets']['diagnostics'] = 1
        self.session = Session.create(self.path, runner=self.runner, target={'kind': 'test'},
                                      declaration=config, development=[{'id': 'public'}],
                                      development_retention=[{'id': 'public-retention'}])
        receipt = self.session.submit(proposal(self.session, .5), key='one')
        self.assertFalse(receipt['selection']['accepted'])
        self.assertEqual(receipt['result']['retention_loss'], .5)
        self.assertEqual(self.session.export()['used']['evaluations'], 3)
        before = len(self.runner.calls)
        self.session.diagnose(key='probe', arm='teacher')
        self.session.diagnose(key='probe', arm='teacher')
        self.assertEqual(len(self.runner.calls), before + 1)
        with self.assertRaises(WeightlessError):
            self.session.diagnose(key='another', arm='teacher')
        self.assertEqual(len(self.runner.calls), before + 1)

    def test_deadline_survives_restart_without_blocking_report(self):
        origin = self.session.export()['origin']
        self.session.close()
        self.session = Session.open(self.path, runner=self.runner, target={'kind': 'test'})
        calls = len(self.runner.calls)
        with patch('wsm.session.time.time', return_value=origin + 61):
            with self.assertRaises(WeightlessError) as error:
                self.session.submit(proposal(self.session), key='late')
            self.assertEqual(error.exception.code, 'deadline_exhausted')
            self.assertFalse(self.session.freeze(key='stop')['complete'])
            self.assertEqual(self.session.export()['used']['attempts'], 0)
        self.assertEqual(calls, len(self.runner.calls))

    def test_clock_regression_refuses_dispatch(self):
        origin = self.session.export()['origin']
        with patch('wsm.session.time.time', return_value=origin - 1):
            with self.assertRaises(WeightlessError) as error:
                self.session.reserve_teacher({}, key='clock', arm='teacher')
            self.assertEqual(error.exception.code, 'clock_regressed')
        self.assertEqual(self.session.export()['used']['teacher_requests'], 0)

    def test_runner_reconstructs_full_checkpoint_without_evaluation(self):
        self.session.submit(proposal(self.session), key='one')
        self.session.freeze(key='stop')
        before = len([a for a, _ in self.runner.calls if a == 'evaluate'])
        proof = verify_session(self.session.export(), runner=self.runner, target={'kind': 'test'})
        self.assertEqual(proof['verified_candidates'], [_hash(1)])
        self.assertEqual(before, len([a for a, _ in self.runner.calls if a == 'evaluate']))

    def test_replay_detects_resealed_wrong_selection(self):
        self.session.submit(proposal(self.session), key='one')
        self.session.freeze(key='stop')
        bundle = self.session.export()
        bundle['operations'][0]['selection']['accepted'] = False
        bundle = _seal({k: v for k, v in bundle.items() if k != 'id'})
        with self.assertRaises(WeightlessError):
            replay_session(bundle)


if __name__ == '__main__':
    unittest.main()
