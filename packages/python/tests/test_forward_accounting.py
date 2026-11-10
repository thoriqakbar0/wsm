import json
import tempfile
import unittest
from pathlib import Path

from wsm.binary64 import WeightlessError
from wsm.session import Session, replay_session, _seal
from wsm.verifier import Verifier
from test_session import Runner, declaration, proposal


class MeasuredRunner(Runner):
    def __init__(self):
        super().__init__()
        self.usage = 'valid'

    def execute(self, action, target, **kwargs):
        if action == 'infer':
            self.calls.append((action, kwargs))
            if self.fail == 'infer':
                raise RuntimeError('uncertain target operation')
            query = kwargs['query']
            return {'checkpoint': query['checkpoint'], 'sequences': [],
                    'prompt_logprobs': [None, -1.0],
                    'usage': {'forward_calls': 1, 'forward_limit': kwargs['forward_limit']}}
        result = super().execute(action, target, **kwargs)
        if action == 'evaluate' and self.usage != 'missing':
            observed = len(kwargs['cases']) if self.usage == 'valid' else kwargs['forward_limit'] + 1
            result['usage'] = {'forward_calls': observed, 'forward_limit': kwargs['forward_limit']}
        return result


class ForwardAccountingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runner = MeasuredRunner()
        self.sessions = []
        self.verifier = None

    def tearDown(self):
        for session in self.sessions:
            session.close()
        if self.verifier:
            self.verifier.close()
        self.tmp.cleanup()

    def session(self, *, budget=300, retention=True, arms=None, envelope=None):
        config = declaration(arms=arms or ['teacher'])
        config['budgets'].update(forward_calls=budget, diagnostics=2, evaluations=20)
        if envelope is not None:
            config['final_forward_calls'] = envelope
        session = Session.create(self.root / ('session-' + str(len(self.sessions))), runner=self.runner,
                                 target={'kind': 'smollm'}, declaration=config,
                                 development=[{'id': 'public-1'}, {'id': 'public-2'}],
                                 development_retention=[{'id': str(i)} for i in range(3)] if retention else None,
                                 verifier=self.verifier)
        self.sessions.append(session)
        return session

    def test_all_development_costs_are_reserved_and_observed_separately(self):
        session = self.session()
        request = proposal(session)
        first = session.submit(request, key='first')
        self.assertEqual(first['costs']['forward_calls'], 119)
        self.assertEqual(first['result']['usage']['forward_calls'], 2)
        self.assertEqual(first['result']['baseline']['usage']['forward_calls'], 2)
        self.assertEqual(first['result']['retention_usage']['forward_calls'], 3)
        self.assertEqual(session.observe('teacher')['history'][0]['retention_usage']['forward_calls'], 3)
        calls = len(self.runner.calls)
        self.assertEqual(session.submit(request, key='first'), first)
        self.assertEqual(len(self.runner.calls), calls)
        session.diagnose(key='diagnostic', arm='teacher')
        session.diagnose(key='diagnostic', arm='teacher')
        self.assertEqual(session.submit(proposal(session, .5), key='second')['costs']['forward_calls'], 85)
        self.assertEqual(session.export()['used']['forward_calls'], 238)
        session.freeze(key='stop')
        replay_session(session.export())

    def test_sdk_inference_is_budgeted_idempotent_and_never_retries_unknown_work(self):
        session = self.session(budget=1, retention=False)
        query = {'checkpoint': session.observe('teacher')['base'], 'tokens': [1, 2],
                 'max_tokens': 0, 'temperature': 0, 'prompt_logprobs': True, 'stop': []}
        first = session.infer(query, key='score', arm='teacher')
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(first['costs'], {'evaluations': 1, 'forward_calls': 1})
        calls = len(self.runner.calls)
        self.assertEqual(session.infer(query, key='score', arm='teacher'), first)
        self.assertEqual(len(self.runner.calls), calls)
        with self.assertRaises(WeightlessError) as error:
            session.infer(query, key='exhausted', arm='teacher')
        self.assertEqual(error.exception.code, 'budget_exhausted')
        self.assertEqual(len(self.runner.calls), calls)
        second = self.session(budget=10, retention=False)
        query['checkpoint'] = second.observe('teacher')['base']
        self.runner.fail = 'infer'
        unknown = second.infer(query, key='unknown', arm='teacher')
        self.assertEqual(unknown['status'], 'outcome_unknown')
        calls = len(self.runner.calls)
        self.assertEqual(second.infer(query, key='unknown', arm='teacher'), unknown)
        self.assertEqual(len(self.runner.calls), calls)
        frozen = second.freeze(key='stop')
        self.assertFalse(frozen['complete'])
        self.assertIn('unknown', frozen['obligations'])
        replay_session(second.export())

    def test_insufficient_reservation_prevents_any_dispatch_or_charge(self):
        session = self.session(budget=118)
        before = len(self.runner.calls)
        with self.assertRaises(WeightlessError) as error:
            session.submit(proposal(session), key='too-large')
        self.assertEqual(error.exception.code, 'budget_exhausted')
        self.assertEqual(len(self.runner.calls), before)
        self.assertEqual(session.export()['used']['forward_calls'], 0)
        self.assertEqual(session.export()['used']['attempts'], 0)

    def test_diagnostic_cannot_spend_reserved_generation_allowance(self):
        session = self.session(budget=119)
        session.submit(proposal(session), key='first')
        before = len(self.runner.calls)
        with self.assertRaises(WeightlessError) as error:
            session.diagnose(key='diagnostic', arm='teacher')
        self.assertEqual(error.exception.code, 'budget_exhausted')
        self.assertEqual(len(self.runner.calls), before)
        self.assertEqual(session.export()['used']['diagnostics'], 0)
        self.assertEqual(session.export()['used']['forward_calls'], 119)

    def test_unknown_and_invalid_usage_keep_the_full_reservation(self):
        for mode in ('unknown', 'missing', 'over-limit'):
            with self.subTest(mode=mode):
                session = self.session()
                self.runner.fail = 'evaluate' if mode == 'unknown' else None
                self.runner.usage = mode
                request = proposal(session)
                result = session.submit(request, key='uncertain')
                self.assertEqual(result['status'], 'outcome_unknown' if mode == 'unknown' else 'invalid_response')
                before = len(self.runner.calls)
                session.submit(request, key='uncertain')
                self.assertEqual(len(self.runner.calls), before)
                self.assertEqual(session.export()['used']['forward_calls'], 119)

    def private_verifier(self):
        self.verifier = Verifier(self.root / 'private.db', runner=self.runner,
                                 datasets={'final': {'task': [{'id': 'private-1', 'target': [1]},
                                                           {'id': 'private-2', 'target': [1]}],
                                                     'retention': [{'id': 'private-r', 'target': [1]}]}})

    def freeze_arms(self, session):
        for arm in ('teacher', 'control'):
            session.submit(proposal(session, arm=arm), key='edit-' + arm)
        session.freeze(key='stop')

    def test_padded_final_envelope_covers_all_arms_and_keeps_actual_usage_private(self):
        self.private_verifier()
        session = self.session(budget=256, retention=False, arms=['teacher', 'control'], envelope=120)
        self.freeze_arms(session)
        before = len(self.runner.calls)
        receipt = session.finalize(key='final', dataset='final')
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(len(self.runner.calls), before + 4)
        self.assertEqual(session.export()['used']['forward_calls'], 256)
        self.assertNotIn('usage', str(receipt))
        claim = json.loads(self.verifier.db.execute('SELECT body FROM claims').fetchone()[0])
        self.assertEqual(sum(usage['forward_calls'] for arm in claim['usage'].values() for usage in arm.values()), 6)
        session.finalize(key='final', dataset='final')
        self.assertEqual(len(self.runner.calls), before + 4)
        self.assertEqual(session.export()['used']['forward_calls'], 256)
        final_calls = [kwargs for _, kwargs in self.runner.calls if kwargs.get('purpose') == 'final']
        self.assertEqual([call['forward_limit'] for call in final_calls], [34, 17, 34, 17])

    def test_final_preflight_rejects_small_envelope_before_private_dispatch(self):
        self.private_verifier()
        session = self.session(retention=False, arms=['teacher', 'control'], envelope=101)
        self.freeze_arms(session)
        before = len(self.runner.calls)
        for _ in range(2):
            with self.assertRaises(WeightlessError) as error:
                session.finalize(key='final', dataset='final')
            self.assertEqual(error.exception.code, 'budget_exhausted')
        self.assertEqual(len(self.runner.calls), before)
        self.assertEqual(self.verifier.status('final')['state'], 'unused')
        self.assertEqual(session.export()['used']['forward_calls'], 237)

    def test_final_reservation_exhaustion_does_not_claim_private_dataset(self):
        self.private_verifier()
        session = self.session(budget=255, retention=False, arms=['teacher', 'control'], envelope=120)
        self.freeze_arms(session)
        before = len(self.runner.calls)
        with self.assertRaises(WeightlessError) as error:
            session.finalize(key='final', dataset='final')
        self.assertEqual(error.exception.code, 'budget_exhausted')
        self.assertEqual(len(self.runner.calls), before)
        self.assertEqual(self.verifier.status('final')['state'], 'unused')
        self.assertEqual(session.export()['used']['forward_calls'], 136)

    def test_budgeted_verifier_requires_predeclared_final_envelope(self):
        self.private_verifier()
        with self.assertRaises(WeightlessError) as error:
            self.session()
        self.assertEqual(error.exception.code, 'invalid_declaration')

    def test_replay_rejects_a_changed_final_envelope(self):
        session = self.session(retention=False, envelope=120)
        session.submit(proposal(session), key='edit')
        session.freeze(key='freeze')
        bundle = session.export()
        bundle['frozen'] = _seal({k: (121 if k == 'final_forward_calls' else v)
                                  for k, v in bundle['frozen'].items() if k != 'id'})
        bundle = _seal({k: v for k, v in bundle.items() if k != 'id'})
        with self.assertRaises(WeightlessError) as error:
            replay_session(bundle)
        self.assertEqual(error.exception.code, 'invalid_artifact')

    def test_replay_rejects_a_reduced_final_reservation(self):
        self.private_verifier()
        session = self.session(budget=256, retention=False, arms=['teacher', 'control'], envelope=120)
        self.freeze_arms(session)
        session.finalize(key='final', dataset='final')
        bundle = session.export()
        operation = next(op for op in bundle['operations'] if op['kind'] == 'finalize')
        operation['costs']['forward_calls'] = 0
        bundle['journal'][operation['dispatch_seq'] - 1]['costs']['forward_calls'] = 0
        bundle['used']['forward_calls'] -= 120
        bundle = _seal({k: v for k, v in bundle.items() if k != 'id'})
        with self.assertRaises(WeightlessError) as error:
            replay_session(bundle)
        self.assertEqual(error.exception.code, 'invalid_artifact')


if __name__ == '__main__':
    unittest.main()
