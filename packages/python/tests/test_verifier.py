import tempfile
import unittest
from pathlib import Path

from wsm.binary64 import WeightlessError
from wsm.session import Session
from wsm.verifier import Verifier
from test_session import Runner, declaration, proposal


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runner = Runner()
        self.verifier = Verifier(self.root / 'private.db', runner=self.runner,
                                 datasets={'final': {'task': [{'id': 'PRIVATE-TASK', 'target': [1]}], 'retention': [{'id': 'PRIVATE-RETENTION', 'target': [1]}]}})
        self.sessions = []

    def tearDown(self):
        for s in self.sessions:
            s.close()
        self.verifier.close()
        self.tmp.cleanup()

    def session(self, name, value=1):
        s = Session.create(self.root / name, runner=self.runner, target={'kind': 'test'},
                           declaration=declaration(claim='confirmatory', criteria={'final_exact_min': 1, 'retention_loss_max': .1}),
                           development=[{'id': 'public'}], verifier=self.verifier)
        self.sessions.append(s)
        s.submit(proposal(s, value), key='one')
        s.freeze(key='stop')
        return s

    def test_private_bounded_report_and_same_claim_retry(self):
        s = self.session('a')
        receipt = s.finalize(key='final', dataset='final')
        calls = len(self.runner.calls)
        self.assertEqual(receipt['arms']['teacher']['conclusion'], 'supported')
        self.assertNotIn('PRIVATE', str(receipt))
        self.assertEqual(s.finalize(key='final', dataset='final'), receipt)
        self.assertEqual(calls, len(self.runner.calls))

    def test_inflight_same_claim_never_releases_a_second_outcome(self):
        session = self.session('concurrent')
        other = Verifier(self.root/'private.db', runner=self.runner, datasets=self.verifier.datasets)
        errors = []
        def race():
            try:
                other.grade(session.freeze(key='stop'), target={'kind':'test'}, key='final', dataset='final')
            except WeightlessError as error:
                errors.append(error.code)
        self.runner.hook = race
        before = len(self.runner.calls)
        try:
            result = session.finalize(key='final', dataset='final')
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(errors, ['verifier_in_progress', 'verifier_in_progress'])
            self.assertEqual(len(self.runner.calls), before+2)
            self.assertEqual(session.finalize(key='final', dataset='final'), result)
        finally:
            self.runner.hook = None
            other.close()
        self.assertEqual(self.verifier.status('final')['state'], 'exposed')
        other = self.session('b')
        calls = len(self.runner.calls)
        with self.assertRaises(WeightlessError):
            other.finalize(key='another', dataset='final')
        self.assertEqual(calls, len(self.runner.calls))

    def test_disclosure_unknown_is_distinct_and_retry_has_no_execution(self):
        s = self.session('a')
        frozen = s.freeze(key='stop')
        receipt = self.verifier.grade(frozen, target={'kind': 'test'}, key='final', dataset='final')
        self.assertEqual(self.verifier.status('final'), {'state': 'disclosure_unknown', 'operation': 'completed'})
        calls = len(self.runner.calls)
        self.assertEqual(self.verifier.grade(frozen, target={'kind': 'test'}, key='final', dataset='final'), receipt)
        self.assertEqual(calls, len(self.runner.calls))

    def test_unknown_completion_spends_dataset_without_retry(self):
        s = self.session('a')
        self.runner.fail = 'evaluate'
        result = s.finalize(key='final', dataset='final')
        self.assertEqual(result['status'], 'outcome_unknown')
        self.assertEqual(self.verifier.status('final')['operation'], 'outcome_unknown')
        calls = len(self.runner.calls)
        s.finalize(key='final', dataset='final')
        self.assertEqual(calls, len(self.runner.calls))

    def test_final_reservation_matches_dispatches_and_retry_does_not_charge(self):
        s = self.session('a')
        before = len([call for call in self.runner.calls if call[0] == 'evaluate'])
        s.finalize(key='final', dataset='final')
        self.assertEqual(s.export()['used']['final_evaluations'], 2)
        self.assertEqual(len([call for call in self.runner.calls if call[0] == 'evaluate']) - before, 2)
        s.finalize(key='final', dataset='final')
        self.assertEqual(s.export()['used']['final_evaluations'], 2)
        finals = [kwargs for action, kwargs in self.runner.calls if kwargs.get('purpose') == 'final']
        self.assertTrue(all(0 < call['timeout'] <= 60 for call in finals))

    def test_exhausted_final_budget_does_not_claim_or_dispatch(self):
        s = self.session('a')
        with s._transaction():
            state = s._state()
            state['used']['final_evaluations'] = state['declaration']['budgets']['final_evaluations']
            s._save(state)
        before = len(self.runner.calls)
        with self.assertRaises(WeightlessError) as error:
            s.finalize(key='final', dataset='final')
        self.assertEqual(error.exception.code, 'budget_exhausted')
        self.assertEqual(self.verifier.status('final')['state'], 'unused')
        self.assertEqual(before, len(self.runner.calls))

    def test_fabricated_exact_metrics_do_not_pass_private_oracle(self):
        s = self.session('a', .5)
        original = self.runner.execute
        def fabricate(action, target, **kwargs):
            result = original(action, target, **kwargs)
            if action == 'evaluate':
                result['metrics']['exact'] = result['metrics']['count']
            return result
        self.runner.execute = fabricate
        receipt = s.finalize(key='final', dataset='final')
        self.assertEqual(receipt['arms']['teacher']['task']['exact'], 0)
        self.assertEqual(receipt['arms']['teacher']['conclusion'], 'bounded_negative')

    def test_claim_persisted_before_crash_blocks_other_session(self):
        s = self.session('a')
        original = self.runner.execute
        def crash(action, target, **kwargs):
            if kwargs.get('purpose') == 'final':
                raise KeyboardInterrupt()
            return original(action, target, **kwargs)
        self.runner.execute = crash
        with self.assertRaises(KeyboardInterrupt):
            s.finalize(key='final', dataset='final')
        calls = len(self.runner.calls)
        receipt = s.finalize(key='final', dataset='final')
        self.assertEqual(receipt['status'], 'outcome_unknown')
        self.assertEqual(calls, len(self.runner.calls))
        self.assertEqual(self.verifier.status('final'), {'state': 'exposed', 'operation': 'outcome_unknown'})
        other = self.session('b')
        calls = len(self.runner.calls)
        with self.assertRaises(WeightlessError):
            other.finalize(key='other-final', dataset='final')
        self.assertEqual(calls, len(self.runner.calls))

    def test_finite_candidate_negative_is_not_impossibility(self):
        result = self.session('a', .5).finalize(key='final', dataset='final')
        arm = result['arms']['teacher']
        self.assertEqual(arm['conclusion'], 'bounded_negative')
        self.assertIn('no population generalization', arm['scope'])


if __name__ == '__main__':
    unittest.main()
