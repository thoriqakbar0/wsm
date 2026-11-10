"""Teacher code must cross the same isolation and selection boundaries as edits."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wsm import BubblewrapRunner, Session, WeightlessError, replay_session
from wsm.client import addition_config


@unittest.skipUnless(os.environ.get('WEIGHTLESS_LOCAL_SANDBOX') == '1',
                     'Set WEIGHTLESS_LOCAL_SANDBOX=1 for native teacher-code isolation')
class NativeCodeTeacherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = addition_config()
        config['declaration']['budgets']['code_executions'] = 4
        self.runner = BubblewrapRunner(timeout=10, memory_bytes=256 * 1024**2)
        self.session = Session.create(Path(self.directory.name) / 'session.db', runner=self.runner,
                                      target=config['target'], declaration=config['declaration'],
                                      development=config['development'])
        self.addCleanup(self.session.close)
        self.view = self.session.observe('researcher')
        self.inspections = [{'tensor': 'output.weight', 'indices': [0, 1]}]

    def run_code(self, source, key='code'):
        return self.session.run_code(source, self.inspections, key=key, arm='researcher',
                                     base=self.view['base'], observation=self.view['observation'])

    def test_code_edits_are_selected_once_and_replay_after_restart(self):
        source = "weights['output.weight'].update({0: 1, 1: -1})"
        result = self.run_code(source)
        self.assertTrue(result['selection']['accepted'])
        self.assertEqual(result['result']['evaluation']['metrics']['loss'], 0)
        self.assertEqual(result['key'], 'code')
        self.assertEqual(result, self.session.status('code', arm='researcher'))
        reopened = self.session.reopen()
        self.session.close()
        self.session = reopened
        self.addCleanup(reopened.close)
        before = self.session.export()['used']
        self.assertEqual(self.run_code(source), result)
        self.assertEqual(self.session.export()['used'], before)
        with self.assertRaises(WeightlessError) as error:
            self.run_code(source, 'stale')
        self.assertEqual(error.exception.code, 'stale_base')
        frozen = self.session.freeze(key='done')
        self.assertEqual(replay_session(self.session.export()), frozen)

    def test_interrupted_submission_resumes_without_reexecuting_python(self):
        source = "print('ran once'); weights['output.weight'].update({0:1,1:-1})"
        with patch.object(self.session, 'submit', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.run_code(source)
        self.assertEqual(self.session.status('code', arm='researcher')['status'], 'awaiting_submission')
        reopened = self.session.reopen()
        self.session.close()
        self.session = reopened
        self.addCleanup(reopened.close)
        result = self.run_code(source)
        self.assertEqual(result, self.session.status('code', arm='researcher'))
        self.assertTrue(result['selection']['accepted'])
        self.assertEqual(result['result']['stdout'], 'ran once\n')
        self.assertEqual(self.session.export()['used']['code_executions'], 1)
        self.assertEqual(self.session.export()['used']['inspections'], 1)

    def test_freeze_tracks_code_waiting_for_submission(self):
        with patch.object(self.session, 'submit', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.run_code("weights['output.weight'].update({0:1,1:-1})")
        frozen = self.session.freeze(key='stop-before-submit')
        self.assertIn('code', frozen['obligations'])
        self.assertFalse(frozen['complete'])
        self.assertEqual(replay_session(self.session.export()), frozen)

    def test_rejected_candidate_has_same_parent_status_and_output_shape(self):
        result = self.run_code("print('candidate'); weights['output.weight'][0] = -1")
        self.assertFalse(result['selection']['accepted'])
        self.assertEqual(result['key'], 'code')
        self.assertEqual(result['result']['stdout'], 'candidate\n')
        self.assertEqual(self.session.status('code', arm='researcher'), result)

    def test_unknown_child_is_visible_through_parent_and_is_not_retried(self):
        source = "print('executed'); weights['output.weight'][0] = 1"
        with patch.object(self.session, '_evaluate_development', side_effect=RuntimeError):
            result = self.run_code(source)
        self.assertEqual(result['key'], 'code')
        self.assertEqual(result['status'], 'outcome_unknown')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['result']['stdout'], 'executed\n')
        self.assertEqual(self.run_code(source), result)
        self.assertEqual(self.session.export()['used']['attempts'], 1)
        self.assertEqual(self.session.export()['used']['code_executions'], 1)
        frozen = self.session.freeze(key='stop-unknown')
        self.assertIn(result['submission'], frozen['obligations'])
        self.assertEqual(replay_session(self.session.export()), frozen)

    def test_freeze_finishes_retained_selection_and_updates_parent_projection(self):
        with patch.object(self.session, '_select'):
            result = self.run_code("weights['output.weight'].update({0:1,1:-1})")
        self.assertEqual(result['status'], 'awaiting_selection')
        frozen = self.session.freeze(key='finish-selection')
        self.assertTrue(frozen['complete'])
        self.assertTrue(self.session.status('code', arm='researcher')['selection']['accepted'])
        self.assertEqual(replay_session(self.session.export()), frozen)

    def test_submission_admission_failure_retains_parent_output(self):
        config = addition_config()
        config['declaration']['budgets'].update(code_executions=1, attempts=0)
        session = Session.create(Path(self.directory.name) / 'no-attempts.db', runner=self.runner,
                                 target=config['target'], declaration=config['declaration'],
                                 development=config['development'])
        self.addCleanup(session.close)
        view = session.observe('researcher')
        result = session.run_code("print('computed'); weights['output.weight'][0]=1", self.inspections,
                                  key='admission', arm='researcher', base=view['base'], observation=view['observation'])
        self.assertEqual(result, session.status('admission', arm='researcher'))
        self.assertEqual(result['status'], 'task_failure')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['error_code'], 'budget_exhausted')
        self.assertEqual(result['result']['stdout'], 'computed\n')
        self.assertEqual(session.observe('researcher')['blocking'], [])
        self.assertEqual(session.export()['used']['attempts'], 0)

    def test_prompt_harness_accepts_teacher_python_and_returns_selection_feedback(self):
        from test_harness import ScriptedTeacher
        from wsm.harness import run
        teacher = ScriptedTeacher([
            {'action': 'code', 'source': "weights['output.weight'].update({0:1,1:-1})",
             'inspections': self.inspections},
            {'action': 'finish'},
        ])
        report = run(self.session, teacher, prompt='Improve addition using Python',
                     arm='researcher', output=Path(self.directory.name) / 'harness')
        self.assertEqual(report['stop'], 'finished')
        self.assertIn('"accepted":true', teacher.prompts[1])
        self.assertEqual(self.session.export()['used']['code_executions'], 1)

    def test_code_cannot_read_private_host_data_or_reach_host_network(self):
        private = Path(self.directory.name) / 'private.txt'
        private.write_text('synthetic-private-canary')
        source = f'''import os,socket
assert not os.path.exists({str(private)!r})
assert not os.path.exists('/target')
assert not os.path.exists('/var/run/docker.sock')
assert 'WSM_TINKER_API_KEY' not in os.environ
s=socket.socket()
s.settimeout(.1)
assert s.connect_ex(('127.0.0.1',9)) != 0
weights['output.weight'].update({{0:1,1:-1}})
'''
        self.assertTrue(self.run_code(source)['selection']['accepted'])

    def test_code_cannot_add_addresses_or_nonfinite_values(self):
        for index, source in enumerate(("weights['output.weight'][2]=1", "weights['hidden.weight']={0:1}",
                                        "weights['output.weight'][0]=float('nan')")):
            result = self.run_code(source, key='invalid-' + str(index))
            self.assertEqual(result['status'], 'task_failure')
            self.assertEqual(self.session.observe('researcher')['base'], self.view['base'])
        self.assertEqual(self.session.export()['used']['attempts'], 0)

    def test_read_only_analysis_returns_bounded_output_without_candidate(self):
        result = self.run_code("print(sum(weights['output.weight'].values()))\nprint('x'*20000)")
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['result']['stdout'].startswith('0.0\n'))
        self.assertEqual(len(result['result']['stdout']), 16384)
        self.assertTrue(result['result']['stdout_truncated'])
        self.assertIsNone(result['result']['evaluation'])
        self.assertIsNone(result['submission'])
        self.assertEqual(self.session.status('code', arm='researcher'), result)
        self.assertEqual(self.session.export()['used']['attempts'], 0)

    def test_inspection_budget_rejection_does_not_reserve_code_or_block_the_arm(self):
        config = addition_config()
        config['declaration']['budgets'].update(code_executions=1, inspections=0)
        session = Session.create(Path(self.directory.name) / 'no-inspections.db', runner=self.runner,
                                 target=config['target'], declaration=config['declaration'],
                                 development=config['development'])
        self.addCleanup(session.close)
        view = session.observe('researcher')
        with self.assertRaises(WeightlessError) as error:
            session.run_code("weights['output.weight'][0]=1", self.inspections, key='no-budget',
                             arm='researcher', base=view['base'], observation=view['observation'])
        self.assertEqual(error.exception.code, 'budget_exhausted')
        self.assertEqual(session.export()['used']['code_executions'], 0)
        self.assertEqual(session.observe('researcher')['blocking'], [])
        proposal = {field: view[field] for field in ('session', 'arm', 'declaration', 'observation', 'base')}
        proposal['edits'] = [{'tensor': 'output.weight', 'index': 0, 'value': 1},
                             {'tensor': 'output.weight', 'index': 1, 'value': -1}]
        self.assertTrue(session.submit(proposal, key='still-usable')['selection']['accepted'])


if __name__ == '__main__':
    unittest.main()
