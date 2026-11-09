import copy
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path

from wsm.binary64 import WeightlessError
from wsm.broker import (ACTIONS, Broker, FORMAT_COMMAND, FORMAT_RESPONSE,
                               parse_command, serve_once, templates_from_view)
from wsm.session import Session, _hash
from wsm.verifier import Verifier
from test_session import Runner, declaration


GRANT = ('observe', 'inspect', 'submit', 'diagnose', 'status', 'freeze', 'cancel', 'finalize', 'export')


def command(action, key, payload):
    return {'format': FORMAT_COMMAND, 'action': action, 'key': key, 'payload': payload}


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runner = Runner()
        budgets = declaration()
        budgets['budgets'] = dict(budgets['budgets'], diagnostics=1)
        self.verifier = Verifier(self.root / 'private.db', runner=self.runner, datasets={
            'final': {'task': [{'id': 'CANARY-PRIVATE', 'target': [1]}], 'retention': []}})
        self.session = Session.create(self.root / 'session.db', runner=self.runner, target={'kind': 'test'},
                                      declaration=budgets, development=[{'id': 'public'}],
                                      verifier=self.verifier, final_dataset='final')
        self.broker = Broker(self.session, arm='teacher', allowed_actions=GRANT)

    def tearDown(self):
        self.session.close()
        self.verifier.close()
        self.tmp.cleanup()

    def test_projection_offers_freeze_before_any_loss(self):
        view = self.session.observe('teacher')
        self.assertEqual(view['blocking'], [])
        self.assertEqual(view['actions'], ['status', 'inspect', 'submit', 'diagnose', 'freeze', 'cancel'])

    def test_developing_menu_is_copyable_and_omits_cancel(self):
        response = self.broker.handle(command('observe', 'wl:observe', {}))
        self.assertEqual(response['format'], FORMAT_RESPONSE)
        self.assertEqual(response['outcome'], 'ok')
        self.assertIsNone(response['error'])
        self.assertEqual([item['command']['action'] for item in response['next']],
                         ['inspect', 'submit', 'diagnose', 'freeze'])
        self.assertEqual(response['next'][1]['fill'], ['key', 'payload.edits'])
        self.assertEqual(response['next'][1]['command']['payload']['base'], response['result']['base'])
        self.assertNotIn('cancel', [item['command']['action'] for item in response['next']])
        names = {name for name, _role, _builder in ACTIONS}
        self.assertTrue(set(response['result']['actions']) <= names)

    def test_unfilled_submit_does_not_reserve_or_run(self):
        observed = self.broker.handle(command('observe', 'wl:observe', {}))
        draft = copy.deepcopy(observed['next'][1]['command'])
        before = len(self.runner.calls)
        refused = self.broker.handle(draft)
        self.assertEqual(refused['outcome'], 'error')
        self.assertEqual(len(self.runner.calls), before)
        self.assertIsNone(self.session._op(''))
        filled = copy.deepcopy(draft)
        filled['key'] = 'edit-1'
        filled['payload']['edits'] = [{'tensor': 'w', 'index': 0, 'value': 1}]
        accepted = self.broker.handle(filled)
        self.assertEqual(accepted['outcome'], 'ok')
        self.assertEqual(accepted['result']['status'], 'committed')
        self.assertEqual(accepted['result']['request_hash'], _hash({'action': 'submit', 'proposal': filled['payload']}))
        calls = len(self.runner.calls)
        again = self.broker.handle(filled)
        self.assertEqual(again['result'], accepted['result'])
        self.assertEqual(len(self.runner.calls), calls)
        self.assertEqual([item['command']['action'] for item in accepted['next']], ['observe', 'status'])

    def test_stale_base_keeps_the_key_free(self):
        observed = self.broker.handle(command('observe', 'wl:observe', {}))
        draft = copy.deepcopy(next(item['command'] for item in observed['next'] if item['command']['action'] == 'submit'))
        draft['key'] = 'edit-1'
        draft['payload']['edits'] = [{'tensor': 'w', 'index': 0, 'value': 1}]
        draft['payload']['base'] = 'injected-stale-base'
        before = len(self.runner.calls)
        refused = self.broker.handle(draft)
        self.assertEqual(refused['error'], {'code': 'stale_base', 'message': 'Proposal observation changed'})
        self.assertEqual([item['command']['action'] for item in refused['next']], ['observe'])
        self.assertEqual(len(self.runner.calls), before)
        self.assertIsNone(self.session._op('edit-1'))
        draft['payload']['base'] = observed['result']['base']
        accepted = self.broker.handle(draft)
        self.assertEqual(accepted['result']['status'], 'committed')

    def test_crash_after_reserve_replays_unknown_and_offers_only_freeze(self):
        observed = self.broker.handle(command('observe', 'wl:observe', {}))
        draft = copy.deepcopy(next(item['command'] for item in observed['next'] if item['command']['action'] == 'submit'))
        draft['key'] = 'edit-1'
        draft['payload']['edits'] = [{'tensor': 'w', 'index': 0, 'value': 1}]
        self.runner.fail = 'evaluate'
        unknown = self.broker.handle(draft)
        self.assertNotIn('private error', json.dumps(unknown))
        self.assertEqual(unknown['outcome'], 'unknown')
        self.assertEqual(unknown['result']['status'], 'outcome_unknown')
        self.assertEqual([item['command']['action'] for item in unknown['next']], ['freeze'])
        calls = len(self.runner.calls)
        self.runner.fail = None
        replayed = self.broker.handle(draft)
        self.assertEqual(replayed['result']['status'], 'outcome_unknown')
        self.assertEqual(len(self.runner.calls), calls)
        blocked = self.broker.handle(command('observe', 'wl:observe', {}))
        self.assertEqual(blocked['result']['blocking'], ['edit-1'])
        self.assertEqual([item['command']['action'] for item in blocked['next']], ['freeze', 'status'])
        frozen = self.broker.handle(command('freeze', 'wl:freeze', {}))
        self.assertFalse(frozen['result']['complete'])
        self.assertEqual(frozen['result']['obligations'], ['edit-1'])
        self.assertEqual(frozen['result']['id'], _hash({k: v for k, v in frozen['result'].items() if k != 'id'}))

    def test_keyboard_interrupt_leaves_a_replayable_reservation(self):
        observed = self.broker.handle(command('observe', 'wl:observe', {}))
        draft = copy.deepcopy(next(item['command'] for item in observed['next'] if item['command']['action'] == 'submit'))
        draft['key'] = 'crash'
        draft['payload']['edits'] = [{'tensor': 'w', 'index': 0, 'value': 1}]
        original = self.runner.execute
        def crash(action, target, **kwargs):
            if action == 'evaluate':
                self.runner.calls.append((action, kwargs))
                raise KeyboardInterrupt()
            return original(action, target, **kwargs)
        self.runner.execute = crash
        with self.assertRaises(KeyboardInterrupt):
            self.broker.handle(draft)
        calls = len(self.runner.calls)
        self.runner.execute = original
        replayed = self.broker.handle(draft)
        self.assertEqual(replayed['outcome'], 'unknown')
        self.assertEqual(len(self.runner.calls), calls)

    def test_changed_key_content_conflicts_without_another_run(self):
        self.test_unfilled_submit_does_not_reserve_or_run()
        calls = len(self.runner.calls)
        observed = self.broker.handle(command('observe', 'wl:observe', {}))
        draft = copy.deepcopy(next(item['command'] for item in observed['next'] if item['command']['action'] == 'submit'))
        draft['key'] = 'edit-1'
        draft['payload']['edits'] = [{'tensor': 'w', 'index': 0, 'value': 0}]
        refused = self.broker.handle(draft)
        self.assertEqual(refused['error']['code'], 'idempotency_conflict')
        self.assertEqual(len(self.runner.calls), calls)

    def test_finish_order_after_a_complete_freeze(self):
        self.test_unfilled_submit_does_not_reserve_or_run()
        frozen = self.broker.handle(command('freeze', 'wl:freeze', {}))
        self.assertTrue(frozen['result']['complete'])
        self.assertEqual([item['command']['action'] for item in frozen['next']], ['observe'])
        menu = self.broker.handle(command('observe', 'wl:observe', {}))
        self.assertEqual([item['command']['action'] for item in menu['next']], ['finalize', 'export'])
        final = self.broker.handle(command('finalize', 'wl:finalize', {}))
        self.assertEqual(final['outcome'], 'ok')
        self.assertNotIn('CANARY-PRIVATE', json.dumps(final))
        exported = self.broker.handle(command('export', 'wl:export', {}))
        self.assertEqual(exported['result']['format'], 'weightless.evidence.v1')
        self.assertNotIn('CANARY-PRIVATE', json.dumps(exported))

    def test_reserved_prefix_never_reaches_dispatch(self):
        before = len(self.runner.calls)
        refused = self.broker.handle(command('submit', 'wl:edit', {'arm': 'teacher'}))
        self.assertEqual(refused['error']['code'], 'invalid_request')
        self.assertEqual(len(self.runner.calls), before)
        self.assertIsNone(self.session._op('wl:edit'))

    def test_all_inspection_and_recovery_targets_are_discoverable(self):
        menu = templates_from_view({'actions': ['inspect', 'status'], 'parameters': {'a': {}, 'b': {}},
                                    'base': 'checkpoint', 'blocking': ['first', 'second']}, GRANT)
        self.assertEqual([item['command']['payload']['tensor'] for item in menu[:2]], ['a', 'b'])
        self.assertEqual([item['command']['payload']['key'] for item in menu[2:]], ['first', 'second'])

    def test_recovery_templates_respect_operator_grants(self):
        broker = Broker(self.session, arm='teacher', allowed_actions=('inspect',))
        response = broker.handle(command('submit', 'edit', {}))
        self.assertEqual(response['error']['code'], 'unauthorized')
        self.assertEqual(response['next'], [])
        refused = self.broker.handle(command('diagnose', 'wl:freeze', {}))
        self.assertEqual(refused['error']['code'], 'invalid_request')
        self.assertIsNone(self.session._op('wl:freeze'))

    def test_duplicate_json_key_and_unknown_field_are_refusals(self):
        raw = b'{"format":"weightless.command.v1","action":"observe","action":"submit","key":"k","payload":{}}'
        with self.assertRaises(WeightlessError) as error:
            parse_command(raw)
        self.assertEqual(error.exception.code, 'invalid_request')
        response = self.broker.handle(b'{"format":"weightless.command.v1","action":"observe","key":"k","payload":{},"extra":1}')
        self.assertEqual(response['error']['code'], 'invalid_request')

    def test_serve_once_writes_one_document(self):
        stdin = io.BytesIO(_json_bytes(command('observe', 'wl:observe', {})))
        stdout = io.BytesIO()
        serve_once(self.broker, stdin, stdout)
        parsed = json.loads(stdout.getvalue())
        self.assertEqual(parsed['command'], {'action': 'observe', 'key': 'wl:observe'})

    def test_broker_does_not_take_a_database_path(self):
        parameters = inspect.signature(Broker.__init__).parameters
        self.assertNotIn('path', parameters)
        self.assertNotIn('database', parameters)
        self.assertNotIn('socket', parameters)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
