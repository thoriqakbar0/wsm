"""Exercise command transport through a subprocess, including native resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CommandTests(unittest.TestCase):
    def call(self, *args):
        return subprocess.run([sys.executable, "-m", "wsm", *map(str, args)],
                              capture_output=True, text=True, timeout=30)

    def test_machine_errors_have_empty_stdout_and_structured_stderr(self):
        for args, code in [(('--unknown',), 'usage'),
                           (('--json', 'submit'), 'usage'),
                           ((), 'terminal_required')]:
            with self.subTest(args=args):
                result = self.call(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')
                self.assertEqual(json.loads(result.stderr)['error']['code'], code)

    def test_model_capabilities_and_help_need_no_workspace_or_optional_runtime(self):
        result = self.call('models', '--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertFalse(record['teacher']['automatic_fallback'])
        self.assertIn('huggingface', [item['kind'] for item in record['targets']])
        help = self.call('--help')
        self.assertEqual(help.returncode, 0, help.stderr)
        self.assertIn('stage-model', help.stdout)

    @unittest.skipUnless(os.environ.get('WEIGHTLESS_LOCAL_SANDBOX') == '1',
                         'Set WEIGHTLESS_LOCAL_SANDBOX=1 for native command execution')
    def test_native_cli_resume_stale_rejection_and_independent_final_grading(self):
        with tempfile.TemporaryDirectory(prefix='wsm-command-') as root:
            root = Path(root)
            workspace, private = root / 'public', root / 'private'

            def run(*args, expected=0):
                result = self.call('--workspace', workspace, '--verifier', private / 'verifier.json', '--json', *args)
                self.assertEqual(result.returncode, expected, result.stderr)
                if expected:
                    self.assertEqual(result.stdout, '')
                    return json.loads(result.stderr)
                self.assertEqual(result.stderr, '')
                return json.loads(result.stdout)

            created = self.call('--workspace', workspace, '--json', 'init', '--private-dir', private)
            self.assertEqual(created.returncode, 0, created.stderr)
            malformed = root / 'malformed-request.json'
            malformed.write_text(json.dumps({'arm': 'researcher', 'command': []}))
            self.assertEqual(run('resume', malformed, expected=2)['error']['code'], 'invalid_request')
            edits = root / 'edits.json'
            edits.write_text(json.dumps([{'tensor': 'output.weight', 'index': 0, 'value': 1},
                                         {'tensor': 'output.weight', 'index': 1, 'value': -1}]))
            proposal_path = root / 'proposal.json'
            proposal = run('propose', '--edits', edits, '--output', proposal_path)
            from wsm import create_example_verifier
            other_verifier = create_example_verifier(root / 'other-private')
            mismatch = run('observe', '--verifier', other_verifier, expected=2)
            self.assertEqual(mismatch['error']['code'], 'verifier_mismatch')
            private_file = private / 'verifier.json'
            original = private_file.read_text()
            changed = json.loads(original)
            changed['cases']['task'][0]['target'] = [999]
            private_file.write_text(json.dumps(changed))
            try:
                mismatch = run('observe', expected=2)
                self.assertEqual(mismatch['error']['code'], 'verifier_mismatch')
            finally:
                private_file.write_text(original)
            accepted = run('submit', proposal_path, '--key', 'one')
            self.assertTrue(accepted['selection']['accepted'])
            self.assertEqual(run('submit', proposal_path, '--key', 'one'), accepted)
            saved = next((workspace / 'requests').glob('*.json'))
            self.assertEqual(run('resume', saved), accepted)
            stale = dict(proposal, base='stale-checkpoint')
            proposal_path.write_text(json.dumps(stale))
            rejected = run('submit', proposal_path, '--key', 'stale', expected=2)
            self.assertEqual(rejected['error']['code'], 'stale_base')
            self.assertEqual(run('observe')['remaining']['attempts'], 7)
            frozen = run('freeze', '--key', 'freeze')
            self.assertTrue(frozen['complete'])
            final = run('finalize', '--key', 'final')
            self.assertEqual(final['arms']['researcher']['task']['exact'], 4)
            bundle = root / 'bundle.json'
            exported = run('export', '--output', bundle)
            self.assertEqual(exported['used']['attempts'], 1)
            self.assertEqual(run('replay', bundle), frozen)
            self.assertIn('verified_candidates', run('verify', bundle))


if __name__ == '__main__':
    unittest.main()
