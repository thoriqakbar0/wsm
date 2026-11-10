"""The installed command supports Python editing and rejects old workflows."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wsm.cli import main


class CommandTests(unittest.TestCase):
    def test_removed_commands_are_rejected(self):
        for command in ('init', 'run', 'observe', 'submit', 'replay', 'export-model', 'teacher'):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main([command])
                self.assertEqual(error.exception.code, 2)

    def test_execute_returns_publication_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            cases = Path(temporary) / 'cases.json'
            cases.write_text('[]')
            for state, expected in [('published', 0), ('failure', 1)]:
                with self.subTest(state=state), patch('wsm.program.execute_program', return_value={'state': state}), contextlib.redirect_stdout(io.StringIO()) as output:
                    status = main(['execute', 'edit.py', '--model', 'model', '--cases', str(cases), '--output', 'output'])
                    self.assertEqual(status, expected)
                    self.assertEqual(json.loads(output.getvalue())['state'], state)

    def test_missing_cases_returns_failure(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(main(['execute', 'edit.py', '--model', 'model', '--cases', '/missing/cases.json', '--output', 'output']), 1)
        self.assertIn('error', json.loads(output.getvalue()))
