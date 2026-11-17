import json
from itertools import chain, repeat
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from wsm.coding_tui import stop
from wsm.tui import _main


@unittest.skipUnless(os.name == 'posix', 'Terminal process groups require POSIX')
class CodingTerminalTests(unittest.TestCase):
    def test_rejects_stale_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = root / "edit.py"
            program.write_text("def edit(model): pass\n")
            cases = root / "cases.json"
            cases.write_text("[]")
            output = root / "previous-run"
            output.mkdir()
            receipt = output / "run.json"
            published = json.dumps({"state": "published"})
            receipt.write_text(published)
            screen = MagicMock()
            screen.getmaxyx.return_value = (24, 80)
            screen.get_wch.side_effect = chain(["p"], repeat("q"))
            answers = [str(program), str(root), str(cases), str(output), "cpu"]

            with (
                patch("wsm.tui.curses.curs_set"),
                patch("wsm.tui.prompt", side_effect=answers),
                patch("wsm.tui.show_result") as show,
            ):
                _main(screen)

            self.assertEqual(show.call_args.args[1:], (
                "Execution error", {"error": "Output directory must be unused"},
            ))
            self.assertEqual(receipt.read_text(), published)

    def test_cleanup_terminates_a_program_that_ignores_sigterm(self):
        process = subprocess.Popen(
            [sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
             'print("ready",flush=True); time.sleep(60)'],
            stdout=subprocess.PIPE, start_new_session=True)
        try:
            self.assertEqual(process.stdout.readline(), b'ready\n')
            stop(process, signal.SIGTERM)
            self.assertEqual(process.returncode, -signal.SIGKILL)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            process.stdout.close()
