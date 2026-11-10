import os
import signal
import subprocess
import sys
import unittest

from wsm.coding_tui import stop


@unittest.skipUnless(os.name == 'posix', 'Terminal process groups require POSIX')
class CodingTerminalTests(unittest.TestCase):
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
