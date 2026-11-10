"""Cleanup checks task counts, including apps already marked stopped."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
spec = importlib.util.spec_from_file_location("wsm_spindle_watchdog", SCRIPTS / "spindle_watchdog.py")
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class CleanupTests(unittest.TestCase):
    def test_stopped_app_with_draining_gpu_tasks_is_not_confirmed(self):
        def rows(tasks):
            return subprocess.CompletedProcess([], 0, stdout=json.dumps([
                {"app_id": "ap-fixture", "description": "wsm-spindle-poc-trainer-test",
                 "state": "stopped", "tasks": str(tasks)}]))
        results = [rows(1), subprocess.CompletedProcess([], 0), rows(0)]
        with patch.object(watchdog.subprocess, "run", side_effect=results), patch.object(watchdog.time, "sleep"):
            self.assertEqual(watchdog.cleanup("wsm-spindle-poc"),
                             {"confirmed": True, "remaining_apps": [], "remaining_tasks": 0})
        with patch.object(watchdog.subprocess, "run", side_effect=[rows(1)]):
            result = watchdog.cleanup("wsm-spindle-poc", seconds=0)
            self.assertFalse(result["confirmed"])
            self.assertEqual(result["remaining_tasks"], 1)

    def test_stale_pid_identity_cannot_terminate_reused_process_group(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            with patch.object(watchdog, "child_groups", return_value={}):
                watchdog.stop_children(0, {child.pid: "stale-identity"})
                self.assertIsNone(child.poll())
                watchdog.stop_children(0, {child.pid: watchdog.process_identity(child.pid)})
            self.assertLess(child.wait(timeout=5), 0)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
