"""Local transport-failure simulations complement the live Modal mechanics proof."""

import json
from pathlib import Path
import tempfile
import unittest

from wsm.modal_trial import ModalTrialRunner
from wsm.trial import TrialExecutionError


class SandboxSimulation:
    def __init__(self, *, missing_response=False, foreign_checkpoint=False):
        self.filesystem = self
        self.missing_response, self.foreign_checkpoint = missing_response, foreign_checkpoint
        self.files = {}
        self.stdout = iter(["OS output before crash\n"])
        self.stderr = iter(["OS diagnostic\n"])
        self.returncode = 1 if missing_response else 0

    def write_text(self, data, path):
        self.files[path] = data.encode()

    def read_bytes(self, path):
        return self.files[path]

    def exec(self, *args, **kwargs):
        request = json.loads(self.files[args[-2]])
        if not self.missing_response:
            checkpoint = "/foreign/checkpoint" if self.foreign_checkpoint else request["output"] + "/checkpoint"
            self.files[args[-1]] = json.dumps({"checkpoint": checkpoint}).encode()
            self.files[checkpoint + "/model.safetensors"] = b"retained tensor bytes"
        return self

    def wait(self):
        return self.returncode


class ModalTransportTests(unittest.TestCase):
    def test_missing_response_and_spoofed_checkpoint_preserve_process_output(self):
        for mode in ("missing_response", "foreign_checkpoint"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                runner = ModalTrialRunner(SandboxSimulation(**{mode: True}), command_seconds=10,
                                          artifacts=Path(temporary) / "weights", reserve=lambda *args: None)
                with self.assertRaises(TrialExecutionError) as raised:
                    runner.execute("def edit(model): pass", "/target", [])
                self.assertIn("OS output before crash", raised.exception.observation["stdout"])
                self.assertIn("OS diagnostic", raised.exception.observation["stderr"])

    def test_export_uses_retained_bytes_after_sandbox_state_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sandbox = SandboxSimulation()
            runner = ModalTrialRunner(sandbox, command_seconds=10, artifacts=root / "weights",
                                      reserve=lambda *args: None)
            result = runner.execute("def edit(model): pass", "/target", [])
            artifact = Path(result["artifact"]["path"])
            sandbox.files[result["checkpoint"] + "/model.safetensors"] = b"later agent mutation"
            runner.export_weights(result["checkpoint"], root / "final.safetensors")
            self.assertEqual(artifact.read_bytes(), b"retained tensor bytes")
            self.assertEqual((root / "final.safetensors").read_bytes(), b"retained tensor bytes")
