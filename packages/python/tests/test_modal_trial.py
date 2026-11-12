"""Local transport-failure simulations complement the live Modal mechanics proof."""

import json
import asyncio
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from wsm.modal_trial import ModalTrialRunner
from wsm.trial import TrialExecutionError
from wsm.errors import WeightlessError


class ExportProcess:
    def __init__(self, command, data):
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "weights"
        path.write_bytes(data)
        self.process = subprocess.Popen([sys.executable, *command[1:-2], str(path), command[-1]],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.returncode = None
        self.transferred = 0
        self.stdout = self.stream(self.process.stdout, count=True)
        self.stderr = self.stream(self.process.stderr)

    def stream(self, pipe, *, count=False):
        try:
            for chunk in iter(lambda: pipe.read(65536), b""):
                if count:
                    self.transferred += len(chunk)
                yield chunk
        finally:
            pipe.close()

    def wait(self):
        self.returncode = self.process.wait()
        self.directory.cleanup()

    def terminate(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()


class SandboxSimulation:
    def __init__(self, *, missing_response=False, foreign_checkpoint=False):
        self.filesystem = self
        self.missing_response, self.foreign_checkpoint = missing_response, foreign_checkpoint
        self.files = {}
        self.stdout = iter(["OS output before crash\n"])
        self.stderr = iter(["OS diagnostic\n"])
        self.returncode = 1 if missing_response else 0
        self.export_process = None
        self.terminated = False

    def write_text(self, data, path):
        self.files[path] = data.encode()

    def read_bytes(self, path):
        if path.endswith("model.safetensors"):
            raise AssertionError("Weights must never use a whole-file read")
        return self.files[path]

    def exec(self, *args, **kwargs):
        if "-c" in args:
            self.export_process = ExportProcess(args, self.files[args[-2]])
            return self.export_process
        request = json.loads(self.files[args[-2]])
        if not self.missing_response:
            checkpoint = "/foreign/checkpoint" if self.foreign_checkpoint else request["output"] + "/checkpoint"
            self.files[args[-1]] = json.dumps({"checkpoint": checkpoint}).encode()
            self.files[checkpoint + "/model.safetensors"] = b"retained tensor bytes"
        return self

    def wait(self):
        return self.returncode

    def terminate(self, *, wait):
        self.terminated = True
        if self.export_process is not None:
            self.export_process.terminate()


class ModalTransportTests(unittest.TestCase):
    def test_successful_process_requires_a_matching_export_receipt(self):
        payload = b"complete binary payload"
        valid = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for receipt in (None, {**valid, "bytes": len(payload) + 1}, {**valid, "sha256": "0" * 64}):
            with self.subTest(receipt=receipt), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                process = SimpleNamespace(stdout=iter([payload]),
                                          stderr=iter([] if receipt is None else [json.dumps(receipt).encode()]),
                                          returncode=0, wait=lambda: None, terminate=lambda: None)
                sandbox = SandboxSimulation()
                sandbox.export_process = process
                sandbox.exec = lambda *args, **kwargs: process
                runner = ModalTrialRunner(sandbox, command_seconds=5, artifacts=root / "weights",
                                          reserve=lambda *args: None)
                output = root / "final.safetensors"
                with self.assertRaises(WeightlessError) as raised:
                    runner.export_weights("/trial/edit-fixture", output, max_bytes=1024)
                self.assertEqual(raised.exception.code, "checkpoint_incomplete")
                self.assertFalse(output.exists())
                self.assertFalse(list(root.glob(".wsm-download-*")))
                self.assertTrue(sandbox.terminated)

    @unittest.skipUnless(importlib.util.find_spec("modal"), "Pinned Modal SDK unavailable")
    def test_sdk_stream_timeout_after_success_does_not_publish_a_prefix(self):
        from modal.exception import ExecTimeoutError
        from modal.io_streams import (_BytesStreamReaderThroughCommandRouter,
                                      _StreamReaderThroughSandboxExecCommandRouterParams)

        class Router:
            timed_out = False

            async def exec_stdio_read(self, *args, **kwargs):
                yield SimpleNamespace(data=b"partial")
                self.timed_out = True
                raise ExecTimeoutError("Synthetic buffered-output deadline")

        router = Router()

        def sdk_chunks():
            loop = asyncio.new_event_loop()
            reader = _BytesStreamReaderThroughCommandRouter(_StreamReaderThroughSandboxExecCommandRouterParams(
                file_descriptor=1, task_id="synthetic-task", object_id="synthetic-exec",
                command_router_client=router, deadline=None))
            iterator = reader.__aiter__()
            try:
                while True:
                    try:
                        yield loop.run_until_complete(anext(iterator))
                    except StopAsyncIteration:
                        break
            finally:
                loop.run_until_complete(iterator.aclose())
                loop.close()

        payload = b"partial plus the remaining checkpoint"
        receipt = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        process = SimpleNamespace(stdout=sdk_chunks(), stderr=iter([json.dumps(receipt).encode()]),
                                  returncode=0, wait=lambda: None, terminate=lambda: None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sandbox = SandboxSimulation()
            sandbox.export_process = process
            sandbox.exec = lambda *args, **kwargs: process
            runner = ModalTrialRunner(sandbox, command_seconds=5, artifacts=root / "weights", reserve=lambda *args: None)
            output = root / "final.safetensors"
            with self.assertRaises(WeightlessError) as raised:
                runner.export_weights("/trial/edit-fixture", output, max_bytes=1024)
            self.assertTrue(router.timed_out)
            self.assertEqual(raised.exception.code, "checkpoint_incomplete")
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".wsm-download-*")))

    def test_export_limit_is_enforced_before_transfer_and_exact_limit_succeeds(self):
        for size, limit in ((1024, 1024), (1025, 1024), (131072, 131072)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                sandbox = SandboxSimulation()
                payload = (bytes(range(256)) * (size // 256 + 1))[:size]
                sandbox.files["/trial/edit-fixture/model.safetensors"] = payload
                reservations = []
                runner = ModalTrialRunner(sandbox, command_seconds=5, artifacts=root / "weights",
                                          reserve=lambda operation, units: reservations.append(units["bytes"]))
                output = root / "final.safetensors"
                if size <= limit:
                    runner.export_weights("/trial/edit-fixture", output, max_bytes=limit)
                    self.assertEqual(output.read_bytes(), payload)
                    self.assertEqual(sandbox.export_process.transferred, size)
                else:
                    with self.assertRaises(WeightlessError) as raised:
                        runner.export_weights("/trial/edit-fixture", output, max_bytes=limit)
                    self.assertEqual(raised.exception.code, "checkpoint_limit")
                    self.assertEqual(sandbox.export_process.transferred, 0)
                    self.assertTrue(sandbox.terminated)
                    self.assertFalse(output.exists())
                self.assertEqual(reservations, [limit + 1 + 8 * 1024**2])
                self.assertFalse(list(root.glob(".wsm-download-*")))

    def test_controller_stops_a_producer_that_ignores_the_export_limit(self):
        class HostileProcess:
            returncode = 0
            stderr = iter(())
            transferred = 0

            @property
            def stdout(self):
                self.transferred += 1024
                yield b"x" * 1024
                self.transferred += 1
                yield b"x"
                raise AssertionError("Controller drained the oversized producer")

            def wait(self):
                pass

            def terminate(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sandbox = SandboxSimulation()
            producer = HostileProcess()
            sandbox.export_process = producer
            sandbox.exec = lambda *args, **kwargs: producer
            reservations = []
            runner = ModalTrialRunner(sandbox, command_seconds=5, artifacts=root / "weights",
                                      reserve=lambda operation, units: reservations.append(units["bytes"]))
            output = root / "final.safetensors"
            with self.assertRaises(WeightlessError) as raised:
                runner.export_weights("/trial/edit-fixture", output, max_bytes=1024)
            self.assertEqual(raised.exception.code, "checkpoint_limit")
            self.assertEqual(producer.transferred, 1025)
            self.assertLessEqual(producer.transferred, reservations[0])
            self.assertTrue(sandbox.terminated)
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".wsm-download-*")))

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
            limited = root / "limited.safetensors"
            with self.assertRaises(WeightlessError) as raised:
                runner.export_weights(result["checkpoint"], limited, max_bytes=1)
            self.assertEqual(raised.exception.code, "checkpoint_limit")
            self.assertFalse(limited.exists())
            with self.assertRaises(FileExistsError):
                runner.export_weights("/trial/edit-other", root / "final.safetensors", max_bytes=1024)
            self.assertEqual((root / "final.safetensors").read_bytes(), b"retained tensor bytes")
