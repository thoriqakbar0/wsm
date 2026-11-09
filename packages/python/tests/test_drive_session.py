import http.server
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from wsm.binary64 import WeightlessError, create_checkpoint
from wsm.sandbox import forward_reservation
from wsm.session import replay_session, verify_session

ROOT = Path(__file__).resolve().parents[3]


def drive_module():
    path = ROOT / "scripts" / "drive_session.py"
    spec = importlib.util.spec_from_file_location("drive_session_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solved_id():
    graph = {"inputSize": 1, "nodes": [{"id": "out", "op": "affine", "input": "input", "weight": "w", "bias": "b"}], "output": "out"}
    return create_checkpoint(graph, {"w": {"shape": [1, 1], "values": [2]}, "b": {"shape": [1], "values": [3]}})["id"]


class DriveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.drive = drive_module()

    def tearDown(self):
        self.tmp.cleanup()

    def example(self):
        config = json.loads((ROOT / "examples" / "drive-session.json").read_text())
        config["output"] = str(self.root / "run")
        return config

    def test_example_selects_the_real_binary_edit(self):
        config = self.example()
        per_case = forward_reservation(config["target"], len(config["development"]), generate=True)
        self.assertGreaterEqual(config["declaration"]["budgets"]["forward_calls"], per_case * 2)
        report = self.drive.drive(config)
        self.assertEqual(report["mode"], "scripted_plumbing")
        self.assertEqual(report["runner"], "inprocess-binary64")
        self.assertIs(report["efficacy_claim"], False)
        self.assertTrue(report["frozen_complete"])
        bundle = json.loads((self.root / "run" / "evidence.json").read_text())
        self.assertEqual(replay_session(bundle)["arms"]["solver"]["checkpoint"], solved_id())
        self.assertNotEqual(bundle["initial"], solved_id())
        self.assertEqual(bundle["frozen"]["arms"]["solver"]["loss"], 0)
        runner = self.drive.InProcessBinaryRunner()
        self.assertEqual(verify_session(bundle, runner=runner, target=config["target"])["frozen"], bundle["frozen"]["id"])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "packages" / "python" / "src")
        destination = self.root / "exported"
        completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_checkpoint.py"),
                                    "--bundle", str(self.root / "run" / "evidence.json"),
                                    "--target", str(self.root / "run" / "target.json"),
                                    "--arm", "solver", "--destination", str(destination), "--max-bytes", "1000000"],
                                   check=True, env=env, capture_output=True, text=True)
        self.assertIn(solved_id(), completed.stdout)
        receipt = json.loads(completed.stdout)
        self.assertEqual(receipt["checkpoint"], solved_id())
        self.assertEqual(json.loads((destination / "checkpoint.json").read_text())["format"], "weightless.checkpoint.v1")
        self.assertIn("does not claim an improvement", (self.root / "run" / "driver.json").read_text())

    def test_rejects_hidden_defaults_and_records_invalid_edits(self):
        config = self.example()
        config["max_new_tokens"] = 32
        with self.assertRaises(WeightlessError) as limited:
            self.drive.drive(config)
        self.assertEqual(limited.exception.code, "unsupported")
        self.assertFalse((self.root / "run").exists())
        missing = self.example()
        del missing["declaration"]["budgets"]["forward_calls"]
        with self.assertRaises(WeightlessError) as budget:
            self.drive.drive(missing)
        self.assertEqual(budget.exception.code, "invalid_declaration")
        both = self.example()
        both["teacher"] = {"instructions": "Edit.", "timeout": 5, "max_output_tokens": 64,
                           "directory": str(self.root / "teacher"), "endpoint": "http://127.0.0.1/v1",
                           "model": "explicit", "api_key_env": "WSM_TEST_KEY"}
        with self.assertRaises(WeightlessError):
            self.drive.drive(both)
        smollm = self.example()
        smollm["target"] = {"kind": "smollm", "source": "/tmp/model", "revision": "other", "model": "other", "dtype": "float32"}
        smollm["runner"] = {"kind": "inprocess-binary64"}
        with self.assertRaises(WeightlessError) as pinned:
            self.drive.drive(smollm)
        self.assertEqual(pinned.exception.code, "unsupported_target")
        bad = self.example()
        bad["proposals"].append({"key": "bad", "edits": [{"tensor": "w", "index": 0, "value": float("nan")}]})
        bad["output"] = str(self.root / "bad-run")
        report = self.drive.drive(bad)
        self.assertTrue(report["frozen_complete"])
        self.assertTrue(any(item.get("error") == "invalid_request" for item in report["feedback"]))
        bundle = json.loads((self.root / "bad-run" / "evidence.json").read_text())
        self.assertEqual(bundle["frozen"]["arms"]["solver"]["checkpoint"], solved_id())

    def test_forward_exhaustion_does_not_apply_a_candidate(self):
        config = self.example()
        config["declaration"]["budgets"]["forward_calls"] = 3
        report = self.drive.drive(config)
        bundle = json.loads((self.root / "run" / "evidence.json").read_text())
        self.assertNotEqual(bundle["frozen"]["arms"]["solver"]["checkpoint"], solved_id())
        self.assertLessEqual(bundle["used"]["forward_calls"], 3)
        self.assertTrue(any(item.get("error") for item in report["feedback"]))

    def test_private_paths_are_not_inside_the_model_mount(self):
        config = self.example()
        config["target"] = {"kind": "smollm", "source": str(self.root),
                            "revision": self.drive._REVISION, "model": self.drive._MODEL, "dtype": "float32"}
        with self.assertRaises(WeightlessError) as error:
            self.drive.drive(config)
        self.assertEqual(error.exception.code, "invalid_request")
        self.assertFalse((self.root / "run").exists())

    def test_teacher_prompt_excludes_private_cases_and_the_credential(self):
        canary = "canary-private-do-not-leak"
        secret = "canary-secret-key"
        requests = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(payload)
                view = json.loads(payload["input"][0]["content"].splitlines()[-1])["view"]
                if len(requests) == 1:
                    command = {"action": "edit", "edits": [{"tensor": "w", "index": 0, "value": 2},
                                                             {"tensor": "b", "index": 0, "value": 3}]}
                else:
                    command = {"action": "finish"}
                event = {"type": "response.completed", "response": {"status": "completed", "model": payload["model"],
                         "tools": [], "tool_choice": "none", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(command)}]}],
                         "usage": {"input_tokens": 3, "output_tokens": 8, "total_tokens": 11}}}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            private = self.root / "private"
            private.mkdir()
            (private / "cases.json").write_text(json.dumps({
                "task": [{"id": canary, "input": [5], "target": [13]}],
                "retention": [{"id": canary + "-retention", "input": [1], "target": [0]}]}))
            config = self.example()
            config["output"] = str(self.root / "teacher-run")
            config["declaration"]["budgets"]["teacher_requests"] = 2
            config["declaration"]["budgets"]["final_evaluations"] = 2
            config["declaration"]["budgets"]["forward_calls"] = 16
            config["declaration"]["final_forward_calls"] = 8
            config["verifier"] = {"database": str(private / "verifier.db"), "dataset": "holdout", "cases": str(private / "cases.json")}
            del config["proposals"]
            config["teacher"] = {"instructions": "Return one JSON command.", "timeout": 5, "max_output_tokens": 256,
                                 "directory": str(self.root / "teacher-private"),
                                 "endpoint": f"http://127.0.0.1:{server.server_port}/v1",
                                 "model": "explicit-test-model", "api_key_env": "WSM_DRIVE_TEST_KEY"}
            os.environ["WSM_DRIVE_TEST_KEY"] = secret
            report = self.drive.drive(config)
        finally:
            os.environ.pop("WSM_DRIVE_TEST_KEY", None)
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(report["mode"], "fresh_agent")
        self.assertIs(report["efficacy_claim"], False)
        self.assertTrue(report["frozen_complete"])
        self.assertEqual(report["final_status"], "completed")
        self.assertEqual(len(requests), 2)
        published = "\n".join(path.read_text(errors="replace") for path in (self.root / "teacher-run").rglob("*") if path.is_file())
        published += "\n".join(json.dumps(item) for item in requests)
        published += "\n".join(path.read_text(errors="replace") for path in (self.root / "teacher-private").rglob("*") if path.is_file())
        self.assertNotIn(canary, published)
        self.assertNotIn(secret, published)
        bundle = json.loads((self.root / "teacher-run" / "evidence.json").read_text())
        self.assertEqual(bundle["frozen"]["arms"]["solver"]["checkpoint"], solved_id())
        self.assertEqual(bundle["final"]["arms"]["solver"]["conclusion"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
